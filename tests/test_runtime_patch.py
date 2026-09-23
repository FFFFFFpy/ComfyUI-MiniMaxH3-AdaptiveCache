"""Prefetch API regression tests; no ComfyUI, torch, or GPU is required."""

import importlib.util
from functools import wraps
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "mmh3_runtime_patch_testpkg"
ORIGINAL_ATTR = "_mmh3_adaptive_cache_original_prefetch_queue_pop"
CALLBACK_ATTR = "_mmh3_adaptive_cache_should_skip"


def legacy_guard(original):
    """Reproduce the v1 guard's signature and closure without importing it."""
    def guarded_prefetch_queue_pop(queue, device, module):
        return original(queue, device, module)

    guarded_prefetch_queue_pop.__qualname__ = (
        "install_prefetch_guard.<locals>.guarded_prefetch_queue_pop"
    )
    return guarded_prefetch_queue_pop


class RuntimePatchTests(unittest.TestCase):
    def setUp(self):
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        self.mp = types.ModuleType("comfy.model_prefetch")
        management = types.ModuleType("comfy.model_management")
        comfy.model_prefetch = self.mp
        comfy.model_management = management
        self.stream = object()
        management.current_stream = lambda device: self.stream

        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(ROOT)]
        cache = types.ModuleType(PACKAGE + ".adaptive_cache")

        class PrefetchSkipContext:
            local = threading.local()

            @classmethod
            def should_skip(cls, module):
                return id(module) in getattr(cls.local, "skipped", set())

            @classmethod
            def set_skipped(cls, modules):
                cls.local.skipped = set(modules)

        cache.PrefetchSkipContext = PrefetchSkipContext
        self.context = PrefetchSkipContext
        modules = {
            "comfy": comfy,
            "comfy.model_prefetch": self.mp,
            "comfy.model_management": management,
            PACKAGE: package,
            PACKAGE + ".adaptive_cache": cache,
        }
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

        self.calls = []
        self.events = []
        self.prefetched = []
        self.result = object()
        self.device = object()
        self.block = object()

        def cleanup(module, modules):
            self.events.append(("cleanup", module, modules))

        self.mp.cleanup_prefetched_modules = cleanup

        def original(queue, device, module, dtype=None, core=None,
                     enable_graph=False, generator=None, malloc_scope=None,
                     **future):
            self.calls.append((queue, device, module, dtype, core,
                               enable_graph, generator, malloc_scope, future))
            if malloc_scope is not None:
                self.events.append(("scope", malloc_scope if module is not None else None))
            if queue is not None:
                consumed = queue.pop(0)
                if isinstance(consumed, tuple):
                    stream, (old_module, modules) = consumed
                    if stream is not None:
                        stream.wait_stream(self.stream)
                    if modules is not None:
                        self.mp.cleanup_prefetched_modules(old_module, modules)
                if queue and queue[0] is not None:
                    self.prefetched.append(queue[0])
                    queue[0] = (None, (module, []))
            if core is not None:
                core()
            return self.result

        self.original = original
        self.mp.prefetch_queue_pop = original
        self.runtime = self.load_runtime()

    def load_runtime(self):
        spec = importlib.util.spec_from_file_location(
            PACKAGE + ".runtime_patch", ROOT / "runtime_patch.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def install(self, skip=True):
        if skip:
            self.context.set_skipped([id(self.block)])
        self.runtime.install_prefetch_guard()
        return self.mp.prefetch_queue_pop

    def test_normal_path_forwards_all_arguments_and_result(self):
        guard = self.install(skip=False)
        core_calls = []
        core = lambda: core_calls.append(True)
        generator = object()
        queue = [None, self.block, None]
        result = guard(queue, self.device, self.block, "dtype", core, True,
                       generator, "block", future_flag="value")
        self.assertIs(result, self.result)
        self.assertEqual(self.calls[0], (queue, self.device, self.block, "dtype",
                         core, True, generator, "block", {"future_flag": "value"}))
        self.assertEqual(core_calls, [True])

    def test_skipped_malloc_scope_advances_once_without_prefetch(self):
        guard = self.install()
        queue = [None, self.block, None]
        self.assertIs(guard(queue, self.device, self.block, malloc_scope="block"), self.result)
        self.assertEqual(queue, [None, None])
        self.assertEqual(self.events, [("scope", "block")])
        self.assertEqual(self.prefetched, [])
        self.assertEqual(len(self.calls), 1)
        self.assertIsNone(self.calls[0][0])

    def test_scope_precedes_cleanup_and_stream_waits_are_preserved(self):
        guard = self.install()
        waits = []
        offload = types.SimpleNamespace(wait_stream=lambda stream: waits.append(stream))
        previous, previous_modules, pending_modules = object(), [object()], [object()]
        queue = [(offload, (previous, previous_modules)),
                 (offload, (self.block, pending_modules)), None]
        guard(queue, self.device, self.block, malloc_scope="block")
        self.assertEqual(self.events, [("scope", "block"),
                         ("cleanup", previous, previous_modules),
                         ("cleanup", self.block, pending_modules)])
        self.assertEqual(waits, [self.stream, self.stream])
        self.assertEqual(queue, [None, None])

    def test_legacy_single_argument_cleanup(self):
        guard = self.install()
        cleaned = []
        self.mp.cleanup_prefetched_modules = lambda modules: cleaned.append(modules)
        first, second = [object()], [object()]
        queue = [(None, (object(), first)), (None, (self.block, second)), None]
        guard(queue, self.device, self.block)
        self.assertEqual(cleaned, [first, second])

    def test_empty_module_lists_are_cleaned_but_none_is_not(self):
        guard = self.install()
        queue = [(None, (object(), None)), (None, (self.block, [])), None]
        guard(queue, self.device, self.block)
        self.assertEqual(self.events, [("cleanup", self.block, [])])

    def test_cleanup_typeerror_is_not_retried_with_a_different_signature(self):
        guard = self.install()
        calls = []
        def broken_cleanup(module, modules):
            calls.append((module, modules))
            raise TypeError("cleanup body failed")
        self.mp.cleanup_prefetched_modules = broken_cleanup
        queue = [(None, (self.block, [])), self.block, None]
        with self.assertRaisesRegex(TypeError, "cleanup body failed"):
            guard(queue, self.device, self.block)
        self.assertEqual(len(calls), 1)

    def test_no_queue_preserves_scope_and_core(self):
        guard = self.install()
        calls = []
        result = guard(None, self.device, self.block, malloc_scope="block",
                       core=lambda: calls.append(True))
        self.assertIs(result, self.result)
        self.assertEqual(calls, [True])
        self.assertEqual(self.events, [("scope", "block")])

    def test_core_and_graph_options_fall_back_without_dropping_callbacks(self):
        guard = self.install()
        calls = []
        queue = [None, self.block, None]
        guard(queue, self.device, self.block, core=lambda: calls.append(True),
              enable_graph=True, malloc_scope="block")
        self.assertIs(self.calls[0][0], queue)
        self.assertEqual(calls, [True])
        self.assertEqual(self.prefetched, [self.block])

    def test_positional_extensions_use_stock_path(self):
        guard = self.install()
        queue = [None, self.block, None]
        guard(queue, self.device, self.block, "dtype", None, False, None, "block")
        self.assertIs(self.calls[0][0], queue)
        self.assertEqual(self.calls[0][3], "dtype")
        self.assertEqual(self.calls[0][7], "block")

    def test_unknown_keyword_extensions_use_stock_path(self):
        guard = self.install()
        queue = [None, self.block, None]
        guard(queue, self.device, self.block, malloc_scope="block", future_flag=123)
        self.assertIs(self.calls[0][0], queue)
        self.assertEqual(self.calls[0][-1], {"future_flag": 123})

    def test_dtype_and_malloc_scope_keep_prefetch_suppressed(self):
        guard = self.install()
        queue = [None, self.block, None]
        guard(queue, self.device, self.block, dtype="dtype", malloc_scope="block")
        self.assertIsNone(self.calls[0][0])
        self.assertEqual(self.calls[0][3], "dtype")
        self.assertEqual(self.prefetched, [])

    def test_legacy_original_without_optional_parameters(self):
        calls = []
        def legacy_original(queue, device, module):
            calls.append((queue, device, module))
        self.mp.prefetch_queue_pop = legacy_original
        guard = self.install()
        queue = [None, self.block, None]
        guard(queue, self.device, self.block)
        self.assertEqual(calls, [(None, self.device, self.block)])
        self.assertEqual(queue, [None, None])

    def test_unsupported_malloc_scope_does_not_mutate_queue(self):
        def legacy_original(queue, device, module):
            pass
        self.mp.prefetch_queue_pop = legacy_original
        guard = self.install()
        queue = [None, self.block, None]
        before = queue[:]
        with self.assertRaisesRegex(TypeError, "malloc_scope"):
            guard(queue, self.device, self.block, malloc_scope="block")
        self.assertEqual(queue, before)

    def test_v1_guard_is_replaced_even_with_stale_v2_marker(self):
        self.mp.prefetch_queue_pop = legacy_guard(self.original)
        setattr(self.mp, "_mmh3_adaptive_cache_prefetch_guard_v2", True)
        guard = self.install(skip=False)
        guard(None, self.device, self.block, malloc_scope="block")
        self.assertEqual(len(self.calls), 1)
        self.assertIs(getattr(self.mp, ORIGINAL_ATTR), self.original)

    def test_nested_legacy_guards_do_not_survive_in_saved_original(self):
        inner = legacy_guard(self.original)
        self.mp.prefetch_queue_pop = legacy_guard(inner)
        setattr(self.mp, ORIGINAL_ATTR, inner)
        guard = self.install(skip=False)
        guard(None, self.device, self.block, malloc_scope="block")
        self.assertEqual(len(self.calls), 1)
        self.assertIs(getattr(self.mp, ORIGINAL_ATTR), self.original)

    def test_reinstallation_repairs_replaced_function_despite_v3_marker(self):
        first = self.install(skip=False)
        self.mp.prefetch_queue_pop = legacy_guard(self.original)
        self.runtime.install_prefetch_guard()
        self.assertIsNot(self.mp.prefetch_queue_pop, first)
        self.mp.prefetch_queue_pop(None, self.device, self.block, malloc_scope="block")
        self.assertEqual(len(self.calls), 1)

    def test_reload_refreshes_callback_without_double_wrapping(self):
        guard = self.install(skip=False)
        self.runtime = self.load_runtime()
        self.runtime.PrefetchSkipContext = types.SimpleNamespace(should_skip=lambda module: True)
        self.runtime.install_prefetch_guard()
        self.assertIs(self.mp.prefetch_queue_pop, guard)
        queue = [None, self.block, None]
        guard(queue, self.device, self.block, malloc_scope="block")
        self.assertEqual(self.prefetched, [])

    def test_old_install_markers_prevent_legacy_rewrapping(self):
        guard = self.install()
        for version in (1, 2):
            self.assertTrue(getattr(self.mp, f"_mmh3_adaptive_cache_prefetch_guard_v{version}"))
        self.runtime.install_prefetch_guard()
        self.assertIs(self.mp.prefetch_queue_pop, guard)

    def test_third_party_wrapper_is_not_discarded_for_saved_stock_ref(self):
        calls = []
        original = self.original
        def third_party(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)
        self.mp.prefetch_queue_pop = third_party
        setattr(self.mp, ORIGINAL_ATTR, self.original)
        guard = self.install(skip=False)
        guard(None, self.device, self.block, malloc_scope="block")
        self.assertEqual(calls, [True])

    def test_functools_wrapped_third_party_wrapper_is_preserved(self):
        first = self.install(skip=False)
        calls = []
        @wraps(first)
        def third_party(*args, **kwargs):
            calls.append(True)
            return first(*args, **kwargs)
        self.mp.prefetch_queue_pop = third_party
        self.runtime.install_prefetch_guard()
        self.mp.prefetch_queue_pop(None, self.device, self.block, malloc_scope="block")
        self.assertEqual(calls, [True])
        self.assertEqual(len(self.calls), 1)

    def test_cyclic_own_guard_fails_with_restart_instruction(self):
        current = legacy_guard(self.original)
        setattr(current, ORIGINAL_ATTR, current)
        self.mp.prefetch_queue_pop = current
        with self.assertRaisesRegex(RuntimeError, "restart ComfyUI"):
            self.runtime.install_prefetch_guard()

    def test_missing_callback_and_final_queue_flush_use_stock_path(self):
        guard = self.install()
        delattr(self.mp, CALLBACK_ATTR)
        queue = [None, self.block, None]
        guard(queue, self.device, self.block, malloc_scope="block")
        guard(queue, self.device, None, malloc_scope="block")
        self.assertEqual(self.events[-1], ("cleanup", self.block, []))
        self.assertIn(("scope", None), self.events)
        self.assertEqual(queue, [None])

    def test_skip_context_is_thread_local(self):
        guard = self.install()
        queue = [None, self.block, None]
        errors = []
        def worker():
            try:
                guard(queue, self.device, self.block, malloc_scope="block")
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(self.prefetched, [self.block])

    def test_install_log_identifies_version_and_source_file(self):
        with self.assertLogs(level="INFO") as captured:
            self.install()
        message = "\n".join(captured.output)
        self.assertIn("prefetch guard v3", message)
        self.assertIn("runtime_patch.py", message)


if __name__ == "__main__":
    unittest.main()
