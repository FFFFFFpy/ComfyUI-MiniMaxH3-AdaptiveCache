"""CPU regression tests for cache ownership, not a real CUDA allocator test."""
import contextlib
import importlib.util
import io
import sys
import threading
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mmh3_compiler_cache_test", ROOT / "adaptive_cache.py")
cache = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cache
SPEC.loader.exec_module(cache)


class GraphState:
    """Model the public pause API; no allocator or GPU behavior is simulated."""
    def __init__(self):
        self.local = threading.local()
        self.calls = []

    @property
    def recording(self):
        return not getattr(self.local, "paused", False)

    @contextlib.contextmanager
    def pause(self):
        if not self.recording:
            raise AssertionError("Nested pause would prematurely resume AIMDO's boolean pause")
        self.calls.append((threading.get_ident(), "pause"))
        self.local.paused = True
        try:
            yield
        finally:
            self.local.paused = False
            self.calls.append((threading.get_ident(), "resume"))


class AllocationOrigins(TorchDispatchMode):
    """Observe whether new, real CPU tensor storage is born inside the scope."""
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.origins = {}

    @staticmethod
    def storages(value):
        flat, _ = tree_flatten(value)
        return {t.untyped_storage()._cdata for t in flat if isinstance(t, torch.Tensor)}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        inputs = self.storages((args, kwargs or {}))
        out = func(*args, **(kwargs or {}))
        for storage in self.storages(out) - inputs:
            self.origins[storage] = self.state.recording
        return out

    def recorded(self, tensor):
        return self.origins[tensor.untyped_storage()._cdata]


class CompilerCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.state = GraphState()
        self.prefetch = types.ModuleType("comfy.model_prefetch")
        self.prefetch.pause_malloc_graph = self.state.pause
        self.prefetch.cleanup_malloc_graph = Mock(side_effect=AssertionError("compiler must stay enabled"))
        self.flags = types.SimpleNamespace(disable_comfy_compiler=False, disable_cuda_graphs=False)
        self.prefetch.args = self.flags
        self.modules = patch.dict(sys.modules, {"comfy.model_prefetch": self.prefetch})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.addCleanup(cache.PrefetchSkipContext.clear)
        self.executed = []
        self.blocks = [object() for _ in range(6)]
        self.controller = self.make_controller()
        self.tracker = AllocationOrigins(self.state)
        self.tracker.__enter__()
        self.addCleanup(self.tracker.__exit__, None, None, None)

    def make_controller(self, **overrides):
        values = dict(cache_depth=0.5, sigma_threshold=1.0, start_percent=0.0,
                      end_percent=1.0, max_consecutive=1, quality_guard="audio_video",
                      content_ceiling=0.5, cache_device="cpu")
        values.update(overrides)
        return cache.AdaptiveTailCacheController(self.blocks, cache.AdaptiveCacheSettings(**values))

    def args(self, hidden=None, sigma=1.0, lane=0):
        if hidden is None:
            hidden = torch.ones(10, 8)
        return {"img": hidden, "mod_segments": [(0, 2, 0), (2, 4, 0), (4, 6, 2), (6, 10, 0)],
                "transformer_options": {"sigmas": torch.tensor([sigma]),
                    "sample_sigmas": torch.tensor([1.0, 0.8, 0.6, 0.4, 0.0]),
                    "cond_or_uncond": [lane], "uuids": [str(lane)]}}

    def run_call(self, sigma=1.0, hidden=None, lane=0, fail_at=None):
        args = self.args(hidden, sigma, lane)
        for index in range(len(self.blocks)):
            def original(a, i=index):
                self.assertTrue(self.state.recording, "Original block must stay in compiler scope")
                self.executed.append(i)
                if fail_at == i:
                    raise RuntimeError("block failure")
                return {"img": a["img"] + 0.1}
            args["img"] = cache.BlockInterceptor(self.controller, index)(args, {"original_block": original})["img"]
        return args["img"]

    def warm_call(self):
        args = self.args()
        self.controller.begin_call(args)
        self.controller.after_warm(args["img"], args)
        return args, self.controller._active_lane()

    def test_warm_snapshot_and_probes_are_not_graph_owned(self):
        _, lane = self.warm_call()
        for t in lane.warm_snapshot + lane.current_probes + lane.previous_probes:
            self.assertFalse(self.tracker.recorded(t), "Persistent cache tensor is graph-owned")
        self.assertTrue(self.state.recording)

    def test_finish_stores_residual_outside_graph(self):
        self.run_call()
        lane = next(iter(self.controller.lanes.values()))
        self.assertTrue(lane.residual.tensors)
        for t in lane.residual.tensors:
            self.assertFalse(self.tracker.recorded(t))
        self.assertTrue(self.state.recording)

    def test_full_cache_refresh_preserves_model_scope_and_target_values(self):
        results = [self.run_call(s) for s in (1.0, 0.8, 0.6)]
        lane = next(iter(self.controller.lanes.values()))
        self.assertEqual((lane.stats.full_calls, lane.stats.cache_calls), (2, 1))
        self.assertEqual(len(self.executed), 6 + 3 + 6)
        for result in results:
            torch.testing.assert_close(result[4:], torch.full_like(result[4:], 1.6))
        torch.testing.assert_close(results[1][:4], torch.full_like(results[1][:4], 1.3))
        self.assertTrue(self.state.calls, "Cache operations never left the allocation graph")
        self.assertFalse(self.flags.disable_comfy_compiler)
        self.assertFalse(self.flags.disable_cuda_graphs)
        self.prefetch.cleanup_malloc_graph.assert_not_called()

    def test_refresh_reuses_residual_storage(self):
        self.run_call()
        lane = next(iter(self.controller.lanes.values()))
        pointers = [t.data_ptr() for t in lane.residual.tensors]
        self.run_call(0.8)
        self.run_call(0.6)
        self.assertEqual(pointers, [t.data_ptr() for t in lane.residual.tensors])
        self.assertTrue(all(not self.tracker.recorded(t) for t in lane.residual.tensors))

    def test_content_rejection_still_executes_full_model(self):
        self.run_call()
        self.run_call(0.8, torch.ones(10, 8) * 100)
        lane = next(iter(self.controller.lanes.values()))
        self.assertEqual(lane.stats.content_rejects, 1)
        self.assertEqual(len(self.executed), 12)
        self.assertTrue(all(not self.tracker.recorded(t) for t in lane.residual.tensors))

    def test_begin_prefetch_runs_outside_graph(self):
        self.run_call()
        lane = next(iter(self.controller.lanes.values()))
        states = []
        with patch.object(lane.residual, "begin_prefetch", side_effect=lambda _: states.append(self.state.recording)):
            self.controller.begin_call(self.args(sigma=0.8))
        self.assertEqual(states, [False])
        self.assertTrue(self.state.recording)

    def test_residual_materialization_runs_outside_graph(self):
        self.run_call()
        lane = next(iter(self.controller.lanes.values()))
        materialize = lane.residual.materialize
        states = []
        def tracked(device):
            states.append(self.state.recording)
            return materialize(device)
        with patch.object(lane.residual, "materialize", side_effect=tracked):
            self.run_call(0.8)
        self.assertEqual(states, [False])

    def test_finish_releases_warm_snapshot_outside_graph(self):
        args, lane = self.warm_call()
        states = []
        weakref.finalize(lane.warm_snapshot[0], lambda: states.append(self.state.recording))
        self.controller.finish_call(args["img"])
        self.assertEqual(states, [False])

    def test_abort_releases_warm_snapshot_and_restores_state(self):
        _, lane = self.warm_call()
        states = []
        weakref.finalize(lane.warm_snapshot[0], lambda: states.append(self.state.recording))
        cache.PrefetchSkipContext.set_skipped([id(self.blocks[-1])])
        self.controller.abort_call()
        self.assertEqual(states, [False])
        self.assertIsNone(self.controller._active_lane())
        self.assertFalse(cache.PrefetchSkipContext.should_skip(self.blocks[-1]))
        self.assertTrue(self.state.recording)

    def test_lifecycle_reset_releases_residual_outside_graph(self):
        for name in ("start_sampling_run", "end_sampling_run"):
            with self.subTest(method=name):
                self.run_call()
                lane = next(iter(self.controller.lanes.values()))
                states = []
                weakref.finalize(lane.residual.tensors[0], lambda: states.append(self.state.recording))
                with contextlib.redirect_stdout(io.StringIO()):
                    getattr(self.controller, name)()
                self.assertEqual(states, [False])
                self.assertFalse(self.controller.lanes)
                self.assertTrue(self.state.recording)

    def test_shape_change_rebuilds_cache_outside_graph(self):
        self.run_call()
        self.run_call(0.8, torch.ones(10, 16))
        lane = next(iter(self.controller.lanes.values()))
        self.assertEqual(lane.stats.full_calls, 2)
        for t in lane.residual.tensors:
            self.assertEqual(t.shape[1], 16)
            self.assertFalse(self.tracker.recorded(t))

    def test_sigma_reset_and_separate_lanes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.run_call()
            self.run_call(0.8)
            self.run_call(1.0)
            self.run_call(1.0, lane=1)
        self.assertEqual(len(self.controller.lanes), 2)
        for lane in self.controller.lanes.values():
            self.assertEqual(lane.stats.full_calls, 1)
            self.assertTrue(all(not self.tracker.recorded(t) for t in lane.residual.tensors))

    def test_block_exception_is_not_swallowed(self):
        with self.assertRaisesRegex(RuntimeError, "block failure"):
            self.run_call(fail_at=4)
        self.assertTrue(self.state.recording)
        self.assertIsNone(self.controller._active_lane())

    def test_cache_operation_exception_resumes_graph(self):
        args = self.args()
        self.controller.begin_call(args)
        with patch.object(self.controller, "_make_probes", side_effect=RuntimeError("probe failure")):
            with self.assertRaisesRegex(RuntimeError, "probe failure"):
                self.controller.after_warm(args["img"], args)
        self.assertTrue(self.state.calls)
        self.assertTrue(self.state.recording)
        self.assertEqual([v for _, v in self.state.calls], ["pause", "resume"] * 2)

    def test_legacy_comfy_without_pause_api(self):
        del self.prefetch.pause_malloc_graph
        result = self.run_call()
        torch.testing.assert_close(result, torch.full_like(result, 1.6))
        self.assertFalse(self.state.calls)

    def test_standalone_without_comfy(self):
        with patch.dict(sys.modules):
            sys.modules.pop("comfy.model_prefetch", None)
            result = self.run_call()
        torch.testing.assert_close(result, torch.full_like(result, 1.6))
        self.assertFalse(self.state.calls)

    def test_nested_cache_operations_do_not_resume_early(self):
        operation = getattr(cache, "_cache_tensor_operation", lambda f: f)
        states = []
        @operation
        def inner():
            states.append(self.state.recording)
        @operation
        def outer():
            states.append(self.state.recording)
            inner()
            states.append(self.state.recording)
        outer()
        self.assertEqual(states, [False, False, False])
        self.assertEqual([v for _, v in self.state.calls], ["pause", "resume"])

    def test_scope_is_thread_local(self):
        operation = getattr(cache, "_cache_tensor_operation", lambda f: f)
        barrier = threading.Barrier(2, timeout=5)
        states, errors = [], []
        @operation
        def body():
            barrier.wait()
            states.append(self.state.recording)
            barrier.wait()
        def run():
            try:
                body()
                states.append(self.state.recording)
            except BaseException as exc:
                errors.append(exc)
        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertFalse(errors)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertEqual(sorted(states), [False, False, True, True])
        self.assertEqual(len(self.state.calls), 4)

    def test_pause_failure_propagates_without_disabling_compiler(self):
        self.prefetch.pause_malloc_graph = Mock(side_effect=RuntimeError("allocator unavailable"))
        with self.assertRaisesRegex(RuntimeError, "allocator unavailable"):
            self.controller.begin_call(self.args())
        self.assertFalse(self.flags.disable_comfy_compiler)
        self.prefetch.cleanup_malloc_graph.assert_not_called()

    def test_pause_api_is_resolved_after_late_import(self):
        factory = self.prefetch.pause_malloc_graph
        del self.prefetch.pause_malloc_graph
        self.controller.begin_call(self.args())
        self.prefetch.pause_malloc_graph = factory
        self.controller.abort_call()
        self.assertEqual([v for _, v in self.state.calls], ["pause", "resume"])

    def test_prefetched_buffers_record_consumer_stream_before_release(self):
        store = cache.ResidualStore()
        store.location = "cpu"
        store.tensors = [object()]
        store._event = Mock()
        store._prefetched = [Mock(), Mock()]
        consumer = Mock()
        manager = Mock()
        manager.attach_mock(consumer, "consumer")
        for index, tensor in enumerate(store._prefetched):
            manager.attach_mock(tensor, f"tensor{index}")
        expected = store._prefetched
        with patch.object(store, "_is_cuda", return_value=True), patch.object(torch.cuda, "current_stream", return_value=consumer):
            self.assertIs(store.materialize(torch.device("cuda:0")), expected)
        self.assertEqual(manager.mock_calls[0][0], "consumer.wait_event")
        for tensor in expected:
            tensor.record_stream.assert_called_once_with(consumer)
        self.assertEqual([call[0] for call in manager.mock_calls],
                         ["consumer.wait_event", "tensor0.record_stream", "tensor1.record_stream"])


    def load_sol_compat(self):
        name = "mmh3_compiler_sol_testpkg"
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT)]
        spec = importlib.util.spec_from_file_location(name + ".sol_attn_compat", ROOT / "sol_attn_compat.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {name: package, name + ".adaptive_cache": cache}):
            spec.loader.exec_module(module)
        return module

    def test_morton_transition_cleans_cache_outside_graph(self):
        sol = self.load_sol_compat()
        self.run_call()
        lane = next(iter(self.controller.lanes.values()))
        states = []
        weakref.finalize(lane.residual.tensors[0], lambda: states.append(self.state.recording))
        args = self.args(sigma=0.8)
        args["transformer_options"]["sol_morton"] = True
        def original(a):
            self.assertTrue(self.state.recording)
            return {"img": a["img"] + 0.1}
        with contextlib.redirect_stdout(io.StringIO()):
            for index in range(6):
                args["img"] = sol.SolAttnCompatibleBlockInterceptor(self.controller, index)(
                    args, {"original_block": original})["img"]
        self.assertEqual(states, [False])
        self.assertFalse(self.controller.lanes)
        self.assertTrue(self.state.recording)
        torch.testing.assert_close(args["img"], torch.full_like(args["img"], 1.6))

    def test_sampling_exception_cleans_cache_outside_graph(self):
        sol = self.load_sol_compat()
        states = []
        def execute():
            _, lane = self.warm_call()
            weakref.finalize(lane.warm_snapshot[0], lambda: states.append(self.state.recording))
            raise RuntimeError("sampling interrupted")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "sampling interrupted"):
                sol.SolAttnCompatibleSamplingLifecycleWrapper(self.controller)(execute)
        self.assertEqual(states, [False])
        self.assertFalse(self.controller.lanes)
        self.assertTrue(self.state.recording)


if __name__ == "__main__":
    unittest.main()
