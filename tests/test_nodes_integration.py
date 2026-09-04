import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_plugin_package():
    name = "mmh3_adaptive_cache_testpkg"
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakePatcher:
    def __init__(self, model):
        self.model = model
        self.replacements = {}
        self.wrappers = {}

    def clone(self):
        return FakePatcher(self.model)

    def set_model_patch_replace(self, patch, family, name, index):
        self.replacements[(family, name, index)] = patch

    def add_wrapper_with_key(self, family, key, wrapper):
        self.wrappers[(family, key)] = wrapper


class NodeIntegrationTests(unittest.TestCase):
    def setUp(self):
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        model_management = types.ModuleType("comfy.model_management")
        model_prefetch = types.ModuleType("comfy.model_prefetch")
        patcher_extension = types.ModuleType("comfy.patcher_extension")

        self.prefetched = []

        def original_prefetch(queue, device, module):
            if queue is None:
                return
            queue.pop(0)
            if queue and queue[0] is not None:
                current = queue[0]
                self.prefetched.append(id(current))
                queue[0] = (None, (current, None))

        model_prefetch.prefetch_queue_pop = original_prefetch
        model_prefetch.cleanup_prefetched_modules = lambda modules: None
        model_management.current_stream = lambda device: None

        class WrappersMP:
            OUTER_SAMPLE = "outer_sample"

        patcher_extension.WrappersMP = WrappersMP
        comfy.model_management = model_management
        comfy.model_prefetch = model_prefetch
        comfy.patcher_extension = patcher_extension

        sys.modules["comfy"] = comfy
        sys.modules["comfy.model_management"] = model_management
        sys.modules["comfy.model_prefetch"] = model_prefetch
        sys.modules["comfy.patcher_extension"] = patcher_extension
        self.comfy = comfy

    def test_node_registers_all_block_and_lifecycle_hooks(self):
        plugin = load_plugin_package()
        nodes = sys.modules[plugin.__name__ + ".nodes"]

        MiniMaxH3Model = type("MiniMaxH3Model", (), {})
        inner = MiniMaxH3Model()
        inner.blocks = [object() for _ in range(50)]
        base = types.SimpleNamespace(diffusion_model=inner)
        source = FakePatcher(base)

        patched, = nodes.MiniMaxH3AdaptiveCache().patch(source, "balanced", "auto")
        self.assertEqual(len(patched.replacements), 50)
        self.assertTrue(any(key[0] == "outer_sample" for key in patched.wrappers))

        cache_module = sys.modules[plugin.__name__ + ".adaptive_cache"]
        skipped_block = inner.blocks[20]
        cache_module.PrefetchSkipContext.set_skipped([id(skipped_block)])
        queue = [None, skipped_block, None]
        self.comfy.model_prefetch.prefetch_queue_pop(queue, None, skipped_block)
        self.assertIsNone(queue[0])

    def test_prefetch_guard_supports_current_two_argument_cleanup(self):
        plugin = load_plugin_package()
        cache_module = sys.modules[plugin.__name__ + ".adaptive_cache"]
        runtime_module = sys.modules[plugin.__name__ + ".runtime_patch"]

        cleaned = []

        def cleanup_prefetched_modules(module, modules):
            cleaned.append((module, modules))

        self.comfy.model_prefetch.cleanup_prefetched_modules = cleanup_prefetched_modules
        runtime_module.install_prefetch_guard()

        previous_block = object()
        skipped_block = object()
        previous_modules = [object()]
        skipped_modules = [object()]
        cache_module.PrefetchSkipContext.set_skipped([id(skipped_block)])

        queue = [
            (None, (previous_block, previous_modules)),
            (None, (skipped_block, skipped_modules)),
            None,
        ]
        self.comfy.model_prefetch.prefetch_queue_pop(queue, None, skipped_block)

        self.assertEqual(
            cleaned,
            [
                (previous_block, previous_modules),
                (skipped_block, skipped_modules),
            ],
        )
        self.assertIsNone(queue[0])

    def test_cache_hit_suppresses_tail_prefetch(self):
        import torch

        plugin = load_plugin_package()
        cache_module = sys.modules[plugin.__name__ + ".adaptive_cache"]
        runtime_module = sys.modules[plugin.__name__ + ".runtime_patch"]
        runtime_module.install_prefetch_guard()

        class AddBlock:
            def __init__(self, amount):
                self.amount = amount

            def __call__(self, hidden):
                return hidden + self.amount

        blocks = [AddBlock(0.01 * (i + 1)) for i in range(10)]
        settings = cache_module.AdaptiveCacheSettings(
            cache_depth=0.6,
            sigma_threshold=1.0,
            start_percent=0.0,
            end_percent=1.0,
            max_consecutive=2,
            quality_guard="off",
            cache_device="cpu",
        )
        controller = cache_module.AdaptiveTailCacheController(blocks, settings)
        schedule = torch.tensor([1.0, 0.8, 0.6, 0.0])
        mod_segments = [(0, 2, 0), (2, 4, 0), (4, 6, 2), (6, 10, 0)]

        def run(hidden, sigma):
            queue = [None] + blocks[:] + [None]
            h = hidden
            for index, block in enumerate(blocks):
                self.comfy.model_prefetch.prefetch_queue_pop(queue, None, block)
                wrapper = cache_module.BlockInterceptor(controller, index)
                args = {
                    "img": h,
                    "mod_segments": mod_segments,
                    "transformer_options": {
                        "sigmas": torch.tensor([sigma]),
                        "sample_sigmas": schedule,
                        "cond_or_uncond": [0],
                        "uuids": ["lane"],
                    },
                }
                h = wrapper(
                    args,
                    {"original_block": lambda a, b=block: {"img": b(a["img"])}}
                )["img"]
            self.comfy.model_prefetch.prefetch_queue_pop(queue, None, None)
            return h

        hidden = torch.zeros(10, 8)
        run(hidden, 1.0)
        self.assertEqual(len(self.prefetched), 10)
        self.prefetched.clear()
        run(hidden + 0.001, 0.8)
        self.assertEqual(len(self.prefetched), controller.warm_blocks)


if __name__ == "__main__":
    unittest.main()
