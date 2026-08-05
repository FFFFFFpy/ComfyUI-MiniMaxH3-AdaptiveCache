import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_modules():
    package_name = "mmh3_adaptive_cache_sol_compat_testpkg"
    for key in list(sys.modules):
        if key == package_name or key.startswith(package_name + "."):
            del sys.modules[key]

    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    sys.modules[package_name] = package

    loaded = {}
    for module_name in ("adaptive_cache", "sol_attn_compat"):
        qualified = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(qualified, ROOT / f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        loaded[module_name] = module
    return loaded["adaptive_cache"], loaded["sol_attn_compat"]


class AddBlock:
    def __init__(self, amount):
        self.amount = amount
        self.calls = 0

    def __call__(self, hidden):
        self.calls += 1
        return hidden + self.amount


class SolAttnCompatibilityTests(unittest.TestCase):
    def _run_call(self, compat, controller, blocks, hidden, sigma, schedule, morton):
        options = {
            "sigmas": torch.tensor([sigma]),
            "sample_sigmas": torch.tensor(schedule),
            "cond_or_uncond": [0],
            "uuids": ["lane"],
            "sol_morton": morton,
        }
        mod_segments = [(0, 2, 0), (2, 4, 0), (4, 6, 2), (6, 10, 0)]
        h = hidden
        for index, block in enumerate(blocks):
            interceptor = compat.SolAttnCompatibleBlockInterceptor(controller, index)
            args = {
                "img": h,
                "mod_segments": mod_segments,
                "transformer_options": options,
            }
            h = interceptor(
                args,
                {"original_block": lambda a, b=block: {"img": b(a["img"])}}
            )["img"]
        return h

    def test_morton_executes_all_blocks_and_invalidates_cache(self):
        cache, compat = load_modules()
        blocks = [AddBlock(0.01 * (index + 1)) for index in range(10)]
        controller = cache.AdaptiveTailCacheController(
            blocks,
            cache.AdaptiveCacheSettings(
                cache_depth=0.6,
                sigma_threshold=1.0,
                start_percent=0.0,
                end_percent=1.0,
                max_consecutive=2,
                quality_guard="off",
                cache_device="cpu",
            ),
        )
        schedule = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        hidden = torch.zeros(10, 8)

        self._run_call(compat, controller, blocks, hidden, 1.0, schedule, morton=False)
        calls_after_full = sum(block.calls for block in blocks)
        self.assertEqual(calls_after_full, controller.block_count)

        self._run_call(compat, controller, blocks, hidden + 0.001, 0.8, schedule, morton=False)
        calls_after_cache = sum(block.calls for block in blocks)
        self.assertEqual(calls_after_cache - calls_after_full, controller.warm_blocks)

        morton_output = self._run_call(
            compat, controller, blocks, hidden, 0.6, schedule, morton=True
        )
        calls_after_morton = sum(block.calls for block in blocks)
        self.assertEqual(calls_after_morton - calls_after_cache, controller.block_count)
        self.assertTrue(torch.allclose(morton_output, torch.full_like(hidden, 0.55)))
        self.assertFalse(controller.lanes)

        self._run_call(compat, controller, blocks, hidden, 0.4, schedule, morton=True)
        calls_after_second_morton = sum(block.calls for block in blocks)
        self.assertEqual(
            calls_after_second_morton - calls_after_morton,
            controller.block_count,
        )

        # Returning to normal token order starts from a mandatory full refresh.
        self._run_call(compat, controller, blocks, hidden, 0.2, schedule, morton=False)
        calls_after_normal = sum(block.calls for block in blocks)
        self.assertEqual(
            calls_after_normal - calls_after_second_morton,
            controller.block_count,
        )

    def test_lifecycle_reports_morton_bypass_count(self):
        cache, compat = load_modules()
        blocks = [AddBlock(0.1) for _ in range(4)]
        controller = cache.AdaptiveTailCacheController(
            blocks,
            cache.AdaptiveCacheSettings(
                cache_depth=0.5,
                sigma_threshold=1.0,
                start_percent=0.0,
                end_percent=1.0,
                max_consecutive=1,
                quality_guard="off",
                cache_device="cpu",
            ),
        )
        lifecycle = compat.SolAttnCompatibleSamplingLifecycleWrapper(controller)

        result = lifecycle(
            lambda: self._run_call(
                compat,
                controller,
                blocks,
                torch.zeros(10, 4),
                1.0,
                [1.0, 0.0],
                morton=True,
            )
        )
        self.assertEqual(result.shape, (10, 4))
        self.assertFalse(getattr(controller, "_mmh3_sol_morton_guard_active"))
        self.assertEqual(getattr(controller, "_mmh3_sol_morton_bypass_calls"), 0)


if __name__ == "__main__":
    unittest.main()
