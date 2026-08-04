import unittest

import torch

from adaptive_cache import (
    AdaptiveCacheSettings,
    AdaptiveTailCacheController,
    BlockInterceptor,
)


class AddBlock:
    def __init__(self, amount):
        self.amount = amount
        self.calls = 0

    def __call__(self, hidden):
        self.calls += 1
        return hidden + self.amount


class AdaptiveCacheTests(unittest.TestCase):
    def _run_call(self, controller, blocks, hidden, sigma, schedule, cond=0):
        options = {
            "sigmas": torch.tensor([sigma]),
            "sample_sigmas": torch.tensor(schedule),
            "cond_or_uncond": [cond],
            "uuids": [f"lane-{cond}"],
        }
        # Text 0:2, condition 2:4, target audio 4:6, target video 6:10.
        mod_segments = [(0, 2, 0), (2, 4, 0), (4, 6, 2), (6, 10, 0)]
        h = hidden
        for index, block in enumerate(blocks):
            wrapper = BlockInterceptor(controller, index)
            args = {
                "img": h,
                "t_emb": None,
                "mod_segments": mod_segments,
                "rope_freqs": None,
                "transformer_options": options,
            }

            def original(a, b=block):
                return {"img": b(a["img"])}

            h = wrapper(args, {"original_block": original})["img"]
        return h

    def test_second_call_can_use_tail_cache(self):
        blocks = [AddBlock(0.01 * (i + 1)) for i in range(10)]
        settings = AdaptiveCacheSettings(
            cache_depth=0.6,
            sigma_threshold=1.0,
            start_percent=0.0,
            end_percent=1.0,
            max_consecutive=2,
            quality_guard="off",
            cache_device="cpu",
        )
        controller = AdaptiveTailCacheController(blocks, settings)
        schedule = [1.0, 0.8, 0.6, 0.0]
        hidden = torch.zeros(10, 8)

        self._run_call(controller, blocks, hidden, 1.0, schedule)
        calls_after_full = sum(b.calls for b in blocks)
        self.assertEqual(calls_after_full, 10)

        self._run_call(controller, blocks, hidden + 0.001, 0.8, schedule)
        calls_after_cache = sum(b.calls for b in blocks)
        self.assertEqual(calls_after_cache - calls_after_full, controller.warm_blocks)

    def test_target_only_residual_does_not_touch_text_rows(self):
        blocks = [AddBlock(0.1) for _ in range(6)]
        settings = AdaptiveCacheSettings(
            cache_depth=0.5,
            sigma_threshold=1.0,
            start_percent=0.0,
            end_percent=1.0,
            max_consecutive=2,
            quality_guard="off",
            cache_device="cpu",
        )
        controller = AdaptiveTailCacheController(blocks, settings)
        schedule = [1.0, 0.5, 0.0]
        hidden = torch.zeros(10, 4)
        self._run_call(controller, blocks, hidden, 1.0, schedule)
        cached = self._run_call(controller, blocks, hidden, 0.5, schedule)

        # Text rows receive only the warm prefix on a cache hit.
        expected_warm = controller.warm_blocks * 0.1
        self.assertTrue(torch.allclose(cached[:4], torch.full_like(cached[:4], expected_warm)))
        # Target rows receive warm prefix plus the cached tail residual.
        self.assertTrue(torch.allclose(cached[4:], torch.full_like(cached[4:], 0.6)))

    def test_lanes_keep_separate_residuals(self):
        blocks = [AddBlock(0.05) for _ in range(8)]
        settings = AdaptiveCacheSettings(
            cache_depth=0.5,
            sigma_threshold=1.0,
            start_percent=0.0,
            end_percent=1.0,
            max_consecutive=2,
            quality_guard="off",
            cache_device="cpu",
        )
        controller = AdaptiveTailCacheController(blocks, settings)
        schedule = [1.0, 0.7, 0.0]
        hidden = torch.zeros(10, 4)
        self._run_call(controller, blocks, hidden, 1.0, schedule, cond=0)
        self._run_call(controller, blocks, hidden + 1.0, 1.0, schedule, cond=1)
        self.assertEqual(len(controller.lanes), 2)
        self.assertTrue(all(lane.residual.tensors for lane in controller.lanes.values()))

    def test_content_spike_forces_full_refresh(self):
        blocks = [AddBlock(0.02 * (i + 1)) for i in range(10)]
        settings = AdaptiveCacheSettings(
            cache_depth=0.6,
            sigma_threshold=1.0,
            start_percent=0.0,
            end_percent=1.0,
            max_consecutive=3,
            quality_guard="audio_video",
            content_multiplier=1.25,
            content_ceiling=0.5,
            cache_device="cpu",
        )
        controller = AdaptiveTailCacheController(blocks, settings)
        schedule = [1.0, 0.8, 0.6, 0.4, 0.0]
        hidden = torch.ones(10, 8)

        self._run_call(controller, blocks, hidden, 1.0, schedule)
        self._run_call(controller, blocks, hidden + 0.001, 0.8, schedule)
        calls_before_spike = sum(block.calls for block in blocks)
        self._run_call(controller, blocks, hidden + 100.0, 0.6, schedule)
        self.assertEqual(sum(block.calls for block in blocks) - calls_before_spike, 10)
        lane = next(iter(controller.lanes.values()))
        self.assertGreaterEqual(lane.stats.content_rejects, 1)

    def test_disabled_cache_runs_all_blocks_and_stores_nothing(self):
        blocks = [AddBlock(0.1) for _ in range(6)]
        settings = AdaptiveCacheSettings(
            cache_depth=0.0,
            sigma_threshold=1.0,
            start_percent=0.0,
            end_percent=1.0,
            max_consecutive=2,
            quality_guard="off",
            cache_device="cpu",
        )
        controller = AdaptiveTailCacheController(blocks, settings)
        schedule = [1.0, 0.5, 0.0]
        hidden = torch.zeros(10, 4)
        self._run_call(controller, blocks, hidden, 1.0, schedule)
        self._run_call(controller, blocks, hidden, 0.5, schedule)
        self.assertEqual(sum(block.calls for block in blocks), 12)
        self.assertFalse(next(iter(controller.lanes.values())).residual.tensors)


if __name__ == "__main__":
    unittest.main()
