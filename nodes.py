from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from .adaptive_cache import (
    AdaptiveCacheSettings,
    AdaptiveTailCacheController,
    BlockInterceptor,
    PRESETS,
    SamplingLifecycleWrapper,
)
from .runtime_patch import install_prefetch_guard


def _find_minimax_dit(model: Any) -> Optional[Any]:
    """Walk common ModelPatcher/BaseModel wrapper chains."""
    current = getattr(model, "model", None)
    seen = set()
    for _ in range(16):
        if current is None or id(current) in seen:
            return None
        seen.add(id(current))
        if type(current).__name__ == "MiniMaxH3Model" and hasattr(current, "blocks"):
            return current
        next_obj = None
        for attr in ("model", "inner_model", "diffusion_model", "unet_model"):
            candidate = getattr(current, attr, None)
            if candidate is not None:
                next_obj = candidate
                break
        current = next_obj
    return None


def _apply_cache_patch(model: Any, settings: AdaptiveCacheSettings, label: str) -> Any:
    inner = _find_minimax_dit(model)
    if inner is None:
        raise ValueError("This node only supports ComfyUI's native MiniMax H3 model")
    if not hasattr(model, "set_model_patch_replace"):
        raise ValueError("The input MODEL does not expose ModelPatcher block replacement")
    if not hasattr(inner, "blocks") or len(inner.blocks) < 2:
        raise ValueError("MiniMax H3 DiT blocks were not found")

    install_prefetch_guard()
    patched = model.clone()
    controller = AdaptiveTailCacheController(inner.blocks, settings, label=label)

    # Stock MiniMax H3 already exposes one replacement point per DiT block.
    # Registering all blocks lets the plugin capture the warm boundary, skip
    # the tail, and suppress its prefetch without modifying model.py.
    for index in range(len(inner.blocks)):
        # Stable index fallback for ModelPatcher deep clones used by multi-GPU.
        try:
            setattr(inner.blocks[index], "_mmh3_adaptive_cache_block_index", index)
        except Exception:
            pass
        patched.set_model_patch_replace(
            BlockInterceptor(controller, index),
            "dit",
            "double_block",
            index,
        )

    try:
        import comfy.patcher_extension
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            f"minimax_h3_adaptive_cache_{id(controller)}",
            SamplingLifecycleWrapper(controller),
        )
    except Exception as exc:
        raise RuntimeError(
            "This ComfyUI build does not expose the OUTER_SAMPLE wrapper API required "
            "for safe cache lifecycle management"
        ) from exc
    return patched


class MiniMaxH3AdaptiveCache:
    """Preset-oriented node for normal workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (["safe", "balanced", "fast", "sigma_only"], {
                    "default": "balanced",
                    "tooltip": "Safe favors fidelity; Balanced is the default; Fast skips more tail blocks; Sigma Only disables the content guard.",
                }),
                "cache_device": (["auto", "gpu", "cpu"], {
                    "default": "auto",
                    "tooltip": "Auto keeps residuals on GPU only when enough free VRAM remains, otherwise uses pinned CPU memory.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "sampling/custom_sampling/minimax_h3"
    DESCRIPTION = "Standalone content-aware tail-block cache for native MiniMax H3. No model.py patch is required."

    def patch(self, model, preset, cache_device):
        settings = replace(PRESETS[preset], cache_device=cache_device)
        patched = _apply_cache_patch(
            model,
            settings,
            label=f"MiniMax H3 Adaptive Cache [{preset}]",
        )
        return (patched,)


class MiniMaxH3AdaptiveCacheAdvanced:
    """Fully configurable node for benchmarking and tuning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "cache_depth": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 0.94, "step": 0.01,
                    "tooltip": "Fraction of trailing DiT blocks skipped on a cache hit. 0.75 recomputes about 12 of 50 blocks.",
                }),
                "sigma_threshold": ("FLOAT", {
                    "default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Maximum normalized sigma change allowed for a cache candidate.",
                }),
                "window_start": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 0.49, "step": 0.01,
                }),
                "window_end": ("FLOAT", {
                    "default": 0.90, "min": 0.51, "max": 1.0, "step": 0.01,
                }),
                "max_consecutive": ("INT", {
                    "default": 2, "min": 0, "max": 10, "step": 1,
                    "tooltip": "Maximum consecutive cache hits before a forced full refresh.",
                }),
                "quality_guard": (["audio_video", "video", "off"], {
                    "default": "audio_video",
                }),
                "content_multiplier": ("FLOAT", {
                    "default": 1.25, "min": 0.25, "max": 4.0, "step": 0.05,
                    "tooltip": "A candidate is rejected when its content delta is much larger than the lane's moving average. Lower is safer.",
                }),
                "content_ceiling": ("FLOAT", {
                    "default": 0.75, "min": 0.05, "max": 2.0, "step": 0.05,
                    "tooltip": "Hard upper bound for symmetric target-content change. Prevents the first candidate from being trusted blindly.",
                }),
                "cache_device": (["auto", "gpu", "cpu"], {"default": "auto"}),
                "gpu_safety_mb": ("INT", {
                    "default": 768, "min": 0, "max": 8192, "step": 64,
                    "tooltip": "Free-VRAM reserve used by Auto cache placement.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "sampling/custom_sampling/minimax_h3"
    DESCRIPTION = "Advanced MiniMax H3 adaptive tail-cache controls. This is an approximate, seed-changing acceleration patch."

    def patch(
        self,
        model,
        cache_depth,
        sigma_threshold,
        window_start,
        window_end,
        max_consecutive,
        quality_guard,
        content_multiplier,
        content_ceiling,
        cache_device,
        gpu_safety_mb,
    ):
        settings = AdaptiveCacheSettings(
            cache_depth=cache_depth,
            sigma_threshold=sigma_threshold,
            start_percent=window_start,
            end_percent=window_end,
            max_consecutive=max_consecutive,
            quality_guard=quality_guard,
            content_multiplier=content_multiplier,
            content_ceiling=content_ceiling,
            cache_device=cache_device,
            gpu_safety_mb=gpu_safety_mb,
        )
        patched = _apply_cache_patch(
            model,
            settings,
            label="MiniMax H3 Adaptive Cache [advanced]",
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3AdaptiveCache": MiniMaxH3AdaptiveCache,
    "MiniMaxH3AdaptiveCacheAdvanced": MiniMaxH3AdaptiveCacheAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3AdaptiveCache": "MiniMax H3 Adaptive Cache",
    "MiniMaxH3AdaptiveCacheAdvanced": "MiniMax H3 Adaptive Cache (Advanced)",
}
