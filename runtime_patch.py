from __future__ import annotations

import inspect
import threading
from typing import Any

from .adaptive_cache import PrefetchSkipContext


_PATCH_LOCK = threading.Lock()
_PATCH_MARKER = "_mmh3_adaptive_cache_prefetch_guard_v2"
_CALLBACK_MARKER = "_mmh3_adaptive_cache_should_skip"
_ORIGINAL_ATTR = "_mmh3_adaptive_cache_original_prefetch_queue_pop"


def _cleanup_prefetch_state(cleanup: Any, prefetch_state: Any) -> None:
    """Clean a prefetched state across old and current ComfyUI APIs."""

    prefetched_module, comfy_modules = prefetch_state
    if comfy_modules is None:
        return

    # ComfyUI 0.34 changed cleanup_prefetched_modules from
    # cleanup_prefetched_modules(comfy_modules) to
    # cleanup_prefetched_modules(module, comfy_modules). Detect the callable
    # shape instead of pinning this custom node to one ComfyUI release.
    try:
        signature = inspect.signature(cleanup)
    except (TypeError, ValueError):
        cleanup(prefetched_module, comfy_modules)
        return

    try:
        signature.bind(prefetched_module, comfy_modules)
    except TypeError:
        cleanup(comfy_modules)
    else:
        cleanup(prefetched_module, comfy_modules)


def install_prefetch_guard() -> None:
    """Install a narrowly-scoped runtime guard around ComfyUI prefetch.

    The guard behaves exactly like ComfyUI's normal prefetch function unless a
    MiniMax H3 cache wrapper has marked the current block as skipped in the
    current thread. No ComfyUI source file is modified.
    """

    try:
        import comfy.model_management
        import comfy.model_prefetch
    except Exception as exc:  # pragma: no cover - only happens outside ComfyUI
        raise RuntimeError("ComfyUI model prefetch modules are unavailable") from exc

    with _PATCH_LOCK:
        # Refresh this on every custom-node reload. The installed guard resolves
        # the callback dynamically, so it never keeps a dead thread-local class
        # from an older module instance.
        setattr(comfy.model_prefetch, _CALLBACK_MARKER, PrefetchSkipContext.should_skip)
        if getattr(comfy.model_prefetch, _PATCH_MARKER, False):
            return

        # If an older version of this plugin already installed a guard in the
        # same Python process, reuse the stored stock ComfyUI function instead
        # of wrapping the stale guard a second time.
        original = getattr(
            comfy.model_prefetch,
            _ORIGINAL_ATTR,
            comfy.model_prefetch.prefetch_queue_pop,
        )

        def guarded_prefetch_queue_pop(
            queue: Any,
            device: Any,
            module: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            should_skip = getattr(comfy.model_prefetch, _CALLBACK_MARKER, None)
            if queue is None or should_skip is None or not should_skip(module):
                return original(queue, device, module, *args, **kwargs)

            cleanup = comfy.model_prefetch.cleanup_prefetched_modules

            # Consume the previously used module exactly as stock ComfyUI does.
            consumed = queue.pop(0)
            if consumed is not None and isinstance(consumed, tuple):
                offload_stream, prefetch_state = consumed
                if offload_stream is not None:
                    offload_stream.wait_stream(comfy.model_management.current_stream(device))
                _cleanup_prefetch_state(cleanup, prefetch_state)

            # queue[0] is the block the stock loop is about to execute. The
            # ModelPatcher wrapper will return an identity for it, so replace
            # the pending prefetch with None while preserving queue alignment.
            if queue:
                pending = queue[0]
                if isinstance(pending, tuple):
                    offload_stream, prefetch_state = pending
                    if offload_stream is not None:
                        offload_stream.wait_stream(comfy.model_management.current_stream(device))
                    _cleanup_prefetch_state(cleanup, prefetch_state)
                queue[0] = None

            return None

        setattr(comfy.model_prefetch, _ORIGINAL_ATTR, original)
        comfy.model_prefetch.prefetch_queue_pop = guarded_prefetch_queue_pop
        setattr(comfy.model_prefetch, _PATCH_MARKER, True)
