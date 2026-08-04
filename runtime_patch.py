from __future__ import annotations

import threading
from typing import Any

from .adaptive_cache import PrefetchSkipContext


_PATCH_LOCK = threading.Lock()
_PATCH_MARKER = "_mmh3_adaptive_cache_prefetch_guard_v1"
_CALLBACK_MARKER = "_mmh3_adaptive_cache_should_skip"


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

        original = comfy.model_prefetch.prefetch_queue_pop

        def guarded_prefetch_queue_pop(queue: Any, device: Any, module: Any) -> None:
            should_skip = getattr(comfy.model_prefetch, _CALLBACK_MARKER, None)
            if queue is None or should_skip is None or not should_skip(module):
                return original(queue, device, module)

            # Consume the previously used module exactly as stock ComfyUI does.
            consumed = queue.pop(0)
            if consumed is not None and isinstance(consumed, tuple):
                offload_stream, prefetch_state = consumed
                if offload_stream is not None:
                    offload_stream.wait_stream(comfy.model_management.current_stream(device))
                _, comfy_modules = prefetch_state
                if comfy_modules is not None:
                    comfy.model_prefetch.cleanup_prefetched_modules(comfy_modules)

            # queue[0] is the block the stock loop is about to execute. The
            # ModelPatcher wrapper will return an identity for it, so replace
            # the pending prefetch with None while preserving queue alignment.
            if queue:
                pending = queue[0]
                if isinstance(pending, tuple):
                    offload_stream, prefetch_state = pending
                    if offload_stream is not None:
                        offload_stream.wait_stream(comfy.model_management.current_stream(device))
                    _, comfy_modules = prefetch_state
                    if comfy_modules is not None:
                        comfy.model_prefetch.cleanup_prefetched_modules(comfy_modules)
                queue[0] = None

        comfy.model_prefetch._mmh3_adaptive_cache_original_prefetch_queue_pop = original
        comfy.model_prefetch.prefetch_queue_pop = guarded_prefetch_queue_pop
        setattr(comfy.model_prefetch, _PATCH_MARKER, True)
