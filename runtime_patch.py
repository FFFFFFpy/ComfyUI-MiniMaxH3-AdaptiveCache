from __future__ import annotations

import inspect
import logging
import threading
from typing import Any

from .adaptive_cache import PrefetchSkipContext


_PATCH_LOCK = threading.Lock()
_PATCH_MARKER = "_mmh3_adaptive_cache_prefetch_guard_v3"
_LEGACY_MARKERS = (
    "_mmh3_adaptive_cache_prefetch_guard_v1",
    "_mmh3_adaptive_cache_prefetch_guard_v2",
)
_INSTALLED_ATTR = "_mmh3_adaptive_cache_installed_prefetch_guard"
_CALLBACK_MARKER = "_mmh3_adaptive_cache_should_skip"
_ORIGINAL_ATTR = "_mmh3_adaptive_cache_original_prefetch_queue_pop"


def _unwrap_own_guard(function: Any) -> Any:
    """Remove only this plugin's guards, including legacy closure-only guards."""

    seen = set()
    while (
        inspect.isfunction(function)
        and function.__code__.co_name == "guarded_prefetch_queue_pop"
        and function.__qualname__.endswith(
            "install_prefetch_guard.<locals>.guarded_prefetch_queue_pop"
        )
    ):
        if id(function) in seen:
            raise RuntimeError("Cyclic Adaptive Cache prefetch guards; restart ComfyUI")
        seen.add(id(function))
        original = getattr(function, _ORIGINAL_ATTR, None)
        if original is None:
            original = inspect.getclosurevars(function).nonlocals.get("original")
        if not callable(original):
            # A third-party decorator may have copied our function name.
            # Do not discard it or replace unrelated plugins with a stale stock ref.
            break
        function = original
    return function


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
        current = comfy.model_prefetch.prefetch_queue_pop
        if (
            getattr(comfy.model_prefetch, _PATCH_MARKER, False)
            and getattr(comfy.model_prefetch, _INSTALLED_ATTR, None) is current
        ):
            return

        # A module-level marker can outlive the function it describes. Recover
        # directly from the active guard chain, not a possibly stale saved ref.
        # Unrelated wrappers remain in the call chain.
        original = _unwrap_own_guard(current)

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

            # Only specialize the plain prefetch call used by native H3.
            # Callback/graph calls and future API extensions keep stock behavior
            # rather than silently losing callbacks or replaying the wrong graph.
            if args or any(key not in {"dtype", "malloc_scope"} for key in kwargs):
                return original(queue, device, module, *args, **kwargs)

            # Advance the allocator scope BEFORE releasing prefetched weights,
            # matching stock ComfyUI's ordering. Its no-queue path does not load
            # weights. Argument validation also happens before we mutate queue.
            result = original(None, device, module, **kwargs)
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

            return result

        setattr(guarded_prefetch_queue_pop, _ORIGINAL_ATTR, original)
        setattr(comfy.model_prefetch, _ORIGINAL_ATTR, original)
        comfy.model_prefetch.prefetch_queue_pop = guarded_prefetch_queue_pop
        setattr(comfy.model_prefetch, _PATCH_MARKER, True)
        setattr(comfy.model_prefetch, _INSTALLED_ATTR, guarded_prefetch_queue_pop)
        # Older copies must not reinstall their three-argument guard over v3.
        for marker in _LEGACY_MARKERS:
            setattr(comfy.model_prefetch, marker, True)
        logging.info(
            "MiniMax H3 Adaptive Cache: prefetch guard v3 installed from %s", __file__
        )
