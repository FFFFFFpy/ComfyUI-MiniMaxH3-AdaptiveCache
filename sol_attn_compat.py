from __future__ import annotations

from typing import Any, Dict

import torch

from .adaptive_cache import (
    AdaptiveTailCacheController,
    BlockInterceptor,
    PrefetchSkipContext,
)


_GUARD_ACTIVE = "_mmh3_sol_morton_guard_active"
_WARNING_PRINTED = "_mmh3_sol_morton_warning_printed"
_BYPASS_CALLS = "_mmh3_sol_morton_bypass_calls"


def _set_state(controller: AdaptiveTailCacheController, name: str, value: Any) -> None:
    setattr(controller, name, value)


def _get_state(controller: AdaptiveTailCacheController, name: str, default: Any) -> Any:
    return getattr(controller, name, default)


def _reset_compatibility_state(controller: AdaptiveTailCacheController) -> None:
    _set_state(controller, _GUARD_ACTIVE, False)
    _set_state(controller, _WARNING_PRINTED, False)
    _set_state(controller, _BYPASS_CALLS, 0)


class SolAttnCompatibleSamplingLifecycleWrapper:
    """Manage cache state and report calls bypassed for Sol-Attn Morton safety."""

    def __init__(self, controller: AdaptiveTailCacheController) -> None:
        self.controller = controller

    def __call__(self, executor: Any, *args: Any, **kwargs: Any) -> Any:
        _reset_compatibility_state(self.controller)
        self.controller.start_sampling_run()
        try:
            return executor(*args, **kwargs)
        finally:
            self.controller.end_sampling_run()
            bypass_calls = int(_get_state(self.controller, _BYPASS_CALLS, 0))
            if bypass_calls:
                print(
                    f"{self.controller.label}: Sol-Attn Morton bypassed Adaptive Cache "
                    f"for {bypass_calls} model call(s); all DiT blocks were executed."
                )
            _reset_compatibility_state(self.controller)


class SolAttnCompatibleBlockInterceptor:
    """Run the normal cache interceptor unless Sol-Attn Morton is active.

    Kijai's Sol-Attn Morton mode reorders the target video tokens before the
    block stack and restores their original order after the final block. A tail
    cache hit would skip that restoration point, while a full cache capture
    would subtract a reordered warm state from a restored final state. Both are
    invalid, so Morton calls bypass Adaptive Cache completely while Sol-Attn
    itself remains active inside every original block.
    """

    def __init__(self, controller: AdaptiveTailCacheController, index: int) -> None:
        self.controller = controller
        self.index = index
        self.delegate = BlockInterceptor(controller, index)

    @staticmethod
    def _morton_enabled(args: Dict[str, Any]) -> bool:
        options = args.get("transformer_options", {})
        return isinstance(options, dict) and bool(options.get("sol_morton"))

    def _enter_morton_guard(self) -> None:
        if not bool(_get_state(self.controller, _GUARD_ACTIVE, False)):
            # Drop any residual captured with the normal token order. This also
            # clears an interrupted controller call before the direct path runs.
            self.controller.start_sampling_run()
            _set_state(self.controller, _GUARD_ACTIVE, True)

        if not bool(_get_state(self.controller, _WARNING_PRINTED, False)):
            print(
                f"{self.controller.label}: Sol-Attn Morton token reordering detected; "
                "Adaptive Cache is bypassed for correctness. Disable Morton to combine "
                "both accelerators."
            )
            _set_state(self.controller, _WARNING_PRINTED, True)

    def _leave_morton_guard_for_normal_order(self) -> None:
        if bool(_get_state(self.controller, _GUARD_ACTIVE, False)):
            # Start from a mandatory full refresh if token order changes back.
            self.controller.start_sampling_run()
            _set_state(self.controller, _GUARD_ACTIVE, False)

    def __call__(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        morton = self._morton_enabled(args)
        if not morton:
            if self.index == 0:
                self._leave_morton_guard_for_normal_order()
            return self.delegate(args, kwargs)

        try:
            if self.index == 0:
                self._enter_morton_guard()

            # Never leave a previous cache-hit prefetch suppression active on
            # the direct path. Morton must execute the complete block stack.
            PrefetchSkipContext.clear()
            hidden = kwargs["original_block"](args)["img"]

            if self.index == self.controller.block_count - 1:
                calls = int(_get_state(self.controller, _BYPASS_CALLS, 0))
                _set_state(self.controller, _BYPASS_CALLS, calls + 1)
            return {"img": hidden}
        except Exception:
            self.controller.abort_call()
            raise
