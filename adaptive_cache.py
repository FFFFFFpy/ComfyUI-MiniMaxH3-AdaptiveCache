from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import torch


TensorRange = Tuple[int, int]


_CACHE_GRAPH_LOCAL = threading.local()


def _cache_tensor_operation(function):
    """Keep plugin-owned tensors out of Comfy's per-block allocation graph.

    Warm snapshots survive a block scope and residuals/probes survive a model
    forward. Their allocation, copy and release cannot belong to those scopes.
    Use the same public pause boundary as native H3 FunControl, not a global
    compiler switch. Original model blocks are deliberately NOT decorated.
    """
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        # Comfy loads model_prefetch before installing our node. Resolve lazily
        # so standalone CPU tests, older Comfy builds and reloads still work.
        prefetch = sys.modules.get("comfy.model_prefetch")
        pause = getattr(prefetch, "pause_malloc_graph", None)
        if pause is None or getattr(_CACHE_GRAPH_LOCAL, "active", False):
            return function(*args, **kwargs)

        # AIMDO's pause is a boolean, not a nesting counter. Nested cache
        # operations must not resume recording until the outer operation ends.
        _CACHE_GRAPH_LOCAL.active = True
        try:
            with pause():
                return function(*args, **kwargs)
        finally:
            _CACHE_GRAPH_LOCAL.active = False

    return wrapped


@dataclass(frozen=True)
class AdaptiveCacheSettings:
    """Runtime settings for the MiniMax H3 tail cache.

    cache_depth is the fraction of trailing DiT blocks that may be skipped.
    The leading blocks are always recomputed and provide a content probe.
    """

    cache_depth: float = 0.75
    sigma_threshold: float = 0.12
    start_percent: float = 0.10
    end_percent: float = 0.90
    max_consecutive: int = 2
    quality_guard: str = "audio_video"  # off / video / audio_video
    content_multiplier: float = 1.25
    content_floor: float = 0.02
    content_ceiling: float = 0.75
    ema_decay: float = 0.85
    video_weight: float = 0.80
    audio_weight: float = 0.20
    probe_tokens: int = 256
    probe_channels: int = 64
    cache_device: str = "auto"  # auto / gpu / cpu
    gpu_safety_mb: int = 768

    def validate(self) -> "AdaptiveCacheSettings":
        if not 0.0 <= self.cache_depth < 1.0:
            raise ValueError("cache_depth must be in [0, 1)")
        if self.sigma_threshold < 0.0:
            raise ValueError("sigma_threshold must be non-negative")
        if not 0.0 <= self.start_percent < self.end_percent <= 1.0:
            raise ValueError("cache window must satisfy 0 <= start < end <= 1")
        if self.max_consecutive < 0:
            raise ValueError("max_consecutive must be non-negative")
        if self.quality_guard not in {"off", "video", "audio_video"}:
            raise ValueError("quality_guard must be off, video, or audio_video")
        if self.content_multiplier <= 0.0:
            raise ValueError("content_multiplier must be positive")
        if self.content_floor < 0.0 or self.content_ceiling <= 0.0:
            raise ValueError("content limits must be positive")
        if self.content_floor > self.content_ceiling:
            raise ValueError("content_floor must not exceed content_ceiling")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.probe_tokens < 1 or self.probe_channels < 1:
            raise ValueError("probe dimensions must be positive")
        if self.cache_device not in {"auto", "gpu", "cpu"}:
            raise ValueError("cache_device must be auto, gpu, or cpu")
        return self


PRESETS: Dict[str, AdaptiveCacheSettings] = {
    "safe": AdaptiveCacheSettings(
        cache_depth=0.60,
        sigma_threshold=0.10,
        start_percent=0.15,
        end_percent=0.85,
        max_consecutive=1,
        quality_guard="audio_video",
        content_multiplier=0.95,
        content_floor=0.01,
        content_ceiling=0.45,
        gpu_safety_mb=1024,
    ),
    "balanced": AdaptiveCacheSettings(),
    "fast": AdaptiveCacheSettings(
        cache_depth=0.84,
        sigma_threshold=0.16,
        start_percent=0.08,
        end_percent=0.92,
        max_consecutive=3,
        quality_guard="audio_video",
        content_multiplier=1.80,
        content_floor=0.04,
        content_ceiling=1.20,
        gpu_safety_mb=512,
    ),
    "sigma_only": AdaptiveCacheSettings(
        cache_depth=0.75,
        sigma_threshold=0.12,
        start_percent=0.10,
        end_percent=0.90,
        max_consecutive=2,
        quality_guard="off",
        content_multiplier=1.0,
    ),
}


class ResidualStore:
    """Stores target-only residuals on GPU or pinned CPU memory.

    CPU residuals can be copied back on a side CUDA stream while the warm
    prefix is being recomputed. The implementation gracefully becomes a plain
    synchronous copy on CPU-only systems.
    """

    def __init__(self) -> None:
        self.tensors: List[torch.Tensor] = []
        self.location: str = "none"
        self.compute_device: Optional[torch.device] = None
        self._prefetched: Optional[List[torch.Tensor]] = None
        self._event: Optional[torch.cuda.Event] = None
        self._stream: Optional[torch.cuda.Stream] = None
        self._stream_device: Optional[torch.device] = None

    def clear(self) -> None:
        self.cancel_prefetch(wait=True)
        self.tensors.clear()
        self.location = "none"
        self.compute_device = None
        self._stream = None
        self._stream_device = None

    @staticmethod
    def _nbytes(tensors: Sequence[torch.Tensor]) -> int:
        return sum(t.numel() * t.element_size() for t in tensors)

    @staticmethod
    def _is_cuda(device: torch.device) -> bool:
        return device.type == "cuda" and torch.cuda.is_available()

    def _choose_location(
        self,
        tensors: Sequence[torch.Tensor],
        requested: str,
        device: torch.device,
        safety_mb: int,
    ) -> str:
        if requested == "cpu" or not self._is_cuda(device):
            return "cpu"
        if requested == "gpu":
            return "gpu"

        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
            required = self._nbytes(tensors) + safety_mb * 1024 * 1024
            return "gpu" if free_bytes >= required else "cpu"
        except Exception:
            return "cpu"

    def set(
        self,
        tensors: Sequence[torch.Tensor],
        requested: str,
        device: torch.device,
        safety_mb: int,
    ) -> None:
        self.cancel_prefetch(wait=True)
        detached = [t.detach().contiguous() for t in tensors]
        location = self._choose_location(detached, requested, device, safety_mb)
        self.compute_device = device

        reusable = (
            self.location == location
            and len(self.tensors) == len(detached)
            and all(old.shape == new.shape and old.dtype == new.dtype
                    for old, new in zip(self.tensors, detached))
        )

        if location == "gpu":
            if reusable and all(t.device == device for t in self.tensors):
                for destination, source in zip(self.tensors, detached):
                    destination.copy_(source)
            else:
                self.tensors = [t.clone() for t in detached]
            self.location = "gpu"
            return

        if reusable and all(t.device.type == "cpu" for t in self.tensors):
            for destination, source in zip(self.tensors, detached):
                destination.copy_(source, non_blocking=False)
        else:
            cpu_tensors: List[torch.Tensor] = []
            for tensor in detached:
                if self._is_cuda(device):
                    cpu = torch.empty_like(tensor, device="cpu", pin_memory=True)
                    cpu.copy_(tensor, non_blocking=False)
                else:
                    cpu = tensor.to("cpu").clone()
                cpu_tensors.append(cpu)
            self.tensors = cpu_tensors
        self.location = "cpu"

    def begin_prefetch(self, device: torch.device) -> None:
        self.cancel_prefetch(wait=True)
        if not self.tensors or self.location != "cpu" or not self._is_cuda(device):
            return

        try:
            if self._stream is None or self._stream_device != device:
                self._stream = torch.cuda.Stream(device=device)
                self._stream_device = device
            stream = self._stream
            with torch.cuda.stream(stream):
                copied = [t.to(device=device, non_blocking=True) for t in self.tensors]
                event = torch.cuda.Event()
                event.record(stream)
            self._prefetched = copied
            self._event = event
        except Exception:
            self.cancel_prefetch()

    def materialize(self, device: torch.device) -> List[torch.Tensor]:
        if not self.tensors:
            return []
        if self.location == "gpu":
            return self.tensors

        if self._prefetched is not None and self._event is not None and self._is_cuda(device):
            consumer = torch.cuda.current_stream(device)
            consumer.wait_event(self._event)
            # The copy event protects the producer only. The tensor was
            # allocated on a side stream and must also outlive the consumer's
            # residual add, even when finish_call drops our Python references.
            for tensor in self._prefetched:
                tensor.record_stream(consumer)
            return self._prefetched
        return [t.to(device=device, non_blocking=False) for t in self.tensors]

    def cancel_prefetch(self, wait: bool = False) -> None:
        if wait and self._event is not None:
            try:
                self._event.synchronize()
            except Exception:
                pass
        self._prefetched = None
        self._event = None


@dataclass
class LaneStats:
    full_calls: int = 0
    cache_calls: int = 0
    content_rejects: int = 0
    sigma_rejects: int = 0
    window_rejects: int = 0
    forced_refreshes: int = 0
    executed_blocks: int = 0
    skipped_blocks: int = 0
    elapsed_seconds: float = 0.0
    content_score_sum: float = 0.0
    content_score_count: int = 0


@dataclass
class LaneState:
    last_sigma: Optional[float] = None
    last_position: Optional[float] = None
    sigma_scale: Optional[float] = None
    consecutive_cache: int = 0
    previous_probes: Optional[List[torch.Tensor]] = None
    content_ema: Optional[float] = None
    residual: ResidualStore = field(default_factory=ResidualStore)
    signature: Optional[Tuple[Any, ...]] = None
    stats: LaneStats = field(default_factory=LaneStats)

    # Per-call transient state.
    call_started_at: float = 0.0
    call_mode: str = "full"
    call_ranges: Optional[List[TensorRange]] = None
    warm_snapshot: Optional[List[torch.Tensor]] = None
    current_probes: Optional[List[torch.Tensor]] = None
    current_content_score: Optional[float] = None
    tentative_cache: bool = False

    def clear_runtime_cache(self) -> None:
        self.residual.clear()
        self.previous_probes = None
        self.content_ema = None
        self.signature = None
        self.sigma_scale = None
        self.consecutive_cache = 0
        self.warm_snapshot = None
        self.current_probes = None


class PrefetchSkipContext:
    """Thread-local block ids whose weight prefetch must be suppressed."""

    _local = threading.local()

    @classmethod
    def set_skipped(cls, module_ids: Iterable[int], skip_from: Optional[int] = None) -> None:
        cls._local.module_ids = set(module_ids)
        cls._local.skip_from = skip_from

    @classmethod
    def clear(cls) -> None:
        cls._local.module_ids = set()
        cls._local.skip_from = None

    @classmethod
    def should_skip(cls, module: Any) -> bool:
        if module is None:
            return False
        if id(module) in getattr(cls._local, "module_ids", set()):
            return True
        skip_from = getattr(cls._local, "skip_from", None)
        block_index = getattr(module, "_mmh3_adaptive_cache_block_index", None)
        return skip_from is not None and block_index is not None and int(block_index) >= int(skip_from)


class AdaptiveTailCacheController:
    """Content-aware target-only tail cache for MiniMax H3.

    The controller is called by one wrapper per DiT block. It recomputes a warm
    prefix, probes the target audio/video rows, and either runs the remaining
    blocks or applies a residual captured on a previous full call.
    """

    def __init__(
        self,
        blocks: Sequence[Any],
        settings: AdaptiveCacheSettings,
        label: str = "MiniMax H3 Adaptive Cache",
    ) -> None:
        self.blocks = list(blocks)
        self.block_count = len(self.blocks)
        self.settings = settings.validate()
        self.label = label
        self.warm_blocks = max(
            1,
            min(
                self.block_count - 1,
                round(self.block_count * (1.0 - self.settings.cache_depth)),
            ),
        )
        self.lanes: Dict[Hashable, LaneState] = {}
        self._lanes_lock = threading.Lock()
        self._runtime = threading.local()

    @_cache_tensor_operation
    def start_sampling_run(self) -> None:
        """Start a fresh sampler invocation and discard all stale cache state."""
        PrefetchSkipContext.clear()
        with self._lanes_lock:
            for lane in self.lanes.values():
                lane.clear_runtime_cache()
            self.lanes = {}
        self._runtime.active_lane_key = None
        self._runtime.active_lane = None
        self._runtime.skip_from = None

    @_cache_tensor_operation
    def end_sampling_run(self) -> None:
        """Print honest block statistics and release residual/probe tensors."""
        PrefetchSkipContext.clear()
        with self._lanes_lock:
            lanes = list(self.lanes.values())
            for lane in lanes:
                self._print_lane_stats(lane)
                lane.clear_runtime_cache()
            self.lanes = {}
        self._runtime.active_lane_key = None
        self._runtime.active_lane = None
        self._runtime.skip_from = None

    def _active_lane(self) -> Optional[LaneState]:
        return getattr(self._runtime, "active_lane", None)

    def _set_active_lane(self, key: Optional[Hashable], lane: Optional[LaneState]) -> None:
        self._runtime.active_lane_key = key
        self._runtime.active_lane = lane

    @property
    def enabled(self) -> bool:
        return (
            self.block_count > 1
            and self.settings.cache_depth > 0.0
            and self.settings.sigma_threshold > 0.0
            and self.settings.max_consecutive > 0
        )

    @staticmethod
    def _to_scalar(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(torch.as_tensor(value).flatten()[0].float())
        except Exception:
            return None

    def _schedule_info(self, transformer_options: Dict[str, Any], lane: LaneState) -> Tuple[Optional[float], float]:
        raw_sigma = self._to_scalar(transformer_options.get("sigmas"))
        if raw_sigma is None:
            return None, 0.0

        sample_sigmas = transformer_options.get("sample_sigmas")
        if sample_sigmas is None:
            return raw_sigma, 0.0

        schedule = torch.as_tensor(sample_sigmas).flatten().float()
        if schedule.numel() < 2:
            return raw_sigma, 0.0

        first = float(schedule[0])
        if lane.sigma_scale is None:
            scale = raw_sigma / first if abs(first) > 1e-12 else 1.0
            lane.sigma_scale = scale if abs(scale) > 1e-12 else 1.0
        sigma = raw_sigma / lane.sigma_scale

        # The final zero is a boundary and normally not evaluated by the model.
        model_sigmas = schedule[:-1]
        idx = int((model_sigmas - sigma).abs().argmin())
        denom = max(1, model_sigmas.numel() - 1)
        position = min(1.0, max(0.0, idx / denom))
        return sigma, position

    @staticmethod
    def _lane_key(transformer_options: Dict[str, Any]) -> Hashable:
        cond = tuple(int(x) for x in transformer_options.get("cond_or_uncond", []))
        uuids = tuple(str(x) for x in transformer_options.get("uuids", []))
        device = str(transformer_options.get("multigpu_thread_device", "default"))
        return device, cond, uuids

    @staticmethod
    def _target_ranges(mod_segments: Sequence[Sequence[int]], rows: int) -> Optional[List[TensorRange]]:
        if len(mod_segments) < 2:
            return None
        audio_segment = mod_segments[-2]
        video_segment = mod_segments[-1]
        if len(audio_segment) < 3 or len(video_segment) < 3:
            return None
        # H3 modulation rows are timestep_row * 3 + modality_tag:
        # video=0, text=1, audio=2. Validate the assumed final segment order.
        if int(audio_segment[2]) % 3 != 2 or int(video_segment[2]) % 3 != 0:
            return None
        ranges = [(int(audio_segment[0]), int(audio_segment[1])),
                  (int(video_segment[0]), int(video_segment[1]))]
        (aa, ab), (va, vb) = ranges
        if not (0 <= aa < ab <= va < vb <= rows):
            return None
        return ranges

    def _signature(self, hidden: torch.Tensor, ranges: Sequence[TensorRange]) -> Tuple[Any, ...]:
        return (
            tuple(hidden.shape),
            tuple(ranges),
            self.warm_blocks,
            str(hidden.dtype),
            str(hidden.device),
        )

    def _probe_one(self, hidden: torch.Tensor, target: TensorRange) -> torch.Tensor:
        start, end = target
        token_count = end - start
        token_stride = max(1, math.ceil(token_count / self.settings.probe_tokens))
        channel_count = hidden.shape[-1]
        channel_stride = max(1, math.ceil(channel_count / self.settings.probe_channels))
        return hidden[start:end:token_stride, ::channel_stride].detach().clone()

    def _make_probes(self, hidden: torch.Tensor, ranges: Sequence[TensorRange]) -> List[torch.Tensor]:
        return [self._probe_one(hidden, target) for target in ranges]

    @staticmethod
    def _relative_delta(current: torch.Tensor, previous: torch.Tensor) -> float:
        if current.shape != previous.shape:
            return float("inf")
        cur = current.float()
        prev = previous.to(device=current.device).float()
        numerator = (cur - prev).abs().mean()
        denominator = (0.5 * (cur.abs().mean() + prev.abs().mean())).clamp_min(1e-5)
        return float((numerator / denominator).item())

    def _content_score(self, probes: Sequence[torch.Tensor], previous: Sequence[torch.Tensor]) -> float:
        audio_delta = self._relative_delta(probes[0], previous[0])
        video_delta = self._relative_delta(probes[1], previous[1])
        if self.settings.quality_guard == "video":
            return video_delta
        total_weight = self.settings.video_weight + self.settings.audio_weight
        if total_weight <= 0:
            return video_delta
        return (
            self.settings.video_weight * video_delta
            + self.settings.audio_weight * audio_delta
        ) / total_weight

    def _content_allowed(self, lane: LaneState, score: Optional[float]) -> bool:
        if self.settings.quality_guard == "off" or score is None or not math.isfinite(score):
            return self.settings.quality_guard == "off" or score is None
        if score > self.settings.content_ceiling:
            return False
        if lane.content_ema is None:
            return True
        limit = max(
            self.settings.content_floor,
            min(self.settings.content_ceiling, lane.content_ema * self.settings.content_multiplier),
        )
        return score <= limit

    def _update_content_ema(self, lane: LaneState, score: Optional[float]) -> None:
        if score is None or not math.isfinite(score):
            return
        lane.stats.content_score_sum += score
        lane.stats.content_score_count += 1
        clipped = min(score, self.settings.content_ceiling)
        if lane.content_ema is None:
            lane.content_ema = clipped
        else:
            d = self.settings.ema_decay
            lane.content_ema = d * lane.content_ema + (1.0 - d) * clipped

    def _reset_lane_for_new_run(self, lane: LaneState) -> None:
        if lane.stats.full_calls or lane.stats.cache_calls:
            self._print_lane_stats(lane)
        lane.last_sigma = None
        lane.last_position = None
        lane.clear_runtime_cache()
        lane.stats = LaneStats()

    @_cache_tensor_operation
    def begin_call(self, args: Dict[str, Any]) -> None:
        PrefetchSkipContext.clear()
        self._runtime.skip_from = None

        options = args.get("transformer_options", {})
        key = self._lane_key(options)
        with self._lanes_lock:
            lane = self.lanes.setdefault(key, LaneState())
        sigma, position = self._schedule_info(options, lane)

        # The outer-sample lifecycle wrapper normally handles run boundaries.
        # Sigma increase remains as a fallback for nonstandard sampler paths.
        if lane.last_sigma is not None and sigma is not None and sigma > lane.last_sigma + 1e-6:
            self._reset_lane_for_new_run(lane)
            sigma, position = self._schedule_info(options, lane)

        lane.call_started_at = time.perf_counter()
        lane.call_mode = "full"
        lane.call_ranges = None
        lane.warm_snapshot = None
        lane.current_probes = None
        lane.current_content_score = None

        in_window = self.settings.start_percent <= position <= self.settings.end_percent
        sigma_delta = float("inf") if lane.last_sigma is None or sigma is None else abs(lane.last_sigma - sigma)
        lane.tentative_cache = (
            self.enabled
            and bool(lane.residual.tensors)
            and in_window
            and sigma_delta < self.settings.sigma_threshold
            and lane.consecutive_cache < self.settings.max_consecutive
        )

        if not in_window:
            lane.stats.window_rejects += 1
        elif sigma_delta >= self.settings.sigma_threshold:
            lane.stats.sigma_rejects += 1
        elif lane.consecutive_cache >= self.settings.max_consecutive:
            lane.stats.forced_refreshes += 1

        if lane.tentative_cache:
            # Copy CPU residuals while the warm prefix is executing.
            lane.residual.begin_prefetch(args["img"].device)

        lane.last_sigma = sigma
        lane.last_position = position
        self._set_active_lane(key, lane)

    @_cache_tensor_operation
    def after_warm(self, hidden: torch.Tensor, args: Dict[str, Any]) -> torch.Tensor:
        lane = self._active_lane()
        if lane is None:
            return hidden

        ranges = self._target_ranges(args.get("mod_segments", []), hidden.shape[0])
        if ranges is None:
            lane.tentative_cache = False
            lane.residual.cancel_prefetch()
            return hidden

        signature = self._signature(hidden, ranges)
        if lane.signature is not None and lane.signature != signature:
            lane.clear_runtime_cache()
            lane.tentative_cache = False
        lane.signature = signature
        lane.call_ranges = ranges

        probes: List[torch.Tensor] = []
        score: Optional[float] = None
        if self.settings.quality_guard != "off":
            probes = self._make_probes(hidden, ranges)
            lane.current_probes = probes
            if lane.previous_probes is not None:
                score = self._content_score(probes, lane.previous_probes)
        lane.current_content_score = score

        content_ok = self._content_allowed(lane, score)
        use_cache = lane.tentative_cache and content_ok
        if lane.tentative_cache and not content_ok:
            lane.stats.content_rejects += 1

        if use_cache:
            residuals = lane.residual.materialize(hidden.device)
            if len(residuals) != len(ranges):
                use_cache = False
            else:
                for (start, end), residual in zip(ranges, residuals):
                    if residual.shape != hidden[start:end].shape:
                        use_cache = False
                        break
                if use_cache:
                    for (start, end), residual in zip(ranges, residuals):
                        hidden[start:end].add_(residual.to(dtype=hidden.dtype))
                    lane.call_mode = "cache"
                    self._runtime.skip_from = self.warm_blocks
                    PrefetchSkipContext.set_skipped(
                        (id(block) for block in self.blocks[self.warm_blocks:]),
                        skip_from=self.warm_blocks,
                    )

        if not use_cache:
            lane.call_mode = "full"
            lane.warm_snapshot = [hidden[a:b].detach().clone() for a, b in ranges]

        if probes:
            lane.previous_probes = [p.detach().clone() for p in probes]
        self._update_content_ema(lane, score)
        return hidden

    def should_skip(self, block_index: int) -> bool:
        skip_from = getattr(self._runtime, "skip_from", None)
        return skip_from is not None and block_index >= skip_from

    @_cache_tensor_operation
    def finish_call(self, hidden: torch.Tensor) -> None:
        lane = self._active_lane()
        if lane is None:
            PrefetchSkipContext.clear()
            return

        try:
            if lane.call_mode == "full" and lane.call_ranges and lane.warm_snapshot:
                residuals = [
                    hidden[a:b].detach() - warm
                    for (a, b), warm in zip(lane.call_ranges, lane.warm_snapshot)
                ]
                lane.residual.set(
                    residuals,
                    requested=self.settings.cache_device,
                    device=hidden.device,
                    safety_mb=self.settings.gpu_safety_mb,
                )
                lane.stats.full_calls += 1
                lane.stats.executed_blocks += self.block_count
                lane.consecutive_cache = 0
            elif lane.call_mode == "cache":
                lane.stats.cache_calls += 1
                lane.stats.executed_blocks += self.warm_blocks
                lane.stats.skipped_blocks += self.block_count - self.warm_blocks
                lane.consecutive_cache += 1
            else:
                lane.stats.full_calls += 1
                lane.stats.executed_blocks += self.block_count
                lane.consecutive_cache = 0

            lane.stats.elapsed_seconds += max(0.0, time.perf_counter() - lane.call_started_at)
            lane.warm_snapshot = None
            lane.residual.cancel_prefetch(wait=True)

        finally:
            PrefetchSkipContext.clear()
            self._runtime.skip_from = None
            self._set_active_lane(None, None)

    @_cache_tensor_operation
    def abort_call(self) -> None:
        lane = self._active_lane()
        if lane is not None:
            lane.residual.cancel_prefetch(wait=True)
            lane.warm_snapshot = None
        PrefetchSkipContext.clear()
        self._runtime.skip_from = None
        self._set_active_lane(None, None)

    def _print_lane_stats(self, lane: LaneState) -> None:
        calls = lane.stats.full_calls + lane.stats.cache_calls
        if calls <= 0:
            return
        possible_blocks = calls * self.block_count
        reduction = 100.0 * lane.stats.skipped_blocks / max(1, possible_blocks)
        avg_score = (
            lane.stats.content_score_sum / lane.stats.content_score_count
            if lane.stats.content_score_count
            else 0.0
        )
        print(
            f"{self.label}: full={lane.stats.full_calls}, cache={lane.stats.cache_calls}, "
            f"executed={lane.stats.executed_blocks}/{possible_blocks} blocks, "
            f"block reduction={reduction:.1f}%, content rejects={lane.stats.content_rejects}, "
            f"avg content delta={avg_score:.5f}, wrapper time={lane.stats.elapsed_seconds:.2f}s, "
            f"residual={lane.residual.location}"
        )


class SamplingLifecycleWrapper:
    """Reset and clean cache state around one complete ComfyUI sampler run."""

    def __init__(self, controller: AdaptiveTailCacheController) -> None:
        self.controller = controller

    def __call__(self, executor: Any, *args: Any, **kwargs: Any) -> Any:
        self.controller.start_sampling_run()
        try:
            return executor(*args, **kwargs)
        finally:
            self.controller.end_sampling_run()


class BlockInterceptor:
    """One ModelPatcher replacement for one MiniMax H3 DiT block."""

    def __init__(self, controller: AdaptiveTailCacheController, index: int) -> None:
        self.controller = controller
        self.index = index

    def __call__(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        try:
            if self.index == 0:
                self.controller.begin_call(args)

            if self.controller.should_skip(self.index):
                hidden = args["img"]
            else:
                hidden = kwargs["original_block"](args)["img"]

            if self.controller.enabled and self.index == self.controller.warm_blocks - 1:
                hidden = self.controller.after_warm(hidden, args)

            if self.index == self.controller.block_count - 1:
                self.controller.finish_call(hidden)
            return {"img": hidden}
        except Exception:
            self.controller.abort_call()
            raise
