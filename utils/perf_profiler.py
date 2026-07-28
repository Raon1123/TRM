"""Bounded, opt-in training profiler support.

This module deliberately has no import-time profiler side effects.  The normal
training path DOES construct :class:`TrainingProfiler` -- ``pretrain.py``
builds one unconditionally on every run -- but with ``enabled=False`` that
instance is inert: ``_profiler`` stays ``None``, no ``torch.profiler.profile``
is ever created, ``enabled`` is False, ``start``/``stop``/``step`` are no-ops,
and ``record`` returns a plain ``nullcontext()``.  Only the profiled train
path calls ``record``, so on the normal path even that null context is never
entered.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import ContextManager, Optional
import warnings

import pydantic
import torch


class PerfProfilerConfig(pydantic.BaseModel):
    """Configuration for a short, performance-only ``torch.profiler`` capture."""

    model_config = pydantic.ConfigDict(extra="forbid")

    enabled: bool = False
    # This acknowledgement prevents accidental profiler traces in a science run.
    performance_only: bool = False
    output_dir: str = "performance_profiles"

    wait: int = pydantic.Field(default=1, ge=0)
    warmup: int = pydantic.Field(default=1, ge=0)
    active: int = pydantic.Field(default=3, ge=1)
    repeat: int = pydantic.Field(default=1, ge=1, le=16)
    max_steps: int = pydantic.Field(default=64, ge=1, le=256)

    record_shapes: bool = False
    profile_memory: bool = True
    with_stack: bool = False
    export_chrome_trace: bool = True
    export_tensorboard_trace: bool = True
    @pydantic.model_validator(mode="after")
    def validate_bounded_capture(self) -> "PerfProfilerConfig":
        if self.enabled and not self.performance_only:
            raise ValueError(
                "perf_profiler.enabled requires perf_profiler.performance_only=true; "
                "profiler timing must not be used as a science run."
            )
        if self.enabled and not self.output_dir.strip():
            raise ValueError("perf_profiler.output_dir must be set when profiling is enabled")
        scheduled_steps = (self.wait + self.warmup + self.active) * self.repeat
        if self.enabled and self.max_steps < scheduled_steps:
            raise ValueError(
                "perf_profiler.max_steps is shorter than its torch.profiler schedule "
                f"({self.max_steps} < {scheduled_steps})"
            )
        return self


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


class TrainingProfiler:
    """Owns one bounded profiler and rank-zero-only trace exporting.

    Call ``step`` once after each profiled train step.  Only scheduled active
    windows are retained, while ``max_steps`` guards against a mistakenly long
    performance run.
    """

    def __init__(
        self,
        config: PerfProfilerConfig,
        *,
        checkpoint_path: Optional[str],
        rank: int,
    ) -> None:
        self.config = config
        self.rank = rank
        self._step_count = 0
        self._trace_count = 0
        self._started = False
        self._profiler: Optional[torch.profiler.profile] = None
        self.output_dir: Optional[Path] = None
        self.session_dir: Optional[Path] = None

        # All ranks execute collectives, but only rank zero may instantiate a
        # profiler or write trace artifacts.  This is intentionally not a
        # configurable policy: multi-rank writers make output ownership and
        # trace interpretation ambiguous.
        active_rank = config.enabled and rank == 0
        if not active_rank:
            return

        output_dir = Path(config.output_dir).expanduser().resolve()
        if checkpoint_path is not None:
            checkpoint_dir = Path(checkpoint_path).expanduser().resolve()
            if _is_relative_to(output_dir, checkpoint_dir) or _is_relative_to(checkpoint_dir, output_dir):
                raise ValueError(
                    "perf_profiler.output_dir must be separate from checkpoint_path so "
                    "performance traces cannot mix with science-run artifacts"
                )
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Never overwrite an earlier capture, including when a user reruns the
        # same Hydra command.  Rank zero is the sole creator of these paths.
        for session_id in range(10_000):
            candidate = output_dir / f"session_{session_id:04d}"
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            self.session_dir = candidate
            break
        if self.session_dir is None:
            raise RuntimeError(f"could not allocate a profiler session under {output_dir}")
        warnings.warn(
            "Performance profiler enabled: this is a performance-only run. Trace output "
            f"will be written to {output_dir}; do not compare its metrics to science runs.",
            stacklevel=2,
        )

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        else:
            warnings.warn("CUDA is unavailable; recording a CPU-only profiler trace.", stacklevel=2)

        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=config.wait,
                warmup=config.warmup,
                active=config.active,
                repeat=config.repeat,
            ),
            on_trace_ready=self._on_trace_ready,
            record_shapes=config.record_shapes,
            profile_memory=config.profile_memory,
            with_stack=config.with_stack,
        )

    @property
    def enabled(self) -> bool:
        return self._profiler is not None

    def record(self, name: str) -> ContextManager[None]:
        if self._profiler is None:
            return nullcontext()
        return torch.profiler.record_function(name)

    def start(self) -> None:
        if self._profiler is not None:
            self._profiler.start()
            self._started = True

    def step(self) -> None:
        if self._profiler is None or self._step_count >= self.config.max_steps:
            return
        self._profiler.step()
        self._step_count += 1
        # The schedule is guaranteed to fit in max_steps by validation.  Stop
        # immediately afterward so a mistakenly long training command cannot
        # keep profiler hooks enabled after its requested capture window.
        if self._step_count >= self.config.max_steps:
            self.stop()

    def stop(self) -> None:
        if self._profiler is not None and self._started:
            self._profiler.stop()
        self._profiler = None
        self._started = False

    def _on_trace_ready(self, profiler: torch.profiler.profile) -> None:
        # torch invokes this only in the active rank, so no other rank writes.
        assert self.session_dir is not None
        trace_dir = self.session_dir / f"capture_{self._trace_count:02d}"
        trace_dir.mkdir(parents=True, exist_ok=False)
        if self.config.export_chrome_trace:
            profiler.export_chrome_trace(str(trace_dir / "trace.json"))
        if self.config.export_tensorboard_trace:
            handler = torch.profiler.tensorboard_trace_handler(str(trace_dir))
            handler(profiler)
        self._trace_count += 1
