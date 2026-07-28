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
from typing import Any, ContextManager, Dict, List, Optional
import json
import warnings

import pydantic
import torch


#: Folded-stack metrics exported when ``export_stacks`` is set.  These are the
#: two metrics ``torch.profiler.profile.export_stacks`` documents; the folded
#: format ("frame;frame;frame <count>" per line) is the canonical flamegraph
#: input.  Order is fixed so a capture directory listing is deterministic.
STACK_EXPORT_METRICS = ("self_cpu_time_total", "self_cuda_time_total")

#: Failures ``export_stacks`` can raise across torch versions: an unsupported
#: metric name is a ``ValueError``, a profile with no materialised function
#: events raises ``AssertionError``, a kineto/profiler-state problem raises
#: ``RuntimeError``, and the write itself can raise ``OSError``.  Anything else
#: is a real bug and is deliberately allowed to propagate.
_STACK_EXPORT_ERRORS = (AssertionError, ValueError, RuntimeError, OSError)


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
    # PERF-DEV-18: default flipped True -> False.  torch permits ONE kineto save
    # per on_trace_ready; with both exports on, the second raises
    # "RuntimeError: Trace is already saved." out of profiler.step() and aborts
    # the run.  The old default therefore made the profiler unusable as shipped,
    # and the pre-registered M3 command (profiling doc section 4.2.1), which sets
    # both to true, could never have run.  Reproduced independently on CPU.
    # The tensorboard handler writes chrome-format JSON anyway, so nothing is
    # lost by preferring the chrome export.
    export_tensorboard_trace: bool = False
    # Folded stacks are the only artifact a flame chart can be built from.
    # Off by default: it requires with_stack, which perturbs the capture.
    #
    # Operator notes for an actual flame-chart capture:
    #   * Also set export_tensorboard_trace=false.  With both chrome and
    #     tensorboard exports on, torch raises "Trace is already saved." from
    #     the second kineto save; that defect is pre-existing (it reproduces
    #     with export_stacks unset), and the folded stacks are written before
    #     the tensorboard handler precisely so they survive it -- but the
    #     capture still dies afterwards until it is fixed upstream.
    #   * Hydra override: the plain ``perf_profiler.export_stacks=true`` form
    #     works only once ``export_stacks`` exists as a key in
    #     config/cfg_pretrain.yaml.  While it is absent, struct mode rejects
    #     that form with ConfigCompositionException and the append form
    #     ``+perf_profiler.export_stacks=true`` is required.
    export_stacks: bool = False

    @pydantic.model_validator(mode="after")
    def validate_bounded_capture(self) -> "PerfProfilerConfig":
        if self.enabled and not self.performance_only:
            raise ValueError(
                "perf_profiler.enabled requires perf_profiler.performance_only=true; "
                "profiler timing must not be used as a science run."
            )
        if self.enabled and self.export_stacks and not self.with_stack:
            # torch records no Python stack frames without with_stack, so the
            # folded-stack export would be empty (or raise).  Fail fast rather
            # than hand back a flamegraph input that silently has no frames.
            raise ValueError(
                "perf_profiler.export_stacks requires perf_profiler.with_stack=true; "
                "without stack recording torch emits no frames and the folded-stack "
                "export would be empty."
            )
        if self.enabled and self.export_chrome_trace and self.export_tensorboard_trace:
            # PERF-DEV-18.  torch allows one kineto save per on_trace_ready; the
            # second raises "Trace is already saved." from profiler.step() and
            # kills the training run mid-capture.  Rejecting the combination up
            # front turns a confusing mid-run abort into an immediate, readable
            # config error.  Verified on this torch build, CPU-only repro.
            raise ValueError(
                "perf_profiler.export_chrome_trace and "
                "perf_profiler.export_tensorboard_trace cannot both be true: torch "
                "permits a single kineto save per capture and the second raises "
                "'Trace is already saved.', aborting the run. Pick one; the "
                "tensorboard handler writes chrome-format JSON, so the chrome "
                "export loses nothing."
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


def _count_folded_lines(path: Path) -> int:
    """Count non-blank folded-stack lines, streaming so a large export is cheap."""
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


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
        # One record per (capture, metric) folded-stack export attempt.  Stays
        # empty unless ``export_stacks`` is set, so a run that never asked for
        # stacks is distinguishable from one whose stacks came back empty.
        self.stack_exports: List[Dict[str, Any]] = []

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

        # Measured on torch 2.10.0+cu126: ``with_stack=True`` alone leaves every
        # ``FunctionEvent.stack`` empty, so ``export_stacks`` writes a zero-line
        # file.  The verbose experimental config is what attaches Python frames
        # to the events.  Gated on ``export_stacks`` so a run that does not ask
        # for folded stacks constructs exactly the profiler it constructed
        # before (``experimental_config=None`` is torch's own default).
        experimental_config = None
        if config.export_stacks:
            experimental_config_cls = getattr(torch.profiler, "_ExperimentalConfig", None)
            if experimental_config_cls is None:
                warnings.warn(
                    "torch.profiler._ExperimentalConfig is unavailable; folded-stack "
                    "exports may contain no frames on this torch build.",
                    stacklevel=2,
                )
            else:
                experimental_config = experimental_config_cls(verbose=True)

        self._profiler = torch.profiler.profile(
            activities=activities,
            experimental_config=experimental_config,
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
        # Ordered before the tensorboard handler on purpose: with both
        # export_chrome_trace and export_tensorboard_trace set, torch raises
        # "Trace is already saved." from the second kineto save.  That is a
        # pre-existing defect (it reproduces with export_stacks unset), but it
        # must not be able to swallow the folded stacks a flame chart needs.
        if self.config.export_stacks:
            self._export_folded_stacks(profiler, trace_dir)
        if self.config.export_tensorboard_trace:
            handler = torch.profiler.tensorboard_trace_handler(str(trace_dir))
            handler(profiler)
        self._trace_count += 1

    def _export_folded_stacks(self, profiler: torch.profiler.profile, trace_dir: Path) -> None:
        """Write folded stacks for every metric, plus a manifest of the attempt.

        ``export_stacks`` opens its output file before it filters events, so a
        metric with no qualifying events (``self_cuda_time_total`` on a CPU-only
        profile) leaves a zero-byte file behind rather than raising.  Such a file
        is removed and the reason recorded: an empty flamegraph input is worse
        than an absent one, but a silently absent one is worse still.
        """
        records: List[Dict[str, Any]] = []
        for metric in STACK_EXPORT_METRICS:
            path = trace_dir / f"stacks_{metric}.txt"
            record: Dict[str, Any] = {
                "capture": trace_dir.name,
                "metric": metric,
                "file": path.name,
                "lines": 0,
            }
            try:
                profiler.export_stacks(str(path), metric)
            except _STACK_EXPORT_ERRORS as error:
                path.unlink(missing_ok=True)
                record["status"] = "unavailable"
                record["reason"] = f"{type(error).__name__}: {error}"
            else:
                lines = _count_folded_lines(path)
                if lines == 0:
                    path.unlink(missing_ok=True)
                    record["status"] = "unavailable"
                    reason = (
                        f"no profiled event carried both a recorded stack and a positive "
                        f"{metric}; the export was empty and was removed"
                    )
                    if "cuda" in metric:
                        reason += f" (cuda_available={torch.cuda.is_available()})"
                    record["reason"] = reason
                else:
                    record["status"] = "written"
                    record["lines"] = lines
            records.append(record)

        self.stack_exports.extend(records)
        manifest = {
            "capture": trace_dir.name,
            "with_stack": self.config.with_stack,
            "cuda_available": torch.cuda.is_available(),
            "metrics": records,
        }
        # Mode "x": trace_dir is created with exist_ok=False per capture, so an
        # existing manifest would mean an unexpected second writer.
        with open(trace_dir / "stacks_manifest.json", "x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")

        missing = [record["metric"] for record in records if record["status"] != "written"]
        if missing:
            warnings.warn(
                "Folded stacks unavailable for "
                + ", ".join(missing)
                + f"; see {trace_dir / 'stacks_manifest.json'} for the reason.",
                stacklevel=2,
            )
