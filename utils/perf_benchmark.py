"""Bounded, opt-in steady-state training benchmark (PERF-001 / P0.1a).

This module is the timing counterpart of :mod:`utils.perf_profiler`: it answers
"how many milliseconds does one canonical update cost, and where do they go?"
*without* attaching a profiler, because CUPTI perturbation invalidates the M2
throughput number we are trying to register.

Design properties that the pre-registration treats as load-bearing:

``synchronize`` is called ONCE per measured window
    CUDA events are *recorded* inside :meth:`TrainingBenchmark.cuda_span` and
    *retained*; their elapsed times are only resolved after the single
    :meth:`TrainingBenchmark._sync` that closes the 200-step window (plus the
    two boundary syncs of an armed :meth:`TrainingBenchmark.event_span`).  A
    per-update ``torch.cuda.synchronize()`` would serialise the pipeline and
    make every measured millisecond a measurement artefact.  To make that
    structurally hard to get wrong, ``_sync`` is the *only* place that calls
    ``sync_fn`` and it increments :attr:`TrainingBenchmark.sync_count`, which
    the test-suite asserts is exactly ``1`` after a full window.

    Read this as a bound on *collector-issued* synchronization, not as a claim
    that the measured update contains no device serialisation at all.  It does.
    TWO pre-existing per-update implicit syncs sit inside the measured window;
    both are present at ``HEAD:pretrain.py`` and neither was introduced here.

    (1) ``metric_values.cpu().numpy()`` is a blocking D2H copy on every update.
    That copy is part of the workload M2 is registered to
    measure, and the frozen timing contract assigns D2H/metrics-W&B a
    ``perf_counter`` bracket, which is what the ``metrics_wandb`` wall span is.
    The pre-registration goes further and treats this copy as a *finding to be
    measured*, not a harness defect: ``2026-07-26_experiment-speed-profiling.md``
    line 74 catalogues "train metric은 매 update D2H 후 W&B log", hypothesis S-2
    (line 109) is "logging/D2H dominated" naming ``metrics.cpu().numpy()``, and
    the candidate remedy of batching it (line 361) is recorded as ``unrun``.
    Eliminating it here would destroy the very baseline that decides S-2 and
    would change the hot path whose cost P0.1a exists to establish.  It also cannot
    corrupt the CUDA numbers: events are recorded on-stream and resolved after
    the window-closing sync, so an intervening blocking copy leaves every
    ``elapsed_time`` intact.

    (2) the ``{k: v.cuda() ...}`` H2D inside ``train_batch``'s ``h2d`` span
    (``pretrain.py:317-318``) is ``non_blocking=False`` and therefore
    synchronizes the stream on every update as well.  See PERF-DEV-07 below.

    CONSEQUENCE, and the only safe reading: ``sync_count == 1`` bounds
    COLLECTOR-ISSUED synchronization only.  It is NOT evidence that the measured
    update is free of device serialisation, and must never be reported as such.

the disabled path is a strict no-op
    ``__init__`` returns before creating anything when the collector is not
    ``config.enabled and rank == 0``; every span method then returns the single
    module-level :data:`_NULL_SPAN` after one frozenset membership test and one
    boolean check.  No ``torch.cuda`` attribute is ever touched, no
    ``perf_counter`` is ever read, no directory is ever created.

CUDA is never touched at import time
    ``torch`` is imported at module scope (as ``perf_profiler`` does) but every
    CUDA entry point is reached through an injectable callable that is resolved
    lazily, so the pure statistics helpers and the writers are unit-testable on
    a host with no free GPU.

``gpu_util_pct`` is a driver-window aggregate, not a per-update measurement
    The column is part of the frozen 25-column schema and is populated once per
    measured row, but its source (``torch.cuda.utilization``, NVML-backed)
    reports "percent of the past sample period during which a kernel ran", and
    that period is between 1 s and 1/6 s.  A 200-row window of millisecond-scale
    updates spans far less than one such period, so consecutive rows repeat the
    same driver-side aggregate.  Downstream analysis must treat it as run-level
    context only: never difference it across rows, never attribute it to a
    single update, and never read its row-to-row variation as signal.

Deviations from the literal frozen spec text are numbered ``PERF-DEV-NN``.  The
AUTHORITY for that numbering is
``lab/reports/2026-07-26_experiment-speed-action-plan.md``, section
"2026-07-28 P0.1a 구현 비준 원장 (PERF-DEV-NN)".  That ledger is the register.
This docstring is NOT the register and claims no completeness: it restates the
entries a reader of this module needs, and the ledger holds the remainder along
with the authoritative status of every ID.  Where the two disagree, the ledger
wins.

Label discipline.  The former module-local working labels ``R1``/``R2``/``R7``/
``R8`` were never pre-registered risk IDs and no risk register defining them
exists.  They are gone.  ``R1`` and ``R2`` are now PERF-DEV-01 and PERF-DEV-02;
the two former ``SPEC-NOTE``s are PERF-DEV-03 and PERF-DEV-04.  ``R7`` (lazy
``torch.cuda.Event`` handle creation, ``_prepare_window``) and ``R8`` (a schema
assertion in ``tests/test_perf_benchmark.py`` that would pass vacuously on a
field-name typo) were never departures at all; they carry no ``PERF-DEV`` ID and
are written out as plain LIMITATIONS at the sites where they occur.

* PERF-DEV-01 -- RATIFIED at implementation level.  ``event_span`` arms on
  ``step >= eval_event_step``, not ``step ==``.  Under the registered M2 preset
  (``epochs=2000 eval_interval=2000``) the single eval block fires near step
  ~4.9k rather than exactly 2000, so an equality test would never arm and the
  six event columns would stay ``evidence_pending``.  What bounds the run there
  is the Hydra ``epochs=2000``, NOT ``perf_benchmark.max_steps`` -- see
  PERF-DEV-06.  First matching block per name wins.  This is the one departure
  that changes which rows exist, so it is listed first.  See ``event_span`` (the
  ``>=`` comment) -- and note that the ``_window_closed`` gate immediately above
  it, not the step comparison, is what makes a mid-window device drain
  unreachable; the ledger ratifies the departure on exactly that ground.
* PERF-DEV-02 -- AWAITING PI SIGN-OFF, not settled; behaviour is frozen until it
  lands.  The ``metrics_device`` CUDA span is collected but has NO CSV column:
  the timing contract asks for a CUDA event pair on metric-device work while the
  25-column header is byte-frozen without such a column, and both cannot hold.
  The span's ``summarize()`` aggregate is published in ``manifest.json`` under
  ``extra_span_summary``.  The ledger *recommends* keeping the byte-freeze and
  ratifying the manifest route, but records the decision as the PI's because it
  reinterprets a pre-registered schema.
* PERF-DEV-03 -- RATIFIED at implementation level.  ``data_wait`` is bracketed
  before its row exists, so it is staged and promoted into the update it fed
  instead of being dropped.
* PERF-DEV-04 -- RATIFIED at implementation level.  Token counts are read before
  the update-wall timer starts, so the label scan cannot inflate
  ``update_wall_ms``.
* PERF-DEV-05 -- RATIFIED at implementation level.  ``TrainingBenchmark`` is
  constructed BEFORE ``profiler.start()`` in ``pretrain.py``, so the
  profiler/benchmark mutual-exclusion ``ValueError`` is raised before any
  profiler resource is acquired and outside the ``try:`` whose
  ``finally: profiler.stop()`` would not cover it.
* PERF-DEV-06 -- RECORDED, NOT RATIFIED.  ``perf_benchmark.max_steps`` bounds
  ROW COLLECTION ONLY.  It does NOT stop training.  This is unlike
  ``perf_profiler.max_steps``, which does bound the run, so the name reads
  backwards and must not be relied on to end a job.  Under the registered M2
  preset the real bound on training length is the Hydra ``epochs=2000``
  (~4.9k optimizer steps); ``perf_benchmark.max_steps=2000`` only caps how many
  measured rows are retained.
* PERF-DEV-07 -- RECORDED, NOT RATIFIED.  This is item ``(2)`` of the sync note
  above, spelled out: a SECOND per-update implicit synchronization inside the
  measured window, alongside item ``(1)``'s ``metric_values.cpu().numpy()`` D2H.
  The ``{k: v.cuda() ...}`` H2D in ``train_batch``'s ``h2d`` span
  (``pretrain.py:317-318``; the ledger cites 317) is ``non_blocking=False`` and
  synchronizes the stream on every update.  It is pre-existing at ``HEAD`` and
  was not introduced by this module.  Therefore ``sync_count == 1`` bounds
  COLLECTOR-ISSUED synchronization ONLY and is NOT evidence that the measured
  update is free of device serialisation.  One line of
  ``torch.cuda.set_sync_debug_mode("warn")`` on an approved M2 run confirms it;
  the item feeds Phase 1 candidate 3 (non-blocking H2D).
* PERF-DEV-10 -- AWAITING PI SIGN-OFF, not settled; behaviour is frozen until it
  lands.  Artifact scope: a completed M2 baseline repeat writes
  ``steady_state.csv`` and ``manifest.json`` only.  ``append_equivalence_ledger``
  and ``capture_resource_manifest`` are deliberately caller-less here --
  ``equivalence_ledger.csv`` is a G1/G2 *gate* artifact (its candidate/control
  IDs and G1/G2 status cannot be filled by a baseline run that has no
  candidate), and the resource manifest is P0.0's.  The ledger finds that
  argument sound and *recommends* moving the equivalence ledger from a P0.1a
  required artifact to a G1 one, but that edits pre-registered contract text and
  so is the PI's call, not the implementer's.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import re
import socket
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import numpy as np
import pydantic
import torch

# --------------------------------------------------------------------------- #
# Module constants (frozen; values are part of the artifact contract)
# --------------------------------------------------------------------------- #

SCHEMA_VERSION: str = "perf_benchmark/1"
RESOURCE_SCHEMA_VERSION: str = "perf_benchmark.resource/1"

#: Written for any counter that was never captured.  Never write ``0.0`` for a
#: span that did not fire -- a silent zero is indistinguishable from a real
#: measurement and corrupts every downstream aggregate.
MISSING: str = "evidence_pending"

#: Local copy of ``models/losses.py:8``.  Copied rather than imported because
#: importing ``models.losses`` from here would be circular via ``pretrain``.
IGNORE_LABEL_ID: int = -100

#: ONE singleton, allocated at import.  Every disabled span returns this exact
#: object, so the disabled path costs an attribute load plus ``nullcontext``
#: enter/exit (~100 ns against a millisecond-scale update).  ``nullcontext`` is
#: stateless and re-entrant, so sharing it is safe.
_NULL_SPAN: ContextManager[None] = contextlib.nullcontext()

STEADY_STATE_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "condition_id",
    "run_id",
    "repeat",
    "seed",
    "step",
    "global_effective_batch",
    "input_tokens",
    "target_tokens",
    "data_wait_ms",
    "update_wall_ms",
    "h2d_cuda_ms",
    "forward_backward_cuda_ms",
    "optimizer_cuda_ms",
    "ema_cuda_ms",
    "metrics_wandb_wall_ms",
    "eval_event_ms",
    "zprobe_event_ms",
    "checkpoint_event_ms",
    "eval_amortized_ms",
    "zprobe_amortized_ms",
    "checkpoint_amortized_ms",
    "max_memory_allocated",
    "max_memory_reserved",
    "gpu_util_pct",
)

EQUIVALENCE_LEDGER_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "control_id",
    "fixed_batch_hash",
    "sample_index_hash",
    "config_diff",
    "schema_version",
    "tolerance",
    "g1_status",
    "g2_status",
)

CUDA_SPAN_NAMES: tuple[str, ...] = (
    "h2d",
    "forward_backward",
    "optimizer",
    "ema",
    "metrics_device",
)
WALL_SPAN_NAMES: tuple[str, ...] = ("data_wait", "metrics_wandb")
EVENT_SPAN_NAMES: tuple[str, ...] = ("eval", "zprobe", "checkpoint")

_CUDA_SPAN_SET = frozenset(CUDA_SPAN_NAMES)
_WALL_SPAN_SET = frozenset(WALL_SPAN_NAMES)
_EVENT_SPAN_SET = frozenset(EVENT_SPAN_NAMES)

#: ``metrics_device`` has no column in the frozen 25-column schema (PERF-DEV-02
#: in the module docstring -- AWAITING PI SIGN-OFF, so this stands unchanged
#: meanwhile).  It is still collected, and its aggregate is
#: published in ``manifest.json`` under ``extra_span_summary`` so that the CSV
#: header stays byte-frozen.
_CUDA_SPAN_COLUMN: dict[str, Optional[str]] = {
    "h2d": "h2d_cuda_ms",
    "forward_backward": "forward_backward_cuda_ms",
    "optimizer": "optimizer_cuda_ms",
    "ema": "ema_cuda_ms",
    "metrics_device": None,
}
_WALL_SPAN_COLUMN: dict[str, str] = {
    "data_wait": "data_wait_ms",
    "metrics_wandb": "metrics_wandb_wall_ms",
}
_EVENT_SPAN_COLUMN: dict[str, tuple[str, str]] = {
    "eval": ("eval_event_ms", "eval_amortized_ms"),
    "zprobe": ("zprobe_event_ms", "zprobe_amortized_ms"),
    "checkpoint": ("checkpoint_event_ms", "checkpoint_amortized_ms"),
}

#: PERF-DEV-03 (RATIFIED at implementation level by the action plan's
#: "2026-07-28 P0.1a 구현 비준 원장" -- see the module docstring for the
#: authority): ``data_wait`` is the only wall span that is bracketed *before*
#: the row it belongs to is opened (``iter_batches``' ``next()`` runs before
#: ``train_batch`` reaches ``begin_update``).  Read literally, "return
#: ``_NULL_SPAN`` unless a measured row is open" would leave ``data_wait_ms``
#: permanently ``evidence_pending`` -- a dead frozen column.  It is therefore
#: staged and promoted into the update it fed.  ``metrics_wandb`` does NOT need
#: this: both of its sites are inside the open row.
_STAGED_WALL_SPANS = frozenset({"data_wait"})

DEFAULT_RESOURCE_MANIFEST_DIR: str = (
    "reports/figures/2026-07-26_experiment-speed-profiling/manifests"
)

#: The three P0.0 read-only probes.  These are shelled out ONLY inside
#: :func:`capture_resource_manifest`, never at import and never from a test
#: (tests inject a fake ``runner``): ``queue_run.sh status`` must not be
#: executed while live training jobs own the GPUs.
DEFAULT_RESOURCE_PROBES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("queue_status", ("scripts/queue_run.sh", "status")),
    ("enqueue_dry_run", ("scripts/sigma_enqueue.sh", "--dry-run", "perf0")),
    (
        "nvidia_smi",
        (
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,utilization.gpu,memory.used,"
            "memory.total,clocks.current.sm,power.draw",
            "--format=csv,noheader",
        ),
    ),
)


# --------------------------------------------------------------------------- #
# Pure statistics (no torch, no CUDA, no I/O)
# --------------------------------------------------------------------------- #


def median(values: Sequence[float]) -> float:
    """Sorted middle; the mean of the two central values when ``N`` is even."""
    n = len(values)
    if n == 0:
        raise ValueError("median() requires at least one value")
    ordered = sorted(float(v) for v in values)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def percentile_nearest_rank(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile -- THE only percentile convention in this module.

    ``index = clamp(ceil(q/100 * N) - 1, 0, N-1)`` on the sorted copy, so
    ``p95 == percentile_nearest_rank(v, 95.0)``.
    """
    n = len(values)
    if n == 0:
        raise ValueError("percentile_nearest_rank() requires at least one value")
    if not (0.0 <= q <= 100.0):
        raise ValueError(f"percentile q must be in [0, 100], got {q}")
    ordered = sorted(float(v) for v in values)
    index = math.ceil(q / 100.0 * n) - 1
    index = min(max(index, 0), n - 1)
    return ordered[index]


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean over all rows."""
    n = len(values)
    if n == 0:
        raise ValueError("mean() requires at least one value")
    return math.fsum(float(v) for v in values) / n


def sample_sd(values: Sequence[float]) -> float:
    """``sqrt(sum((x-mean)^2) / (N-1))``; ``0.0`` when ``N < 2``."""
    n = len(values)
    if n < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(math.fsum((float(v) - mu) ** 2 for v in values) / (n - 1))


def coefficient_of_variation(values: Sequence[float]) -> float:
    """``sample_sd / mean``; ``nan`` when the mean is exactly zero."""
    mu = mean(values)
    if mu == 0.0:
        return float("nan")
    return sample_sd(values) / mu


def bootstrap_median_ci(
    values: Sequence[float],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Non-parametric percentile CI for the median; deterministic given ``seed``."""
    n = len(values)
    if n == 0:
        raise ValueError("bootstrap_median_ci() requires at least one value")
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    # np.median's even-N convention matches the pure ``median`` above.
    draws = np.median(arr[rng.integers(0, n, size=(resamples, n))], axis=1)
    lo_q = 100.0 * (1.0 - confidence) / 2.0
    hi_q = 100.0 * (1.0 + confidence) / 2.0
    medians = draws.tolist()
    return (
        percentile_nearest_rank(medians, lo_q),
        percentile_nearest_rank(medians, hi_q),
    )


def hierarchical_bootstrap_median_ci(
    repeats: Sequence[Sequence[float]],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Two-stage bootstrap: resample repeats, then rows within each drawn repeat.

    The 600 rows of a three-repeat condition are NOT independent -- they share a
    process launch, a data-loader shuffle and a thermal state.  Flattening them
    before resampling understates the interval, so this function must never see
    a flat list.
    """
    groups = [np.asarray(r, dtype=np.float64) for r in repeats]
    if not groups:
        raise ValueError("hierarchical_bootstrap_median_ci() requires at least one repeat")
    if any(g.ndim != 1 for g in groups):
        # A flat ``[1.0, 2.0, 3.0]`` yields 0-d elements whose ``.size`` is 1, so
        # the emptiness check below waves it through and stage 2 then dies with
        # an opaque IndexError.  Reject it here: passing pooled rows to a
        # hierarchical bootstrap is precisely the error this function exists to
        # make impossible.
        raise ValueError(
            "hierarchical_bootstrap_median_ci() requires a nested sequence of "
            "per-repeat row sequences, not a flat sequence of rows"
        )
    if any(g.size == 0 for g in groups):
        raise ValueError("hierarchical_bootstrap_median_ci() requires non-empty repeats")
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    n_repeats = len(groups)
    rng = np.random.default_rng(seed)
    repeat_draws = rng.integers(0, n_repeats, size=(resamples, n_repeats))
    medians: list[float] = []
    for draw in repeat_draws:
        pooled = np.concatenate(
            [
                groups[j][rng.integers(0, groups[j].size, size=groups[j].size)]
                for j in draw
            ]
        )
        medians.append(float(np.median(pooled)))
    lo_q = 100.0 * (1.0 - confidence) / 2.0
    hi_q = 100.0 * (1.0 + confidence) / 2.0
    return (
        percentile_nearest_rank(medians, lo_q),
        percentile_nearest_rank(medians, hi_q),
    )


def summarize(
    values: Sequence[float], *, resamples: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """The only aggregation entrypoint reports may call."""
    if len(values) == 0:
        raise ValueError("summarize() requires at least one value")
    ci_low, ci_high = bootstrap_median_ci(values, resamples=resamples, seed=seed)
    return {
        "n": len(values),
        "p50": median(values),
        "p95": percentile_nearest_rank(values, 95.0),
        "mean": mean(values),
        "sd": sample_sd(values),
        "cv": coefficient_of_variation(values),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


_REPEAT_SUFFIX = re.compile(r"_r\d+$")


def derive_condition_id(run_name: Optional[str]) -> str:
    """``perf0_..._r2`` -> ``perf0_...``; ``"unknown"`` when there is no run name."""
    if not run_name:
        return "unknown"
    return _REPEAT_SUFFIX.sub("", run_name)


def derive_repeat(run_name: Optional[str], configured_repeat: int) -> int:
    """An explicit ``repeat`` always wins; otherwise read the ``_r<n>`` suffix."""
    if configured_repeat != 1:
        return configured_repeat
    if run_name:
        match = re.search(r"_r(\d+)$", run_name)
        if match:
            return int(match.group(1))
    return 1


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class PerfBenchmarkConfig(pydantic.BaseModel):
    """Configuration for a bounded, unprofiled steady-state timing run (M2)."""

    model_config = pydantic.ConfigDict(extra="forbid")

    enabled: bool = False
    # This acknowledgement prevents a benchmark run being read as science.
    performance_only: bool = False
    output_dir: str = "performance_benchmarks"

    warmup_steps: int = pydantic.Field(default=50, ge=0, le=100_000)
    measured_steps: int = pydantic.Field(default=200, ge=1, le=100_000)
    eval_event_step: int = pydantic.Field(default=2000, ge=1)
    max_steps: int = pydantic.Field(default=2000, ge=1, le=10_000_000)

    condition_id: str = ""
    repeat: int = pydantic.Field(default=1, ge=1, le=16)

    bootstrap_resamples: int = pydantic.Field(default=10_000, ge=100, le=100_000)
    bootstrap_seed: int = pydantic.Field(default=0, ge=0)

    @pydantic.model_validator(mode="after")
    def validate_bounded_benchmark(self) -> "PerfBenchmarkConfig":
        # Every check is gated on ``enabled`` so a disabled incoherent config
        # validates clean, exactly as perf_profiler.py:42/47/50 does.
        if self.enabled and not self.performance_only:
            raise ValueError(
                "perf_benchmark.enabled requires perf_benchmark.performance_only=true; "
                "benchmark timing must not be used as a science run."
            )
        if self.enabled and not self.output_dir.strip():
            raise ValueError(
                "perf_benchmark.output_dir must be set when benchmarking is enabled"
            )
        scheduled = self.warmup_steps + self.measured_steps
        if self.enabled and self.max_steps < scheduled:
            raise ValueError(
                "perf_benchmark.max_steps is shorter than its measurement schedule "
                f"({self.max_steps} < {scheduled})"
            )
        if self.enabled and self.eval_event_step <= scheduled:
            # An armed ``event_span`` synchronizes on both of its boundaries.  If
            # it can fire while the measured window is still collecting, those
            # two device-wide drains land *inside* the window and every row after
            # them is measured against a freshly drained pipeline -- the exact
            # failure class the once-per-window sync budget exists to prevent.
            # ``train_state.step`` advances by one per update (pretrain.py:305),
            # so this is a schedule check in the same units.  It is a guard, not
            # a proof: the ``total_steps`` early return advances ``step`` without
            # opening a row, so ``step >= update_index``.  The unconditional
            # ``_window_closed`` gate in ``event_span`` is what actually makes
            # a mid-window drain unreachable.
            raise ValueError(
                "perf_benchmark.eval_event_step must be greater than "
                "warmup_steps + measured_steps so an event span cannot "
                f"synchronize inside the measured window ({self.eval_event_step} <= {scheduled})"
            )
        return self


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Span context managers
# --------------------------------------------------------------------------- #


class _CudaSpan:
    """Records a retained CUDA event pair.  NEVER synchronizes."""

    __slots__ = ("_bench", "_row", "_name", "_start")

    def __init__(self, bench: "TrainingBenchmark", row: dict, name: str) -> None:
        self._bench = bench
        self._row = row
        self._name = name
        self._start: Any = None

    def __enter__(self) -> None:
        self._start = self._bench._take_event()
        self._start.record()

    def __exit__(self, exc_type, exc, tb) -> bool:
        end = self._bench._take_event()
        end.record()
        # RETAIN the pair; ``elapsed_time`` is only called after the single
        # window-closing synchronize.  Resolving here would require a sync.
        self._bench._pending_pairs.append((self._row, self._name, self._start, end))
        return False


class _WallSpan:
    """Accumulates ``perf_counter`` deltas into one row column (sums repeats)."""

    __slots__ = ("_bench", "_row", "_name", "_t0")

    def __init__(self, bench: "TrainingBenchmark", row: Optional[dict], name: str) -> None:
        self._bench = bench
        self._row = row
        self._name = name
        self._t0 = 0.0

    def __enter__(self) -> None:
        self._t0 = time.perf_counter()

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed_ms = (time.perf_counter() - self._t0) * 1e3
        target = self._bench._staging if self._row is None else self._row["_wall"]
        target[self._name] = target.get(self._name, 0.0) + elapsed_ms
        return False


class _EventSpan:
    """Brackets a rare, expensive event; the only legal sync site besides the window close."""

    __slots__ = ("_bench", "_name", "_t0")

    def __init__(self, bench: "TrainingBenchmark", name: str) -> None:
        self._bench = bench
        self._name = name
        self._t0 = 0.0

    def __enter__(self) -> None:
        self._bench._event_armed.add(self._name)
        self._bench._sync()
        self._t0 = time.perf_counter()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._bench._sync()
        self._bench._event_ms[self._name] = (time.perf_counter() - self._t0) * 1e3
        return False


# --------------------------------------------------------------------------- #
# Collector
# --------------------------------------------------------------------------- #


class TrainingBenchmark:
    """Rank-zero steady-state collector: retains event pairs, syncs once per window."""

    def __init__(
        self,
        config: PerfBenchmarkConfig,
        *,
        checkpoint_path: Optional[str],
        rank: int,
        run_name: Optional[str] = None,
        seed: int = 0,
        eval_interval: Optional[int] = None,
        resolved_config: Optional[Mapping[str, Any]] = None,
        data_paths: Sequence[str] = (),
        profiler_enabled: bool = False,
        event_factory: Optional[Callable[[], Any]] = None,
        sync_fn: Optional[Callable[[], None]] = None,
        utilization_fn: Optional[Callable[[], float]] = None,
        memory_stats_fn: Optional[Callable[[], tuple[int, int]]] = None,
        reset_peak_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        self.config = config
        self.rank = rank
        self._active = False
        self._finalized = False
        self._session_dir: Optional[Path] = None
        self._sync_count = 0
        self.output_dir: Optional[Path] = None
        # Set BEFORE the early return: ``cuda_span``/``wall_span`` read
        # ``self._row`` on the disabled path, so the attribute must exist even
        # when nothing else is constructed.
        self._row: Optional[dict] = None

        # All ranks run the same training path; only rank zero may own timing
        # state or write artifacts (perf_profiler.py:94 pattern).  A non-active
        # collector creates NOTHING -- no directory, no CUDA call, no timer.
        if not (config.enabled and rank == 0):
            return

        if profiler_enabled:
            raise ValueError(
                "perf_benchmark cannot run while perf_profiler.enabled=true; "
                "profiler perturbation invalidates M2 timing"
            )

        output_dir = Path(config.output_dir).expanduser().resolve()
        if checkpoint_path is not None:
            checkpoint_dir = Path(checkpoint_path).expanduser().resolve()
            if _is_relative_to(output_dir, checkpoint_dir) or _is_relative_to(
                checkpoint_dir, output_dir
            ):
                raise ValueError(
                    "perf_benchmark.output_dir must be separate from checkpoint_path so "
                    "benchmark artifacts cannot mix with science-run artifacts"
                )
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        # Never overwrite an earlier session, including a rerun of the same
        # Hydra command.  Rank zero is the sole creator of these paths.
        for session_id in range(10_000):
            candidate = output_dir / f"session_{session_id:04d}"
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            self._session_dir = candidate
            break
        if self._session_dir is None:
            raise RuntimeError(f"could not allocate a benchmark session under {output_dir}")

        warnings.warn(
            "Performance benchmark enabled: this is a performance-only run. Timing output "
            f"will be written to {output_dir}; its science metrics are void and must not be "
            "compared to science runs.",
            stacklevel=2,
        )

        self._active = True
        self._run_id = run_name if run_name else MISSING
        self._condition_id = config.condition_id or derive_condition_id(run_name)
        self._repeat = derive_repeat(run_name, config.repeat)
        self._seed = int(seed)
        self._eval_interval = eval_interval
        self._resolved_config = dict(resolved_config or {})
        self._data_paths = list(data_paths)
        self._start_utc = _utc_now_iso()

        # Injectables.  Resolved here (never at import) and only ever CALLED on
        # the measured path, so a CPU-only test that injects fakes never reaches
        # torch.cuda at all.
        self._event_factory = event_factory or _default_event_factory
        self._sync_fn = sync_fn or torch.cuda.synchronize
        self._utilization_fn = utilization_fn or torch.cuda.utilization
        self._memory_stats_fn = memory_stats_fn or _default_memory_stats
        self._reset_peak_fn = reset_peak_fn or torch.cuda.reset_peak_memory_stats

        # State machine.
        self._update_index = 0
        self._rows: list[dict] = []
        self._row_t0 = 0.0
        self._staging: dict[str, float] = {}
        self._pending_pairs: list[tuple[dict, str, Any, Any]] = []
        self._event_pool: list[Any] = []
        self._event_cursor = 0
        self._event_pool_overflow = 0
        self._prepared = False
        self._window_closed = False
        self._event_ms: dict[str, float] = {}
        self._event_armed: set[str] = set()
        self._rows_dropped_incomplete = 0

    # -- predicates -------------------------------------------------------- #

    @property
    def enabled(self) -> bool:
        """Live-state predicate (False on rank != 0 and after ``finalize``)."""
        return self._active

    @property
    def sync_count(self) -> int:
        """Number of ``sync_fn`` calls so far -- the enforceable headline property."""
        return self._sync_count

    @property
    def session_dir(self) -> Optional[Path]:
        """The claimed non-overwriting ``session_%04d`` directory, else ``None``.

        Retained after ``finalize`` so callers can locate the written artifacts.
        """
        return self._session_dir

    # -- the single synchronization funnel --------------------------------- #

    def _sync(self) -> None:
        """The ONLY place ``sync_fn`` is called.

        Routing every synchronize through one counted method is what makes the
        headline correctness property enforceable rather than conventional:
        ``cuda_span`` can only ``record()`` events, and ``elapsed_time`` is only
        reachable from ``_resolve_pending`` which runs after this call.  There
        are exactly three call sites -- window close in ``end_update`` and the
        enter/exit of an armed ``event_span``.  A per-update synchronize would
        serialise the CUDA pipeline and invalidate every measured millisecond,
        so adding a fourth call site must be a deliberate, reviewed act.
        """
        self._sync_count += 1
        self._sync_fn()

    # -- spans -------------------------------------------------------------- #

    def cuda_span(self, name: str) -> ContextManager[None]:
        # Name validation is unconditional (before the active check) so a bad
        # span name fails loudly on the disabled path too.  A frozenset test is
        # ~40 ns, which keeps the disabled-path no-op claim intact.
        if name not in _CUDA_SPAN_SET:
            raise KeyError(f"unknown cuda_span name {name!r}; legal names: {CUDA_SPAN_NAMES}")
        row = self._row
        if row is None:
            return _NULL_SPAN
        return _CudaSpan(self, row, name)

    def wall_span(self, name: str) -> ContextManager[None]:
        if name not in _WALL_SPAN_SET:
            raise KeyError(f"unknown wall_span name {name!r}; legal names: {WALL_SPAN_NAMES}")
        row = self._row
        if row is not None:
            return _WallSpan(self, row, name)
        # See PERF-DEV-03 at ``_STAGED_WALL_SPANS``: the loader wait is
        # measured before its update opens a row, so it is staged and promoted.
        # Once the window is closed no further row can consume a staged value,
        # so stop reading the clock entirely rather than accumulating into a
        # dict nobody drains.
        if self._active and not self._window_closed and name in _STAGED_WALL_SPANS:
            return _WallSpan(self, None, name)
        return _NULL_SPAN

    def event_span(self, name: str, step: int) -> ContextManager[None]:
        if name not in _EVENT_SPAN_SET:
            raise KeyError(f"unknown event_span name {name!r}; legal names: {EVENT_SPAN_NAMES}")
        if not self._active:
            return _NULL_SPAN
        if not self._window_closed:
            # Defence in depth for the validator's eval_event_step check: an
            # armed event span drains the device on both boundaries, so it must
            # never open while the measured window is still collecting rows.
            # Refusing to arm leaves the six event columns MISSING -- an honest,
            # visible gap -- instead of silently corrupting the rows that follow.
            return _NULL_SPAN
        # ``>=``, never ``==``: with epochs=2000/eval_interval=2000 the single
        # eval block fires at step ~4.9k for k=6, not at 2000.  An equality test
        # would never arm and the six event columns would stay MISSING.  First
        # matching block per name wins; later ones are ignored.  This is
        # PERF-DEV-01, ratified at implementation level by the action plan's
        # "2026-07-28 P0.1a 구현 비준 원장" (see the module docstring).
        if step < self.config.eval_event_step:
            return _NULL_SPAN
        if name in self._event_armed:
            return _NULL_SPAN
        return _EventSpan(self, name)

    # -- loader wrapper ----------------------------------------------------- #

    def iter_batches(self, loader: Iterable[Any]) -> Iterator[Any]:
        """Bracket ONLY ``next()`` in ``wall_span("data_wait")``.

        ``pretrain.py`` calls this only when ``bench.enabled``, so the disabled
        path never even enters this generator frame.
        """
        iterator = iter(loader)
        while True:
            with self.wall_span("data_wait"):
                try:
                    item = next(iterator)
                except StopIteration:
                    return
            yield item

    # -- update lifecycle --------------------------------------------------- #

    def begin_update(self, batch: Mapping[str, Any], global_effective_batch: int) -> None:
        if not self._active:
            return

        index = self._update_index
        self._update_index += 1

        staged = self._staging
        self._staging = {}

        measured = (
            not self._window_closed
            and index >= self.config.warmup_steps
            and index < self.config.max_steps
            and len(self._rows) < self.config.measured_steps
        )
        if not measured:
            # Warmup / past-max_steps updates consume no event budget and the
            # loader wait they staged is discarded with them.
            return

        if not self._prepared:
            self._prepare_window()

        row: dict[str, Any] = {
            "step": MISSING,
            "global_effective_batch": int(global_effective_batch),
            "input_tokens": MISSING,
            "target_tokens": MISSING,
            "update_wall_ms": MISSING,
            "max_memory_allocated": MISSING,
            "max_memory_reserved": MISSING,
            "gpu_util_pct": MISSING,
            "_cuda": {},
            "_wall": dict(staged),
        }

        # Token counts come off the CPU batch BEFORE ``.cuda()`` -- no CUDA is
        # touched here.  PERF-DEV-04 (RATIFIED at implementation level by the
        # action plan's "2026-07-28 P0.1a 구현 비준 원장"; see the module
        # docstring for the authority): counted BEFORE the update-wall timer
        # starts, so the ``!=``/``sum`` scan over a 2048-row label tensor cannot
        # inflate ``update_wall_ms``.  This mirrors the spec's own stated
        # principle for the ``end_update`` samplers.
        row["input_tokens"] = _count_input_tokens(batch)
        row["target_tokens"] = _count_target_tokens(batch)

        self._row = row
        self._row_t0 = time.perf_counter()

    def end_update(self, step: int) -> None:
        if not self._active:
            return
        row = self._row
        if row is None:
            # Called unconditionally by the train loop: a skipped update (the
            # total_steps early return), a warmup update, or a post-window
            # update all land here with nothing open.
            return

        # First statement, so nothing below is charged to the update.
        row["update_wall_ms"] = (time.perf_counter() - self._row_t0) * 1e3
        row["step"] = int(step)
        self._row = None

        # Sampled after every span is closed, so their cost lands in neither
        # ``update_wall_ms`` nor ``data_wait_ms``.
        try:
            allocated, reserved = self._memory_stats_fn()
            row["max_memory_allocated"] = int(allocated)
            row["max_memory_reserved"] = int(reserved)
        except Exception:  # pragma: no cover - defensive, provenance not science
            row["max_memory_allocated"] = MISSING
            row["max_memory_reserved"] = MISSING
        # Run-level context, NOT a per-update measurement: NVML's sample period
        # is 1 s to 1/6 s, far longer than one update, so consecutive rows repeat
        # one driver-side aggregate.  See the module docstring -- never difference
        # this column across rows.
        try:
            row["gpu_util_pct"] = float(self._utilization_fn())
        except Exception:
            row["gpu_util_pct"] = MISSING

        self._rows.append(row)
        if len(self._rows) >= self.config.measured_steps:
            self._close_window()

    # -- window management -------------------------------------------------- #

    def _prepare_window(self) -> None:
        """Arm the measured window once, at the first measured update."""
        self._prepared = True
        # Peak memory describes the steady state, not the torch.compile/startup
        # peak, so reset exactly once here.
        self._reset_peak_fn()
        budget = self.config.measured_steps * len(CUDA_SPAN_NAMES) * 2
        # Bulk preallocation hoists only the PYTHON-object allocation out of the
        # measured rows.  ``torch.cuda.Event`` is lazy -- it defers
        # ``cudaEventCreateWithFlags`` until the first ``record()`` -- so the
        # CUDA-side handle is still created inside the first span that uses each
        # event.  Allocation-cost mitigation is therefore only partial.  This is
        # a KNOWN LIMITATION, not a departure from the spec: it carries no
        # PERF-DEV ID because nothing the spec asks for changes.  The residue is
        # host-side microseconds charged to ``update_wall_ms``,
        # and it cannot inflate a CUDA elapsed_time because the handle exists
        # before the start event is stamped.  Warming the pool with real
        # ``record()`` calls would fire thousands of stream operations on a
        # timing-critical path and is deliberately not done.
        self._event_pool = [self._event_factory() for _ in range(budget)]
        self._event_cursor = 0

    def _take_event(self) -> Any:
        if self._event_cursor < len(self._event_pool):
            event = self._event_pool[self._event_cursor]
            self._event_cursor += 1
            return event
        # Never silently skip a span: allocate late and surface the count in
        # ``extra_span_summary`` so an over-budget window is visible.
        self._event_pool_overflow += 1
        return self._event_factory()

    def _resolve_pending(self) -> None:
        """Turn retained event pairs into milliseconds.  Callers MUST sync first."""
        for row, name, start, end in self._pending_pairs:
            bucket = row["_cuda"]
            bucket[name] = bucket.get(name, 0.0) + float(start.elapsed_time(end))
        self._pending_pairs = []
        # Drop the event objects; the window is over.
        self._event_pool = []
        self._event_cursor = 0

    def _close_window(self) -> None:
        self._window_closed = True
        self._sync()  # exactly once per measured window
        self._resolve_pending()

    # -- teardown ----------------------------------------------------------- #

    def finalize(self) -> None:
        if not self._active or self._finalized:
            return
        self._finalized = True

        if self._row is not None:
            # An update that began but never ended (exception on the train
            # path): its partial row is dropped rather than written half-filled.
            self._rows_dropped_incomplete += 1
            self._row = None

        if self._pending_pairs:
            # Only reachable when the run ended before ``measured_steps`` rows
            # were collected.  This is the one other legal sync site; it cannot
            # occur in a complete M2 run, which is why the suite asserts
            # ``sync_count == 1`` for the full-window case.  Resolving here is
            # strictly better than discarding real measurements.
            self._close_window()

        extra_span_summary = self._build_extra_span_summary()
        rows = [self._build_csv_row(r) for r in self._rows]

        assert self._session_dir is not None
        write_steady_state_csv(self._session_dir / "steady_state.csv", rows)

        device = _device_provenance()
        git_sha, git_dirty = git_provenance(_repo_root())
        payload = build_manifest(
            command=" ".join(sys.argv) if sys.argv else MISSING,
            resolved_config=self._resolved_config,
            data_hash=data_manifest_hash(self._data_paths),
            git_sha=git_sha,
            git_dirty=git_dirty,
            torch_version=device["torch_version"],
            cuda_version=device["cuda_version"],
            cudnn_version=device["cudnn_version"],
            driver_version=device["driver_version"],
            gpu_name=device["gpu_name"],
            gpu_clock_mhz=device["gpu_clock_mhz"],
            gpu_power_w=device["gpu_power_w"],
            hostname=socket.gethostname(),
            start_utc=self._start_utc,
            end_utc=_utc_now_iso(),
            condition_id=self._condition_id,
            run_id=self._run_id,
            repeat=self._repeat,
            seed=self._seed,
            warmup_steps=self.config.warmup_steps,
            measured_steps=self.config.measured_steps,
            rows_written=len(rows),
            sync_count=self._sync_count,
            extra_span_summary=extra_span_summary,
        )
        payload["bootstrap_resamples"] = self.config.bootstrap_resamples
        payload["bootstrap_seed"] = self.config.bootstrap_seed
        payload["eval_interval"] = self._eval_interval
        payload["eval_event_step"] = self.config.eval_event_step
        payload["max_steps"] = self.config.max_steps
        write_manifest(self._session_dir / "manifest.json", payload)

        self._active = False

    # -- row/manifest assembly ---------------------------------------------- #

    def _amortized(self, event_ms: Union[str, float]) -> Union[str, float]:
        if event_ms == MISSING:
            return MISSING
        if not self._eval_interval:
            return MISSING
        return float(event_ms) / float(self._eval_interval)

    def _build_csv_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        cuda = row["_cuda"]
        wall = row["_wall"]
        out: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "condition_id": self._condition_id,
            "run_id": self._run_id,
            "repeat": self._repeat,
            "seed": self._seed,
            "step": row["step"],
            "global_effective_batch": row["global_effective_batch"],
            "input_tokens": row["input_tokens"],
            "target_tokens": row["target_tokens"],
            "update_wall_ms": row["update_wall_ms"],
            "max_memory_allocated": row["max_memory_allocated"],
            "max_memory_reserved": row["max_memory_reserved"],
            "gpu_util_pct": row["gpu_util_pct"],
        }
        for name, column in _WALL_SPAN_COLUMN.items():
            out[column] = wall.get(name, MISSING)
        for name, column in _CUDA_SPAN_COLUMN.items():
            if column is None:
                continue  # metrics_device -> manifest extra_span_summary (PERF-DEV-02)
            out[column] = cuda.get(name, MISSING)
        for name, (event_col, amortized_col) in _EVENT_SPAN_COLUMN.items():
            event_ms = self._event_ms.get(name, MISSING)
            out[event_col] = event_ms
            out[amortized_col] = self._amortized(event_ms)
        return out

    def _build_extra_span_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "event_pool_overflow": self._event_pool_overflow,
            "rows_dropped_incomplete": self._rows_dropped_incomplete,
            "window_closed": self._window_closed,
        }
        # metrics_device has no CSV column by design (PERF-DEV-02); publish its
        # aggregate here so the frozen 25-column header stays byte-stable.
        values = [
            r["_cuda"]["metrics_device"]
            for r in self._rows
            if "metrics_device" in r["_cuda"]
        ]
        if values:
            summary["metrics_device_cuda_ms"] = summarize(
                values,
                resamples=self.config.bootstrap_resamples,
                seed=self.config.bootstrap_seed,
            )
        else:
            summary["metrics_device_cuda_ms"] = MISSING
        return summary


def _default_event_factory() -> Any:
    return torch.cuda.Event(enable_timing=True)


def _default_memory_stats() -> tuple[int, int]:
    return (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())


def _count_input_tokens(batch: Mapping[str, Any]) -> Union[int, str]:
    try:
        return int(batch["inputs"].numel())
    except Exception:
        return MISSING


def _count_target_tokens(batch: Mapping[str, Any]) -> Union[int, str]:
    try:
        return int((batch["labels"] != IGNORE_LABEL_ID).sum().item())
    except Exception:
        return MISSING


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _device_provenance() -> dict[str, Any]:
    """Best-effort device provenance that never initialises a CUDA context.

    ``torch.cuda.get_device_name`` and ``nvidia-smi`` are only consulted when a
    CUDA context already exists (i.e. a real training run).  On a CPU-only test
    host every GPU field is ``MISSING`` and no subprocess is spawned.
    """
    info: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": None,
        "driver_version": MISSING,
        "gpu_name": MISSING,
        "gpu_clock_mhz": None,
        "gpu_power_w": None,
    }
    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:  # pragma: no cover
        info["cudnn_version"] = None

    try:
        initialised = torch.cuda.is_available() and torch.cuda.is_initialized()
    except Exception:  # pragma: no cover
        initialised = False
    if not initialised:
        return info

    try:
        info["gpu_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:  # pragma: no cover
        info["gpu_name"] = MISSING
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=" + str(torch.cuda.current_device()),
                "--query-gpu=driver_version,clocks.current.sm,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            driver, clock, power = (
                part.strip() for part in result.stdout.strip().splitlines()[0].split(",")
            )
            info["driver_version"] = driver or MISSING
            info["gpu_clock_mhz"] = int(float(clock)) if clock else None
            info["gpu_power_w"] = float(power) if power else None
    except Exception:  # pragma: no cover
        pass
    return info


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #

_FLOAT_COLUMNS = frozenset(
    {c for c in STEADY_STATE_COLUMNS if c.endswith("_ms")} | {"gpu_util_pct"}
)


def _format_cell(column: str, value: Any) -> Any:
    if value == MISSING:
        return MISSING
    if column in _FLOAT_COLUMNS:
        return f"{float(value):.6f}"
    return value


def write_steady_state_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the frozen 25-column steady-state CSV.

    ``DictWriter`` silently writes ``''`` for a *missing* fieldname, so the
    explicit key-set check below is what actually guarantees schema fidelity.
    """
    path = Path(path)
    expected = set(STEADY_STATE_COLUMNS)
    for index, row in enumerate(rows):
        if set(row) != expected:
            missing = sorted(expected - set(row))
            unexpected = sorted(set(row) - expected)
            raise ValueError(
                f"steady_state row {index} does not match the frozen schema "
                f"(missing={missing}, unexpected={unexpected})"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(STEADY_STATE_COLUMNS),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _format_cell(c, row[c]) for c in STEADY_STATE_COLUMNS})


def build_manifest(
    *,
    command: str,
    resolved_config: Mapping[str, Any],
    data_hash: str,
    git_sha: str,
    git_dirty: bool,
    torch_version: str,
    cuda_version: Optional[str],
    cudnn_version: Optional[int],
    driver_version: str,
    gpu_name: str,
    gpu_clock_mhz: Optional[int],
    gpu_power_w: Optional[float],
    hostname: str,
    start_utc: str,
    end_utc: str,
    condition_id: str,
    run_id: str,
    repeat: int,
    seed: int,
    warmup_steps: int,
    measured_steps: int,
    rows_written: int,
    sync_count: int,
    extra_span_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure dict builder -- no I/O, no inference.

    Any field the caller could not determine is passed in as ``MISSING``; this
    function never guesses a value.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "resolved_config": dict(resolved_config),
        "data_hash": data_hash,
        "git": {"sha": git_sha, "dirty": bool(git_dirty)},
        "versions": {
            "torch": torch_version,
            "cuda": cuda_version,
            "cudnn": cudnn_version,
            "driver": driver_version,
        },
        "gpu": {
            "name": gpu_name,
            "clock_mhz": gpu_clock_mhz,
            "power_w": gpu_power_w,
        },
        "hostname": hostname,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "condition_id": condition_id,
        "run_id": run_id,
        "repeat": repeat,
        "seed": seed,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "rows_written": rows_written,
        "sync_count": sync_count,
        "extra_span_summary": dict(extra_span_summary),
    }


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``manifest.json``; mode ``"x"`` so an existing manifest is never clobbered."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def append_equivalence_ledger(path: Path, row: Mapping[str, Any]) -> None:
    """Append one equivalence-ledger row, writing the frozen header on creation."""
    path = Path(path)
    expected = set(EQUIVALENCE_LEDGER_COLUMNS)
    if set(row) != expected:
        missing = sorted(expected - set(row))
        unexpected = sorted(set(row) - expected)
        raise ValueError(
            "equivalence ledger row does not match the frozen schema "
            f"(missing={missing}, unexpected={unexpected})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(EQUIVALENCE_LEDGER_COLUMNS),
            extrasaction="raise",
            lineterminator="\n",
        )
        if not exists:
            writer.writeheader()
        writer.writerow({c: row[c] for c in EQUIVALENCE_LEDGER_COLUMNS})


def git_provenance(
    repo_root: Path, *, runner: Callable[..., Any] = subprocess.run
) -> tuple[str, bool]:
    """``(HEAD sha, dirty)``; ``(MISSING, True)`` when git cannot be consulted.

    Failing closed on ``dirty=True`` is deliberate: an unknown worktree state
    must never be recorded as clean provenance.
    """
    try:
        sha_result = runner(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        status_result = runner(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return (MISSING, True)
    try:
        if sha_result.returncode != 0 or status_result.returncode != 0:
            return (MISSING, True)
        return (sha_result.stdout.strip(), bool(status_result.stdout.strip()))
    except AttributeError:
        return (MISSING, True)


def data_manifest_hash(data_paths: Sequence[str]) -> str:
    """sha256 over each path's sorted relative filename + size + mtime_ns.

    Returns :data:`MISSING` when any listed path is absent (or when no path was
    given at all): a benchmark whose data provenance cannot be established must
    say so rather than emit a confident hash of nothing.
    """
    paths = list(data_paths)
    if not paths:
        return MISSING
    digest = hashlib.sha256()
    for raw in paths:
        root = Path(raw)
        if not root.exists():
            return MISSING
        digest.update(str(raw).encode("utf-8"))
        digest.update(b"\0")
        if root.is_file():
            entries = [(root.name, root)]
        else:
            entries = sorted(
                (entry.relative_to(root).as_posix(), entry)
                for entry in root.rglob("*")
                if entry.is_file()
            )
        for rel, entry in entries:
            stat = entry.stat()
            digest.update(f"{rel}\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode("utf-8"))
    return digest.hexdigest()


def capture_resource_manifest(
    *,
    output_dir: Union[str, Path] = DEFAULT_RESOURCE_MANIFEST_DIR,
    repo_root: Union[str, Path] = ".",
    probes: Sequence[tuple[str, Sequence[str]]] = DEFAULT_RESOURCE_PROBES,
    now: Optional[datetime] = None,
    runner: Callable[..., Any] = subprocess.run,
    hostname_fn: Callable[[], str] = socket.gethostname,
) -> Path:
    """P0.0: write ``resource_<UTC>.json`` from three read-only probes.

    Probes are shelled out at capture time only.  Tests MUST inject a fake
    ``runner``: the default probe set includes ``scripts/queue_run.sh status``,
    which must not be executed while live training jobs own the GPUs.
    """
    stamp = now or datetime.now(timezone.utc)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"resource_{stamp.strftime('%Y%m%dT%H%M%SZ')}.json"

    probe_payload: dict[str, Any] = {}
    for name, argv in probes:
        argv_list = list(argv)
        entry: dict[str, Any] = {
            "argv": argv_list,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "status": MISSING,
        }
        try:
            result = runner(
                argv_list,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            entry["stderr"] = f"{type(exc).__name__}: {exc}"
            probe_payload[name] = entry
            continue
        entry["returncode"] = getattr(result, "returncode", None)
        entry["stdout"] = getattr(result, "stdout", "") or ""
        entry["stderr"] = getattr(result, "stderr", "") or ""
        entry["status"] = "ok" if entry["returncode"] == 0 else MISSING
        probe_payload[name] = entry

    # ``runner`` is deliberately NOT forwarded to git: it exists so a test can
    # fake the three read-only PROBES (``queue_run.sh status`` must never
    # actually execute while live training jobs own the GPUs).  Git is not one
    # of those probes and lives under a separate top-level payload key, so
    # feeding it a probe mock would silently record the probe's stdout as the
    # worktree SHA -- garbage in the one field whose whole purpose is
    # provenance.  ``git_provenance`` carries its own injectable ``runner`` for
    # callers that genuinely need to fake git, and it fails closed to
    # ``(MISSING, True)`` when ``repo_root`` is not a git worktree.
    sha, dirty = git_provenance(Path(repo_root))
    payload = {
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "captured_utc": stamp.isoformat().replace("+00:00", "Z"),
        "hostname": hostname_fn(),
        "git": {"sha": sha, "dirty": dirty},
        "probes": probe_payload,
    }
    # Mode "x": a same-second re-capture raises FileExistsError rather than
    # overwriting an earlier resource manifest.
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return path
