"""Unit tests for utils/perf_benchmark.py (PERF-001 / P0.1a).

Every test here is GPU-free by construction:

* an autouse fixture forces ``torch.cuda.is_available()`` to ``False`` so the
  manifest's device-provenance probe never queries a device;
* every *enabled* collector injects ``event_factory``/``sync_fn``/
  ``utilization_fn``/``memory_stats_fn``/``reset_peak_fn``, so no CUDA entry
  point is ever reached (this is the fake-injection pattern the pre-registration
  requires -- CUDA integration stays evidence-pending until a reserved GPU runs
  it -- and the reason this file does NOT inherit
  ``test_perf_profiler.py::test_cpu_trace_is_bounded_and_never_overwrites``,
  which attaches a real CUPTI profiler);
* every ``capture_resource_manifest`` call injects a fake ``runner`` AND a
  ``tmp_path`` output dir, because the default probe set shells out to
  ``scripts/queue_run.sh status`` and the default output dir is a real repo
  path.  ``runner.call_count == 3`` is asserted so "the real queue script was
  never invoked" is a test property, not a convention.

Config-rejection tests match on the frozen message text, never on a bare field
name: under ``extra="forbid"`` a ``pytest.raises(ValidationError, match=...)``
on a field name passes vacuously.  That vacuity is a KNOWN LIMITATION of this
test file, not a departure from the spec: it carries no ``PERF-DEV`` ID (see the
``utils.perf_benchmark`` module docstring's label-discipline note).
"""

from __future__ import annotations

import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pydantic
import pytest
import torch
import yaml

from utils import perf_benchmark as pb
from utils.perf_benchmark import (
    EQUIVALENCE_LEDGER_COLUMNS,
    IGNORE_LABEL_ID,
    MISSING,
    SCHEMA_VERSION,
    STEADY_STATE_COLUMNS,
    PerfBenchmarkConfig,
    TrainingBenchmark,
    append_equivalence_ledger,
    bootstrap_median_ci,
    coefficient_of_variation,
    derive_condition_id,
    derive_repeat,
    hierarchical_bootstrap_median_ci,
    mean,
    median,
    percentile_nearest_rank,
    sample_sd,
    summarize,
    write_manifest,
    write_steady_state_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The 26 registered columns, transcribed from the pre-registration rather than
#: imported, so this file fails if the module constant is ever edited.
#: Amended 25 -> 26 by PI decision 2026-07-28 (PERF-DEV-02): the timing contract
#: mandates a metric-device CUDA bracket, so ``metrics_device_cuda_ms`` gets a
#: first-class column, ordered after ``ema_cuda_ms`` so the five CUDA columns
#: follow the contract's enumeration.
FROZEN_HEADER = (
    "schema_version,condition_id,run_id,repeat,seed,step,global_effective_batch,"
    "input_tokens,target_tokens,data_wait_ms,update_wall_ms,h2d_cuda_ms,"
    "forward_backward_cuda_ms,optimizer_cuda_ms,ema_cuda_ms,"
    "metrics_device_cuda_ms,metrics_wandb_wall_ms,"
    "eval_event_ms,zprobe_event_ms,checkpoint_event_ms,eval_amortized_ms,"
    "zprobe_amortized_ms,checkpoint_amortized_ms,max_memory_allocated,"
    "max_memory_reserved,gpu_util_pct"
)

_FLOAT_CELL = re.compile(r"^-?\d+\.\d{6}$")

#: Distinct powers of two, so a transposed or accidentally-summed CUDA column is
#: immediately visible instead of hiding behind a uniform fake duration.
SPAN_MS = {
    "h2d": 1.0,
    "forward_backward": 2.0,
    "optimizer": 4.0,
    "ema": 8.0,
    "metrics_device": 16.0,
}


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch: pytest.MonkeyPatch):
    """Guarantee the device-provenance probe in ``finalize`` never queries a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


class _Clock:
    """Test-driven virtual CUDA clock, in milliseconds."""

    def __init__(self) -> None:
        self.now = 0.0


class _FakeEvent:
    """Stands in for ``torch.cuda.Event(enable_timing=True)``.

    ``record()`` only *stamps* the clock; the test body advances it.  So
    ``with bench.cuda_span("h2d"): clock.now += 1.0`` yields exactly 1.0 ms, with
    no ambiguity about whether the enter or the exit stamp carries the duration.
    """

    __slots__ = ("_clock", "t")

    def __init__(self, clock: _Clock) -> None:
        self._clock = clock
        self.t: float | None = None

    def record(self) -> None:
        self.t = self._clock.now

    def elapsed_time(self, other: "_FakeEvent") -> float:
        assert self.t is not None and other.t is not None
        return other.t - self.t


class _Calls(dict):
    def bump(self, key: str) -> None:
        self[key] = self.get(key, 0) + 1


def make_bench(
    tmp_path: Path,
    *,
    rank: int = 0,
    enabled: bool = True,
    run_name: str | None = "perf0_m2_tf_z_iter_k6_s1_r1",
    eval_interval: int | None = None,
    output_dir: str | None = None,
    profiler_enabled: bool = False,
    resolved_config: dict | None = None,
    **config_kwargs,
) -> tuple[TrainingBenchmark, _Clock, _Calls]:
    """Build a fully-injected collector; returns ``(bench, clock, calls)``."""
    defaults = dict(warmup_steps=0, measured_steps=2, max_steps=100)
    defaults.update(config_kwargs)
    # The validator forbids an eval_event_step that could fire inside the
    # measured window, so the fixture default has to track the schedule the
    # caller actually asked for rather than being a fixed literal.
    defaults.setdefault(
        "eval_event_step", defaults["warmup_steps"] + defaults["measured_steps"] + 1
    )
    config = PerfBenchmarkConfig(
        enabled=enabled,
        performance_only=enabled,
        output_dir=output_dir or str(tmp_path / "bench"),
        **defaults,
    )
    clock = _Clock()
    calls = _Calls()
    bench = TrainingBenchmark(
        config,
        checkpoint_path=str(tmp_path / "checkpoints"),
        rank=rank,
        run_name=run_name,
        seed=1,
        eval_interval=eval_interval,
        resolved_config=resolved_config if resolved_config is not None else {"probe": 1},
        data_paths=(),
        profiler_enabled=profiler_enabled,
        event_factory=lambda: _FakeEvent(clock),
        sync_fn=lambda: calls.bump("sync"),
        utilization_fn=lambda: 42.0,
        memory_stats_fn=lambda: (111, 222),
        reset_peak_fn=lambda: calls.bump("reset_peak"),
    )
    return bench, clock, calls


def make_batch(rows: int = 4, seq: int = 8, ignored: int = 3) -> dict:
    """A CPU batch shaped like pretrain.py's: ``inputs`` + ``labels``."""
    inputs = torch.zeros((rows, seq), dtype=torch.long)
    labels = torch.zeros((rows, seq), dtype=torch.long)
    labels.view(-1)[:ignored] = IGNORE_LABEL_ID
    return {"inputs": inputs, "labels": labels}


def drive_update(
    bench: TrainingBenchmark,
    clock: _Clock,
    step: int,
    *,
    batch: dict | None = None,
    global_batch: int = 2048,
    spans: dict | None = None,
) -> None:
    """Replay pretrain.py's span order for one update."""
    durations = SPAN_MS if spans is None else spans
    bench.begin_update(batch if batch is not None else make_batch(), global_batch)
    for name in ("h2d", "forward_backward", "optimizer", "metrics_device"):
        with bench.cuda_span(name):
            clock.now += durations[name]
    with bench.wall_span("metrics_wandb"):
        pass
    with bench.cuda_span("ema"):
        clock.now += durations["ema"]
    bench.end_update(step)


def read_csv(session_dir: Path) -> tuple[list[str], list[dict]]:
    text = (session_dir / "steady_state.csv").read_text(encoding="utf-8")
    header = text.splitlines()[0]
    rows = list(csv.DictReader(text.splitlines()))
    return header.split(","), rows


def read_manifest(session_dir: Path) -> dict:
    return json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))


class _FakeResult:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# --------------------------------------------------------------------------- #
# 1. config bounds validation
# --------------------------------------------------------------------------- #


def test_enabled_requires_performance_only_acknowledgement():
    with pytest.raises(pydantic.ValidationError, match="performance_only=true"):
        PerfBenchmarkConfig(enabled=True)


def test_schedule_must_fit_within_max_steps():
    with pytest.raises(
        pydantic.ValidationError, match="shorter than its measurement schedule"
    ):
        PerfBenchmarkConfig(
            enabled=True,
            performance_only=True,
            warmup_steps=50,
            measured_steps=200,
            max_steps=100,
        )


def test_eval_event_step_must_lie_outside_the_measured_window():
    """An armed event span syncs twice; it must not be able to do so mid-window."""
    with pytest.raises(pydantic.ValidationError, match="cannot synchronize inside"):
        PerfBenchmarkConfig(
            enabled=True,
            performance_only=True,
            warmup_steps=50,
            measured_steps=200,
            eval_event_step=100,
            max_steps=2000,
        )
    # Boundary: eval_event_step == warmup + measured is still inside.
    with pytest.raises(pydantic.ValidationError, match="cannot synchronize inside"):
        PerfBenchmarkConfig(
            enabled=True,
            performance_only=True,
            warmup_steps=50,
            measured_steps=200,
            eval_event_step=250,
            max_steps=2000,
        )
    # One past the window is legal.
    PerfBenchmarkConfig(
        enabled=True,
        performance_only=True,
        warmup_steps=50,
        measured_steps=200,
        eval_event_step=251,
        max_steps=2000,
    )


def test_enabled_requires_a_non_blank_output_dir():
    with pytest.raises(pydantic.ValidationError, match="output_dir must be set"):
        PerfBenchmarkConfig(enabled=True, performance_only=True, output_dir="   ")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warmup_steps": -1},
        {"warmup_steps": 100_001},
        {"measured_steps": 0},
        {"measured_steps": 100_001},
        {"eval_event_step": 0},
        {"max_steps": 0},
        {"max_steps": 10_000_001},
        {"repeat": 0},
        {"repeat": 17},
        {"bootstrap_resamples": 99},
        {"bootstrap_resamples": 100_001},
        {"bootstrap_seed": -1},
    ],
)
def test_out_of_range_fields_are_rejected(kwargs):
    with pytest.raises(pydantic.ValidationError):
        PerfBenchmarkConfig(**kwargs)


def test_unknown_keys_are_forbidden():
    with pytest.raises(pydantic.ValidationError):
        PerfBenchmarkConfig(measured_stepz=200)


def test_disabled_incoherent_config_validates_clean():
    """Every cross-field check is gated on ``enabled`` (perf_profiler.py:42 pattern)."""
    config = PerfBenchmarkConfig(
        enabled=False, warmup_steps=50, measured_steps=200, max_steps=1, output_dir=""
    )
    assert config.enabled is False
    assert config.performance_only is False


def test_canonical_m2_preset_validates_on_yaml_defaults_alone():
    config = PerfBenchmarkConfig(
        enabled=True,
        performance_only=True,
        warmup_steps=50,
        measured_steps=200,
        eval_event_step=2000,
        max_steps=2000,
        output_dir=(
            "reports/figures/2026-07-26_experiment-speed-profiling/data/"
            "m2_tf_z_iter_k6_s1"
        ),
    )
    assert config.bootstrap_resamples == 10_000
    assert config.bootstrap_seed == 0
    assert config.repeat == 1
    assert config.condition_id == ""


# --------------------------------------------------------------------------- #
# 2. enabled defaults to False in config/cfg_pretrain.yaml
# --------------------------------------------------------------------------- #


def test_cfg_pretrain_yaml_perf_benchmark_is_off_and_matches_pydantic_defaults():
    block = yaml.safe_load((REPO_ROOT / "config" / "cfg_pretrain.yaml").read_text())[
        "perf_benchmark"
    ]

    assert block["enabled"] is False
    assert block["performance_only"] is False

    # Round-trip: catches YAML/pydantic drift in one assertion, and ``extra=forbid``
    # makes a stray YAML key fail here too.
    assert PerfBenchmarkConfig(**block) == PerfBenchmarkConfig()
    assert list(block) == list(PerfBenchmarkConfig.model_fields)


# --------------------------------------------------------------------------- #
# 3. disabled path is a strict no-op
# --------------------------------------------------------------------------- #


def test_disabled_path_is_a_strict_no_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the disabled benchmark path touched torch.cuda")

    for attr in (
        "Event",
        "synchronize",
        "utilization",
        "max_memory_allocated",
        "max_memory_reserved",
        "reset_peak_memory_stats",
    ):
        monkeypatch.setattr(torch.cuda, attr, boom)

    bench = TrainingBenchmark(
        PerfBenchmarkConfig(),  # default => disabled
        checkpoint_path=str(tmp_path / "checkpoints"),
        rank=0,
        run_name="whatever",
        eval_interval=2,
    )

    assert bench.enabled is False
    assert bench.session_dir is None
    assert bench.sync_count == 0
    # Nothing was constructed: no state machine, no injectables.
    assert not hasattr(bench, "_rows")

    # Every span is the SAME module-level singleton -- identity, not equality.
    for name in pb.CUDA_SPAN_NAMES:
        assert bench.cuda_span(name) is pb._NULL_SPAN
    for name in pb.WALL_SPAN_NAMES:
        assert bench.wall_span(name) is pb._NULL_SPAN
    for name in pb.EVENT_SPAN_NAMES:
        assert bench.event_span(name, 10_000) is pb._NULL_SPAN

    # A full simulated training loop through the disabled collector.
    for step in range(1, 4):
        bench.begin_update(make_batch(), 2048)
        for name in pb.CUDA_SPAN_NAMES:
            with bench.cuda_span(name):
                pass
        for name in pb.WALL_SPAN_NAMES:
            with bench.wall_span(name):
                pass
        for name in pb.EVENT_SPAN_NAMES:
            with bench.event_span(name, 10_000):
                pass
        bench.end_update(step)
    bench.finalize()
    bench.finalize()  # idempotent

    assert bench.sync_count == 0
    assert not (tmp_path / "bench").exists()
    assert not (tmp_path / "performance_benchmarks").exists()
    assert not (Path.cwd() / "performance_benchmarks").exists()
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# 4. steady_state.csv header
# --------------------------------------------------------------------------- #


def test_steady_state_header_is_the_registered_26_columns(tmp_path: Path):
    bench, clock, _ = make_bench(tmp_path, measured_steps=2)
    drive_update(bench, clock, 1)
    drive_update(bench, clock, 2)
    bench.finalize()

    header_cells, rows = read_csv(bench.session_dir)
    assert header_cells == FROZEN_HEADER.split(",")
    assert len(header_cells) == 26
    assert tuple(header_cells) == STEADY_STATE_COLUMNS  # module constant agrees
    assert len(rows) == 2
    assert rows[0]["schema_version"] == SCHEMA_VERSION


def test_write_steady_state_csv_rejects_a_schema_mismatch(tmp_path: Path):
    good = {c: 0 for c in STEADY_STATE_COLUMNS}

    short = dict(good)
    short.pop("gpu_util_pct")
    with pytest.raises(ValueError, match="frozen schema"):
        write_steady_state_csv(tmp_path / "a.csv", [short])

    wide = dict(good)
    # Any column outside the registered set is rejected.  This used to use
    # metrics_device_cuda_ms, which PERF-DEV-02 has since promoted to a real
    # column (PI 2026-07-28), so the probe now uses a genuinely unknown name.
    wide["not_a_registered_column"] = 1.0
    with pytest.raises(ValueError, match="frozen schema"):
        write_steady_state_csv(tmp_path / "b.csv", [wide])


def test_cuda_spans_land_in_their_own_columns(tmp_path: Path):
    """Distinct powers of two: a transposition or double-count cannot pass."""
    bench, clock, _ = make_bench(tmp_path, measured_steps=1)
    drive_update(bench, clock, 7, global_batch=2048)
    bench.finalize()

    _, rows = read_csv(bench.session_dir)
    row = rows[0]

    assert _FLOAT_CELL.match(row["h2d_cuda_ms"])  # frozen f"{v:.6f}" writer format
    assert float(row["h2d_cuda_ms"]) == pytest.approx(1.0)
    assert float(row["forward_backward_cuda_ms"]) == pytest.approx(2.0)
    assert float(row["optimizer_cuda_ms"]) == pytest.approx(4.0)
    assert float(row["ema_cuda_ms"]) == pytest.approx(8.0)

    assert row["step"] == "7"
    assert row["global_effective_batch"] == "2048"
    assert row["input_tokens"] == "32"          # 4 x 8
    assert row["target_tokens"] == "29"         # minus the 3 IGNORE_LABEL_ID cells
    assert row["max_memory_allocated"] == "111"
    assert row["max_memory_reserved"] == "222"
    assert float(row["gpu_util_pct"]) == pytest.approx(42.0)
    assert _FLOAT_CELL.match(row["update_wall_ms"])
    assert _FLOAT_CELL.match(row["metrics_wandb_wall_ms"])


def test_metrics_device_has_its_own_csv_column(tmp_path: Path):
    """PERF-DEV-02 (PI 2026-07-28): metrics_device is a first-class column.

    This test previously asserted the OPPOSITE -- that the span was absent from
    the CSV and lived only in the manifest -- because the 25-column header was
    byte-frozen without a slot for it.  The PI resolved the contract's internal
    contradiction by amending the schema instead, so the assertion is inverted
    deliberately, not relaxed.
    """
    bench, clock, _ = make_bench(tmp_path, measured_steps=2)
    drive_update(bench, clock, 1)
    drive_update(bench, clock, 2)
    bench.finalize()

    header_cells, rows = read_csv(bench.session_dir)
    assert "metrics_device_cuda_ms" in header_cells
    # Ordered after ema_cuda_ms, so the five CUDA columns run in the timing
    # contract's own enumeration order.
    assert (
        header_cells.index("metrics_device_cuda_ms")
        == header_cells.index("ema_cuda_ms") + 1
    )
    for row in rows:
        # 16.0 is the metrics_device duration; it must now land in that column.
        assert float(row["metrics_device_cuda_ms"]) == pytest.approx(16.0)

    # The manifest aggregate is retained as a convenience, but is no longer the
    # authoritative home for the span.
    summary = read_manifest(bench.session_dir)["extra_span_summary"][
        "metrics_device_cuda_ms"
    ]
    assert set(summary) == {"n", "p50", "p95", "mean", "sd", "cv", "ci_low", "ci_high"}
    assert summary["p50"] == pytest.approx(16.0)


def test_missing_counters_are_written_verbatim(tmp_path: Path):
    """Never a silent 0.0 for a span that did not fire."""
    bench, clock, _ = make_bench(tmp_path, measured_steps=1, eval_interval=2)
    drive_update(bench, clock, 1)
    bench.finalize()

    _, rows = read_csv(bench.session_dir)
    row = rows[0]
    for column in (
        "eval_event_ms",
        "zprobe_event_ms",
        "checkpoint_event_ms",
        "eval_amortized_ms",
        "zprobe_amortized_ms",
        "checkpoint_amortized_ms",
        "data_wait_ms",  # no iter_batches in this run
    ):
        assert row[column] == MISSING == "evidence_pending"


def test_repeated_wall_spans_sum_into_one_column(tmp_path: Path):
    """pretrain.py opens metrics_wandb TWICE per update (D2H at :361, wandb.log at :782)."""
    bench, clock, _ = make_bench(tmp_path, measured_steps=1)
    bench.begin_update(make_batch(), 2048)
    with bench.wall_span("metrics_wandb"):
        time.sleep(0.01)
    with bench.wall_span("metrics_wandb"):
        time.sleep(0.01)
    bench.end_update(1)
    bench.finalize()

    _, rows = read_csv(bench.session_dir)
    # ~20 ms when summed; ~10 ms if the second span overwrote the first.
    assert float(rows[0]["metrics_wandb_wall_ms"]) > 18.0


def test_finalize_resolves_a_window_that_never_filled(tmp_path: Path):
    """A run that dies before measured_steps rows still gets resolved CUDA times."""
    bench, clock, _ = make_bench(tmp_path, measured_steps=5, max_steps=10)
    drive_update(bench, clock, 1)
    drive_update(bench, clock, 2)
    assert bench.sync_count == 0  # window still open, pairs still retained

    bench.finalize()
    assert bench.sync_count == 1  # the other legal sync site

    _, rows = read_csv(bench.session_dir)
    assert len(rows) == 2
    assert float(rows[0]["h2d_cuda_ms"]) == pytest.approx(1.0)
    assert float(rows[1]["ema_cuda_ms"]) == pytest.approx(8.0)


def test_iter_batches_populates_data_wait(tmp_path: Path):
    """PERF-DEV-03: the loader wait is staged before the row opens, then promoted."""
    bench, clock, _ = make_bench(tmp_path, measured_steps=3)
    loader = [make_batch() for _ in range(3)]

    for index, batch in enumerate(bench.iter_batches(loader), start=1):
        drive_update(bench, clock, index, batch=batch)
    bench.finalize()

    _, rows = read_csv(bench.session_dir)
    assert len(rows) == 3
    for row in rows:
        assert _FLOAT_CELL.match(row["data_wait_ms"])
        assert float(row["data_wait_ms"]) >= 0.0


def test_each_data_wait_is_promoted_into_the_row_it_actually_fed(tmp_path: Path):
    """Distinct, well-separated loader waits pin the promotion to the right row.

    An off-by-one promotion would shift the bands; a failure to clear
    ``_staging`` would make them accumulate (5, 30, 80 instead of 5, 25, 50).
    """
    sleeps = [0.005, 0.025, 0.050]

    def slow_loader():
        for delay in sleeps:
            time.sleep(delay)
            yield make_batch()

    bench, clock, _ = make_bench(tmp_path, measured_steps=3)
    for index, batch in enumerate(bench.iter_batches(slow_loader()), start=1):
        drive_update(bench, clock, index, batch=batch)
    bench.finalize()

    _, rows = read_csv(bench.session_dir)
    measured = [float(r["data_wait_ms"]) for r in rows]
    assert len(measured) == 3

    # Disjoint bands: each row carries its OWN wait, generously bounded above so
    # a loaded host cannot flake, but tightly enough that any permutation fails.
    assert 5.0 <= measured[0] < 20.0
    assert 25.0 <= measured[1] < 45.0
    assert 50.0 <= measured[2] < 75.0
    assert measured[0] < measured[1] < measured[2]
    # Accumulation would put row 3 at ~80 ms; the band above already excludes it.


# --------------------------------------------------------------------------- #
# 5. manifest.json
# --------------------------------------------------------------------------- #


def test_manifest_contains_every_required_field(tmp_path: Path):
    bench, clock, _ = make_bench(
        tmp_path,
        measured_steps=2,
        eval_interval=4,
        resolved_config={"arch": {"name": "trm"}, "seed": 1},
    )
    drive_update(bench, clock, 1)
    drive_update(bench, clock, 2)
    bench.finalize()

    m = read_manifest(bench.session_dir)

    # The frozen contract's manifest items, mapped to their key paths.
    for key in (
        "command",
        "resolved_config",
        "data_hash",
        "git",
        "versions",
        "gpu",
        "hostname",
        "start_utc",
        "end_utc",
    ):
        assert key in m, key
    assert set(m["git"]) == {"sha", "dirty"}
    assert {"torch", "cuda", "cudnn", "driver"} <= set(m["versions"])
    assert {"name", "clock_mhz", "power_w"} <= set(m["gpu"])

    # Provenance that must never be silently pooled.
    for key in (
        "schema_version",
        "condition_id",
        "run_id",
        "repeat",
        "seed",
        "warmup_steps",
        "measured_steps",
        "rows_written",
        "sync_count",
        "extra_span_summary",
        "bootstrap_resamples",
        "bootstrap_seed",
        "eval_interval",
        "eval_event_step",
        "max_steps",
    ):
        assert key in m, key

    assert m["schema_version"] == SCHEMA_VERSION
    assert m["resolved_config"] == {"arch": {"name": "trm"}, "seed": 1}
    assert m["rows_written"] == 2
    assert m["sync_count"] == 1
    assert m["seed"] == 1
    assert m["repeat"] == 1
    assert m["eval_interval"] == 4
    assert m["hostname"]
    assert isinstance(m["git"]["dirty"], bool)
    assert m["git"]["sha"] == MISSING or re.fullmatch(r"[0-9a-f]{40}", m["git"]["sha"])
    # No data_paths were given, so the hash must say so rather than hash nothing.
    assert m["data_hash"] == MISSING
    # CPU-only host: device fields are MISSING/None, never invented.
    assert m["gpu"]["name"] == MISSING
    assert m["versions"]["torch"] == str(torch.__version__)
    for stamp in (m["start_utc"], m["end_utc"]):
        assert stamp.endswith("Z")
        datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def test_build_manifest_is_a_pure_dict_builder(tmp_path: Path):
    payload = pb.build_manifest(
        command="uv run pretrain.py",
        resolved_config={"a": 1},
        data_hash=MISSING,
        git_sha="0" * 40,
        git_dirty=True,
        torch_version="2.7.0",
        cuda_version="12.6",
        cudnn_version=90000,
        driver_version="550.0",
        gpu_name="fake",
        gpu_clock_mhz=1400,
        gpu_power_w=250.0,
        hostname="host",
        start_utc="2026-07-28T00:00:00Z",
        end_utc="2026-07-28T00:01:00Z",
        condition_id="cond",
        run_id="run",
        repeat=3,
        seed=1,
        warmup_steps=50,
        measured_steps=200,
        rows_written=200,
        sync_count=1,
        extra_span_summary={},
    )
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["git"] == {"sha": "0" * 40, "dirty": True}
    assert list(tmp_path.iterdir()) == []  # pure: wrote nothing


# --------------------------------------------------------------------------- #
# 6. equivalence_ledger.csv
# --------------------------------------------------------------------------- #


def test_equivalence_ledger_schema_and_append(tmp_path: Path):
    path = tmp_path / "equivalence_ledger.csv"
    row = {
        "candidate_id": "cand",
        "control_id": "ctrl",
        "fixed_batch_hash": "abc",
        "sample_index_hash": "def",
        "config_diff": "arch.L_layers: 2 -> 1",
        "schema_version": SCHEMA_VERSION,
        "tolerance": "0.02",
        "g1_status": "PASS",
        "g2_status": "PENDING",
    }
    append_equivalence_ledger(path, row)
    append_equivalence_ledger(path, dict(row, candidate_id="cand2"))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(EQUIVALENCE_LEDGER_COLUMNS)
    assert EQUIVALENCE_LEDGER_COLUMNS == (
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
    assert len(lines) == 3  # header written exactly once
    parsed = list(csv.DictReader(lines))
    assert [r["candidate_id"] for r in parsed] == ["cand", "cand2"]


def test_equivalence_ledger_rejects_a_key_mismatch(tmp_path: Path):
    path = tmp_path / "ledger.csv"
    with pytest.raises(ValueError, match="frozen schema"):
        append_equivalence_ledger(path, {"candidate_id": "only"})
    assert not path.exists()


# --------------------------------------------------------------------------- #
# 7. non-overwrite
# --------------------------------------------------------------------------- #


def test_sessions_never_clobber_an_existing_artifact_directory(tmp_path: Path):
    output_dir = tmp_path / "bench"
    first, clock, _ = make_bench(tmp_path, measured_steps=1, output_dir=str(output_dir))
    assert first.session_dir.name == "session_0000"
    drive_update(first, clock, 1)
    first.finalize()
    csv_bytes = (first.session_dir / "steady_state.csv").read_bytes()

    second, clock2, _ = make_bench(tmp_path, measured_steps=1, output_dir=str(output_dir))
    assert second.session_dir.name == "session_0001"
    assert second.session_dir != first.session_dir
    drive_update(second, clock2, 1)
    second.finalize()

    # The first session's artifacts are untouched.
    assert (first.session_dir / "steady_state.csv").read_bytes() == csv_bytes
    assert sorted(p.name for p in output_dir.iterdir()) == ["session_0000", "session_0001"]


def test_session_claim_skips_a_preexisting_directory(tmp_path: Path):
    output_dir = tmp_path / "bench"
    (output_dir / "session_0000").mkdir(parents=True)
    (output_dir / "session_0000" / "steady_state.csv").write_text("PRECIOUS")

    bench, _, _ = make_bench(tmp_path, output_dir=str(output_dir))
    assert bench.session_dir.name == "session_0001"
    assert (output_dir / "session_0000" / "steady_state.csv").read_text() == "PRECIOUS"


def test_write_manifest_refuses_to_overwrite(tmp_path: Path):
    path = tmp_path / "manifest.json"
    write_manifest(path, {"a": 1})
    with pytest.raises(FileExistsError):
        write_manifest(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


def test_finalize_is_idempotent(tmp_path: Path):
    """A second finalize must not re-open manifest.json in mode "x"."""
    bench, clock, _ = make_bench(tmp_path, measured_steps=1)
    drive_update(bench, clock, 1)
    bench.finalize()
    assert bench.enabled is False
    bench.finalize()  # must not raise FileExistsError
    assert len(read_csv(bench.session_dir)[1]) == 1


# --------------------------------------------------------------------------- #
# 8. amortized == event_ms / eval_interval
# --------------------------------------------------------------------------- #


def test_amortized_fields_equal_event_ms_over_eval_interval(tmp_path: Path):
    bench, clock, calls = make_bench(
        tmp_path, measured_steps=1, eval_event_step=100, eval_interval=2
    )
    drive_update(bench, clock, 1)

    with bench.event_span("eval", 150):
        time.sleep(0.01)  # ~10 ms, so the 6-dp CSV format cannot quantize it away
    with bench.event_span("zprobe", 150):
        time.sleep(0.01)
    with bench.event_span("checkpoint", 150):
        time.sleep(0.01)
    bench.finalize()

    _, rows = read_csv(bench.session_dir)
    row = rows[0]
    for event_col, amortized_col in (
        ("eval_event_ms", "eval_amortized_ms"),
        ("zprobe_event_ms", "zprobe_amortized_ms"),
        ("checkpoint_event_ms", "checkpoint_amortized_ms"),
    ):
        event_ms = float(row[event_col])
        assert event_ms > 5.0  # the sleep really was measured
        assert float(row[amortized_col]) == pytest.approx(event_ms / 2, abs=2e-6)

    # Event-span boundaries are the only other legal sync sites: 1 window + 3x2.
    assert bench.sync_count == 7 == calls["sync"]


def test_amortized_is_missing_when_eval_interval_is_unknown(tmp_path: Path):
    bench, clock, _ = make_bench(
        tmp_path, measured_steps=1, eval_event_step=100, eval_interval=None
    )
    drive_update(bench, clock, 1)
    with bench.event_span("eval", 150):
        time.sleep(0.005)
    bench.finalize()

    _, rows = read_csv(bench.session_dir)
    assert float(rows[0]["eval_event_ms"]) > 1.0
    assert rows[0]["eval_amortized_ms"] == MISSING


# --------------------------------------------------------------------------- #
# 9. p95 nearest rank
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "n, expected_nearest_rank, numpy_linear",
    [
        (20, 19.0, 19.05),  # ceil(0.95*20)-1 = 18
        (10, 10.0, 9.55),   # ceil(0.95*10)-1 = 9
        (7, 7.0, 6.70),     # ceil(0.95*7)-1  = 6
    ],
)
def test_p95_uses_nearest_rank_not_interpolation(n, expected_nearest_rank, numpy_linear):
    values = [float(v) for v in range(1, n + 1)]
    got = percentile_nearest_rank(values, 95.0)

    assert got == expected_nearest_rank
    # The naive interpolating percentile gives a different answer -- which is the
    # whole point of pinning the convention.
    assert float(np.percentile(values, 95)) == pytest.approx(numpy_linear)
    assert got != pytest.approx(float(np.percentile(values, 95)))

    # summarize() must use the same convention.
    assert summarize(values, resamples=200)["p95"] == expected_nearest_rank


def test_percentile_is_order_independent_and_clamped():
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert percentile_nearest_rank(values, 0.0) == 1.0
    assert percentile_nearest_rank(values, 100.0) == 5.0
    assert percentile_nearest_rank(values, 50.0) == 3.0


def test_pure_statistics_helpers():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([4.0, 1.0, 3.0, 2.0]) == 2.5  # even N -> mean of the two centres
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert sample_sd([2.0]) == 0.0
    assert sample_sd([1.0, 3.0]) == pytest.approx(np.std([1.0, 3.0], ddof=1))
    assert coefficient_of_variation([1.0, 3.0]) == pytest.approx(
        sample_sd([1.0, 3.0]) / 2.0
    )
    assert np.isnan(coefficient_of_variation([-1.0, 1.0]))  # mean == 0 -> nan

    for fn in (median, mean, percentile_nearest_rank):
        with pytest.raises(ValueError):
            fn([], 50.0) if fn is percentile_nearest_rank else fn([])

    out = summarize([1.0, 2.0, 3.0, 4.0], resamples=500, seed=0)
    assert set(out) == {"n", "p50", "p95", "mean", "sd", "cv", "ci_low", "ci_high"}
    assert out["n"] == 4
    assert out["ci_low"] <= out["p50"] <= out["ci_high"]


# --------------------------------------------------------------------------- #
# 10. bootstrap determinism + hierarchical resampling
# --------------------------------------------------------------------------- #


def test_bootstrap_median_ci_is_reproducible_under_a_fixed_seed():
    rng = np.random.default_rng(1234)
    values = (rng.normal(loc=10.0, scale=1.0, size=200)).tolist()

    a = bootstrap_median_ci(values, resamples=2000, seed=0)
    b = bootstrap_median_ci(values, resamples=2000, seed=0)
    assert a == b  # bit-identical, not merely close

    c = bootstrap_median_ci(values, resamples=2000, seed=7)
    assert c != a  # the seed actually drives the resampling

    lo, hi = a
    assert lo <= median(values) <= hi
    # And the sample-level determinism propagates through summarize().
    assert summarize(values, resamples=2000, seed=0)["ci_low"] == lo


def test_hierarchical_bootstrap_resamples_repeats_then_rows_and_never_pools():
    """Two repeats at 1 ms, one at 100 ms.

    Pooling the 600 rows makes the median 1.0 in every draw, so a flat bootstrap
    reports the degenerate interval (1.0, 1.0).  Resampling repeats first draws
    >= 2 copies of the 100 ms repeat with probability 7/27 ~ 26% >> 2.5%, so the
    honest upper bound is 100.0.  An implementation that flattens fails here.
    """
    repeat_a = [1.0] * 200
    repeat_b = [1.0] * 200
    repeat_c = [100.0] * 200
    repeats = [repeat_a, repeat_b, repeat_c]
    pooled = repeat_a + repeat_b + repeat_c

    flat_lo, flat_hi = bootstrap_median_ci(pooled, resamples=2000, seed=0)
    assert (flat_lo, flat_hi) == (1.0, 1.0)

    hier_lo, hier_hi = hierarchical_bootstrap_median_ci(repeats, resamples=2000, seed=0)
    assert hier_lo == 1.0
    assert hier_hi == 100.0
    assert hier_hi > flat_hi  # the whole point: 600 rows are not independent

    # Deterministic for a fixed seed, seed-sensitive otherwise.
    assert hierarchical_bootstrap_median_ci(repeats, resamples=2000, seed=0) == (
        hier_lo,
        hier_hi,
    )

    with pytest.raises(ValueError):
        hierarchical_bootstrap_median_ci([])
    with pytest.raises(ValueError):
        hierarchical_bootstrap_median_ci([[1.0], []])


def _stage_one_only_median_ci(
    repeats, *, resamples: int = 2000, confidence: float = 0.95, seed: int = 0
):
    """Reference implementation with stage 2 (the row resample) DELETED.

    Byte-for-byte the real ``hierarchical_bootstrap_median_ci`` except that each
    drawn repeat contributes its rows verbatim instead of a with-replacement
    resample of them.  Its only purpose is to be *distinguishable* from the real
    function, which is what pins the second stage of the frozen contract.
    """
    groups = [np.asarray(r, dtype=np.float64) for r in repeats]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(resamples, len(groups)))
    medians = [
        float(np.median(np.concatenate([groups[j] for j in draw]))) for draw in draws
    ]
    return (
        percentile_nearest_rank(medians, 100.0 * (1.0 - confidence) / 2.0),
        percentile_nearest_rank(medians, 100.0 * (1.0 + confidence) / 2.0),
    )


def test_hierarchical_bootstrap_actually_resamples_rows_within_each_repeat():
    """Stage 2 in isolation: three IDENTICAL, internally bimodal repeats.

    The previous fixture (constant-valued repeats) could not test this: with every
    row inside a repeat equal, a with-replacement resample of those rows is an
    identity operation, so deleting stage 2 left the assertions green.

    Here each repeat is exactly balanced, ``[0.0]*100 + [100.0]*100``.  Because
    the three repeats are identical, stage 1 is *provably* a no-op -- any drawn
    multiset of repeats pools the same 300 zeros and 300 hundreds -- so any
    difference in the interval is attributable to stage 2 alone:

    * stage-1-only: the pooled median is always ``(0+100)/2 == 50.0`` -> (50, 50)
    * two-stage:    each drawn repeat's zero-count is ~Binomial(200, 0.5), so the
      pooled zero-count has sd ~ 12 and the pooled median flips between the two
      modes -> (0, 100)
    """
    balanced = [0.0] * 100 + [100.0] * 100
    repeats = [list(balanced), list(balanced), list(balanced)]

    # Stage 1 alone cannot move the interval off the tipping point.
    assert _stage_one_only_median_ci(repeats, resamples=2000, seed=0) == (50.0, 50.0)

    # The real implementation resamples rows too, so it sees both modes.
    lo, hi = hierarchical_bootstrap_median_ci(repeats, resamples=2000, seed=0)
    assert (lo, hi) == (0.0, 100.0)
    assert (lo, hi) != _stage_one_only_median_ci(repeats, resamples=2000, seed=0)

    # Still deterministic under a fixed seed.
    assert hierarchical_bootstrap_median_ci(repeats, resamples=2000, seed=0) == (lo, hi)


def test_hierarchical_bootstrap_rejects_a_flat_sequence_of_rows():
    """A flat list is pooled rows, which this estimator exists to refuse.

    ``np.asarray(1.0)`` is 0-d with ``.size == 1``, so the emptiness check waves a
    flat sequence through and stage 2 then dies with an opaque ``IndexError``.
    """
    with pytest.raises(ValueError, match="not a flat sequence of rows"):
        hierarchical_bootstrap_median_ci([1.0, 2.0, 3.0])

    # The nested form of the same data is accepted.
    assert hierarchical_bootstrap_median_ci(
        [[1.0], [2.0], [3.0]], resamples=200, seed=0
    ) == pytest.approx((1.0, 3.0))


# --------------------------------------------------------------------------- #
# 11. rank-0 ownership
# --------------------------------------------------------------------------- #


def test_nonzero_rank_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("a non-zero rank touched torch.cuda")

    for attr in ("Event", "synchronize", "utilization", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, attr, boom)

    bench, clock, calls = make_bench(tmp_path, rank=1, measured_steps=2)

    assert bench.enabled is False
    assert bench.session_dir is None
    assert bench.sync_count == 0
    assert not (tmp_path / "bench").exists()

    for step in range(1, 4):
        drive_update(bench, clock, step)
    with bench.event_span("eval", 10_000):
        pass
    bench.finalize()

    assert calls == {}
    assert not (tmp_path / "bench").exists()
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# Synchronization budget -- the headline correctness property
# --------------------------------------------------------------------------- #


def test_sync_is_never_called_per_update(tmp_path: Path):
    bench, clock, calls = make_bench(
        tmp_path, warmup_steps=2, measured_steps=3, max_steps=50
    )

    for step in (1, 2):  # warmup
        drive_update(bench, clock, step)
    assert bench.sync_count == 0

    for step in (3, 4):  # measured, window still open
        drive_update(bench, clock, step)
    assert bench.sync_count == 0, "sync_fn must never be called per update"

    drive_update(bench, clock, 5)  # closes the 3-row window
    assert bench.sync_count == 1
    assert calls["sync"] == 1

    bench.finalize()
    assert bench.sync_count == 1  # nothing left pending to resolve

    _, rows = read_csv(bench.session_dir)
    assert [r["step"] for r in rows] == ["3", "4", "5"]  # warmup rows discarded
    # Retained event pairs resolved after the single sync.
    assert float(rows[0]["h2d_cuda_ms"]) == pytest.approx(1.0)


def test_reset_peak_memory_is_called_once_at_the_first_measured_update(tmp_path: Path):
    bench, clock, calls = make_bench(
        tmp_path, warmup_steps=2, measured_steps=3, max_steps=50
    )
    for step in (1, 2):
        drive_update(bench, clock, step)
    assert "reset_peak" not in calls

    for step in (3, 4, 5):
        drive_update(bench, clock, step)
    assert calls["reset_peak"] == 1
    bench.finalize()
    assert calls["reset_peak"] == 1


# --------------------------------------------------------------------------- #
# Event-step gating, row-drop rule, span-name validation, mutual exclusion
# --------------------------------------------------------------------------- #


def test_event_span_arms_on_ge_and_captures_the_first_block_per_name(tmp_path: Path):
    """PERF-DEV-01: with epochs=2000/eval_interval=2000 the eval block fires at ~4.9k."""
    bench, clock, _ = make_bench(
        tmp_path, measured_steps=1, eval_event_step=2000, eval_interval=2
    )
    drive_update(bench, clock, 1)
    assert bench.sync_count == 1

    # Below the threshold: inert.
    assert bench.event_span("eval", 1999) is pb._NULL_SPAN
    with bench.event_span("eval", 1999):
        time.sleep(0.005)
    assert bench.sync_count == 1

    # First block at or above the threshold wins.
    with bench.event_span("eval", 4900):
        time.sleep(0.02)
    assert bench.sync_count == 3

    # A later block for the same name is ignored, and does not sync.
    assert bench.event_span("eval", 9800) is pb._NULL_SPAN
    with bench.event_span("eval", 9800):
        time.sleep(0.05)
    assert bench.sync_count == 3

    bench.finalize()
    _, rows = read_csv(bench.session_dir)
    eval_ms = float(rows[0]["eval_event_ms"])
    assert 15.0 < eval_ms < 45.0, "the second, longer block must have been ignored"
    assert rows[0]["zprobe_event_ms"] == MISSING
    assert rows[0]["checkpoint_event_ms"] == MISSING


def test_event_span_never_arms_while_the_measured_window_is_open(tmp_path: Path):
    """Defence in depth behind the eval_event_step validator.

    An armed event span synchronizes on both boundaries.  If it could fire before
    the window closed, those drains would land inside the window and every later
    row would be measured against a freshly drained pipeline.
    """
    bench, clock, _ = make_bench(
        tmp_path, warmup_steps=0, measured_steps=3, eval_event_step=4, max_steps=50
    )
    drive_update(bench, clock, 1)

    # Step is far past eval_event_step, but only two of three rows are collected.
    assert bench.event_span("eval", 10_000) is pb._NULL_SPAN
    with bench.event_span("eval", 10_000):
        time.sleep(0.01)
    assert bench.sync_count == 0, "an event span drained the device mid-window"

    drive_update(bench, clock, 2)
    drive_update(bench, clock, 3)  # closes the window
    assert bench.sync_count == 1

    # Now that the window is closed the same call arms normally.
    with bench.event_span("eval", 10_000):
        time.sleep(0.01)
    assert bench.sync_count == 3
    bench.finalize()

    _, rows = read_csv(bench.session_dir)
    assert len(rows) == 3
    assert float(rows[0]["eval_event_ms"]) > 1.0


def test_extra_span_summary_counters_are_clean_on_a_complete_window(tmp_path: Path):
    bench, clock, _ = make_bench(tmp_path, measured_steps=2)
    drive_update(bench, clock, 1)
    drive_update(bench, clock, 2)
    bench.finalize()

    summary = read_manifest(bench.session_dir)["extra_span_summary"]
    assert summary["event_pool_overflow"] == 0
    assert summary["rows_dropped_incomplete"] == 0
    assert summary["window_closed"] is True


def test_event_pool_overflow_is_counted_rather_than_silently_absorbed(tmp_path: Path):
    """The budget is measured_steps * len(CUDA_SPAN_NAMES) * 2, with zero slack.

    Anyone who adds a sixth CUDA span per update overruns it; the count is the
    tripwire that says so instead of the pool quietly growing.
    """
    bench, clock, _ = make_bench(tmp_path, measured_steps=1)
    bench.begin_update(make_batch(), 2048)
    # Six spans against a five-span budget: h2d is opened twice (the row bucket
    # accumulates), because there are only five legal span names.
    for name in ("h2d", "h2d", "forward_backward", "optimizer", "ema", "metrics_device"):
        with bench.cuda_span(name):
            clock.now += 1.0
    bench.end_update(1)
    bench.finalize()

    summary = read_manifest(bench.session_dir)["extra_span_summary"]
    assert summary["event_pool_overflow"] == 2  # one pair beyond the budget
    _, rows = read_csv(bench.session_dir)
    # The over-budget span was still measured, not dropped: h2d summed 1.0 + 1.0.
    assert float(rows[0]["h2d_cuda_ms"]) == pytest.approx(2.0)


def test_a_row_begun_but_never_ended_is_dropped_and_counted(tmp_path: Path):
    """An exception on the train path leaves a half-filled row; it must not be written."""
    bench, clock, _ = make_bench(tmp_path, measured_steps=3, max_steps=50)
    drive_update(bench, clock, 1)

    # Second update begins, opens spans, then the train loop dies before
    # end_update -- pretrain.py's `finally: bench.finalize()` still runs.
    bench.begin_update(make_batch(), 2048)
    with bench.cuda_span("h2d"):
        clock.now += 1.0
    bench.finalize()

    _, rows = read_csv(bench.session_dir)
    assert [r["step"] for r in rows] == ["1"], "the partial row must not be written"

    summary = read_manifest(bench.session_dir)["extra_span_summary"]
    assert summary["rows_dropped_incomplete"] == 1
    assert summary["window_closed"] is True  # finalize resolved the open window
    assert read_manifest(bench.session_dir)["rows_written"] == 1


def test_end_update_without_begin_update_produces_no_row(tmp_path: Path):
    """Row-drop rule: pretrain.py's total_steps early return skips begin_update."""
    bench, clock, _ = make_bench(tmp_path, measured_steps=3, max_steps=50)
    drive_update(bench, clock, 1)
    bench.end_update(2)  # the skipped update
    bench.end_update(3)
    drive_update(bench, clock, 4)
    bench.finalize()

    _, rows = read_csv(bench.session_dir)
    assert [r["step"] for r in rows] == ["1", "4"]


def test_row_collection_stops_but_the_object_stays_alive_for_a_later_event(tmp_path: Path):
    # The validator forbids max_steps < warmup_steps + measured_steps, so the cap
    # can never truncate the window below measured_steps; what it does guarantee
    # is that surplus updates produce nothing while the collector stays alive.
    bench, clock, _ = make_bench(
        tmp_path, warmup_steps=1, measured_steps=2, max_steps=3, eval_event_step=4
    )
    for step in range(1, 7):
        drive_update(bench, clock, step)
    assert bench.sync_count == 1  # still exactly one window-closing sync

    # Collection has stopped, but a later event_span can still fire.
    with bench.event_span("eval", 10):
        time.sleep(0.005)
    bench.finalize()

    _, rows = read_csv(bench.session_dir)
    assert [r["step"] for r in rows] == ["2", "3"]  # warmup dropped, surplus dropped
    assert float(rows[0]["eval_event_ms"]) > 1.0


def test_unknown_span_names_raise_key_error(tmp_path: Path):
    bench, _, _ = make_bench(tmp_path)
    with pytest.raises(KeyError):
        bench.cuda_span("data_wait")        # a wall name is not a cuda name
    with pytest.raises(KeyError):
        bench.wall_span("h2d")
    with pytest.raises(KeyError):
        bench.event_span("nope", 10_000)

    disabled = TrainingBenchmark(
        PerfBenchmarkConfig(), checkpoint_path=str(tmp_path / "ckpt"), rank=0
    )
    with pytest.raises(KeyError):
        disabled.cuda_span("typo")  # validated even on the disabled path


def test_benchmark_refuses_to_run_alongside_the_profiler(tmp_path: Path):
    with pytest.raises(ValueError, match="perf_profiler.enabled=true"):
        make_bench(tmp_path, profiler_enabled=True)


def test_nonzero_rank_does_not_raise_the_profiler_mutual_exclusion(tmp_path: Path):
    """Contracted behaviour: the rank gate precedes the profiler check.

    The frozen collector contract raises "when *active* and profiler_enabled",
    and a non-active collector "returns immediately having created nothing", so
    the check is rank-0-only by design.  Under torchrun the rank-0 ValueError is
    what aborts the job; see the reconciling comment at the ``TrainingBenchmark``
    construction site in pretrain.py.
    """
    bench, _, _ = make_bench(tmp_path, rank=1, profiler_enabled=True)
    assert bench.enabled is False
    assert not (tmp_path / "bench").exists()


def test_output_dir_must_be_separate_from_checkpoint_path(tmp_path: Path):
    with pytest.raises(ValueError, match="separate from checkpoint_path"):
        make_bench(tmp_path, output_dir=str(tmp_path / "checkpoints" / "bench"))
    with pytest.raises(ValueError, match="separate from checkpoint_path"):
        make_bench(tmp_path, output_dir=str(tmp_path))  # parent of checkpoint_path


# --------------------------------------------------------------------------- #
# Provenance derivation (never silently pooled)
# --------------------------------------------------------------------------- #


def test_condition_id_and_repeat_are_derived_from_run_name():
    assert derive_condition_id("perf0_m2_tf_z_iter_k6_s1_r2") == "perf0_m2_tf_z_iter_k6_s1"
    assert derive_condition_id("no_suffix") == "no_suffix"
    assert derive_condition_id(None) == "unknown"
    assert derive_condition_id("") == "unknown"

    assert derive_repeat("perf0_m2_tf_z_iter_k6_s1_r3", 1) == 3
    assert derive_repeat("perf0_m2_tf_z_iter_k6_s1_r3", 2) == 2  # explicit wins
    assert derive_repeat("no_suffix", 1) == 1
    assert derive_repeat(None, 1) == 1


def test_derived_provenance_reaches_the_csv(tmp_path: Path):
    bench, clock, _ = make_bench(
        tmp_path, measured_steps=1, run_name="perf0_m2_tf_z_iter_k6_s1_r2"
    )
    drive_update(bench, clock, 1)
    bench.finalize()

    _, rows = read_csv(bench.session_dir)
    assert rows[0]["run_id"] == "perf0_m2_tf_z_iter_k6_s1_r2"
    assert rows[0]["condition_id"] == "perf0_m2_tf_z_iter_k6_s1"
    assert rows[0]["repeat"] == "2"
    assert rows[0]["seed"] == "1"

    m = read_manifest(bench.session_dir)
    assert (m["condition_id"], m["repeat"], m["seed"]) == (
        "perf0_m2_tf_z_iter_k6_s1",
        2,
        1,
    )


def test_explicit_condition_id_overrides_derivation(tmp_path: Path):
    bench, clock, _ = make_bench(
        tmp_path, measured_steps=1, run_name="ignored_r9", condition_id="explicit", repeat=4
    )
    drive_update(bench, clock, 1)
    bench.finalize()
    _, rows = read_csv(bench.session_dir)
    assert rows[0]["condition_id"] == "explicit"
    assert rows[0]["repeat"] == "4"


def test_git_provenance_fails_closed(tmp_path: Path):
    def failing_runner(*args, **kwargs):
        raise OSError("git not found")

    assert pb.git_provenance(tmp_path, runner=failing_runner) == (MISSING, True)

    def nonzero_runner(*args, **kwargs):
        return _FakeResult(returncode=128, stdout="", stderr="not a repository")

    assert pb.git_provenance(tmp_path, runner=nonzero_runner) == (MISSING, True)

    def ok_runner(argv, **kwargs):
        if argv[1] == "rev-parse":
            return _FakeResult(0, "a" * 40 + "\n")
        return _FakeResult(0, " M pretrain.py\n")

    assert pb.git_provenance(tmp_path, runner=ok_runner) == ("a" * 40, True)


def test_data_manifest_hash_is_stable_and_fails_closed(tmp_path: Path):
    assert pb.data_manifest_hash([]) == MISSING
    assert pb.data_manifest_hash([str(tmp_path / "absent")]) == MISSING

    root = tmp_path / "data"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "a.npy").write_bytes(b"0123")
    first = pb.data_manifest_hash([str(root)])
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert pb.data_manifest_hash([str(root)]) == first

    (root / "sub" / "b.npy").write_bytes(b"4567")
    assert pb.data_manifest_hash([str(root)]) != first


# --------------------------------------------------------------------------- #
# P0.0 resource manifest -- shells out ONLY through the injected runner
# --------------------------------------------------------------------------- #

FIXED_NOW = datetime(2026, 7, 28, 3, 4, 5, tzinfo=timezone.utc)

#: The P0.0 output root, transcribed from the contract rather than imported --
#: same treatment as ``FROZEN_HEADER``, so a typo in the module constant fails
#: here instead of shipping silently (every other test injects its own tmp dir).
FROZEN_RESOURCE_MANIFEST_DIR = (
    "reports/figures/2026-07-26_experiment-speed-profiling/manifests"
)


def test_default_resource_manifest_dir_is_the_frozen_contract_path():
    assert pb.DEFAULT_RESOURCE_MANIFEST_DIR == FROZEN_RESOURCE_MANIFEST_DIR


def test_capture_resource_manifest_uses_only_the_three_readonly_probes(tmp_path: Path):
    seen: list[tuple[list[str], dict]] = []

    def runner(argv, **kwargs):
        seen.append((list(argv), kwargs))
        return _FakeResult(0, f"stdout::{argv[0]}", "")

    path = pb.capture_resource_manifest(
        output_dir=tmp_path / "manifests",
        repo_root=tmp_path,
        now=FIXED_NOW,
        runner=runner,
        hostname_fn=lambda: "test-host",
    )

    # The real scripts/queue_run.sh was never executed: exactly three calls, and
    # their argv equals the frozen probe table.
    assert len(seen) == 3
    assert [argv for argv, _ in seen] == [list(a) for _, a in pb.DEFAULT_RESOURCE_PROBES]
    for _, kwargs in seen:
        assert kwargs["cwd"] == str(tmp_path)
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["timeout"] == 120
        assert kwargs["check"] is False

    assert path == tmp_path / "manifests" / "resource_20260728T030405Z.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == pb.RESOURCE_SCHEMA_VERSION == "perf_benchmark.resource/1"
    assert payload["captured_utc"] == "2026-07-28T03:04:05Z"
    assert payload["hostname"] == "test-host"
    assert set(payload["git"]) == {"sha", "dirty"}
    assert set(payload["probes"]) == {"queue_status", "enqueue_dry_run", "nvidia_smi"}
    for name, entry in payload["probes"].items():
        assert entry["status"] == "ok"
        assert entry["returncode"] == 0
        assert entry["stdout"].startswith("stdout::")
        assert isinstance(entry["argv"], list)

    # ``runner`` fakes the PROBES only: git provenance is never taken from probe
    # stdout (that would record garbage in the one provenance field).
    sha = payload["git"]["sha"]
    assert not sha.startswith("stdout::")
    assert sha == MISSING or re.fullmatch(r"[0-9a-f]{40}", sha)


def test_capture_resource_manifest_marks_a_failing_probe_missing(tmp_path: Path):
    def runner(argv, **kwargs):
        if argv[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi missing")
        if argv[0].endswith("queue_run.sh"):
            return _FakeResult(2, "partial output", "boom")
        return _FakeResult(0, "ok", "")

    path = pb.capture_resource_manifest(
        output_dir=tmp_path / "manifests",
        repo_root=tmp_path,
        now=FIXED_NOW,
        runner=runner,
        hostname_fn=lambda: "test-host",
    )
    probes = json.loads(path.read_text(encoding="utf-8"))["probes"]

    assert probes["queue_status"]["status"] == MISSING
    assert probes["queue_status"]["returncode"] == 2
    assert probes["queue_status"]["stdout"] == "partial output"  # text preserved
    assert probes["queue_status"]["stderr"] == "boom"

    assert probes["nvidia_smi"]["status"] == MISSING
    assert probes["nvidia_smi"]["returncode"] is None
    assert "FileNotFoundError" in probes["nvidia_smi"]["stderr"]

    assert probes["enqueue_dry_run"]["status"] == "ok"


def test_capture_resource_manifest_never_overwrites(tmp_path: Path):
    runner = lambda argv, **kwargs: _FakeResult(0, "x", "")  # noqa: E731
    kwargs = dict(
        output_dir=tmp_path / "manifests",
        repo_root=tmp_path,
        now=FIXED_NOW,
        runner=runner,
        hostname_fn=lambda: "test-host",
    )
    first = pb.capture_resource_manifest(**kwargs)
    before = first.read_bytes()
    with pytest.raises(FileExistsError):
        pb.capture_resource_manifest(**kwargs)
    assert first.read_bytes() == before


def test_capture_resource_manifest_probe_table_is_read_only(tmp_path: Path):
    """The frozen probe table must never contain a queue-mutating command."""
    names = [name for name, _ in pb.DEFAULT_RESOURCE_PROBES]
    argvs = [list(argv) for _, argv in pb.DEFAULT_RESOURCE_PROBES]

    assert names == ["queue_status", "enqueue_dry_run", "nvidia_smi"]
    assert argvs[0] == ["scripts/queue_run.sh", "status"]
    assert argvs[1][:2] == ["scripts/sigma_enqueue.sh", "--dry-run"]
    assert argvs[2][0] == "nvidia-smi"
    # No probe may start a worker or write a job file.
    flat = " ".join(" ".join(a) for a in argvs)
    for forbidden in ("queue_run.sh start", "queue_run.sh\n", "--enqueue"):
        assert forbidden not in flat
    assert "--dry-run" in argvs[1]


# --------------------------------------------------------------------------- #
# pretrain.py wiring -- the other named owner in the frozen contract
# --------------------------------------------------------------------------- #


def test_pretrain_wires_the_benchmark_in_and_leaves_it_off_by_default():
    """pretrain.py has no other automated coverage; pin the two contract facts.

    The import is inside the test body, not at module scope, so a heavy or
    broken import cannot break collection of the rest of this file.  It is
    GPU-free: importing pretrain does not initialise a CUDA context.
    """
    import inspect

    # A delta, not an absolute: another test in the session may legitimately have
    # taken a CUDA context (test_perf_profiler attaches a real profiler), and
    # sys.modules may already hold pretrain from an earlier import.
    cuda_was_initialized = torch.cuda.is_initialized()
    import pretrain

    assert (
        torch.cuda.is_initialized() == cuda_was_initialized
    ), "importing pretrain took a CUDA context"

    # Opt-in only: the default PretrainConfig must carry a disabled benchmark.
    field = pretrain.PretrainConfig.model_fields["perf_benchmark"]
    default = field.default_factory()
    assert isinstance(default, PerfBenchmarkConfig)
    assert default.enabled is False
    assert default.performance_only is False

    # train_batch gained a required trailing `bench` parameter, so a missed
    # out-of-tree caller fails loudly instead of silently measuring nothing.
    params = inspect.signature(pretrain.train_batch).parameters
    assert "bench" in params
    assert params["bench"].default is inspect.Parameter.empty
    assert list(params)[-1] == "bench"
