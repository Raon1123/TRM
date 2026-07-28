"""Folded-stack export tests for :mod:`utils.perf_profiler` (GPU-free).

Every test here forces ``torch.cuda.is_available()`` to ``False`` before a
profiler is constructed, so a capture only ever records CPU activity.  That is
deliberate: these tests must not attach to a GPU that a training run owns, and
it also makes the "CUDA metric unavailable" path the deterministic default on
any host.
"""

from pathlib import Path
import contextlib
import json
import warnings

import pydantic
import pytest
import torch

from utils.perf_profiler import (
    STACK_EXPORT_METRICS,
    PerfProfilerConfig,
    TrainingProfiler,
)


@pytest.fixture
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the CPU-only profiler path; never touch a GPU from a unit test."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _stack_config(tmp_path: Path, **overrides) -> PerfProfilerConfig:
    kwargs = dict(
        enabled=True,
        performance_only=True,
        output_dir=str(tmp_path / "profiles"),
        wait=0,
        warmup=0,
        active=1,
        repeat=1,
        max_steps=1,
        with_stack=True,
        export_stacks=True,
        export_tensorboard_trace=False,
    )
    kwargs.update(overrides)
    return PerfProfilerConfig(**kwargs)


def _run_one_capture(config: PerfProfilerConfig, tmp_path: Path) -> TrainingProfiler:
    profiler = TrainingProfiler(config, checkpoint_path=str(tmp_path / "checkpoints"), rank=0)
    profiler.start()
    with profiler.record("unit/work"):
        tensor = torch.randn(32, 32, requires_grad=True)
        (tensor @ tensor).sum().backward()
    profiler.step()
    profiler.stop()
    return profiler


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_export_stacks_defaults_to_off():
    assert PerfProfilerConfig().export_stacks is False


def test_export_stacks_without_with_stack_is_a_validation_error():
    with pytest.raises(pydantic.ValidationError, match="with_stack"):
        PerfProfilerConfig(
            enabled=True, performance_only=True, export_stacks=True, with_stack=False
        )


def test_export_stacks_with_with_stack_validates():
    config = PerfProfilerConfig(
        enabled=True, performance_only=True, export_stacks=True, with_stack=True
    )
    assert config.export_stacks and config.with_stack


def test_disabled_run_is_not_forced_to_set_with_stack():
    # The validator is gated on `enabled`, like every other check in it: an
    # inert config carrying export_stacks=True must not break config loading.
    assert PerfProfilerConfig(export_stacks=True).with_stack is False


# --------------------------------------------------------------------------
# a real (CPU-only) capture
# --------------------------------------------------------------------------


def test_capture_writes_folded_cpu_stacks_and_manifest(tmp_path: Path, cpu_only: None):
    config = _stack_config(tmp_path)
    with pytest.warns(UserWarning):
        profiler = _run_one_capture(config, tmp_path)

    captures = sorted(Path(config.output_dir).glob("session_*/capture_*"))
    assert len(captures) == 1
    capture = captures[0]

    cpu_stacks = capture / "stacks_self_cpu_time_total.txt"
    assert cpu_stacks.exists()
    lines = [line for line in cpu_stacks.read_text().splitlines() if line.strip()]
    assert lines, "with_stack capture produced no folded CPU frames"
    for line in lines:
        frames, _, value = line.rpartition(" ")
        assert frames, f"folded line has no stack: {line!r}"
        assert int(value) > 0

    manifest = json.loads((capture / "stacks_manifest.json").read_text())
    assert manifest["with_stack"] is True
    assert [entry["metric"] for entry in manifest["metrics"]] == list(STACK_EXPORT_METRICS)
    cpu_entry = manifest["metrics"][0]
    assert cpu_entry["status"] == "written"
    assert cpu_entry["lines"] == len(lines)
    assert profiler.stack_exports == manifest["metrics"]


def test_missing_cuda_metric_is_recorded_not_silently_swallowed(tmp_path: Path, cpu_only: None):
    config = _stack_config(tmp_path)
    with pytest.warns(UserWarning, match="self_cuda_time_total"):
        profiler = _run_one_capture(config, tmp_path)

    capture = sorted(Path(config.output_dir).glob("session_*/capture_*"))[0]
    # No zero-byte flamegraph input is left behind ...
    assert not (capture / "stacks_self_cuda_time_total.txt").exists()
    # ... but the reason it is absent is on disk and on the object.
    cuda_entry = next(
        entry for entry in profiler.stack_exports if entry["metric"] == "self_cuda_time_total"
    )
    assert cuda_entry["status"] == "unavailable"
    assert cuda_entry["lines"] == 0
    assert "self_cuda_time_total" in cuda_entry["reason"]
    manifest = json.loads((capture / "stacks_manifest.json").read_text())
    assert manifest["cuda_available"] is False
    assert manifest["metrics"][1]["reason"] == cuda_entry["reason"]


def test_capture_without_export_stacks_writes_no_stack_artifacts(tmp_path: Path, cpu_only: None):
    """A run that never asked for stacks is distinguishable from an empty one."""
    config = _stack_config(tmp_path, export_stacks=False)
    with pytest.warns(UserWarning):
        profiler = _run_one_capture(config, tmp_path)

    capture = sorted(Path(config.output_dir).glob("session_*/capture_*"))[0]
    assert sorted(path.name for path in capture.iterdir()) == ["trace.json"]
    assert profiler.stack_exports == []


def test_both_trace_exports_enabled_is_rejected(tmp_path: Path):
    """PERF-DEV-18: the combination cannot work, so it must not be constructible.

    torch permits one kineto save per ``on_trace_ready``; the second raises
    "Trace is already saved." from ``profiler.step()`` and aborts the training
    run mid-capture.  This previously escaped config validation, which is why
    the pre-registered M3 command (profiling doc section 4.2.1) -- which sets
    both flags true -- could never have run.  Rejecting it up front turns a
    confusing mid-run abort into an immediate, readable error.
    """
    with pytest.raises(ValueError, match="cannot both be true"):
        _stack_config(tmp_path, export_chrome_trace=True, export_tensorboard_trace=True)

    # The rejection must be specific to the combination, not to either flag.
    _stack_config(tmp_path, export_chrome_trace=True, export_tensorboard_trace=False)
    _stack_config(tmp_path, export_chrome_trace=False, export_tensorboard_trace=True)

    # ... and it must not fire on a disabled profiler, mirroring how every other
    # bound in this config is gated on `enabled`.
    PerfProfilerConfig(
        enabled=False, export_chrome_trace=True, export_tensorboard_trace=True
    )


def test_stacks_are_written_under_a_tensorboard_only_export(
    tmp_path: Path, cpu_only: None
):
    """The folded-stack export must not depend on which trace exporter is used."""
    config = _stack_config(
        tmp_path, export_chrome_trace=False, export_tensorboard_trace=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _run_one_capture(config, tmp_path)

    capture = sorted(Path(config.output_dir).glob("session_*/capture_*"))[0]
    assert (capture / "stacks_self_cpu_time_total.txt").exists()
    assert (capture / "stacks_manifest.json").exists()


def test_nonzero_rank_writes_no_stacks(tmp_path: Path, cpu_only: None):
    profiler = TrainingProfiler(
        _stack_config(tmp_path), checkpoint_path=str(tmp_path / "checkpoints"), rank=1
    )
    assert not profiler.enabled
    assert profiler.stack_exports == []
    assert not (tmp_path / "profiles").exists()


# --------------------------------------------------------------------------
# export failure modes, driven through a stub profiler
# --------------------------------------------------------------------------


class _StubProfiler:
    """Stands in for ``torch.profiler.profile`` in the export-only code path."""

    def __init__(self, *, raises=(), empty=()) -> None:
        self.raises = set(raises)
        self.empty = set(empty)
        self.calls: list[str] = []

    def export_stacks(self, path: str, metric: str) -> None:
        self.calls.append(metric)
        if metric in self.raises:
            # torch opens the file before it can fail in some versions.
            Path(path).write_text("")
            raise RuntimeError(f"no kineto results for {metric}")
        # Mirrors torch: the file is opened before events are filtered, so an
        # unavailable metric leaves a zero-byte file behind.
        Path(path).write_text("" if metric in self.empty else "a;b;c 12\na;b 3\n")


def _export_dir(tmp_path: Path, cpu_only_config: PerfProfilerConfig) -> tuple[TrainingProfiler, Path]:
    profiler = TrainingProfiler(
        cpu_only_config, checkpoint_path=str(tmp_path / "checkpoints"), rank=0
    )
    profiler.stop()
    trace_dir = Path(profiler.session_dir) / "capture_00"
    trace_dir.mkdir()
    return profiler, trace_dir


def test_export_error_removes_file_and_records_reason(tmp_path: Path, cpu_only: None):
    profiler, trace_dir = _export_dir(tmp_path, _stack_config(tmp_path))
    stub = _StubProfiler(raises={"self_cuda_time_total"})

    with pytest.warns(UserWarning, match="self_cuda_time_total"):
        profiler._export_folded_stacks(stub, trace_dir)

    assert stub.calls == list(STACK_EXPORT_METRICS)
    assert (trace_dir / "stacks_self_cpu_time_total.txt").exists()
    assert not (trace_dir / "stacks_self_cuda_time_total.txt").exists()
    manifest = json.loads((trace_dir / "stacks_manifest.json").read_text())
    cpu_entry, cuda_entry = manifest["metrics"]
    assert cpu_entry["status"] == "written" and cpu_entry["lines"] == 2
    assert cuda_entry["status"] == "unavailable"
    assert "RuntimeError" in cuda_entry["reason"]


def test_empty_export_is_reported_as_unavailable(tmp_path: Path, cpu_only: None):
    profiler, trace_dir = _export_dir(tmp_path, _stack_config(tmp_path))
    stub = _StubProfiler(empty=set(STACK_EXPORT_METRICS))

    with pytest.warns(UserWarning, match="self_cpu_time_total"):
        profiler._export_folded_stacks(stub, trace_dir)

    assert sorted(path.name for path in trace_dir.iterdir()) == ["stacks_manifest.json"]
    manifest = json.loads((trace_dir / "stacks_manifest.json").read_text())
    assert [entry["status"] for entry in manifest["metrics"]] == ["unavailable", "unavailable"]
    assert all("empty" in entry["reason"] for entry in manifest["metrics"])


def test_full_success_still_writes_a_manifest(tmp_path: Path, cpu_only: None):
    profiler, trace_dir = _export_dir(tmp_path, _stack_config(tmp_path))
    stub = _StubProfiler()

    profiler._export_folded_stacks(stub, trace_dir)

    manifest = json.loads((trace_dir / "stacks_manifest.json").read_text())
    assert [entry["status"] for entry in manifest["metrics"]] == ["written", "written"]
    assert all("reason" not in entry for entry in manifest["metrics"])
    assert len(profiler.stack_exports) == 2


def test_manifest_is_never_clobbered(tmp_path: Path, cpu_only: None):
    profiler, trace_dir = _export_dir(tmp_path, _stack_config(tmp_path))
    profiler._export_folded_stacks(_StubProfiler(), trace_dir)
    with pytest.raises(FileExistsError):
        profiler._export_folded_stacks(_StubProfiler(), trace_dir)
