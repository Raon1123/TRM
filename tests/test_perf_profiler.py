from pathlib import Path

import pydantic
import pytest

from utils.perf_profiler import PerfProfilerConfig, TrainingProfiler


def test_disabled_profiler_has_no_output_or_torch_profiler(tmp_path: Path):
    profiler = TrainingProfiler(
        PerfProfilerConfig(), checkpoint_path=str(tmp_path / "checkpoints"), rank=0
    )

    assert not profiler.enabled
    with profiler.record("ignored"):
        pass
    profiler.start()
    profiler.step()
    profiler.stop()
    assert not (tmp_path / "performance_profiles").exists()


def test_enabled_requires_performance_only_acknowledgement():
    with pytest.raises(pydantic.ValidationError, match="performance_only"):
        PerfProfilerConfig(enabled=True)


def test_rank_zero_trace_ownership_cannot_be_disabled():
    with pytest.raises(pydantic.ValidationError, match="rank0_only"):
        PerfProfilerConfig(enabled=True, performance_only=True, rank0_only=False)


def test_schedule_must_fit_within_max_steps():
    with pytest.raises(pydantic.ValidationError, match="max_steps"):
        PerfProfilerConfig(
            enabled=True, performance_only=True, wait=1, warmup=1, active=3, repeat=1, max_steps=4
        )


def test_profiler_rejects_checkpoint_output_overlap(tmp_path: Path):
    config = PerfProfilerConfig(
        enabled=True,
        performance_only=True,
        output_dir=str(tmp_path / "checkpoints" / "profiles"),
    )
    with pytest.raises(ValueError, match="separate from checkpoint_path"):
        TrainingProfiler(config, checkpoint_path=str(tmp_path / "checkpoints"), rank=0)


def test_profiler_rejects_checkpoint_when_output_is_its_parent(tmp_path: Path):
    config = PerfProfilerConfig(
        enabled=True, performance_only=True, output_dir=str(tmp_path / "artifacts")
    )
    with pytest.raises(ValueError, match="separate from checkpoint_path"):
        TrainingProfiler(
            config, checkpoint_path=str(tmp_path / "artifacts" / "checkpoints"), rank=0
        )


def test_cpu_trace_is_bounded_and_never_overwrites(tmp_path: Path):
    output_dir = tmp_path / "profiles"
    config = PerfProfilerConfig(
        enabled=True,
        performance_only=True,
        output_dir=str(output_dir),
        wait=0,
        warmup=0,
        active=1,
        repeat=1,
        max_steps=1,
        export_tensorboard_trace=False,
    )

    profiler = TrainingProfiler(config, checkpoint_path=str(tmp_path / "checkpoints"), rank=0)
    profiler.start()
    with profiler.record("unit/work"):
        _ = sum(range(10))
    profiler.step()
    profiler.stop()

    assert len(list(output_dir.glob("session_*/capture_*/trace.json"))) == 1
    assert not profiler.enabled

    next_profiler = TrainingProfiler(config, checkpoint_path=str(tmp_path / "checkpoints"), rank=0)
    assert next_profiler.session_dir is not profiler.session_dir
    next_profiler.stop()


def test_nonzero_rank_does_not_create_trace_directory(tmp_path: Path):
    profiler = TrainingProfiler(
        PerfProfilerConfig(enabled=True, performance_only=True, output_dir=str(tmp_path / "profiles")),
        checkpoint_path=str(tmp_path / "checkpoints"),
        rank=1,
    )
    assert not profiler.enabled
    assert not (tmp_path / "profiles").exists()
