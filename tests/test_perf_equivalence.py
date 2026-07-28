"""PERF-001 T1: paired enabled/disabled equivalence of the training path.

Specification: ``lab/reports/2026-07-28_perf001-resource-plan.md`` §5.5 / T1,
which escalates the plan's G1 *manual* frozen-state gate into an automated one.
The PI's requirement, verbatim: "동등한 실험 결과가 나온다는 것 또한 테스트에서
보장하기를 기대한다."

``tests/test_pretrain_perf_disabled_path.py`` proves the harness is INERT when
off.  That is only half the claim.  This file proves the other half: turning the
harness ON leaves the experiment unchanged.  The same fixed training sequence is
replayed twice from identical starting state -- once with
``perf_benchmark.enabled=False``, once with ``enabled=True`` -- and the two runs
are compared as whole traces.

WHAT "INDISTINGUISHABLE" MEANS HERE (the trace fields, per T1's enumeration):

* every model call: its ordinal, ``carry``, the batch keys/identities it saw,
  and ``return_keys``;
* ``optimizer.step_calls`` / ``zero_grad_calls`` counts, and the SEQUENCE of
  ``lr`` values written into ``param_groups`` (the config used here has
  ``lr_warmup_steps=2`` and ``lr_min_ratio=0.1`` precisely so that sequence is
  non-constant -- under the ``minimal_config()`` default it would be a constant
  and a scrambled schedule would compare equal);
* the FULL metrics dict returned by every ``train_batch`` call -- every key,
  every value, bit-exact;
* ``train_state.step`` progression and ``train_state.carry`` after each update;
* per-batch-value ``.cuda()`` call counts (the ``h2d`` span must neither add
  nor drop a transfer);
* the exception that escapes a failing update, and whether the update that
  raised was rolled back the same way.

BIT-EXACT, NOT APPROXIMATE.  Instrumentation does not touch arithmetic, so any
numerical drift is already a defect.  Floats are canonicalized through
``_canon`` as ``(type name, dtype string, float.hex())`` and compared with
``==`` on that tuple, which is strictly stronger than ``==`` on the values:
it separates ``-0.0`` from ``0.0`` and makes ``NaN`` compare equal to ``NaN``.
``pytest.approx`` appears nowhere in an equivalence assertion.

GPU-FREE, AND UNDER THE DISABLED FILE'S TRIPWIRE.  ``_cuda_tripwire`` is
imported and rebound at module level, so pytest registers it as an autouse
fixture for this file too: the ten ``torch.cuda`` attributes plus
``torch.Tensor.cuda`` raise, and the process CUDA-init state is asserted
unchanged across every test here.  The ENABLED collector is therefore driven
the way ``tests/test_perf_benchmark.py::make_bench`` drives it -- with injected
``event_factory``/``sync_fn``/``utilization_fn``/``memory_stats_fn``/
``reset_peak_fn`` -- so no CUDA entry point is reached even with the harness on.
``test_the_cuda_tripwire_is_active_in_this_module`` exists so that this
mechanism cannot silently go dead (an import refactor that dropped the rebind
would otherwise make every other test here weaker with no visible failure).

NON-VACUITY.  A paired test passes trivially if the "enabled" run never
actually collected anything.  ``_assert_bench_really_collected`` is the guard:
it requires ``steady_state.csv`` to hold exactly ``measured_steps`` data rows,
with ``input_tokens``/``target_tokens`` parsed as real ints (not ``MISSING``)
and all five CUDA columns populated.  Token counts are the only place enabled
code reads the experiment's own data (``batch["inputs"].numel()`` and
``(batch["labels"] != IGNORE_LABEL_ID).sum().item()`` in ``begin_update``), and
both are wrapped in ``except: return MISSING`` -- so a batch value that cannot
answer them degrades silently.  That is why ``_TensorBatchValue`` exists rather
than plain ``_FakeBatchValue``: it is tensor-backed, so ``MISSING`` in those
columns means the enabled path did not really run.

FAKES ARE IMPORTED, NEVER COPIED.  ``_FakeBatchValue``, ``_FakeModel``,
``_FakeOptimizer``, ``minimal_config`` and ``_cuda_tripwire`` come from
``tests.test_pretrain_perf_disabled_path``; ``_Clock``, ``_FakeEvent`` and
``make_bench`` from ``tests.test_perf_benchmark``.  The recording classes here
are SUBCLASSES that add a trace, so the behaviour under test is the same object
the existing 13 tests exercise.

WHAT THIS FILE DOES NOT CLAIM.  The span objects hold only
``(bench, row, name, event)`` -- they are handed no reference to ``metrics`` or
to the batch dict -- so no perturbation *inside a span* can change a metric
value.  The metrics-equality assertions are therefore a guard against a future
rewiring that gives a span such a handle, not a test that today's spans could
plausibly fail.  The levers that do reach the experiment today are the ones the
made-it-fail demonstration uses: ``begin_update``'s batch scan, and a span
``__exit__`` swallowing an exception.  Real CUDA event semantics remain
evidence-pending exactly as in the sibling files.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import pretrain
from utils import perf_benchmark as pb

from tests.test_perf_benchmark import _Clock, make_bench
from tests.test_pretrain_perf_disabled_path import (
    _FakeBatchValue,
    _FakeModel,
    _FakeOptimizer,
    _cuda_tripwire,
    disabled_bench,
    minimal_config,
)

#: Rebound so pytest registers the sibling file's autouse CUDA tripwire here as
#: well.  Without this line every test below would run with real ``torch.cuda``
#: attributes and the "GPU-free" claim would rest on inspection alone.
_cuda_tripwire = _cuda_tripwire

#: Window shape for the enabled collector.  ``warmup_steps=1`` and
#: ``measured_steps=3`` over a 6-update sequence means the trace crosses all
#: three regimes -- warmup (no row open), measured (spans armed), post-window
#: (row closed again) -- so equivalence is asserted across the regime changes,
#: not just inside the measured window.
WARMUP_STEPS = 1
MEASURED_STEPS = 3

#: Updates driven per run.  ``TOTAL_STEPS`` is deliberately smaller, so the last
#: update takes ``train_batch``'s ``total_steps`` early return: the branch where
#: enabled bookkeeping diverges most (``end_update`` called with no row open).
UPDATES = 6
TOTAL_STEPS = 5

#: Non-degenerate lr schedule; see the module docstring.
LR_SCHEDULE_OVERRIDES = dict(lr_warmup_steps=2, lr_min_ratio=0.1)


@pytest.fixture(autouse=True)
def _no_cuda_available(monkeypatch: pytest.MonkeyPatch):
    """Keep ``finalize``'s device-provenance probe off the driver, deterministically.

    ``_device_provenance`` consults ``torch.cuda.is_available() and
    is_initialized()``.  Both are pure queries and the tripwire deliberately
    leaves them alone, and every deeper call it would make is tripwired and
    swallowed by its own ``except Exception`` -- so this fixture is belt-and-
    braces, not load-bearing.  It matters for one reason: the manifest written
    by the enabled run must not vary with what an unrelated earlier test in the
    session did to the process CUDA state.  Mirrors
    ``tests/test_perf_benchmark.py::_no_cuda``.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


# --------------------------------------------------------------------------- #
# Fakes: subclasses of the sibling file's, adding a recorded trace
# --------------------------------------------------------------------------- #


class _TensorBatchValue(_FakeBatchValue):
    """A ``_FakeBatchValue`` that can answer ``begin_update``'s token scan.

    ``begin_update`` reads ``batch["inputs"].numel()`` and
    ``(batch["labels"] != IGNORE_LABEL_ID).sum().item()`` off the CPU batch,
    each inside ``except Exception: return MISSING``.  The plain
    ``_FakeBatchValue`` answers neither, so an enabled run over it writes
    ``MISSING`` into both token columns and the non-vacuity guard cannot tell a
    working collector from a broken one.  Backing the value with a real CPU
    tensor closes that hole while keeping ``.cuda()`` a counter that returns
    ``self`` -- so ``torch.Tensor.cuda``, which the tripwire patches to raise,
    is never reached.
    """

    __slots__ = ("tensor",)

    def __init__(self, name: str, tensor: torch.Tensor) -> None:
        super().__init__(name)
        self.tensor = tensor

    def numel(self) -> int:
        return self.tensor.numel()

    def __ne__(self, other: Any):  # noqa: D105 - delegates to the backing tensor
        return self.tensor != other

    __hash__ = object.__hash__


class _RecordingModel(_FakeModel):
    """``_FakeModel`` plus a per-call trace and step-dependent metric values.

    The base class returns the SAME two metric values on every call, so a
    comparator that silently dropped or transposed a value would still compare
    equal.  Here ``count`` and ``lm_loss`` vary with the call ordinal, and a
    third key is added, so the returned-metrics comparison has something to
    distinguish.  ``initial_carry`` stays string-valued (no tensor): it is
    called inside ``with torch.device("cuda")`` in ``pretrain.train_batch``,
    which the sibling file documents as an unpatchable route to a context.
    """

    def __init__(self) -> None:
        super().__init__()
        self.trace: list[dict[str, Any]] = []
        self.initial_carry_trace: list[Any] = []

    def initial_carry(self, batch):
        self.initial_carry_trace.append(_batch_signature(batch))
        return super().initial_carry(batch)

    def __call__(self, *, carry, batch, return_keys):
        ordinal = self.calls  # pre-increment: 0-based call index
        self.trace.append(
            {
                "ordinal": ordinal,
                "carry": carry,
                "batch": _batch_signature(batch),
                "return_keys": list(return_keys),
            }
        )
        self.calls += 1
        self.last_carry = carry
        self.last_batch = batch
        # A real CPU autograd graph, so ``((1 / gbs) * loss).backward()`` runs
        # for real without a device -- same construction as ``_FakeModel``.
        weight = torch.ones(2, requires_grad=True)
        loss = (weight * (3.0 + ordinal)).sum()
        metrics = {
            "count": torch.tensor(4.0 + ordinal),
            "lm_loss": torch.tensor(8.0 - 0.25 * ordinal),
            "accuracy": torch.tensor(0.5 + 0.125 * ordinal),
        }
        return f"carry-after-forward-{ordinal}", loss, metrics, None, None


class _RecordingOptimizer(_FakeOptimizer):
    """``_FakeOptimizer`` plus the sequence of lr values actually applied.

    Recorded in ``step()``, i.e. after ``train_batch`` has written
    ``param_group['lr']``, so the trace is "the lr this update stepped with"
    rather than "an lr that was assigned at some point".
    """

    def __init__(self) -> None:
        super().__init__()
        self.lr_trace: list[Any] = []

    def step(self):
        self.lr_trace.append(_canon(self.param_groups[0]["lr"]))
        super().step()


class _RaisingModel(_RecordingModel):
    """Raises inside the ``forward_backward`` span on a chosen call ordinal."""

    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call

    def __call__(self, *, carry, batch, return_keys):
        if self.calls == self.fail_on_call:
            self.calls += 1
            raise RuntimeError(f"synthetic forward failure on call {self.fail_on_call}")
        return super().__call__(carry=carry, batch=batch, return_keys=return_keys)


class _RecordingLoader:
    """A plain iterable of pre-built batches; records how often it was iterated."""

    def __init__(self, batches: list[dict[str, _TensorBatchValue]]) -> None:
        self.batches = batches
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return iter(self.batches)


# --------------------------------------------------------------------------- #
# Canonicalization
# --------------------------------------------------------------------------- #


def _canon(value: Any) -> Any:
    """Bit-exact, type-aware canonical form for a metric/lr value.

    Floats become ``(type name, dtype string, float.hex())``.  The hex mantissa
    is exact, so this is strictly stronger than ``==``: it separates ``-0.0``
    from ``0.0``, and it makes ``NaN`` compare equal to ``NaN`` instead of
    unequal to itself.  Non-float values are returned with their type name
    attached, so ``1`` and ``1.0`` and ``np.float32(1.0)`` never compare equal.
    """
    if isinstance(value, (float, np.floating)):
        dtype = np.asarray(value).dtype.str
        return (type(value).__name__, dtype, float(value).hex())
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return (type(value).__name__, int(value))
    if isinstance(value, torch.Tensor):
        return (
            "Tensor",
            str(value.dtype),
            tuple(value.shape),
            tuple(float(x).hex() for x in value.reshape(-1).tolist()),
        )
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tobytes())
    return (type(value).__name__, value)


def _canon_metrics(metrics: Any) -> Any:
    if metrics is None:
        return None
    return {key: _canon(value) for key, value in sorted(metrics.items())}


def _batch_signature(batch: Any) -> Any:
    """Identity-free description of the batch dict handed to the model.

    Keys in order, each value's ``name``, its ``.cuda()`` count at the moment of
    the call, and the backing tensor's exact contents.  ``id()`` is deliberately
    NOT used -- object addresses differ between two runs for reasons that have
    nothing to do with the harness.  Instead,
    ``test_..._passes_through_the_same_batch_objects`` asserts identity
    (``is``) WITHIN a run, which is the property that matters: the h2d span
    must hand the model the very objects the loader produced.
    """
    if not isinstance(batch, dict):
        return ("not-a-dict", repr(batch))
    return tuple(
        (
            key,
            value.name,
            value.cuda_calls,
            _canon(value.tensor) if isinstance(value, _TensorBatchValue) else None,
        )
        for key, value in batch.items()
    )


# --------------------------------------------------------------------------- #
# The paired driver
# --------------------------------------------------------------------------- #


def _make_batches(count: int) -> list[dict[str, _TensorBatchValue]]:
    """A FIXED batch sequence: contents depend only on the index, never on RNG.

    Built fresh for each run.  Sharing the objects between the two runs would
    let ``cuda_calls`` accumulate across runs (1 in the first, 2 in the second)
    and the H2D-count comparison would go red for a reason unrelated to the
    harness.
    """
    batches = []
    for i in range(count):
        inputs = torch.arange(8, dtype=torch.long).reshape(2, 4) + i
        labels = torch.arange(8, dtype=torch.long).reshape(2, 4) - i
        # Give ``_count_target_tokens`` something to exclude, so the column is
        # not simply ``numel``.
        labels.view(-1)[0] = pb.IGNORE_LABEL_ID
        batches.append(
            {
                "inputs": _TensorBatchValue("inputs", inputs),
                "labels": _TensorBatchValue("labels", labels),
                "puzzle_identifiers": _TensorBatchValue(
                    "puzzle_identifiers", torch.full((2,), i, dtype=torch.long)
                ),
            }
        )
    return batches


def _run_training(
    bench: pb.TrainingBenchmark,
    clock: _Clock | None,
    *,
    model: _RecordingModel | None = None,
    updates: int = UPDATES,
    total_steps: int = TOTAL_STEPS,
    event_step: int | None = None,
) -> dict[str, Any]:
    """Replay ``pretrain.py``'s train loop over a fixed batch sequence.

    Everything ``launch()`` does around ``train_batch`` that carries a span is
    mirrored here in the ratified order: the ``iter_batches`` loader
    conditional, ``train_batch`` itself (which owns ``begin_update`` and the
    h2d/forward_backward/optimizer/metrics_device/metrics_wandb spans), then the
    loop-level ``wall_span("metrics_wandb")``, ``cuda_span("ema")``,
    ``end_update``, and the three ``event_span``s.  ``clock`` is advanced only
    where the test owns the code (the ema span); the spans inside
    ``train_batch`` therefore measure 0.0 ms, which is irrelevant to
    equivalence and is why no timing value is compared between runs.

    Returns ``(trace, objects)``.  ``trace`` is the deliverable -- the two runs
    are compared by ``==`` on it, so every field it carries is part of the
    equivalence claim and a field added here is compared automatically.
    ``objects`` is deliberately OUTSIDE the compared trace: it holds live
    objects (batch dicts, model, optimizer, loader) for the within-run identity
    assertions, and object addresses are not a property of the harness.
    """
    config = minimal_config(**LR_SCHEDULE_OVERRIDES)
    model = model if model is not None else _RecordingModel()
    optimizer = _RecordingOptimizer()
    train_state = pretrain.TrainState(
        model=model,
        optimizers=[optimizer],
        optimizer_lrs=[config.lr],
        carry=None,
        step=0,
        total_steps=total_steps,
    )
    batches = _make_batches(updates)
    loader = _RecordingLoader(batches)

    # The exact conditional from pretrain.py's train loop.
    iterable = bench.iter_batches(loader) if bench.enabled else loader

    updates_trace: list[dict[str, Any]] = []
    seen_batches: list[Any] = []
    #: The LIVE metrics dicts ``train_batch`` returned, kept unconverted.  The
    #: compared trace stores their canonical (frozen) form, so comparing that
    #: across ``finalize()`` would be a tautology; these references are what
    #: makes the "finalize does not reach back into the run" assertion real.
    raw_metrics: list[Any] = []
    pre_finalize_metrics: list[Any] = []
    raised: Any = None

    try:
        for batch in iterable:
            seen_batches.append(batch)
            metrics = pretrain.train_batch(
                config,
                train_state,
                batch,
                config.global_batch_size,
                rank=0,
                world_size=1,
                bench=bench,
            )
            with bench.wall_span("metrics_wandb"):  # stands in for wandb.log
                pass
            with bench.cuda_span("ema"):  # stands in for ema_helper.update
                if clock is not None:
                    clock.now += 8.0
            bench.end_update(train_state.step)
            raw_metrics.append(metrics)

            if event_step is not None and train_state.step >= event_step:
                for name in ("eval", "zprobe", "checkpoint"):
                    with bench.event_span(name, train_state.step):
                        pass

            updates_trace.append(
                {
                    "metrics": _canon_metrics(metrics),
                    "step": train_state.step,
                    "carry": train_state.carry,
                    "batch_cuda_calls": [v.cuda_calls for v in batch.values()],
                    "batch_keys": list(batch),
                }
            )
    except Exception as exc:  # recorded, not swallowed: compared between runs
        raised = (type(exc).__name__, str(exc))
    finally:
        # Snapshot taken BEFORE finalize, from the live dicts, so the caller can
        # tell whether finalize wrote into them.
        pre_finalize_metrics = [_canon_metrics(m) for m in raw_metrics]
        # PERF-DEV-08: the real train loop finalizes from a ``finally``.
        bench.finalize()

    trace = {
        "updates": updates_trace,
        "model_calls": model.calls,
        "model_trace": model.trace,
        "initial_carry_calls": model.initial_carry_calls,
        "initial_carry_trace": model.initial_carry_trace,
        "step_calls": optimizer.step_calls,
        "zero_grad_calls": optimizer.zero_grad_calls,
        "lr_trace": optimizer.lr_trace,
        "final_step": train_state.step,
        "final_carry": train_state.carry,
        "loader_iter_calls": loader.iter_calls,
        "batches_consumed": len(seen_batches),
        "raised": raised,
    }
    # Objects, not part of the compared trace -- returned for the within-run
    # identity assertions.
    return trace, {
        "batches": batches,
        "seen_batches": seen_batches,
        "model": model,
        "optimizer": optimizer,
        "train_state": train_state,
        "loader": loader,
        "raw_metrics": raw_metrics,
        "pre_finalize_metrics": pre_finalize_metrics,
    }


def _run_disabled(tmp_path: Path, **kwargs):
    bench = disabled_bench(tmp_path / "never_created")
    assert bench.enabled is False
    trace, objects = _run_training(bench, None, **kwargs)
    assert not (tmp_path / "never_created").exists()
    return trace, objects, bench


def _run_enabled(tmp_path: Path, **kwargs):
    bench, clock, calls = make_bench(
        tmp_path,
        warmup_steps=WARMUP_STEPS,
        measured_steps=MEASURED_STEPS,
        max_steps=100,
        eval_event_step=WARMUP_STEPS + MEASURED_STEPS + 1,
    )
    assert bench.enabled is True
    trace, objects = _run_training(bench, clock, **kwargs)
    return trace, objects, bench, calls


def _read_rows(session_dir: Path) -> list[dict[str, str]]:
    with (session_dir / "steady_state.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_bench_really_collected(bench: pb.TrainingBenchmark) -> list[dict[str, str]]:
    """Non-vacuity guard: the enabled run must have produced real rows.

    Without this, every equivalence assertion in this file would pass on a
    collector that silently did nothing -- which is precisely the failure mode
    a paired test is blind to.  ``input_tokens``/``target_tokens`` are the
    strongest available signal: they are the ONLY place enabled code reads the
    experiment's own data, and both degrade to ``MISSING`` inside a bare
    ``except`` rather than raising.
    """
    assert bench.session_dir is not None
    rows = _read_rows(bench.session_dir)
    assert len(rows) == MEASURED_STEPS, f"expected {MEASURED_STEPS} measured rows, got {len(rows)}"
    for row in rows:
        assert row["input_tokens"] != pb.MISSING
        assert row["target_tokens"] != pb.MISSING
        assert int(row["input_tokens"]) == 8, "begin_update did not read the real batch"
        assert int(row["target_tokens"]) == 7, "the IGNORE_LABEL_ID scan did not run"
        for column in (
            "h2d_cuda_ms",
            "forward_backward_cuda_ms",
            "optimizer_cuda_ms",
            "ema_cuda_ms",
            "metrics_device_cuda_ms",
        ):
            assert row[column] != pb.MISSING, f"{column} was never armed"
    assert (bench.session_dir / "manifest.json").exists()
    return rows


# --------------------------------------------------------------------------- #
# Tripwire liveness
# --------------------------------------------------------------------------- #


def test_the_cuda_tripwire_is_active_in_this_module():
    """The imported autouse fixture really applies here.

    Every GPU-free claim in this file rests on the rebind of ``_cuda_tripwire``
    at module scope.  An import refactor that dropped it would leave the other
    tests passing while silently running against real ``torch.cuda``.  This
    test is the only thing that would go red.
    """
    with pytest.raises(AssertionError, match="reached torch.cuda.synchronize"):
        torch.cuda.synchronize()
    with pytest.raises(AssertionError, match="reached torch.cuda.Event"):
        torch.cuda.Event(enable_timing=True)
    with pytest.raises(AssertionError, match="moved a real torch.Tensor to CUDA"):
        torch.ones(1).cuda()


# --------------------------------------------------------------------------- #
# T1: the equivalence property
# --------------------------------------------------------------------------- #


def test_enabling_perf_benchmark_does_not_change_training_results(tmp_path: Path):
    """THE T1 property: two runs, identical start, identical fixed batches.

    Compared as whole traces with ``==`` on canonicalized values -- see the
    module docstring for the field list and for why the comparison is bit-exact
    rather than approximate.
    """
    disabled_trace, _, disabled_bench_obj = _run_disabled(tmp_path / "off")
    enabled_trace, _, enabled_bench_obj, calls = _run_enabled(tmp_path / "on")

    # Non-vacuity first: if the enabled run collected nothing, everything below
    # is a comparison of two disabled runs.
    _assert_bench_really_collected(enabled_bench_obj)
    assert disabled_bench_obj.session_dir is None

    # The whole trace at once, so a field added to ``_run_training`` is compared
    # automatically instead of needing a new assertion.
    assert enabled_trace == disabled_trace

    # Positive controls: the sequence really ran, and really hit the
    # ``total_steps`` early return on its last update.
    assert disabled_trace["batches_consumed"] == UPDATES
    assert disabled_trace["model_calls"] == TOTAL_STEPS
    assert disabled_trace["step_calls"] == TOTAL_STEPS
    assert disabled_trace["final_step"] == UPDATES
    assert disabled_trace["updates"][-1]["metrics"] is None, (
        "the total_steps early-return branch was never taken"
    )
    assert disabled_trace["raised"] is None
    # The enabled collector did its own work: one window-closing synchronize.
    assert calls.get("sync") == 1
    assert calls.get("reset_peak") == 1


def test_enabled_and_disabled_apply_the_same_lr_sequence(tmp_path: Path):
    """The lr sequence specifically, asserted to be non-constant first.

    Subsumed by the whole-trace comparison above, but stated separately because
    a constant schedule would make that comparison vacuous for this field, and
    nothing else in the file would notice.
    """
    disabled_trace, _, _ = _run_disabled(tmp_path / "off")
    enabled_trace, _, bench, _ = _run_enabled(tmp_path / "on")
    _assert_bench_really_collected(bench)

    lrs = disabled_trace["lr_trace"]
    assert len(lrs) == TOTAL_STEPS
    assert len(set(lrs)) > 1, (
        "the lr schedule is constant; a scrambled schedule would compare equal"
    )
    assert enabled_trace["lr_trace"] == lrs


def test_enabled_and_disabled_return_bit_identical_metrics(tmp_path: Path):
    """Every returned metrics dict: same keys, same values, bit for bit."""
    disabled_trace, _, _ = _run_disabled(tmp_path / "off")
    enabled_trace, _, bench, _ = _run_enabled(tmp_path / "on")
    _assert_bench_really_collected(bench)

    disabled_metrics = [u["metrics"] for u in disabled_trace["updates"]]
    enabled_metrics = [u["metrics"] for u in enabled_trace["updates"]]
    assert enabled_metrics == disabled_metrics

    populated = [m for m in disabled_metrics if m is not None]
    assert len(populated) == TOTAL_STEPS
    for metrics in populated:
        assert set(metrics) == {"train/count", "train/lm_loss", "train/accuracy", "train/lr"}
    # Non-constant across updates, so a comparator that compared only the first
    # element -- or an off-by-one in the trace -- cannot pass.
    assert len({tuple(sorted(m.items())) for m in populated}) == TOTAL_STEPS


def test_enabled_run_transfers_each_batch_value_exactly_once(tmp_path: Path):
    """The h2d span neither adds nor drops a ``.cuda()``."""
    disabled_trace, disabled_objs, _ = _run_disabled(tmp_path / "off")
    enabled_trace, enabled_objs, bench, _ = _run_enabled(tmp_path / "on")
    _assert_bench_really_collected(bench)

    per_update_disabled = [u["batch_cuda_calls"] for u in disabled_trace["updates"]]
    per_update_enabled = [u["batch_cuda_calls"] for u in enabled_trace["updates"]]
    assert per_update_enabled == per_update_disabled

    # Absolute, not merely equal: every value of every executed update moved
    # exactly once, and the update past ``total_steps`` moved nothing.
    assert per_update_disabled[:TOTAL_STEPS] == [[1, 1, 1]] * TOTAL_STEPS
    assert per_update_disabled[TOTAL_STEPS:] == [[0, 0, 0]] * (UPDATES - TOTAL_STEPS)

    for objs in (disabled_objs, enabled_objs):
        final = [v.cuda_calls for batch in objs["batches"] for v in batch.values()]
        assert final == [1] * (3 * TOTAL_STEPS) + [0] * (3 * (UPDATES - TOTAL_STEPS))


def test_enabled_loader_wrapper_passes_through_the_same_batch_objects(tmp_path: Path):
    """``iter_batches`` yields the loader's own dicts, in order, once each.

    Within a run this is an identity (``is``) property: the generator must not
    copy, reorder, buffer or drop.  ``bench.enabled`` selects the generator, so
    only the enabled run exercises it -- which is exactly why the disabled run
    cannot cover it.
    """
    _, disabled_objs, _ = _run_disabled(tmp_path / "off")
    _, enabled_objs, bench, _ = _run_enabled(tmp_path / "on")
    _assert_bench_really_collected(bench)

    for objs in (disabled_objs, enabled_objs):
        assert len(objs["seen_batches"]) == UPDATES
        for produced, seen in zip(objs["batches"], objs["seen_batches"]):
            assert seen is produced
        assert objs["loader"].iter_calls == 1


def test_enabled_spans_do_not_mutate_the_batch_or_the_metrics(tmp_path: Path):
    """Spans measure; they must not touch what they measure.

    Two properties, both within the enabled run: (a) the batch dicts the loader
    produced are the same objects with the same backing tensors afterwards --
    ``train_batch`` rebinds ``batch`` locally, so the caller's dict must be
    untouched apart from the ``.cuda()`` counter; (b) the metrics dicts handed
    back are unchanged by the ``metrics_device``/``metrics_wandb`` brackets, and
    stay unchanged after ``finalize()`` runs.
    """
    before = [
        {key: (value.name, value.tensor.clone()) for key, value in batch.items()}
        for batch in _make_batches(UPDATES)
    ]

    _, objs, bench, _ = _run_enabled(tmp_path / "on")
    _assert_bench_really_collected(bench)

    for expected, batch in zip(before, objs["batches"]):
        assert list(batch) == list(expected)
        for key, value in batch.items():
            name, tensor = expected[key]
            assert value.name == name
            # Bit-exact: the span brackets must not have written into the batch.
            assert torch.equal(value.tensor, tensor)
            assert value.tensor.dtype == tensor.dtype

    # The metrics dicts the run handed back, re-canonicalized from the LIVE
    # objects now, versus the snapshot the driver took just before
    # ``finalize()``.  Reading live state is the point: comparing the trace's
    # already-frozen copies to themselves would be a tautology.
    live = objs["raw_metrics"]
    assert len(live) == UPDATES
    assert [_canon_metrics(m) for m in live] == objs["pre_finalize_metrics"]

    # Idempotent second finalize, as the train loop's ``finally`` allows: still
    # no write-back into the run's metrics.
    bench.finalize()
    assert [_canon_metrics(m) for m in live] == objs["pre_finalize_metrics"]


def test_enabled_and_disabled_fail_identically_when_the_model_raises(tmp_path: Path):
    """Failure behaviour is part of "indistinguishable".

    A span ``__exit__`` that returned truthy would swallow the exception and
    let the train loop continue on corrupt state.  Both runs must surface the
    same exception, at the same point in the sequence, with the same
    bookkeeping up to it -- and the enabled collector must drop the row that
    was open when the update died rather than write it half-filled.
    """
    fail_on = 2  # inside the measured window (warmup=1)

    disabled_trace, _, _ = _run_disabled(
        tmp_path / "off", model=_RaisingModel(fail_on_call=fail_on)
    )
    enabled_trace, _, bench, _ = _run_enabled(
        tmp_path / "on", model=_RaisingModel(fail_on_call=fail_on)
    )

    assert disabled_trace["raised"] == (
        "RuntimeError",
        f"synthetic forward failure on call {fail_on}",
    ), "the exception did not propagate out of the disabled run"
    assert enabled_trace == disabled_trace

    # Only the updates before the failure completed.
    assert len(disabled_trace["updates"]) == fail_on
    assert disabled_trace["step_calls"] == fail_on

    # The enabled collector's own bookkeeping: the update that raised had a row
    # open, and ``finalize`` dropped it instead of writing a partial row.
    assert bench.session_dir is not None
    rows = _read_rows(bench.session_dir)
    assert len(rows) == fail_on - WARMUP_STEPS
    assert bench._rows_dropped_incomplete == 1


def test_enabled_and_disabled_agree_when_event_spans_fire(tmp_path: Path):
    """The eval/zprobe/checkpoint brackets, armed, change nothing either.

    ``event_span`` is the one span kind that synchronizes, and it arms only
    after the measured window has closed.  Driving the loop with
    ``event_step`` set exercises it on the real path; the disabled run takes
    the ``_NULL_SPAN`` branch for the same call sites.
    """
    event_step = WARMUP_STEPS + MEASURED_STEPS + 1

    disabled_trace, _, _ = _run_disabled(tmp_path / "off", event_step=event_step)
    enabled_trace, _, bench, calls = _run_enabled(tmp_path / "on", event_step=event_step)

    _assert_bench_really_collected(bench)
    assert enabled_trace == disabled_trace

    # Positive control: the event spans really armed.  One window-closing sync
    # plus two per armed event span (enter and exit), for three names.
    assert calls.get("sync") == 1 + 2 * 3
    manifest_rows = _read_rows(bench.session_dir)
    assert all(row["eval_event_ms"] != pb.MISSING for row in manifest_rows)
