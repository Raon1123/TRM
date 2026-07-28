"""PERF-DEV-12: regression test for the DISABLED normal path in ``pretrain.py``.

The completion gate in ``lab/reports/2026-07-26_experiment-speed-action-plan.md``
reads "disabled normal path is regression-tested".  Before this file, no test
imported ``pretrain.py`` at all: ``test_perf_benchmark.py::
test_disabled_path_is_a_strict_no_op`` exercises the *collector* in isolation,
so the actual call sites in ``pretrain.py`` were verified by reading alone.
This file closes that gap by importing ``pretrain`` and driving the real
functions.

Scope of "disabled normal path" here, matching the ratified wiring:

* ``PerfBenchmarkConfig``/``PerfProfilerConfig`` default to off, and a
  ``PretrainConfig`` built with no perf overrides inherits both;
* every span ``pretrain.py`` actually opens -- ``cuda_span`` at ``h2d``,
  ``forward_backward``, ``optimizer``, ``metrics_device``, ``ema``;
  ``wall_span`` at ``metrics_wandb``; ``event_span`` at ``eval``, ``zprobe``,
  ``checkpoint`` -- returns the SAME module-level ``perf_benchmark._NULL_SPAN``
  object (identity, not equality, and the same object as each other);
* the loader conditional at the top of the train loop yields the loader
  object ITSELF, so ``TrainingBenchmark.iter_batches``'s generator frame is
  never entered;
* the whole update lifecycle (``begin_update`` / ``end_update`` /
  ``finalize``, the last one reached through the PERF-DEV-08 ``finally``)
  is inert and writes nothing;
* importing ``pretrain`` creates no CUDA context.

GPU-free by construction.  ``train_batch`` is driven with fake batch values
whose ``.cuda()`` returns ``self``, real CPU tensors for the metric stack, and
an autouse fixture that (a) patches ten named ``torch.cuda`` attributes plus
``torch.Tensor.cuda`` to raise -- not every route to a context; see the scope
note on ``_CUDA_TRIPWIRES`` -- and (b) asserts the process CUDA-init state is
unchanged across each test, which is what catches the unpatchable routes.
No test here allocates on a device or creates a CUDA context.  The
claim is deliberately that narrow: ``test_importing_pretrain_creates_no_cuda_context``
restores full GPU *visibility* for its child process -- otherwise its
``is_initialized()`` assertion would be vacuous whenever the suite is run with
``CUDA_VISIBLE_DEVICES=""`` -- and that child calls ``torch.cuda.device_count()``,
which queries the driver without initializing a context.

Only the ``train_batch``-resident wiring is covered by executing real code.
Everything that lives in ``launch()`` is not, because ``launch()`` cannot be
driven without a GPU, a dataset and W&B.  That is ALL of: the three
``event_span`` sites (``eval``, ``zprobe``, ``checkpoint``); the loop-level
``wall_span("metrics_wandb")`` in the train loop (distinct from the
``train_batch`` one); ``cuda_span("ema")``; ``end_update(train_state.step)``;
and the ``finalize()`` in the PERF-DEV-08 ``finally``.  None of them is ever
executed from real code -- ``test_train_batch_full_disabled_loop_body_is_inert``
re-types the loop body by hand ("stands in for wandb.log" / "stands in for
ema_helper.update").  The ``launch()``-resident SPANS get two GPU-free checks:
AST enumeration of the call sites
(``test_pretrain_span_call_sites_match_the_ratified_set``) plus collector-level
identity of what those names return.  ``end_update`` and ``finalize`` get only
the collector-level inertness check (``test_disabled_update_lifecycle_is_inert``);
their call sites are NOT AST enumerated, because ``PRETRAIN_SPANS`` covers spans
alone.  Those are the strongest GPU-free properties available for the
``launch()`` sites, and all of them are weaker than the ``train_batch``
coverage, which does execute the real function body.

DEVIATIONS from the literal task text, stated so they are not silently dropped:

* ``PretrainConfig()`` cannot be constructed bare -- ``arch``, ``data_paths``,
  ``global_batch_size``, ``epochs``, ``lr``, ``lr_min_ratio``,
  ``lr_warmup_steps``, ``weight_decay``, ``beta1``, ``beta2``,
  ``puzzle_emb_lr`` and ``puzzle_emb_weight_decay`` are required fields.  The
  clause "``PretrainConfig().perf_benchmark.enabled`` is False" is therefore
  tested as (a) the two config classes' own defaults and (b) a minimal valid
  ``PretrainConfig`` with no perf keys supplied.

STILL UNCOVERED -- properties that are GPU-only and are NOT claimed here:

* real ``torch.cuda.Event.record``/``elapsed_time`` behaviour (every CUDA
  entry point is faked or tripwired);
* the actual H2D transfer at the ``cuda_span("h2d")`` site.  The fake batch
  value's ``.cuda()`` returns ``self`` and cannot observe PERF-DEV-07 -- the
  second, pre-existing per-update implicit stream sync caused by
  ``non_blocking=False``.  ``sync_count == 1`` therefore continues to mean
  "the collector issued one synchronize", never "the measured window contains
  no device serialization";
* ``torch.compile`` graph behaviour of the real model around the span
  brackets.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import torch

import pretrain
from utils import perf_benchmark as pb
from utils.perf_benchmark import PerfBenchmarkConfig, TrainingBenchmark
from utils.perf_profiler import PerfProfilerConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_PATH = REPO_ROOT / "pretrain.py"

#: The ratified span wiring, transcribed from the plan's timing contract rather
#: than imported, so a silent rewire of ``pretrain.py`` fails this file.
PRETRAIN_SPANS: frozenset[tuple[str, str]] = frozenset(
    {
        ("cuda_span", "h2d"),
        ("cuda_span", "forward_backward"),
        ("cuda_span", "optimizer"),
        ("cuda_span", "metrics_device"),
        ("cuda_span", "ema"),
        ("wall_span", "metrics_wandb"),
        ("event_span", "eval"),
        ("event_span", "zprobe"),
        ("event_span", "checkpoint"),
    }
)

_SPAN_METHODS = ("cuda_span", "wall_span", "event_span")


# --------------------------------------------------------------------------- #
# Tripwires and fakes
# --------------------------------------------------------------------------- #


#: CUDA entry points the disabled path must never reach.  Patched to raise.
#:
#: SCOPE, precisely: these ten ``torch.cuda`` attributes plus ``torch.Tensor.cuda``
#: (patched in the fixture below).  That is NOT every route to a CUDA context, and
#: this list must not be described as if it were.  Verified escapes: factory
#: ``device=`` kwargs (``torch.zeros(1, device="cuda")``), ``Tensor.to("cuda")``,
#: the ``torch.device("cuda")`` context manager (a C ``method_descriptor``, so it
#: is unpatchable from the ``torch.cuda`` Python namespace -- and ``pretrain.py``
#: uses one on the disabled path), and ``torch.cuda._lazy_init``, which is the
#: real initialization route (``torch.cuda.init`` merely delegates to it, so
#: patching ``init`` is decorative).
#: The ``_cuda_baseline`` delta assertion in the fixture is what covers those.
#: ``is_available``/``is_initialized`` are deliberately left alone: they are pure
#: queries and never allocate.
_CUDA_TRIPWIRES = (
    "synchronize",
    "Event",
    "utilization",
    "reset_peak_memory_stats",
    "max_memory_allocated",
    "max_memory_reserved",
    "current_device",
    "init",
    "set_device",
    "memory_allocated",
)


@pytest.fixture(autouse=True)
def _cuda_tripwire(monkeypatch: pytest.MonkeyPatch):
    def _trip(name: str):
        def _boom(*_args, **_kwargs):
            raise AssertionError(f"disabled path reached torch.cuda.{name}")

        return _boom

    for name in _CUDA_TRIPWIRES:
        if hasattr(torch.cuda, name):
            monkeypatch.setattr(torch.cuda, name, _trip(name), raising=False)

    def _tensor_cuda(*_args, **_kwargs):
        raise AssertionError("disabled path moved a real torch.Tensor to CUDA")

    monkeypatch.setattr(torch.Tensor, "cuda", _tensor_cuda, raising=False)

    # Order-independent CUDA-context guard.  A bare
    # ``assert torch.cuda.is_initialized() is False`` is process-global, so any
    # earlier test in the session that initializes CUDA (tests/test_perf_profiler.py
    # does, once kineto/CUPTI engages at profiler ``step()``/``stop()``) makes it
    # fail for reasons unrelated to this code path.  Comparing against the
    # baseline captured here is immune to that ordering while still detecting the
    # routes the tripwire list cannot patch (factory ``device=`` kwargs,
    # ``Tensor.to``, ``torch.device`` ctx, ``_lazy_init``).  Applies to every test
    # in this file, not just the two that once asserted it inline.
    baseline = torch.cuda.is_initialized()
    yield baseline
    assert torch.cuda.is_initialized() is baseline, (
        "disabled path created a CUDA context"
    )


class _FakeBatchValue:
    """Stands in for a pinned CPU tensor: records ``.cuda()`` and returns self."""

    __slots__ = ("name", "cuda_calls")

    def __init__(self, name: str) -> None:
        self.name = name
        self.cuda_calls = 0

    def cuda(self):
        self.cuda_calls += 1
        return self


class _FakeModel:
    """Minimal stand-in for the trained module: no parameters, no device work."""

    def __init__(self) -> None:
        self.calls = 0
        self.initial_carry_calls = 0
        self.last_carry: Any = None
        self.last_batch: Any = None

    def initial_carry(self, batch):
        self.initial_carry_calls += 1
        self.last_batch = batch
        return "carry-from-initial_carry"

    def __call__(self, *, carry, batch, return_keys):
        self.calls += 1
        self.last_carry = carry
        self.last_batch = batch
        # A real CPU autograd graph so ``((1 / gbs) * loss).backward()`` runs
        # for real without a device.
        weight = torch.ones(2, requires_grad=True)
        loss = (weight * 3.0).sum()
        metrics = {
            "count": torch.tensor(4.0),
            "lm_loss": torch.tensor(8.0),
        }
        return "carry-after-forward", loss, metrics, None, None


class _FakeOptimizer:
    def __init__(self) -> None:
        self.param_groups = [{"lr": 0.0}]
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self):
        self.step_calls += 1

    def zero_grad(self):
        self.zero_grad_calls += 1


def minimal_config(**overrides) -> "pretrain.PretrainConfig":
    """A valid ``PretrainConfig`` with NO perf keys supplied."""
    base = dict(
        arch={"name": "trm@TinyRecursiveReasoningModel", "loss": {"name": "losses@ACTLossHead"}},
        data_paths=["data/sigma_k_10/6"],
        global_batch_size=8,
        epochs=2,
        lr=1e-4,
        lr_min_ratio=1.0,
        lr_warmup_steps=0,
        weight_decay=1.0,
        beta1=0.9,
        beta2=0.95,
        puzzle_emb_lr=1e-4,
        puzzle_emb_weight_decay=1.0,
    )
    base.update(overrides)
    return pretrain.PretrainConfig(**base)


def disabled_bench(output_dir: Path) -> TrainingBenchmark:
    """The collector ``pretrain.py`` constructs on the normal path.

    ``output_dir`` points somewhere that must stay untouched: a disabled
    collector creates nothing, so the directory must not come into existence.
    """
    return TrainingBenchmark(
        PerfBenchmarkConfig(output_dir=str(output_dir)),
        checkpoint_path=None,
        rank=0,
        run_name="disabled_path_probe_r1",
        seed=1,
        eval_interval=2000,
        resolved_config={},
        data_paths=[],
        profiler_enabled=False,
    )


def _run_in_subprocess(
    script: str,
    env_overrides: dict[str, str],
    *,
    unrestrict_gpus: bool = False,
) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if unrestrict_gpus:
        # The suite may be run with ``CUDA_VISIBLE_DEVICES=""`` (the profiler
        # test attaches CUPTI otherwise).  Under that restriction an
        # ``is_initialized()`` assertion is vacuous, so the child is given the
        # unrestricted view back.  Safe: the child only imports, and importing
        # ``pretrain`` is exactly the thing being asserted not to allocate.
        #
        # OPERATING CONSTRAINT for anyone editing a script passed with
        # ``unrestrict_gpus=True``: the child sees EVERY GPU on the host,
        # including devices reserved by live training jobs.  It must never
        # allocate, never create a context, and never call anything past
        # ``torch.cuda.device_count()`` (a driver query that does not
        # initialize).  A single ``.cuda()``/``torch.zeros(..., device=...)``
        # added here would land on a reserved device.
        env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(env_overrides)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


def test_perf_features_are_off_by_default():
    """Both perf blocks default to disabled, and PretrainConfig inherits that.

    See the DEVIATIONS note in the module docstring: bare ``PretrainConfig()``
    is not constructible, so the clause is tested on the config classes'
    defaults plus a minimal config with no perf keys supplied.
    """
    assert PerfBenchmarkConfig().enabled is False
    assert PerfProfilerConfig().enabled is False

    config = minimal_config()
    assert config.perf_benchmark.enabled is False
    assert config.perf_profiler.enabled is False
    # The defaults come from the field factories, not from anything the test
    # passed in.
    assert config.perf_benchmark == PerfBenchmarkConfig()
    assert config.perf_profiler == PerfProfilerConfig()


def test_a_default_config_builds_an_inert_collector(tmp_path: Path):
    """``TrainingBenchmark(config.perf_benchmark, ...)`` as pretrain.py calls it."""
    config = minimal_config()
    session_root = tmp_path / "never_created"
    bench = TrainingBenchmark(
        config.perf_benchmark,
        checkpoint_path=config.checkpoint_path,
        rank=0,
        run_name=config.run_name,
        seed=config.seed,
        eval_interval=config.eval_interval,
        resolved_config=config.model_dump(),
        data_paths=config.data_paths,
        profiler_enabled=config.perf_profiler.enabled,
    )
    assert bench.enabled is False
    assert bench.session_dir is None
    assert bench.output_dir is None
    assert bench.sync_count == 0
    assert not session_root.exists()


# --------------------------------------------------------------------------- #
# Import hygiene
# --------------------------------------------------------------------------- #


def test_importing_pretrain_creates_no_cuda_context():
    """Hermetic: a fresh interpreter, so no earlier test can have initialized CUDA.

    Deliberately NOT run with ``CUDA_VISIBLE_DEVICES=""`` -- under that
    override ``is_initialized()`` is False vacuously.  This is the real
    property on the real machine.
    """
    result = _run_in_subprocess(
        """
        import torch
        assert not torch.cuda.is_initialized(), "torch itself initialized CUDA on import"
        import pretrain
        assert not torch.cuda.is_initialized(), "importing pretrain created a CUDA context"
        assert pretrain.PerfBenchmarkConfig().enabled is False
        print("VISIBLE_DEVICE_COUNT", torch.cuda.device_count())
        print("NO_CUDA_CONTEXT_OK")
        """,
        env_overrides={},
        unrestrict_gpus=True,
    )
    assert result.returncode == 0, result.stderr
    assert "NO_CUDA_CONTEXT_OK" in result.stdout
    # Guard against the assertion passing because the child saw no GPU at all.
    # If this host genuinely has none, the check is inherently weaker and says
    # so rather than pretending otherwise.
    count = int(result.stdout.split("VISIBLE_DEVICE_COUNT")[1].split()[0])
    if count == 0:
        pytest.skip("no visible CUDA device: 'creates no CUDA context' is vacuous here")


def test_pretrain_imports_with_no_visible_gpus():
    """Module scope has no hard CUDA dependency: import succeeds with zero GPUs."""
    result = _run_in_subprocess(
        """
        import torch
        assert torch.cuda.is_available() is False
        import pretrain
        assert pretrain.PretrainConfig.model_fields["perf_benchmark"] is not None
        print("NO_GPU_IMPORT_OK")
        """,
        env_overrides={"CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode == 0, result.stderr
    assert "NO_GPU_IMPORT_OK" in result.stdout


# --------------------------------------------------------------------------- #
# Span identity on the disabled path
# --------------------------------------------------------------------------- #


def test_pretrain_span_call_sites_match_the_ratified_set():
    """Completeness guard for the identity test below.

    Parsed from the AST rather than grepped, so a span added to ``pretrain.py``
    but not to ``PRETRAIN_SPANS`` fails here instead of quietly escaping the
    identity assertions.
    """
    tree = ast.parse(PRETRAIN_PATH.read_text(encoding="utf-8"), filename=str(PRETRAIN_PATH))
    found: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in _SPAN_METHODS:
            continue
        if not isinstance(func.value, ast.Name) or func.value.id != "bench":
            continue
        assert node.args, f"bench.{func.attr}() called with no span name"
        first = node.args[0]
        assert isinstance(first, ast.Constant) and isinstance(first.value, str), (
            f"bench.{func.attr}() called with a non-literal span name; the "
            "disabled-path identity test can no longer enumerate the call sites"
        )
        found.add((func.attr, first.value))

    assert found == set(PRETRAIN_SPANS), (
        "pretrain.py's span wiring drifted from the ratified set; "
        f"only in code={sorted(found - set(PRETRAIN_SPANS))}, "
        f"only in test={sorted(set(PRETRAIN_SPANS) - found)}"
    )


def test_every_disabled_span_is_the_same_null_span_object(tmp_path: Path):
    """Identity (``is``), not equality: one shared module-level nullcontext."""
    bench = disabled_bench(tmp_path / "never_created")
    assert bench.enabled is False

    spans = []
    for kind, name in sorted(PRETRAIN_SPANS):
        method = getattr(bench, kind)
        span = method(name, 4900) if kind == "event_span" else method(name)
        assert span is pb._NULL_SPAN, f"{kind}({name!r}) did not return the shared _NULL_SPAN"
        spans.append(span)

    # Redundant by construction, kept only as a literal restatement of the
    # docstring's "the same object as each other" clause: every element of
    # ``spans`` was already asserted ``is pb._NULL_SPAN`` above, so this cannot
    # fail unless that assertion did.  It is NOT an independent guard against a
    # future per-call ``nullcontext()`` -- such a change would fail the ``is``
    # assertion on the first iteration and never reach this line.
    assert len({id(s) for s in spans}) == 1

    # ``data_wait`` is only opened from ``iter_batches``, which pretrain.py
    # never enters when disabled -- but the direct call must be inert as well.
    assert bench.wall_span("data_wait") is pb._NULL_SPAN

    # Entering and leaving each one is a no-op that records nothing.
    for kind, name in sorted(PRETRAIN_SPANS):
        method = getattr(bench, kind)
        ctx = method(name, 4900) if kind == "event_span" else method(name)
        with ctx:
            pass
    assert bench.sync_count == 0
    assert bench.session_dir is None
    assert not (tmp_path / "never_created").exists()


def test_disabled_spans_still_reject_unknown_names(tmp_path: Path):
    """Name validation is unconditional; a typo fails loudly even when off."""
    bench = disabled_bench(tmp_path / "never_created")
    with pytest.raises(KeyError):
        bench.cuda_span("not_a_span")
    with pytest.raises(KeyError):
        bench.wall_span("not_a_span")
    with pytest.raises(KeyError):
        bench.event_span("not_a_span", 1)


# --------------------------------------------------------------------------- #
# Loader conditional
# --------------------------------------------------------------------------- #


class _ExplodingLoader:
    """Iterating this is the failure: the disabled path must not iterate it here."""

    def __iter__(self):
        raise AssertionError("the loader was iterated by the benchmark wrapper")


def test_disabled_loader_conditional_yields_the_loader_itself(tmp_path: Path):
    """``bench.iter_batches(l) if bench.enabled else l`` must be ``l`` itself."""
    bench = disabled_bench(tmp_path / "never_created")
    train_loader = _ExplodingLoader()

    batches = bench.iter_batches(train_loader) if bench.enabled else train_loader

    assert batches is train_loader
    # A generator object would be falsy-safe to hold; the raw loader is not an
    # iterator at all, which is the point -- the generator frame was never made.
    assert not hasattr(batches, "send")
    with pytest.raises(AssertionError, match="iterated by the benchmark wrapper"):
        iter(batches)


def test_pretrain_still_guards_iter_batches_behind_bench_enabled():
    """The conditional itself, whitespace-normalized (no line-number anchor)."""
    source = " ".join(PRETRAIN_PATH.read_text(encoding="utf-8").split())
    assert (
        "batches = bench.iter_batches(train_loader) if bench.enabled else train_loader" in source
    ), "the loader conditional in pretrain.py changed; re-verify the disabled-path claim"


# --------------------------------------------------------------------------- #
# Update lifecycle
# --------------------------------------------------------------------------- #


def test_disabled_update_lifecycle_is_inert(tmp_path: Path):
    """``begin_update``/``end_update``/``finalize`` on a disabled collector.

    ``__init__`` returns early before most collector state exists, so every one
    of these must check ``_active`` first or a disabled run raises
    ``AttributeError`` on the very first update.
    """
    session_root = tmp_path / "never_created"
    bench = disabled_bench(session_root)
    batch = {"inputs": _FakeBatchValue("inputs"), "labels": _FakeBatchValue("labels")}

    for step in range(1, 6):
        bench.begin_update(batch, 2048)
        with bench.cuda_span("forward_backward"):
            pass
        bench.end_update(step)

    # PERF-DEV-08: reached through the train loop's ``finally``, including on
    # runs that never enabled the benchmark.  Idempotent.
    bench.finalize()
    bench.finalize()

    assert bench.enabled is False
    assert bench.sync_count == 0
    assert bench.session_dir is None
    assert bench.output_dir is None
    assert not session_root.exists()
    assert list(tmp_path.iterdir()) == []
    # The token-count scan in ``begin_update`` must not have run.
    assert all(v.cuda_calls == 0 for v in batch.values())


# --------------------------------------------------------------------------- #
# train_batch driven end to end
# --------------------------------------------------------------------------- #


def _drive_train_batch(bench: TrainingBenchmark, *, total_steps: int = 10, step: int = 0):
    config = minimal_config()
    model = _FakeModel()
    optimizer = _FakeOptimizer()
    train_state = pretrain.TrainState(
        model=model,
        optimizers=[optimizer],
        optimizer_lrs=[config.lr],
        carry=None,
        step=step,
        total_steps=total_steps,
    )
    batch = {
        "inputs": _FakeBatchValue("inputs"),
        "labels": _FakeBatchValue("labels"),
        "puzzle_identifiers": _FakeBatchValue("puzzle_identifiers"),
    }
    metrics = pretrain.train_batch(
        config,
        train_state,
        batch,
        config.global_batch_size,
        rank=0,
        world_size=1,
        bench=bench,
    )
    return metrics, train_state, model, optimizer, batch


def test_train_batch_disabled_completes_without_touching_cuda(tmp_path: Path):
    """Drive the real ``train_batch`` with a disabled bench, then assert BOTH
    that no CUDA was touched and -- positively -- that the whole body ran.

    Without the positive controls a wrong ``total_steps`` would make every
    "did not touch CUDA" assertion pass on a function that returned at line 2.
    """
    session_root = tmp_path / "never_created"
    bench = disabled_bench(session_root)

    metrics, train_state, model, optimizer, batch = _drive_train_batch(bench)

    # -- positive controls: the body really executed -----------------------
    assert train_state.step == 1
    assert model.initial_carry_calls == 1, "the carry-init branch did not run"
    assert model.calls == 1, "the forward did not run"
    assert train_state.carry == "carry-after-forward"
    assert optimizer.step_calls == 1
    assert optimizer.zero_grad_calls == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(minimal_config().lr)
    assert metrics is not None
    assert set(metrics) == {"train/count", "train/lm_loss", "train/lr"}
    assert metrics["train/lr"] == pytest.approx(minimal_config().lr)
    # count=4 divided by count -> 1.0; lm_loss=8 divided by global_batch_size=8.
    assert float(metrics["train/count"]) == pytest.approx(1.0)
    assert float(metrics["train/lm_loss"]) == pytest.approx(1.0)
    # Every batch value went through the h2d span exactly once.
    assert [v.cuda_calls for v in batch.values()] == [1, 1, 1]

    # -- the disabled-path property ---------------------------------------
    # No bare ``is_initialized() is False`` here: it is process-global and so
    # order-dependent.  The autouse fixture's ``_cuda_baseline`` delta assertion
    # carries the property instead, for every test in this file.
    assert bench.enabled is False
    assert bench.sync_count == 0
    assert bench.session_dir is None
    assert not session_root.exists()


def test_train_batch_full_disabled_loop_body_is_inert(tmp_path: Path):
    """Mirror the disabled branch of the train loop, spans and all."""
    session_root = tmp_path / "never_created"
    bench = disabled_bench(session_root)
    config = minimal_config()
    model = _FakeModel()
    optimizer = _FakeOptimizer()
    train_state = pretrain.TrainState(
        model=model,
        optimizers=[optimizer],
        optimizer_lrs=[config.lr],
        carry=None,
        step=0,
        total_steps=10,
    )

    for _ in range(4):
        batch = {"inputs": _FakeBatchValue("inputs"), "labels": _FakeBatchValue("labels")}
        metrics = pretrain.train_batch(
            config, train_state, batch, config.global_batch_size,
            rank=0, world_size=1, bench=bench,
        )
        assert metrics is not None
        with bench.wall_span("metrics_wandb"):      # stands in for wandb.log
            pass
        with bench.cuda_span("ema"):                # stands in for ema_helper.update
            pass
        bench.end_update(train_state.step)

    # The event spans from the eval block.
    for name in ("eval", "zprobe", "checkpoint"):
        assert bench.event_span(name, train_state.step) is pb._NULL_SPAN
        with bench.event_span(name, train_state.step):
            pass

    bench.finalize()

    assert train_state.step == 4
    assert model.calls == 4
    assert optimizer.step_calls == 4
    assert bench.sync_count == 0
    assert bench.session_dir is None
    assert not session_root.exists()
    assert list(tmp_path.iterdir()) == []
    # CUDA-context guard lives in the autouse fixture's ``_cuda_baseline`` delta
    # assertion -- order-independent, and applied to every test in this file.


def test_train_batch_past_total_steps_returns_before_any_span(tmp_path: Path):
    """The ``total_steps`` early return opens no row and moves nothing to CUDA."""
    session_root = tmp_path / "never_created"
    bench = disabled_bench(session_root)

    metrics, train_state, model, optimizer, batch = _drive_train_batch(
        bench, total_steps=3, step=3
    )

    assert metrics is None
    assert train_state.step == 4
    assert model.calls == 0
    assert optimizer.step_calls == 0
    assert all(v.cuda_calls == 0 for v in batch.values())
    assert bench.sync_count == 0
    assert not session_root.exists()
