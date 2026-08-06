"""Tests for the additive sequence/trajectory metrics in ``utils/z_logging.py``.

These discharge the four families added under the ``z_seq_metrics`` gate:

* (A) full-sequence participation ratio per ACT step -- ``zseq/*``
* (B) mutual information per ACT step -- ``zmi/*``
* (C) tau(h) recurrence-power alignment -- ``ztau/*``
* (D) permutation-structure metrics on the decode -- ``zperm/*``

plus the hard constraints they must not violate: the frozen keys ``z/eff_rank``,
``z/pca_top2_var``, ``z/mean_norm`` and ``z/delta_step_<t>`` keep byte-identical
meaning (74 historical runs are compared against them), nothing is computed when
the gate is off, and ``halt_max_steps == 1`` degrades to a single honest
observation rather than a fabricated trajectory.

GPU-FREE BY CONSTRUCTION
------------------------
``torch.cuda.is_available()`` is True on the development box, so "runs without a
GPU" is a property this file ENFORCES rather than one the environment supplies.
Every test drives CPU tensors; ``_probe_forward`` is exercised through its
``device="cpu"`` parameter, which exists for exactly this reason and whose
default reproduces the historical CUDA behaviour.  The autouse fixture below
patches the ``torch.cuda`` entry points this code could reach and, because that
list is provably incomplete (factory ``device=`` kwargs, ``Tensor.to("cuda")``,
the ``torch.device("cuda")`` context manager -- a C ``method_descriptor``, hence
unpatchable from the Python namespace -- and ``torch.cuda._lazy_init``), also
asserts the process CUDA-init state is unchanged across each test.  That
baseline-delta form is what catches the unpatchable routes, and it is
order-independent: an absolute ``is_initialized() is False`` assertion would
fail whenever an earlier module in the session (``tests/test_perf_profiler.py``)
has engaged CUPTI.

STILL UNCOVERED -- read this before treating a green run as full verification
---------------------------------------------------------------------------
* THE REAL ACT LOOP.  ``test_probe_forward_pairs_each_act_step_latent_with_its
  _own_decode`` drives ``_probe_forward``'s genuine two-loop body, but against a
  SCRIPTED stub model, not ``TinyRecursiveReasoningModel_ACTV1``.  The claim it
  supports is "the capture indexes and pairs the two loops correctly", not "the
  model produces those latents".  Nothing here executes a real forward pass, a
  real ``initial_carry``, ``torch.compile`` or the EMA copy.
* EVAL-MODE HALTING.  The alignment of ``z_history[h]`` with ``preds_traj[h]``
  rests on eval-mode halting being deterministic and batch-synchronous
  (``trm.py:275`` gates adaptive halting on ``self.training``).  The stub is
  deterministic by construction, so these tests cannot detect a regression in
  that precondition; ``test_train_mode_suppresses_tau_and_perm`` only checks
  that the code refuses to report when it knows the precondition is violated.
* MI ACCURACY.  At B=512, r=32, C=10 there is no ground truth to test against.
  The MI tests pin the estimator's SIGN behaviour, its null floor and its
  ceiling -- i.e. the bias disclosure required by the honesty constraint -- and
  deliberately assert nothing about the estimate's closeness to a true MI.
* THE MULTI-GPU / distributed path, snapshot writing, and wandb transport.

Run this module on its own::

    rtk uv run pytest tests/test_z_logging_seq_metrics.py -q

Whole-suite green is NOT an achievable gate in this checkout: ``pytest tests/``
exits 2 during collection because of four PRE-EXISTING failures unrelated to
this work (``test_figpipe_contracts``, ``test_figure_pipeline_v3``,
``test_extract_evals_web``, ``test_pull_wandb_tau``), all fallout from the
``lab/`` symlink reorganisation.  Use per-module invocation or
``--continue-on-collection-errors``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch

from utils import z_logging as zl


# --------------------------------------------------------------------------- #
# Frozen contract, transcribed rather than imported
# --------------------------------------------------------------------------- #

#: The keys 74 historical runs are compared against.  Transcribed as a literal
#: on purpose: importing them from the module under test would make a silent
#: rename invisible here, which is the exact regression this guards against.
#: ``z/delta_step_<t>`` and the two PCA image keys are matched by prefix below.
_FROZEN_SCALAR_KEYS = frozenset({
    "probe/train_exact",
    "probe/test_exact",
    "phase/index",
    "z/eff_rank",
    "z/pca_top2_var",
    "z/mean_norm",
})
_FROZEN_KEY_PREFIXES = ("z/delta_step_", "z/pca_by_label_", "z/pca_by_correct_")

#: Return keys ``_probe_forward`` promised before this work.  New keys may be
#: added; none of these may disappear.
_FROZEN_PROBE_RESULT_KEYS = frozenset({
    "z_H", "z_L", "labels", "correct_mask", "step_deltas", "exact_acc",
})


# --------------------------------------------------------------------------- #
# GPU-free enforcement
# --------------------------------------------------------------------------- #

_CUDA_TRIPWIRES = (
    "synchronize", "current_device", "init", "set_device",
    "memory_allocated", "max_memory_allocated",
)


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)

    def _trip(name: str):
        def _boom(*_a, **_kw):
            raise AssertionError(f"seq-metrics path reached torch.cuda.{name}")
        return _boom

    for name in _CUDA_TRIPWIRES:
        if hasattr(torch.cuda, name):
            monkeypatch.setattr(torch.cuda, name, _trip(name), raising=False)

    def _tensor_cuda(*_a, **_kw):
        raise AssertionError("seq-metrics path moved a tensor to CUDA")

    monkeypatch.setattr(torch.Tensor, "cuda", _tensor_cuda, raising=False)

    baseline = torch.cuda.is_initialized()
    yield
    assert torch.cuda.is_initialized() is baseline, (
        "seq-metrics path created a CUDA context")


# --------------------------------------------------------------------------- #
# Synthetic-probe helpers
# --------------------------------------------------------------------------- #

def _random_perms(n: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.stack([rng.permutation(n) for _ in range(count)])


def _encode(perm: np.ndarray, seq_len: int) -> np.ndarray:
    """Dataset encoding: PAD=0, row[:n] = perm + 1 (build_sigma_k_dataset.py:133)."""
    out = np.zeros((perm.shape[0], seq_len), dtype=np.int64)
    out[:, :perm.shape[1]] = perm + 1
    return out


def _zero_mean_orthonormal(rows: int, cols: int, seed: int) -> np.ndarray:
    """``rows x cols`` with orthonormal, mutually orthogonal, ZERO-MEAN columns.

    Built by QR-ing ``[ones | random]`` and discarding the first column, so every
    retained column is orthogonal to the all-ones vector.  Feeding this through
    an orthonormal row map gives a batch-centered Gram whose nonzero eigenvalues
    are all equal, i.e. a participation ratio of exactly ``cols`` -- which is
    what lets the tests assert a planted rank instead of a fuzzy monotonicity.
    """
    rng = np.random.RandomState(seed)
    basis = np.concatenate(
        [np.ones((rows, 1)), rng.randn(rows, cols)], axis=1)
    q, _ = np.linalg.qr(basis)
    return q[:, 1:cols + 1]


def _latent_with_planted_pr(B: int, P: int, seq_len: int, D: int,
                            rank: int, seed: int) -> torch.Tensor:
    """(B, P+seq_len, D) latent whose TOKEN slice has participation ratio ``rank``.

    Puzzle-embedding slots are left at zero: they are constant across the batch,
    so they contribute nothing to the batch-centered Gram and ``pr_joint_all``
    equals ``pr_joint_tok`` for this construction.
    """
    F = seq_len * D
    W = _zero_mean_orthonormal(B, rank, seed)             # (B, rank)
    rng = np.random.RandomState(seed + 7)
    V, _ = np.linalg.qr(rng.randn(F, rank))               # (F, rank), orthonormal
    X = W @ V.T                                           # (B, F)
    Z = torch.zeros(B, P + seq_len, D, dtype=torch.float32)
    Z[:, P:, :] = torch.from_numpy(X.reshape(B, seq_len, D)).float()
    return Z


# --------------------------------------------------------------------------- #
# (A) Participation ratio
# --------------------------------------------------------------------------- #

def test_trace_form_participation_ratio_matches_the_frozen_covariance_chain():
    """PR = tr(G)^2 / ||G||_F^2 reproduces ``_effective_rank(_pca(X)[0])``.

    A RELATIVE tolerance, not equality: the two travel different numerical
    paths.  ``_effective_rank`` clips negative eigenvalues to zero and the trace
    form cannot (it never forms eigenvalues), so exact agreement is not a
    property either implementation has, and asserting it would make this test a
    round-off detector rather than a correctness one.
    """
    rng = np.random.RandomState(0)
    X = (rng.randn(64, 20) @ rng.randn(20, 40)).astype(np.float64)

    legacy = zl._effective_rank(zl._pca(X)[0])

    Xc = torch.from_numpy(X)
    Xc = Xc - Xc.mean(dim=0, keepdim=True)
    trace_form = float(zl._pr_from_gram(Xc @ Xc.T))

    assert trace_form == pytest.approx(legacy, rel=1e-9)


def test_participation_ratio_is_invariant_to_the_ambient_dimension():
    """The same latent structure embedded in F=40 and F=4000 gives one PR.

    This is the property that makes the Gram form exact rather than an
    approximation: PR depends only on the nonzero spectrum, and the nonzero
    spectra of the F x F covariance and the B x B Gram coincide.  The 13824^2
    covariance the naive "just don't mean-pool" fix would build is never needed.
    """
    rng = np.random.RandomState(1)
    W = rng.randn(48, 6)

    prs = []
    for F, seed in ((40, 2), (4000, 3)):
        V, _ = np.linalg.qr(np.random.RandomState(seed).randn(F, 6))
        X = torch.from_numpy(W @ V.T)
        Xc = X - X.mean(dim=0, keepdim=True)
        prs.append(float(zl._pr_from_gram(Xc @ Xc.T)))

    assert prs[0] == pytest.approx(prs[1], rel=1e-9)


@pytest.mark.parametrize("rank", [1, 3, 9])
def test_participation_ratio_recovers_a_planted_rank(rank: int):
    Z = _latent_with_planted_pr(B=40, P=2, seq_len=4, D=5, rank=rank, seed=rank)
    G = zl._position_grams(Z)
    assert float(zl._pr_from_gram(G[2:].sum(dim=0))) == pytest.approx(
        float(rank), rel=1e-6)


def test_per_position_grams_sum_to_the_flattened_gram():
    """The identity the implementation leans on to compute every PR in one pass.

    Flattening (B, S, D) then centering over the batch is the same operation as
    centering each position over the batch and concatenating, so the Gram of any
    subset of positions is the sum of that subset's per-position Grams.
    """
    Z = torch.randn(30, 5, 7, dtype=torch.float64)
    G = zl._position_grams(Z)

    flat = Z.reshape(30, -1)
    flat = flat - flat.mean(dim=0, keepdim=True)
    assert torch.allclose(G.sum(dim=0), flat @ flat.T)

    sub = Z[:, 2:, :].reshape(30, -1)
    sub = sub - sub.mean(dim=0, keepdim=True)
    assert torch.allclose(G[2:].sum(dim=0), sub @ sub.T)


def test_across_position_pr_is_uninterpretable_without_its_dispersion_companion():
    """PR ~ 1.0 means two OPPOSITE things; ``pos_var_frac`` is what separates them.

    This is why the dispersion key is mandatory rather than decorative.  When the
    positions are exactly identical the position-centered block is zero and the
    ``s2 < 1e-30`` guard rescues the ratio to 1.0; when the positions vary along
    a single direction the ratio is genuinely 1.0.  "Mean-pooling is lossless"
    is true only in the first case, and the PR alone cannot tell them apart.
    """
    B, S, D = 16, 5, 4
    base = torch.randn(B, 1, D)

    identical = base.expand(B, S, D).contiguous()
    pr_id, frac_id = zl._across_position_stats(identical)

    direction = torch.randn(1, S, 1)
    rank1 = base.expand(B, S, D).contiguous() + direction * torch.ones(B, S, D)
    pr_r1, frac_r1 = zl._across_position_stats(rank1)

    assert pr_id == pytest.approx(1.0)
    assert pr_r1 == pytest.approx(1.0, rel=1e-6)
    # Identical positions carry no position-axis energy; the rank-1 spread does.
    assert frac_id < 1e-12
    assert frac_r1 > 1e-3


def test_pooled_and_unpooled_participation_ratios_measure_different_things():
    """The motivating claim, made empirical rather than asserted.

    A latent whose token positions carry INDEPENDENT batch structure has a joint
    participation ratio far above the mean-pooled one, because mean-pooling
    projects the position axis onto the single all-ones direction before the
    spectrum is ever formed.  That is the sense in which a low ``z/eff_rank``
    does not imply a low-capacity state.

    What is deliberately NOT asserted: that unpooled PR >= pooled PR in general.
    Mean-pooling is a linear projection and the participation ratio is not
    monotone under projection, so the general inequality is false.  The claim
    here is about this construction only.
    """
    B, P, seq_len, D, per_pos = 60, 2, 5, 8, 3
    rng = np.random.RandomState(30)
    Z = torch.zeros(B, P + seq_len, D)
    # Every token position writes a rank-3 code into the SAME D-subspace but
    # along batch directions that are mutually orthogonal ACROSS positions.  The
    # joint state therefore spans seq_len * 3 = 15 directions, while their
    # average spans only 3: the information lives in how the positions differ,
    # which is exactly what the pooling step discards.
    W_all = _zero_mean_orthonormal(B, seq_len * per_pos, seed=40)
    V, _ = np.linalg.qr(rng.randn(D, per_pos))
    for p in range(seq_len):
        W = W_all[:, p * per_pos:(p + 1) * per_pos]
        Z[:, P + p, :] = torch.from_numpy(W @ V.T).float()

    G = zl._position_grams(Z)
    pr_joint = float(zl._pr_from_gram(G[P:].sum(dim=0)))
    pr_perpos = float(zl._pr_from_gram(G[P:]).mean())
    pr_pooled = zl._pooled_pr_legacy(Z)

    assert pr_joint > 3.0 * pr_pooled, (
        f"joint {pr_joint} vs pooled {pr_pooled}: pooling should collapse this")
    # Decomposition bound: pr_joint / (seq_len * pr_perpos_mean) lies in (0, 1]
    # by Cauchy-Schwarz, and separates "each position independently rich"
    # (ratio near 1) from "all positions share one global code" (near 1/seq_len).
    ratio = pr_joint / (seq_len * pr_perpos)
    assert 0.0 < ratio <= 1.0 + 1e-9, ratio


def test_decomposition_ratio_bound_holds_on_randomised_latents():
    """pr_joint <= seq_len * mean_p pr_p, for arbitrary latents."""
    rng = np.random.RandomState(31)
    for trial in range(25):
        B = int(rng.randint(12, 40))
        S = int(rng.randint(2, 7))
        D = int(rng.randint(2, 9))
        Z = torch.from_numpy(rng.randn(B, S, D)).float()
        if trial % 3 == 0:                      # also probe low-rank latents
            Z = Z[:, :, :1].expand(B, S, D).contiguous()
        G = zl._position_grams(Z)
        joint = float(zl._pr_from_gram(G.sum(dim=0)))
        perpos = float(zl._pr_from_gram(G).mean())
        assert joint <= S * perpos + 1e-6, (trial, joint, S * perpos)


# --------------------------------------------------------------------------- #
# Permutation algebra
# --------------------------------------------------------------------------- #

def test_permutation_power_algebra():
    n, N = 10, 8
    sigma = _random_perms(n, N, seed=4)
    identity = np.tile(np.arange(n), (N, 1))

    assert np.array_equal(zl._perm_power(sigma, 0), identity)
    assert np.array_equal(zl._perm_power(sigma, 1), sigma)
    assert np.array_equal(zl._perm_power(sigma, 3),
                          zl._perm_compose(sigma, zl._perm_compose(sigma, sigma)))
    # sigma^(a+b) == sigma^a . sigma^b
    assert np.array_equal(
        zl._perm_power(sigma, 11),
        zl._perm_compose(zl._perm_power(sigma, 4), zl._perm_power(sigma, 7)))
    with pytest.raises(ValueError):
        zl._perm_power(sigma, -1)


def test_landau_bound_makes_the_contiguous_exponent_grid_exhaustive():
    """g(10) = 30, so j = 0..30 covers every DISTINCT power of any sigma in S_10.

    Exhaustiveness is the claim, not "sigma^30 == id": the order of a given
    permutation need not divide g(n) (a 7-cycle has order 7, and 7 does not
    divide 30).  What g(n) bounds is the order itself, so the distinct powers
    sigma^0 .. sigma^(ord-1) always fit inside the grid.
    """
    assert zl._landau_g(10) == 30
    # The naive "max over any part q <= m of q * g(m-q)" recursion double-counts
    # a repeated prime and would return 9 here; the true value is 6.
    assert zl._landau_g(6) == 6
    assert zl._landau_g(12) == 60

    n = 10
    grid = [zl._perm_power(_random_perms(n, 1, seed=99), j) for j in range(31)]
    for seed in range(20):
        sigma = _random_perms(n, 1, seed=100 + seed)
        powers = {zl._perm_power(sigma, j).tobytes() for j in range(31)}
        # Every power, however large the exponent, is already in the grid.
        for j in (37, 91, 512, 2 ** 15):
            assert zl._perm_power(sigma, j).tobytes() in powers
    assert len(grid) == 31


@pytest.mark.parametrize("k", [1, 2, 3, 7, 20])
def test_detect_k_recovers_the_dataset_exponent(k: int):
    sigma = _random_perms(10, 64, seed=5)
    assert zl._detect_k(sigma, zl._perm_power(sigma, k)) == k


def test_detect_k_returns_none_rather_than_a_sentinel_when_there_is_no_power():
    """None, never -1: a negative exponent in the grid would break ``_perm_power``."""
    sigma = _random_perms(10, 32, seed=6)
    target = sigma.copy()
    target[0] = np.roll(target[0], 1)   # row 0 is no longer any power of sigma
    assert zl._detect_k(sigma, target) is None


def test_detect_permutation_probe_rejects_a_non_permutation_probe():
    good = _encode(_random_perms(6, 12, seed=7), seq_len=7)
    assert zl._detect_permutation_probe(good) == 6

    bad = good.copy()
    bad[3, 2] = bad[3, 1]        # duplicated symbol -> not a bijection
    assert zl._detect_permutation_probe(bad) is None

    ragged = good.copy()
    ragged[5, 6] = 4             # junk past the PAD boundary
    assert zl._detect_permutation_probe(ragged) is None


# --------------------------------------------------------------------------- #
# (D) Permutation structure
# --------------------------------------------------------------------------- #

def test_every_power_of_sigma_is_cycle_consistent_and_a_valid_permutation():
    sigma = _random_perms(10, 64, seed=8)
    for j in range(0, 12):
        valid, cyclic = zl._perm_structure(sigma, zl._perm_power(sigma, j))
        assert valid == 1.0, f"sigma^{j} should be a permutation"
        assert cyclic == 1.0, f"sigma^{j} should commute with sigma"


def test_valid_permutation_check_catches_a_non_permutation_decode():
    sigma = _random_perms(10, 4, seed=9)
    yhat = zl._perm_power(sigma, 2).copy()
    yhat[0, 3] = yhat[0, 4]                   # row 0: duplicate -> not a bijection
    yhat[1, :] = 0                            # row 1: constant map
    yhat[2, 0] = -1                           # row 2: PAD decode, out of range

    valid, _ = zl._perm_structure(sigma, yhat)
    assert valid == pytest.approx(0.25)       # only row 3 survives


def test_cycle_consistency_and_validity_are_independent_not_nested():
    """A constant map onto a fixed point commutes with sigma yet is not bijective.

    This is the sharpest available case for the (D) family: it pins that
    cycle-consistency is a NECESSARY condition on a power of sigma and never a
    sufficient one, and that the two rates must be read side by side.  The same
    point at population scale: the centralizer of sigma in S_10 is strictly
    larger than the cyclic group it generates for essentially every cycle type
    (type [5,5] gives 50 elements against 5), so a cycle-consistency rate of 1.0
    does not mean the decodes are powers of sigma.
    """
    n = 6
    # sigma fixes 0 and rotates the rest, so 0 is a fixed point.
    sigma = np.array([[0, 2, 3, 4, 5, 1]], dtype=np.int64)
    assert sigma[0, 0] == 0

    constant = np.zeros((1, n), dtype=np.int64)          # yhat(i) = 0 for all i
    valid, cyclic = zl._perm_structure(sigma, constant)
    assert cyclic == 1.0, "a constant map onto a fixed point commutes with sigma"
    assert valid == 0.0, "...while being maximally non-bijective"


# --------------------------------------------------------------------------- #
# (C) tau(h) -- the separation the whole measurement exists for
# --------------------------------------------------------------------------- #

def _sigma_k_probe(n: int, seq_len: int, B: int, k: int, seed: int):
    sigma = _random_perms(n, B, seed=seed)
    inputs = torch.from_numpy(_encode(sigma, seq_len))
    labels = torch.from_numpy(_encode(zl._perm_power(sigma, k), seq_len))
    return sigma, inputs, labels


def _decode_traj(sigma: np.ndarray, exponents: List[int],
                 seq_len: int) -> List[torch.Tensor]:
    return [torch.from_numpy(_encode(zl._perm_power(sigma, e), seq_len)
                             ).to(torch.int16) for e in exponents]


def _transpose_one_pair(perm: np.ndarray, seed: int) -> np.ndarray:
    """Swap the values at two distinct positions of every row.

    Because a permutation's entries are distinct, swapping two of them changes
    EXACTLY two positions per row.  So the result disagrees with ``perm`` at
    exactly 2 of n positions and agrees at exactly n-2 -- an agreement of
    (n-2)/n that is exact, not statistical, which is what lets the tests below
    assert the numeric SCALE of the agreement metric rather than only its
    endpoints.  The result is still a valid permutation, so it also cannot be
    rejected by the bijectivity screen.
    """
    rng = np.random.RandomState(seed)
    out = perm.copy()
    n = perm.shape[1]
    for r in range(perm.shape[0]):
        a, b = rng.choice(n, size=2, replace=False)
        out[r, a], out[r, b] = out[r, b], out[r, a]
    assert np.all((out != perm).sum(axis=1) == 2)
    return out


def _perm_power_naive(sigma: np.ndarray, j: int) -> np.ndarray:
    """sigma^j by NAIVE repeated application, independent of the shipped code.

    Deliberately shares no code with ``zl._perm_power`` (binary exponentiation)
    or with the logger's incremental repeated squaring: it applies sigma one
    step at a time, j mod ord(row) times per row.  A reference that reused the
    implementation's own algorithm could not detect an error in that algorithm.

    Reducing the exponent mod the row's own order first is what makes this
    tractable at j = 2^15 = 32768 while remaining exact: sigma^j = sigma^(j mod
    ord) for every row, and ord divides Landau's g(n) <= 30 at n=10.
    """
    N, n = sigma.shape
    out = np.empty_like(sigma)
    for r in range(N):
        row = sigma[r]
        # Order of this row = lcm of its cycle lengths, found by walking cycles.
        seen = np.zeros(n, dtype=bool)
        order = 1
        for start in range(n):
            if seen[start]:
                continue
            length, cur = 0, start
            while not seen[cur]:
                seen[cur] = True
                cur = row[cur]
                length += 1
            order = int(np.lcm(order, length))
        cur = np.arange(n, dtype=sigma.dtype)
        for _ in range(j % order):
            cur = row[cur]
        out[r] = cur
    return out


def test_tau_separates_a_sequential_decoder_from_a_doubling_decoder():
    """The categorical question: one power per ACT step, or squaring each step?

    Two hand-built decode trajectories over the SAME probe:
      sequential  yhat^(h) = sigma^h        -> 1, 2, 3,  4
      doubling    yhat^(h) = sigma^(2^(h-1)) -> 1, 2, 4,  8

    They agree at h=1 and h=2 by construction and first diverge at h=3, which is
    why ``ztau/discriminable`` exists.  The metric must score each hypothesis at
    1.0 on its own trajectory and strictly below 1.0 on the other's, and
    ``best_exp`` must read back the exponent that was actually planted.
    """
    n, seq_len, B, T = 10, 11, 128, 4
    sigma, inputs, labels = _sigma_k_probe(n, seq_len, B, k=3, seed=11)

    seq_exp = [1, 2, 3, 4]
    dbl_exp = [1, 2, 4, 8]

    out_seq = zl._extra_step_metrics(
        z_traj=[], preds_traj=_decode_traj(sigma, seq_exp, seq_len),
        inputs=inputs, labels=labels)
    out_dbl = zl._extra_step_metrics(
        z_traj=[], preds_traj=_decode_traj(sigma, dbl_exp, seq_len),
        inputs=inputs, labels=labels)

    for h in range(1, T + 1):
        sk = f"{h:02d}"
        assert out_seq[f"ztau/train/agree_seq_step_{sk}"] == 1.0
        assert out_dbl[f"ztau/train/agree_dbl_step_{sk}"] == 1.0
        # best_exp reads back the planted exponent, from an exhaustive grid.
        assert out_seq[f"ztau/train/best_exp_step_{sk}"] == float(seq_exp[h - 1])
        assert out_dbl[f"ztau/train/best_exp_step_{sk}"] == float(dbl_exp[h - 1])

    # The hypotheses coincide at h=1,2 -- so separation may only be claimed at
    # h>=3, and only then.
    for h in (1, 2):
        sk = f"{h:02d}"
        assert out_seq[f"ztau/train/agree_dbl_step_{sk}"] == 1.0
        assert out_dbl[f"ztau/train/agree_seq_step_{sk}"] == 1.0

    # Separation is asserted WITH A MARGIN, not merely as "< 1.0".  A bare
    # "< 1.0" is satisfied by a degenerate all-or-nothing indicator (1.0 on an
    # exact match, 0.0 otherwise) -- a metric that would read 0.0/0.0 at every
    # step of any real run and carry no discriminative content whatsoever.  The
    # margins below say the wrong hypothesis must sit DOWN AT CHANCE while the
    # right one sits at 1.0, which is the property the campaign actually needs.
    # Chance agreement between two distinct powers of a random 10-permutation is
    # ~1/n = 0.1 (measured 0.104 on this probe); 0.35 is a wide bound on that.
    for h in (3, 4):
        sk = f"{h:02d}"
        wrong_dbl = out_seq[f"ztau/train/agree_dbl_step_{sk}"]
        wrong_seq = out_dbl[f"ztau/train/agree_seq_step_{sk}"]
        assert wrong_dbl < 0.35, (
            "a sequential trajectory must NOT look like doubling at h>=3; "
            f"agree_dbl={wrong_dbl} is above chance")
        assert wrong_seq < 0.35, (
            "a doubling trajectory must NOT look like sequential at h>=3; "
            f"agree_seq={wrong_seq} is above chance")
        assert out_seq[f"ztau/train/agree_seq_step_{sk}"] - wrong_dbl > 0.5
        assert out_dbl[f"ztau/train/agree_dbl_step_{sk}"] - wrong_seq > 0.5

    assert out_seq["ztau/discriminable"] == 1.0
    assert out_dbl["ztau/discriminable"] == 1.0


def test_agreement_is_graded_and_its_numeric_scale_is_pinned():
    """The agreement keys are a POSITION FRACTION -- pinned to an exact value.

    Every other tau test feeds either an exact power of sigma (agreement 1.0) or
    uniform-random garbage (agreement ~chance), which pins only the endpoints
    and leaves the whole middle of the range -- the regime a real, partially
    converged model actually occupies -- undefined.  Two mutually contradictory
    definitions survive endpoint-only tests: a row-exact rate (``.all(axis=1)``,
    which reads 0.0 here) and any monotone rescaling (e.g. ``min(1, 2*mean)``,
    which reads 1.0 here).  They disagree by the full range of the metric.

    A decode that is sigma^h with exactly one transposition per row therefore
    has to read exactly (n-2)/n = 0.8 -- asserted on the hypothesis columns
    (``agree_seq``/``agree_dbl``), on a heatmap cell, and on ``best_exp_agree``,
    so the scale is pinned wherever it is emitted and not only inside the
    private helper.
    """
    n, seq_len, B, T = 10, 11, 128, 3
    sigma, inputs, labels = _sigma_k_probe(n, seq_len, B, k=3, seed=21)
    expected = (n - 2) / n
    assert expected == 0.8

    # h -> exponent under BOTH hypotheses at once: they coincide at h=1,2 and
    # h=3 uses 3 (sequential), so agree_seq is the graded column at every h and
    # agree_dbl is graded at h=1,2.
    perturbed = [
        torch.from_numpy(
            _encode(_transpose_one_pair(zl._perm_power(sigma, e), seed=30 + e),
                    seq_len)).to(torch.int16)
        for e in (1, 2, 3)
    ]
    out = zl._extra_step_metrics(z_traj=[], preds_traj=perturbed,
                                 inputs=inputs, labels=labels)

    for h in range(1, T + 1):
        sk = f"{h:02d}"
        # The planted exponent is still identified: 0.8 beats every rival.
        assert out[f"ztau/train/best_exp_step_{sk}"] == float(h)
        assert out[f"ztau/train/best_exp_agree_step_{sk}"] == pytest.approx(
            expected, abs=1e-12)
        assert out[f"ztau/train/agree_seq_step_{sk}"] == pytest.approx(
            expected, abs=1e-12)
        # The heatmap cell for the planted exponent carries the same scale.
        assert out[f"ztau/train/agree_h{sk}_j{h:02d}"] == pytest.approx(
            expected, abs=1e-12)
        # Two positions per row are wrong, so no row is exact.
        assert out[f"ztau/train/exact_target_step_{sk}"] == 0.0

    # agree_dbl is the same graded column while the hypotheses coincide (h=1,2),
    # so the doubling key's scale is pinned too, not just the sequential one.
    for h in (1, 2):
        assert out[f"ztau/train/agree_dbl_step_{h:02d}"] == pytest.approx(
            expected, abs=1e-12)

    # A transposition preserves bijectivity, so this decode must still be a
    # valid permutation -- the graded agreement is not an artefact of garbage.
    assert out["zperm/train/valid_perm_step_01"] == 1.0


def test_doubling_column_is_exact_at_every_step_of_the_deepest_campaign_arm():
    """The doubling hypothesis, checked to T=16 against an independent oracle.

    The campaign's ablation_act sweep runs halt_max_steps in {2,4,8,16}, so the
    doubling column reaches exponent 2^15 = 32768.  Testing only to T=4 checks
    the hypothesis no further than 2^3 = 8 and leaves the halt=8 and halt=16
    arms -- the deep arms the whole depth question is about -- outside every
    assertion.  Two distinct wrong implementations survive a T<=4 suite:

      * skipping a squaring at any single step (correct through h=4, wrong from
        h=5 on);
      * indexing the precomputed power table with a CLAMPED literal exponent,
        ``powers[min(2**(h-1), jmax)]``.  That is exact while 2^(h-1) <= g(n)=30,
        i.e. for h<=5, and wrong from h=6 -- where it silently substitutes
        sigma^30 for sigma^32.

    Both are caught here because every step to 16 is compared against
    ``_perm_power_naive``, which reduces mod each row's own order and applies
    sigma one step at a time, sharing no algorithm with the implementation.
    """
    n, seq_len, B, T = 10, 11, 96, 16
    sigma, inputs, labels = _sigma_k_probe(n, seq_len, B, k=3, seed=22)

    dbl_exponents = [2 ** (h - 1) for h in range(1, T + 1)]
    assert dbl_exponents[-1] == 32768
    traj = [torch.from_numpy(_encode(_perm_power_naive(sigma, e), seq_len)
                             ).to(torch.int16) for e in dbl_exponents]

    out = zl._extra_step_metrics(z_traj=[], preds_traj=traj,
                                 inputs=inputs, labels=labels)

    for h in range(1, T + 1):
        sk = f"{h:02d}"
        assert out[f"ztau/train/agree_dbl_step_{sk}"] == 1.0, (
            f"doubling column disagrees with sigma^(2^{h - 1}) at h={h}")

    # ...and the sequential column must NOT also read 1.0 once the hypotheses
    # have diverged, or "agreement 1.0 everywhere" would be vacuous.  Exponents
    # 2^(h-1) and h coincide only at h=1,2, and past the row orders they can
    # re-alias, so this is asserted where divergence is guaranteed.
    for h in (3, 4, 5):
        assert out[f"ztau/train/agree_seq_step_{h:02d}"] < 0.35

    # The exhaustive grid is not silently truncated at this depth: 31 exponents
    # x 16 steps = 496 keys, under the 1024 cap.
    assert out["ztau/n_exponents"] == float(zl._landau_g(n) + 1)
    assert (zl._landau_g(n) + 1) * T <= zl._TAU_MAX_AGREE_KEYS


def test_repeated_squaring_matches_an_independent_oracle_past_the_grid():
    """``_perm_power`` itself is exact where the exponent grid cannot reach.

    The tau grid stops at Landau's g(10) = 30, but the doubling hypothesis needs
    sigma^32768.  This pins the primitive both hypotheses are built from against
    the naive oracle at every doubling exponent the deepest arm uses, including
    the ones with no column in the grid.
    """
    n, B = 10, 64
    sigma = _random_perms(n, B, seed=23)
    for h in range(1, 17):
        j = 2 ** (h - 1)
        assert np.array_equal(zl._perm_power(sigma, j),
                              _perm_power_naive(sigma, j)), f"exponent {j}"

    # And the incremental repeated squaring the logger uses is the same object:
    # dbl_{h+1} = dbl_h o dbl_h starting from sigma.
    dbl = sigma.copy()
    for h in range(1, 17):
        assert np.array_equal(dbl, _perm_power_naive(sigma, 2 ** (h - 1)))
        dbl = zl._perm_compose(dbl, dbl)


def test_exact_target_honours_the_ignore_label_mask_of_real_labels():
    """Production labels pad with IGNORE_LABEL_ID = -100, and no other test does.

    ``_probe_forward`` remaps the file's pad id 0 to IGNORE_LABEL_ID before the
    reduction ever sees the labels, so every real ``exact_target`` reading is
    computed against a label row whose tail is -100 while the decode's tail is
    whatever the model emitted there.  Every other test in this module builds
    labels through ``_encode``, which pads with 0, leaving the mask all-True and
    the masking branch completely unexercised -- a reduction that ignored the
    mask would score a PERFECT model at 0.0 in production and 1.0 in the tests.
    """
    n, seq_len, B = 10, 13, 64
    sigma, inputs, labels = _sigma_k_probe(n, seq_len, B, k=3, seed=24)

    labels_prod = labels.clone()
    labels_prod[labels_prod == 0] = zl.IGNORE_LABEL_ID
    assert (labels_prod == zl.IGNORE_LABEL_ID).any(), "test must exercise pad"

    # A perfect decode of sigma^3 that emits junk in the ignored tail, which is
    # exactly what an unconstrained decoder does at masked positions.
    perfect = _encode(zl._perm_power(sigma, 3), seq_len)
    perfect[:, n:] = 7
    traj = [torch.from_numpy(perfect).to(torch.int16)]

    out = zl._extra_step_metrics(z_traj=[], preds_traj=traj,
                                 inputs=inputs, labels=labels_prod)
    assert out["ztau/train/exact_target_step_01"] == 1.0, (
        "a perfect decode must score 1.0; junk at IGNORE_LABEL_ID positions is "
        "outside the loss and must not count against it")

    # And the mask must not excuse a genuine error inside the loss region.
    wrong = perfect.copy()
    wrong[:, 0] = (wrong[:, 0] % n) + 1
    out_w = zl._extra_step_metrics(
        z_traj=[], preds_traj=[torch.from_numpy(wrong).to(torch.int16)],
        inputs=inputs, labels=labels_prod)
    assert out_w["ztau/train/exact_target_step_01"] == 0.0


@pytest.mark.parametrize("T, expected", [(1, 0.0), (2, 0.0), (3, 1.0), (4, 1.0)])
def test_discriminable_flips_exactly_at_the_depth_the_hypotheses_diverge(
        T: int, expected: float):
    """T=2 is the boundary, and it is the campaign's halt=2 arm.

    Sequential and doubling predict the same exponent at h=1 (1) and h=2 (2) and
    first differ at h=3 (3 vs 4), so a run with T<=2 cannot distinguish them even
    in principle.  Asserting only the far endpoints leaves the threshold free to
    drift onto the halt=2 arm and label an undecidable run decidable -- which is
    the single reading error this key exists to prevent.
    """
    n, seq_len, B = 10, 11, 32
    sigma, inputs, labels = _sigma_k_probe(n, seq_len, B, k=2, seed=25)
    out = zl._extra_step_metrics(
        z_traj=[], preds_traj=_decode_traj(sigma, list(range(1, T + 1)), seq_len),
        inputs=inputs, labels=labels)
    assert out["ztau/discriminable"] == expected


def test_tau_heatmap_grid_is_emitted_uncollapsed():
    """The per-(h, j) agreement cells exist, so a heatmap is drawable downstream.

    Deliberately not collapsed to a single verdict scalar in the logger.
    """
    n, seq_len, B = 10, 11, 64
    sigma, inputs, labels = _sigma_k_probe(n, seq_len, B, k=2, seed=12)
    out = zl._extra_step_metrics(
        z_traj=[], preds_traj=_decode_traj(sigma, [1, 2, 3], seq_len),
        inputs=inputs, labels=labels)

    n_exp = int(out["ztau/n_exponents"])
    assert n_exp == zl._landau_g(n) + 1
    for h in (1, 2, 3):
        cells = [k for k in out if k.startswith(f"ztau/train/agree_h{h:02d}_j")]
        assert len(cells) == n_exp
    # j = h is the planted exponent and must be the maximum of its row.
    assert out["ztau/train/agree_h02_j02"] == 1.0
    assert out["ztau/train/agree_h02_j03"] < 1.0


def test_exponent_identification_distinguishes_off_by_one_from_random():
    """``best_exp`` names the wrong power when the model is off by an exponent."""
    n, seq_len, B = 10, 11, 96
    sigma, inputs, labels = _sigma_k_probe(n, seq_len, B, k=3, seed=13)

    off_by_one = _decode_traj(sigma, [4], seq_len)          # target is sigma^3
    out = zl._extra_step_metrics(z_traj=[], preds_traj=off_by_one,
                                 inputs=inputs, labels=labels)
    assert out["zperm/detected_k"] == 3.0
    assert out["ztau/train/best_exp_step_01"] == 4.0
    assert out["ztau/train/best_exp_agree_step_01"] == 1.0
    assert out["ztau/train/exact_target_step_01"] == 0.0

    rng = np.random.RandomState(14)
    garbage = [torch.from_numpy(
        _encode(np.stack([rng.permutation(n) for _ in range(B)]), seq_len)
    ).to(torch.int16)]
    out_g = zl._extra_step_metrics(z_traj=[], preds_traj=garbage,
                                   inputs=inputs, labels=labels)
    assert out_g["ztau/train/best_exp_agree_step_01"] < 0.5


# --------------------------------------------------------------------------- #
# (B) Mutual information -- bias disclosure, not accuracy
# --------------------------------------------------------------------------- #

def _mi_probe(B: int, P: int, D: int, n: int, seed: int, *,
              deterministic: bool = False, signal: float = 0.0):
    """(out, sigma) for one ACT step over a latent with a controlled signal.

    ``deterministic`` plants position i = codes[sigma[:, i]] exactly, so the
    symbol is recoverable with certainty and the TRUE I(sigma(i); z) is exactly
    H(sigma(i)) = log2(n) -- the reference case for the attainable ceiling.
    """
    seq_len = n + 1
    sigma, inputs, labels = _sigma_k_probe(n, seq_len, B, k=2, seed=seed)
    torch.manual_seed(seed)
    if deterministic:
        Z = torch.zeros(B, P + seq_len, D)
        codes = torch.randn(n, D)
        for i in range(n):
            Z[:, P + i, :] = codes[torch.from_numpy(sigma[:, i])]
    else:
        Z = torch.randn(B, P + seq_len, D)
        if signal:
            codes = torch.randn(n, D)
            for i in range(n):
                Z[:, P + i, :] += signal * codes[torch.from_numpy(sigma[:, i])]
    return zl._extra_step_metrics(z_traj=[Z], preds_traj=[],
                                  inputs=inputs, labels=labels), sigma


def test_mi_floor_is_materially_negative_and_zero_is_not_the_reference():
    """The bias floor is NOT ~0, and an uninformative latent sits ON it.

    This is the estimator's single most misreadable property.  The naive
    reading -- "0 bits means no information, negative means worse than the
    marginal" -- is wrong: the cross-fit decoder's held-out cross-entropy
    exceeds H(X) even on pure noise, so the floor is materially BELOW zero
    (measured -1.10 bits at the production configuration B=512/D=512/n=10,
    a third of the 3.32-bit ceiling).  Without this test the one-sided
    ``null < 0.05`` check above is satisfied by an estimator whose floor is at
    0, at -1, or at -100 alike, and every weakly-informative ACT step would
    render as a meaningful plateau instead of a reading at the instrument's
    floor.

    Pinned here as: the null is strongly negative, AND a latent that carries no
    information reads essentially the same value as the null.  Those two
    together say "the floor is negative and it is where uninformative latents
    land", which is exactly the fact the dashboard reader needs.  Any future
    clipping of the estimate at zero -- the obvious "cleanup" -- fails this.
    """
    out, _ = _mi_probe(B=256, P=3, D=24, n=8, seed=15)
    lb = out["zmi/train/sym_decode_lb_bits_step_01"]
    null = out["zmi/train/sym_decode_null_bits_step_01"]

    assert null < -0.3, (
        f"null {null:+.3f} is not materially negative -- if the floor really "
        "moved to ~0 the estimator note's reading rule must be re-derived, not "
        "the tolerance widened")
    assert lb < -0.3, (
        f"an uninformative latent read {lb:+.3f}; a value at or above 0 means "
        "the estimate is being clipped, which hides the floor")
    assert abs(lb - null) < 0.4, (
        f"uninformative latent ({lb:+.3f}) should sit at the shuffle floor "
        f"({null:+.3f}); a large gap means the null is not measuring the floor")


def test_mi_saturation_key_reports_the_attainable_not_the_mathematical_ceiling():
    """``zmi/sym_decode_saturated_bits`` is the top endpoint of the instrument.

    log2(n) is the ceiling of the ESTIMAND; this key is the ceiling of the
    ESTIMATOR.  A latent from which the symbol is exactly recoverable has true
    MI equal to log2(n) and yet reads materially below it, so without this key
    that shortfall is indistinguishable on the dashboard from real model
    shortfall.

    Also pins that the value is MEASURED, not a constant: it must move with the
    probe size, because r itself is B-dependent.  A hardcoded number (the
    tempting "just document 2.85") fails both the ceiling gap and the
    B-dependence assertion.
    """
    n, P, D = 8, 3, 24
    ceiling = float(np.log2(n))

    out_rand, _ = _mi_probe(B=256, P=P, D=D, n=n, seed=15)
    out_det, _ = _mi_probe(B=256, P=P, D=D, n=n, seed=15, deterministic=True)

    sat = out_rand["zmi/sym_decode_saturated_bits"]
    # Emitted whenever an lb is emitted -- the value can never be read alone.
    assert "zmi/train/sym_decode_lb_bits_step_01" in out_rand
    assert out_det["zmi/sym_decode_saturated_bits"] == pytest.approx(sat)

    assert 0.0 < sat < ceiling - 0.2, (
        f"saturated {sat:.3f} must sit materially below the mathematical "
        f"ceiling {ceiling:.3f}: that gap IS the finite-sample bias this key "
        "exists to expose")

    # An exactly-recoverable latent lands near the saturation point, far above
    # an uninformative one -- i.e. the key really is the top of the range.
    lb_det = out_det["zmi/train/sym_decode_lb_bits_step_01"]
    lb_rand = out_rand["zmi/train/sym_decode_lb_bits_step_01"]
    assert abs(lb_det - sat) < 0.5, (
        f"a deterministic latent read {lb_det:.3f} against a saturation "
        f"reference of {sat:.3f}")
    assert lb_det > lb_rand + 1.0
    # ...and it is still short of the mathematical ceiling, which is the whole
    # point: the gap is the instrument, not the model.
    assert lb_det < ceiling

    # Measured, not hardcoded: halving the probe size must move it.
    sat_small, _ = _mi_probe(B=128, P=P, D=D, n=n, seed=15)
    assert abs(sat_small["zmi/sym_decode_saturated_bits"] - sat) > 0.25, (
        "saturation must track the probe size; a constant would not")


class _capture_warnings:
    """Collect ``log.warning`` messages emitted by utils.z_logging in a block."""

    def __enter__(self) -> List[str]:
        self.records: List[str] = []
        outer = self

        class _H(logging.Handler):
            def emit(self, record):
                outer.records.append(record.getMessage())

        self.handler = _H()
        self.logger = zl.log
        self.logger.addHandler(self.handler)
        self.prev = self.logger.level
        self.logger.setLevel(logging.WARNING)
        return self.records

    def __exit__(self, *exc):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.prev)
        return False


def test_mi_fold_leak_alarm_is_checked_in_code_not_only_documented():
    """A positive null must actually raise a warning, not be silently logged.

    The estimator note calls the null "a live alarm"; an alarm nothing inspects
    is prose.  A positive null means the fold split leaked and every lb on that
    curve stops being a bound in any direction, so it must reach the log.
    Warning, never raising: aborting a multi-day run over a diagnostic is the
    failure mode this module's containment design exists to prevent.
    """
    with _capture_warnings() as rec:
        zl._mi_alarm_check({"01": -1.10, "02": -0.98, "03": 0.0}, "train")
    assert rec == [], f"a non-positive null must not warn, got {rec}"

    with _capture_warnings() as rec:
        zl._mi_alarm_check({"01": -1.10, "02": +0.42}, "train")
    assert len(rec) == 1, "a positive null must warn exactly once"
    assert "02" in rec[0] and "fold" in rec[0].lower(), rec[0]

    # Jitter around the floor is not an alarm -- the threshold sits above the
    # null's own sampling noise, so this warns on leaks and not on noise.
    with _capture_warnings() as rec:
        zl._mi_alarm_check({"01": 0.02}, "train")
    assert rec == []

    # NaN (a degenerate fold) is not an alarm either.
    with _capture_warnings() as rec:
        zl._mi_alarm_check({"01": float("nan")}, "train")
    assert rec == []


def test_mi_lower_bound_is_signed_rises_with_signal_and_respects_its_ceiling():
    """Pins the estimator's DISCLOSURE properties, not its accuracy.

    A held-out Barber-Agakov bound has a guaranteed sign, so with no planted
    signal it must not be materially positive; with a planted mean shift it must
    rise; and being a lower bound on I(symbol ; z) it must never exceed
    log2(n).  Nothing here asserts closeness to a true MI -- at this sample size
    that is not a testable property, which is precisely why the null and the
    ceiling are logged as sibling keys rather than a single number.
    """
    n, seq_len, P, D, B = 8, 9, 3, 24, 256
    sigma, inputs, labels = _sigma_k_probe(n, seq_len, B, k=2, seed=15)
    ceiling = float(np.log2(n))

    values = []
    for signal in (0.0, 1.0, 4.0):
        torch.manual_seed(16)
        Z = torch.randn(B, P + seq_len, D)
        if signal:
            codes = torch.randn(n, D)
            for i in range(n):
                Z[:, P + i, :] += signal * codes[torch.from_numpy(sigma[:, i])]
        out = zl._extra_step_metrics(z_traj=[Z], preds_traj=[],
                                     inputs=inputs, labels=labels)
        lb = out["zmi/train/sym_decode_lb_bits_step_01"]
        null = out["zmi/train/sym_decode_null_bits_step_01"]
        values.append(lb)

        assert lb <= ceiling + 1e-9, "a lower bound cannot exceed log2(n)"
        # Shuffled labels carry zero information, so the null is a valid bound on
        # zero: it must not be materially positive.  A positive null is the
        # fold-leak alarm, not a tolerance to be widened.
        assert null < 0.05, f"null {null} is positive -- the fold split leaked"

    assert values[0] < 0.25, "no planted signal must not read as information"
    assert values[1] > values[0] + 0.5
    assert values[2] > values[1]


def test_mi_keys_never_appear_without_their_null_sibling_and_ceilings():
    """The honesty constraint is discharged by the key namespace, not by prose."""
    n, seq_len, P, D, B = 8, 9, 3, 16, 128
    _, inputs, labels = _sigma_k_probe(n, seq_len, B, k=2, seed=17)
    out = zl._extra_step_metrics(z_traj=[torch.randn(B, P + seq_len, D)],
                                 preds_traj=[], inputs=inputs, labels=labels)

    for key in out:
        if key.startswith("zmi/train/sym_decode_lb_bits_step_"):
            assert key.replace("_lb_", "_null_") in out

    assert out["zmi/probe_ceiling_bits"] == pytest.approx(float(np.log2(B)))
    assert out["zmi/sym_ceiling_bits"] == pytest.approx(float(np.log2(n)))
    assert out["zmi/perm_entropy_bits"] == pytest.approx(
        float(np.log2(np.arange(1, n + 1)).sum()))
    # The instrument's hard ceiling sits BELOW the quantity the task cares about:
    # log2(n!) is not reachable from a probe of this size, by construction.
    assert out["zmi/probe_ceiling_bits"] < out["zmi/perm_entropy_bits"]
    # No key claims to be a mutual information without naming its estimator.
    for key in out:
        if key.startswith("zmi/") and key.endswith(tuple("0123456789")):
            assert "decode" in key


# --------------------------------------------------------------------------- #
# Frozen-contract locks
# --------------------------------------------------------------------------- #

def test_new_metrics_never_emit_a_frozen_key_or_leave_their_namespaces():
    n, seq_len, P, D, B = 10, 11, 4, 12, 64
    sigma, inputs, labels = _sigma_k_probe(n, seq_len, B, k=2, seed=18)
    out = zl._extra_step_metrics(
        z_traj=[torch.randn(B, P + seq_len, D) for _ in range(3)],
        preds_traj=_decode_traj(sigma, [1, 2, 3], seq_len),
        inputs=inputs, labels=labels)

    assert out, "the reduction produced no keys at all"
    assert not (set(out) & _FROZEN_SCALAR_KEYS)
    for key in out:
        assert not key.startswith(_FROZEN_KEY_PREFIXES)
        assert key.split("/", 1)[0] in zl._SEQ_KEY_NAMESPACES, key


def test_pooled_pr_reproduces_the_frozen_eff_rank_bitwise():
    """``==``, not ``approx`` -- this is a lock on hard constraint 1.

    ``_pooled_pr_legacy`` runs the frozen chain verbatim (bf16 round-trip ->
    ``_mean_pool_z`` -> ``_pca`` -> ``_effective_rank``), which is why it must
    NOT be converted to the cheap trace form used by every other PR in the
    module: the trace form does not clip negative eigenvalues and agrees only to
    ~1e-7, so the exact-equality lock would silently disappear.  If a future
    change breaks this assertion, fix the change, not the test.
    """
    B, P, seq_len, D = 48, 5, 7, 9
    torch.manual_seed(19)
    z = torch.randn(B, P + seq_len, D)

    frozen = zl._effective_rank(zl._pca(zl._mean_pool_z(z.to(torch.bfloat16)))[0])
    assert zl._pooled_pr_legacy(z) == frozen


def test_the_frozen_probe_result_is_byte_identical_with_and_without_the_gate():
    """Turning the new metrics on must not perturb a single frozen value.

    Drives the real ``_probe_forward`` twice over the same scripted model and
    compares ``z_H``, ``z_L``, ``step_deltas``, ``exact_acc`` and -- the key that
    74 historical runs are compared on -- the ``z/eff_rank`` recomputed from the
    returned tensor by the frozen chain.
    """
    model, probe = _scripted_setup(n_steps=3)

    off = zl._probe_forward(model, probe, device="cpu", compute_extra=False)
    on = zl._probe_forward(model, probe, device="cpu", compute_extra=True)

    assert _FROZEN_PROBE_RESULT_KEYS <= set(off)
    assert torch.equal(off["z_H"], on["z_H"])
    assert torch.equal(off["z_L"], on["z_L"])
    assert torch.equal(off["labels"], on["labels"])
    assert torch.equal(off["correct_mask"], on["correct_mask"])
    assert off["step_deltas"] == on["step_deltas"]
    assert off["exact_acc"] == on["exact_acc"]

    eff_off = zl._effective_rank(zl._pca(zl._mean_pool_z(off["z_H"]))[0])
    eff_on = zl._effective_rank(zl._pca(zl._mean_pool_z(on["z_H"]))[0])
    assert eff_off == eff_on

    # ...and the gate really is a gate.
    assert off["extra_metrics"] == {}
    assert off["preds_traj"] is None
    assert on["extra_metrics"]["zseq/status_ok"] == 1.0


# --------------------------------------------------------------------------- #
# Scripted model: the only test that drives the real capture loop
# --------------------------------------------------------------------------- #

class _InnerCarry:
    def __init__(self, z_H: torch.Tensor, z_L: torch.Tensor) -> None:
        self.z_H = z_H
        self.z_L = z_L


class _OnlyZH:
    """transformers_baseline / trm_hier6: a z_H with no plain ``z_L`` sibling."""
    def __init__(self, z_H: torch.Tensor) -> None:
        self.z_H = z_H


class _OnlyZL:
    """trm_singlez: the InnerCarry declares only ``z_L``."""
    def __init__(self, z_L: torch.Tensor) -> None:
        self.z_L = z_L


class _Carry:
    def __init__(self, B: int) -> None:
        self.step = 0
        self.inner_carry: Optional[_InnerCarry] = None
        self.halted = torch.zeros(B, dtype=torch.bool)


class _ScriptedModel:
    """Deterministic stand-in for ``ACTLossHead`` with a known per-step identity.

    At ACT step h the latent's token slice has participation ratio exactly h and
    the decode is exactly ``sigma^h``.  Both facts are recoverable from the
    emitted metrics, which is what turns "the capture pairs loop 1's latent with
    loop 2's decode at the same index" into an assertion instead of an
    inspection.  Halting is batch-synchronous and step-count-driven, mirroring
    eval-mode ``trm.py:272`` where ``halted = is_last_step``.
    """

    training = False
    #: Which latent fields the carry exposes, mirroring the real arch grid:
    #: "both" (trm), "z_h_only" (transformers_baseline, trm_hier6),
    #: "z_l_only" (trm_singlez), "no_inner_attr" (a carry shape we have not seen).
    latent_fields = "both"

    def __init__(self, n_steps: int, sigma: np.ndarray, seq_len: int,
                 P: int, D: int) -> None:
        self.n_steps = n_steps
        self.sigma = sigma
        self.seq_len = seq_len
        self.P = P
        self.D = D
        self.B = sigma.shape[0]
        self.n = sigma.shape[1]
        self.calls = 0

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> _Carry:
        return _Carry(self.B)

    def __call__(self, carry: _Carry, batch: Dict[str, torch.Tensor],
                 return_keys: Any):
        self.calls += 1
        carry.step += 1
        h = carry.step
        z = _latent_with_planted_pr(self.B, self.P, self.seq_len, self.D,
                                    rank=h, seed=1000 + h)
        zb = z.to(torch.bfloat16)
        if self.latent_fields == "both":
            carry.inner_carry = _InnerCarry(z_H=zb, z_L=(z * 0.5).to(torch.bfloat16))
        elif self.latent_fields == "z_h_only":
            carry.inner_carry = _OnlyZH(zb)
        elif self.latent_fields == "z_l_only":
            carry.inner_carry = _OnlyZL(zb)
        else:
            # The attribute is REMOVED, not set to None.  A None-valued
            # attribute would leave `carry.inner_carry` legal and never exercise
            # the crash path; only a genuinely absent attribute does, which is
            # what the frozen try/except above is catching in the first place.
            if hasattr(carry, "inner_carry"):
                del carry.inner_carry
        all_finish = h >= self.n_steps
        carry.halted = torch.full((self.B,), all_finish, dtype=torch.bool)

        preds: Dict[str, torch.Tensor] = {}
        if "preds" in list(return_keys):
            decoded = _encode(zl._perm_power(self.sigma, h), self.seq_len)
            preds["preds"] = torch.from_numpy(decoded)
        return carry, torch.zeros(()), {}, preds, torch.tensor(all_finish)


def _scripted_setup(n_steps: int, n: int = 6, seq_len: int = 7, B: int = 40,
                    P: int = 3, D: int = 5, k: int = 2, seed: int = 21):
    sigma = _random_perms(n, B, seed=seed)
    probe = {
        "inputs": torch.from_numpy(_encode(sigma, seq_len)).to(torch.int32),
        "labels": torch.from_numpy(
            _encode(zl._perm_power(sigma, k), seq_len)).to(torch.int32),
        "puzzle_identifiers": torch.zeros(B, dtype=torch.int32),
    }
    return _ScriptedModel(n_steps, sigma, seq_len, P, D), probe


def test_probe_forward_pairs_each_act_step_latent_with_its_own_decode():
    """The capture-path test: index h holds step h's latent AND step h's decode.

    Both halves are asserted from planted, step-dependent structure rather than
    from the shapes lining up: the latent at step h has participation ratio
    exactly h, and the decode at step h is exactly ``sigma^h``.  A one-step skew
    between the two ACT loops would break both assertions.
    """
    T = 4
    model, probe = _scripted_setup(n_steps=T)
    result = zl._probe_forward(model, probe, device="cpu", compute_extra=True)
    extra = result["extra_metrics"]

    assert extra["zseq/status_ok"] == 1.0
    assert extra["zseq/n_act_steps"] == float(T)
    assert extra["zseq/has_trajectory"] == 1.0
    assert extra["zseq/latent_is_z_h"] == 1.0
    assert extra["zseq/model_in_eval_mode"] == 1.0
    # Two ACT loops of T steps each, exactly as the pre-existing implementation.
    assert model.calls == 2 * T

    for h in range(1, T + 1):
        sk = f"{h:02d}"
        assert extra[f"zseq/train/pr_joint_tok_step_{sk}"] == pytest.approx(
            float(h), rel=2e-2), "latent at index h is not step h's latent"
        assert extra[f"ztau/train/agree_seq_step_{sk}"] == 1.0, (
            "decode at index h is not step h's decode")
        assert extra[f"ztau/train/best_exp_step_{sk}"] == float(h)

    # The decode trajectory is captured for the snapshot at (T, B, seq_len).
    assert result["preds_traj"].shape == (T, 40, 7)
    assert result["preds_traj"].dtype == torch.int16


@pytest.mark.parametrize("arch,expect_zseq,expect_is_zh", [
    # trm: both fields -> the frozen paired read succeeds, we reuse its tensor.
    ("both", True, 1.0),
    # transformers_baseline / trm_hier6: z_H present, plain z_L absent.  The
    # frozen block discards BOTH and emits no z metrics for these cohorts; the
    # new capture recovers the z_H that was there all along.
    ("z_h_only", True, 1.0),
    # trm_singlez: only z_L exists, and it is the latent that carries the state.
    ("z_l_only", True, 0.0),
    # A carry shape with no inner_carry at all must degrade, never raise.
    ("no_inner_attr", False, 1.0),
])
def test_new_capture_covers_the_architectures_the_frozen_paired_read_blinds(
        arch: str, expect_zseq: bool, expect_is_zh: float):
    """Independent getattr per field, without repairing the frozen paired read.

    Repairing the frozen block would start emitting ``z/eff_rank`` for cohorts
    that have never had it, which changes what a frozen key means for those
    runs -- so the frozen block stays byte-identical and only the NEW capture
    reads each field separately.  The frozen keys must therefore stay ABSENT
    for these architectures while the new ones appear.
    """
    T = 2
    model, probe = _scripted_setup(n_steps=T)
    model.latent_fields = arch

    result = zl._probe_forward(model, probe, device="cpu", compute_extra=True)
    extra = result["extra_metrics"]

    # The frozen contract is untouched: z_H/z_L stay None whenever the paired
    # read could not satisfy BOTH fields, exactly as before this work.
    if arch != "both":
        assert result["z_H"] is None and result["z_L"] is None

    assert extra["zseq/status_ok"] == 1.0
    assert extra["zseq/latent_is_z_h"] == expect_is_zh
    assert ("zseq/train/pr_joint_tok_step_01" in extra) is expect_zseq
    # Decode-only families survive every architecture.
    assert extra["ztau/train/agree_seq_step_01"] == 1.0
    assert extra["zperm/train/valid_perm_final"] == 1.0


def test_single_act_step_emits_one_honest_point_and_fabricates_no_trajectory():
    """halt_max_steps == 1: one point per family, no *_step_02, no crash.

    Note the deliberate asymmetry with the frozen ``z/delta_step_<t>``, which
    emits NOTHING here: a delta needs two states, whereas a participation ratio,
    an MI and a decode agreement are each well defined at a single state.
    Emitting the single observation the run can actually make is not fabricating
    a trajectory; suppressing it would discard the only measurement available.
    """
    model, probe = _scripted_setup(n_steps=1)
    result = zl._probe_forward(model, probe, device="cpu", compute_extra=True)
    extra = result["extra_metrics"]

    assert result["step_deltas"] == [], "frozen delta behaviour must not change"
    assert extra["zseq/status_ok"] == 1.0
    assert extra["zseq/n_act_steps"] == 1.0
    assert extra["zseq/has_trajectory"] == 0.0

    step1 = [k for k in extra if k.endswith("_step_01")]
    assert step1, "the single honest observation must still be reported"
    for key in extra:
        assert "_step_02" not in key and "_h02_" not in key, key

    # Sequential and doubling predict the same exponent at h=1 and h=2 and first
    # diverge at h=3, so a T<=2 run cannot distinguish them even in principle.
    assert extra["ztau/discriminable"] == 0.0
    assert extra["ztau/train/agree_seq_step_01"] == extra[
        "ztau/train/agree_dbl_step_01"]


def test_train_mode_suppresses_the_decode_dependent_families():
    """z comes from loop 1 and preds from loop 2 -- pairing needs eval mode."""
    model, probe = _scripted_setup(n_steps=3)
    model.training = True
    result = zl._probe_forward(model, probe, device="cpu", compute_extra=True)
    extra = result["extra_metrics"]

    assert extra["zseq/model_in_eval_mode"] == 0.0
    assert extra["zseq/status_ok"] == 0.0, (
        "a swallowed precondition must be visible in wandb, not inferred")
    assert not [k for k in extra if k.startswith(("ztau/", "zperm/"))]
    # The latent-only families are unaffected: they never touch the decode.
    assert "zseq/train/pr_joint_tok_step_01" in extra


def test_status_keys_survive_a_failure_inside_the_reduction(
        monkeypatch: pytest.MonkeyPatch):
    """A bug in the new code must be visible as status_ok=0, not as key absence.

    Absent keys are ambiguous between four conditions (bug, no usable latent,
    feature off, probe not permutation-shaped); a status scalar is not.
    """
    def _boom(*_a, **_kw):
        raise RuntimeError("synthetic failure inside the reduction")

    monkeypatch.setattr(zl, "_extra_step_metrics", _boom)
    model, probe = _scripted_setup(n_steps=2)
    result = zl._probe_forward(model, probe, device="cpu", compute_extra=True)

    # Training survives...
    assert result["exact_acc"] is not None
    assert result["z_H"] is not None
    # ...and the failure is legible.
    assert result["extra_metrics"]["zseq/status_ok"] == 0.0
    assert result["extra_metrics"]["zseq/n_act_steps"] == 2.0


def test_reduction_raises_rather_than_swallowing_so_tests_cannot_pass_vacuously():
    """Containment lives at the call site, never inside the pure reduction."""
    with pytest.raises(Exception):
        zl._extra_step_metrics(
            z_traj=[torch.randn(4, 2, 3)],     # latent shorter than the labels
            preds_traj=[],
            inputs=torch.zeros(4, 5, dtype=torch.int64),
            labels=torch.zeros(4, 5, dtype=torch.int64))


def test_non_permutation_probe_degrades_to_omission_without_raising():
    B, seq_len, P, D = 32, 6, 2, 8
    rng = np.random.RandomState(22)
    inputs = torch.from_numpy(rng.randint(1, 5, size=(B, seq_len)))
    labels = torch.from_numpy(rng.randint(1, 5, size=(B, seq_len)))
    preds = [torch.from_numpy(rng.randint(0, 5, size=(B, seq_len))).to(torch.int16)]

    out = zl._extra_step_metrics(z_traj=[torch.randn(B, P + seq_len, D)],
                                 preds_traj=preds, inputs=inputs, labels=labels)

    assert not [k for k in out if k.startswith(("ztau/", "zperm/"))]
    assert not [k for k in out if k.startswith("zmi/train/")]
    assert "zseq/train/pr_joint_tok_step_01" in out   # latent family unaffected


def test_missing_latent_still_yields_the_decode_dependent_families():
    """Arch coverage: tau and perm need only the decode and the probe inputs."""
    n, seq_len, B = 6, 7, 40
    sigma, inputs, labels = _sigma_k_probe(n, seq_len, B, k=2, seed=23)
    out = zl._extra_step_metrics(
        z_traj=[None, None], preds_traj=_decode_traj(sigma, [1, 2], seq_len),
        inputs=inputs, labels=labels)

    assert not [k for k in out if k.startswith("zseq/train/")]
    assert out["ztau/train/agree_seq_step_02"] == 1.0
    assert out["zperm/train/valid_perm_final"] == 1.0


# --------------------------------------------------------------------------- #
# Tier 2: key emission through ZDynamicsLogger.log (with _probe_forward stubbed)
# --------------------------------------------------------------------------- #

def _make_logger(seq_metrics: bool) -> zl.ZDynamicsLogger:
    """Bypass ``__init__`` so no probe files, no dataset and no disk are needed."""
    logger = object.__new__(zl.ZDynamicsLogger)
    logger._checkpoint_path = None
    logger._seq_metrics = seq_metrics
    logger._phase_tracker = zl.PhaseTracker(0.999, 2)
    logger._train_probe = {}
    logger._test_probe = {}
    logger._train_fp = "deadbeef"
    logger._test_fp = "cafef00d"
    return logger


def _install_probe_stub(monkeypatch: pytest.MonkeyPatch,
                        extra: Dict[str, float]) -> Dict[str, Any]:
    """Stub ``_probe_forward``.  NOTE: the real ACT loop is NOT exercised here.

    The seven-key return contract is reproduced exactly so the stub cannot drift
    from the implementation without this test noticing.
    """
    torch.manual_seed(24)
    z = torch.randn(24, 9, 6).to(torch.bfloat16)

    def _stub(model, probe, *, device="cuda", compute_extra=False):
        return {
            "z_H": z,
            "z_L": z * 0,
            "labels": torch.zeros(24, 5, dtype=torch.int32),
            "correct_mask": torch.ones(24, dtype=torch.bool),
            "step_deltas": [0.5, 0.25],
            "exact_acc": 0.75,
            "extra_metrics": dict(extra) if compute_extra else {},
            "preds_traj": None,
        }

    monkeypatch.setattr(zl, "_probe_forward", _stub)
    monkeypatch.setattr(zl, "_make_pca_scatter", lambda *a, **kw: [])
    return {"z": z}


def _capture_wandb(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
    import wandb
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(wandb, "log",
                        lambda d, step=None: captured.append(dict(d)))
    return captured


def test_logged_frozen_keys_are_identical_with_and_without_the_new_metrics(
        monkeypatch: pytest.MonkeyPatch):
    """The end-to-end version of hard constraint 1, at the wandb boundary."""
    ctx = _install_probe_stub(monkeypatch, {})
    captured = _capture_wandb(monkeypatch)
    _make_logger(seq_metrics=False).log(model=None, step=7,
                                        save_train_state_fn=None,
                                        train_state=None)
    baseline = captured[-1]

    n_steps = 3
    extra = {
        "zseq/status_ok": 1.0,
        "zseq/n_act_steps": float(n_steps),
        f"zseq/train/pr_pooled_all_step_{n_steps:02d}": zl._pooled_pr_legacy(ctx["z"]),
        "zseq/train/pr_joint_tok_step_01": 12.5,
        "ztau/train/agree_seq_step_01": 1.0,
    }
    _install_probe_stub(monkeypatch, extra)
    captured2 = _capture_wandb(monkeypatch)
    _make_logger(seq_metrics=True).log(model=None, step=7,
                                       save_train_state_fn=None,
                                       train_state=None)
    withnew = captured2[-1]

    frozen = _FROZEN_SCALAR_KEYS | {"z/delta_step_1", "z/delta_step_2"}
    assert frozen <= set(baseline)
    for key in frozen:
        assert withnew[key] == baseline[key], key

    # Strictly additive, and only inside the four new namespaces.
    added = set(withnew) - set(baseline)
    assert added
    for key in added:
        assert key.split("/", 1)[0] in zl._SEQ_KEY_NAMESPACES, key

    # The live drift detector reads exactly zero: the per-step pooled PR at h=T
    # travels the identical chain from the identical tensor as z/eff_rank.
    assert withnew["zseq/legacy_pr_reldiff"] == 0.0


def test_logger_refuses_to_overwrite_a_frozen_key_or_admit_a_foreign_namespace(
        monkeypatch: pytest.MonkeyPatch):
    """Two independent structural locks, tested by trying to defeat both."""
    _install_probe_stub(monkeypatch, {
        "z/eff_rank": -999.0,            # frozen name: must be refused
        "probe/train_exact": -999.0,     # frozen name: must be refused
        "custom/thing": 1.0,             # foreign namespace: must be dropped
        "zseq/train/pr_joint_tok_step_01": 3.0,   # legitimate
    })
    captured = _capture_wandb(monkeypatch)
    _make_logger(seq_metrics=True).log(model=None, step=11,
                                       save_train_state_fn=None,
                                       train_state=None)
    logged = captured[-1]

    assert logged["z/eff_rank"] != -999.0
    assert logged["probe/train_exact"] == 0.75
    assert "custom/thing" not in logged
    assert logged["zseq/train/pr_joint_tok_step_01"] == 3.0


def test_logger_emits_no_new_keys_when_the_gate_is_off(
        monkeypatch: pytest.MonkeyPatch):
    _install_probe_stub(monkeypatch, {"zseq/train/pr_joint_tok_step_01": 3.0})
    captured = _capture_wandb(monkeypatch)
    _make_logger(seq_metrics=False).log(model=None, step=3,
                                        save_train_state_fn=None,
                                        train_state=None)
    logged = captured[-1]

    for key in logged:
        assert key.split("/", 1)[0] not in zl._SEQ_KEY_NAMESPACES, key
