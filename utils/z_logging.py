"""
z_logging.py — z dynamics / convergence logging + learning phase auto-detection.

Usage: imported by pretrain.py; all public entry points check rank and
log_z_dynamics internally.  No model file is modified.

Key design choices:
- PhaseTracker is pure Python (no torch), unit-testable in isolation.
- Probe inputs are loaded once at construction time and pinned; the same
  tensor (by hash/ptr) is reused every eval, satisfying Gate 5.
- Snapshots are saved in bfloat16 to stay well under 50 MB (Gate 4).
- PCA uses numpy (no sklearn needed).
- matplotlib uses the Agg (non-GUI) backend unconditionally.

SEQUENCE-AWARE / TRAJECTORY METRICS (added under the ``z_seq_metrics`` gate)
---------------------------------------------------------------------------
The four legacy keys ``z/eff_rank``, ``z/pca_top2_var``, ``z/mean_norm`` and
``z/delta_step_<t>`` are FROZEN: 74 historical runs are compared against them,
so their names, their inputs and their arithmetic are untouched here.  Every
quantity added below is additive under a new namespace (``zseq/``, ``zmi/``,
``ztau/``, ``zperm/``) and is emitted only when ``z_seq_metrics`` is on, which
is off by default and stacks on top of ``log_z_dynamics``.

Motivation: ``z/eff_rank`` is a participation ratio of the MEAN-POOLED latent
(``_mean_pool_z`` below collapses (B, S, D) -> (B, D)), so the sequence axis is
destroyed before the spectrum is ever formed.  A low pooled effective rank
therefore does not imply the full state is low-capacity.

DO NOT claim unpooled PR >= pooled PR.  Mean-pooling is a linear projection
``X_flat @ M`` and the participation ratio is NOT monotone under projection.
The correct claim is that the pooled and unpooled views share the same support
and differ only in the pooling; the empirical answer is whatever the two
numbers say.
"""

from __future__ import annotations

import os
import hashlib
import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PhaseTracker — pure Python, no torch dependency
# ---------------------------------------------------------------------------

class PhaseTracker:
    """
    Tracks grokking phases based on exact accuracy thresholds.

    Phase 0: train_exact < theta
    Phase 1: train_exact >= theta  AND  test_exact < theta
    Phase 2: train_exact >= theta  AND  test_exact >= theta

    A transition is committed only after `patience` consecutive evals
    satisfy the next phase's conditions (prevents flapping).
    Phases are monotone — once committed, they never decrease.
    """

    def __init__(self, phase_threshold: float = 0.999, phase_patience: int = 2):
        self.theta = phase_threshold
        self.patience = phase_patience

        self._phase: int = 0          # committed phase (monotone)
        self._candidate: int = 0      # next candidate phase
        self._candidate_count: int = 0  # consecutive evals satisfying candidate
        self._transition_steps: list[int] = []  # history of committed transitions
        self._last_step: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_raw_phase(self, train_exact: float, test_exact: float) -> int:
        """Phase from current accuracy values (no patience)."""
        if train_exact >= self.theta and test_exact >= self.theta:
            return 2
        if train_exact >= self.theta:
            return 1
        return 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, train_exact: float, test_exact: float, step: int = 0) -> Tuple[int, bool]:
        """
        Feed a new observation.

        Returns:
            (current_phase, transitioned)
            transitioned=True means this call produced a new committed transition.
        """
        self._last_step = step
        raw = self._compute_raw_phase(train_exact, test_exact)

        # Only ever go forward (monotone)
        next_candidate = max(raw, self._phase)

        if next_candidate > self._phase:
            # We're looking at a potential transition
            if next_candidate == self._candidate:
                self._candidate_count += 1
            else:
                # New candidate resets the patience counter
                self._candidate = next_candidate
                self._candidate_count = 1

            if self._candidate_count >= self.patience:
                # Commit the transition
                self._phase = self._candidate
                self._candidate_count = 0
                self._transition_steps.append(step)
                return self._phase, True
        else:
            # Raw phase matches committed phase — reset candidate
            self._candidate = self._phase
            self._candidate_count = 0

        return self._phase, False

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def transition_steps(self) -> list[int]:
        return list(self._transition_steps)


# ---------------------------------------------------------------------------
# Probe dataset loader (rank-0 only, loaded once)
# ---------------------------------------------------------------------------

IGNORE_LABEL_ID = -100  # matches losses.py


def _load_probe_tensors(data_path: str, split: str, probe_size: int,
                        ignore_label_id_in_file: Optional[int]) -> Dict[str, torch.Tensor]:
    """
    Load the first `probe_size` examples from a dataset split.

    Returns tensors: inputs, labels, puzzle_identifiers
    All on CPU, int32 (same as PuzzleDataset._collate_batch).

    NOTE: num_puzzle_identifiers==1 for sigma_k datasets, so direct
    slice of puzzle_identifiers.npy is safe (all=blank_identifier_id=0).
    """
    split_dir = os.path.join(data_path, split)
    inputs = np.load(os.path.join(split_dir, "all__inputs.npy"), mmap_mode="r")[:probe_size].astype(np.int32)
    labels = np.load(os.path.join(split_dir, "all__labels.npy"), mmap_mode="r")[:probe_size].astype(np.int32)
    puzzle_identifiers = np.load(
        os.path.join(split_dir, "all__puzzle_identifiers.npy"), mmap_mode="r"
    )[:probe_size].astype(np.int32)

    # Replicate _collate_batch label remap
    if ignore_label_id_in_file is not None:
        labels[labels == ignore_label_id_in_file] = IGNORE_LABEL_ID

    result = {
        "inputs": torch.from_numpy(inputs.copy()),
        "labels": torch.from_numpy(labels.copy()),
        "puzzle_identifiers": torch.from_numpy(puzzle_identifiers.copy()),
    }

    # Compute and log a fingerprint once for probe-fixity verification (Gate 5)
    fingerprint = hashlib.md5(inputs.tobytes()).hexdigest()[:8]
    log.info("Probe loaded: split=%s path=%s n=%d input_hash=%s",
             split, data_path, len(inputs), fingerprint)

    return result, fingerprint


# ---------------------------------------------------------------------------
# Z-probe forward (inference-mode, eval model)
# ---------------------------------------------------------------------------

def _probe_forward(model: torch.nn.Module, probe: Dict[str, torch.Tensor],
                   *, device: str = "cuda", compute_extra: bool = False
                   ) -> Tuple[Dict[str, Any], bool]:
    """
    Run the model on `probe` tensors (already on CPU).
    Returns dict with z_H, z_L (bfloat16 CPU), labels, correct_mask,
    per-step delta list, and exact accuracy.

    Mirrors evaluate()'s ACT loop: initial_carry → forward until all_finish.

    ``device`` exists ONLY so this function is reachable from a CPU unit test.
    The default "cuda" reproduces the historical behaviour byte for byte (the
    three sites below were hard-coded to CUDA), and production never passes it.
    Without the parameter the per-ACT-step capture -- the code that decides
    whether ``z_history[h]`` really is step h's latent and whether it lines up
    with ``preds_traj[h]`` -- would have zero execution coverage, since a green
    suite that stubs this function out would falsely imply it was verified.

    ``compute_extra`` gates the additive sequence/trajectory metrics.  When it
    is False this function does exactly what it did before, allocation for
    allocation.
    """
    batch = {k: v.to(device) for k, v in probe.items()}

    with torch.inference_mode(), torch.device(device):
        carry = model.initial_carry(batch)  # type: ignore

    z_history: list[Tuple[torch.Tensor, torch.Tensor]] = []
    # Separate, independently-read latent trajectory for the NEW metrics only.
    # See the arch-coverage comment inside the loop.
    extra_z_traj: list[Optional[torch.Tensor]] = []
    extra_latent_is_z_h = True

    with torch.inference_mode():
        # Capture initial z (after reset on first step)
        while True:
            carry, _loss, metrics, _preds, all_finish = model(
                carry=carry, batch=batch, return_keys=[]
            )
            # Grab z after this step
            try:
                z_H = carry.inner_carry.z_H.detach().float()
                z_L = carry.inner_carry.z_L.detach().float()
                z_history.append((z_H, z_L))
            except AttributeError:
                # Fallback for models that don't have inner_carry.z_H/z_L
                z_history.append((None, None))

            if compute_extra:
                # ARCH COVERAGE.  The frozen try/except above reads z_H and THEN
                # z_L under a single guard, so any carry missing either field
                # yields (None, None) and throws away the field that WAS there:
                # trm_singlez declares only z_L, transformers_baseline only z_H,
                # trm_hier6 has z_H plus z_L1..z_L5 and no plain z_L.  Three of
                # the four architectures therefore emit no z metrics at all.
                # We deliberately do NOT repair the frozen block -- repairing it
                # would start emitting z/eff_rank for cohorts that have never had
                # it, which changes what a frozen key means for those runs.
                # Instead the NEW capture reads each field with its own getattr
                # (the analysis/pr_recompute.py:207-213 pattern).  When the frozen
                # paired read succeeded we reuse ITS tensor rather than upcasting
                # a second copy: at B=512, S=27, D=512 a duplicate fp32 z_H costs
                # 28 MB per ACT step, i.e. ~450 MB at halt_max_steps=16.
                reused = z_history[-1][0]
                if reused is not None:
                    extra_z_traj.append(reused)
                else:
                    # `carry.inner_carry` itself may be absent -- the frozen
                    # block above swallows that as AttributeError, so this path
                    # must not assume the attribute exists or a carry shape we
                    # have not seen would raise INSIDE the ACT loop, outside any
                    # guard, and take the probe (and the run) down.  Chained
                    # getattr, never attribute access.
                    inner = getattr(carry, "inner_carry", None)
                    raw = getattr(inner, "z_H", None)
                    if isinstance(raw, torch.Tensor):
                        extra_z_traj.append(raw.detach().float())
                    else:
                        raw = getattr(inner, "z_L", None)
                        if isinstance(raw, torch.Tensor):
                            # Flipped ONLY when a z_L was really substituted, so
                            # zseq/latent_is_z_h means what its name says.  A
                            # carry with no latent at all leaves it untouched
                            # rather than claiming "the latent is z_L".
                            extra_latent_is_z_h = False
                            extra_z_traj.append(raw.detach().float())
                        else:
                            extra_z_traj.append(None)

            if all_finish:
                break

    # --- Exact accuracy ---
    labels_gpu = batch["labels"]  # (B, seq_len), already remapped
    mask = labels_gpu != IGNORE_LABEL_ID

    # Rerun last forward to get logits (carry was deleted above)
    with torch.inference_mode(), torch.device(device):
        carry2 = model.initial_carry(batch)  # type: ignore

    preds_traj: list[torch.Tensor] = []

    with torch.inference_mode():
        while True:
            carry2, _loss, metrics2, preds2, all_finish2 = model(
                carry=carry2, batch=batch, return_keys=["preds"]
            )
            # Loop 2 ALREADY passes return_keys=["preds"], so every ACT step's
            # decode is materialised here and then discarded on all but the last
            # iteration.  Capturing it costs one int16 copy per step and adds no
            # forward pass.  We deliberately do NOT add return_keys to loop 1:
            # loop 1 is the code path that produces the frozen z/delta_step_<t>
            # key, and it stays untouched.
            if compute_extra and "preds" in preds2:
                preds_traj.append(preds2["preds"].detach().to(torch.int16).cpu())
            if all_finish2:
                break

    preds = preds2["preds"]  # (B, seq_len)
    is_correct = mask & (preds == labels_gpu)
    loss_counts = mask.sum(-1)
    seq_correct = is_correct.sum(-1) == loss_counts
    halted = carry2.halted
    valid = halted & (loss_counts > 0)

    exact_acc = (valid & seq_correct).sum().item() / max(valid.sum().item(), 1)
    correct_mask = seq_correct.cpu()

    # --- Per-step z deltas ---
    step_deltas: list[float] = []
    for t in range(1, len(z_history)):
        zh_prev, _ = z_history[t - 1]
        zh_cur, _ = z_history[t]
        if zh_prev is None or zh_cur is None:
            continue
        # Mean relative change across batch, averaged over sequence & hidden
        delta = (
            (zh_cur - zh_prev).norm(dim=(-1, -2))
            / (zh_prev.norm(dim=(-1, -2)) + 1e-8)
        ).mean().item()
        step_deltas.append(delta)

    # Final z (last step)
    final_z_H, final_z_L = z_history[-1] if z_history else (None, None)

    # ------------------------------------------------------------------ #
    # Sequence-aware / trajectory metrics (additive, opt-in).
    #
    # Everything above this point is byte-identical to the pre-existing
    # implementation.  The block below cannot alter a frozen value: it only
    # reads tensors, and its whole body is wrapped so that a failure degrades
    # to "no new keys" rather than killing a multi-day run at an eval boundary
    # (log() is called from pretrain.py with no try/except at either end).
    # ------------------------------------------------------------------ #
    extra: Dict[str, Any] = {}
    preds_traj_stacked: Optional[torch.Tensor] = None
    if compute_extra:
        model_is_training = bool(getattr(model, "training", False))
        # Status keys are built BEFORE the try and are therefore emitted even
        # when the reduction raises.  Without them, a genuine bug in the new
        # code is indistinguishable in wandb from (a) an architecture with no
        # usable latent, (b) the feature being switched off, and (c) a probe
        # that is not permutation-shaped -- all four collapse to "keys absent".
        extra = {
            "zseq/status_ok": 0.0,
            # Observed trajectory length, NOT config.halt_max_steps: the two can
            # disagree and only the observed one describes the emitted keys.
            "zseq/n_act_steps": float(len(z_history)),
            "zseq/has_trajectory": 1.0 if len(z_history) > 1 else 0.0,
            "zseq/latent_is_z_h": 1.0 if extra_latent_is_z_h else 0.0,
            "zseq/model_in_eval_mode": 0.0 if model_is_training else 1.0,
        }
        try:
            if preds_traj and len(preds_traj) == len(z_history):
                preds_traj_stacked = torch.stack(preds_traj, dim=0)
            extra.update(_extra_step_metrics(
                z_traj=extra_z_traj,
                preds_traj=preds_traj,
                inputs=batch["inputs"],
                labels=batch["labels"],
                model_is_training=model_is_training,
            ))
            # status_ok stays 0 in train mode: z comes from loop 1 and preds from
            # loop 2, and they are only guaranteed to line up because eval-mode
            # halting is deterministic and batch-synchronous
            # (trm.py:275 gates adaptive halting on self.training).
            extra["zseq/status_ok"] = 0.0 if model_is_training else 1.0
        except Exception:
            log.exception(
                "z_logging: sequence/trajectory metrics failed; "
                "frozen keys are unaffected and training continues")

    return {
        "z_H": final_z_H.to(torch.bfloat16).cpu() if final_z_H is not None else None,
        "z_L": final_z_L.to(torch.bfloat16).cpu() if final_z_L is not None else None,
        "labels": batch["labels"].cpu(),
        "correct_mask": correct_mask,
        "step_deltas": step_deltas,
        "exact_acc": exact_acc,
        # --- additive; absent-by-default consumers use .get() ---
        "extra_metrics": extra,
        "preds_traj": preds_traj_stacked,
    }


# ---------------------------------------------------------------------------
# PCA helpers (numpy only)
# ---------------------------------------------------------------------------

def _mean_pool_z(z: torch.Tensor) -> np.ndarray:
    """z: (B, S, D) → (B, D) via mean-pool over sequence dimension."""
    return z.float().mean(dim=1).numpy()


def _pca(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin PCA over rows of X (B, D).
    Returns (eigenvalues_descending, eigenvectors shape (D, k)).
    """
    X_centered = X - X.mean(axis=0, keepdims=True)
    cov = (X_centered.T @ X_centered) / max(len(X) - 1, 1)
    # Symmetric eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order — reverse to descending
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


def _effective_rank(eigenvalues: np.ndarray) -> float:
    """Participation ratio: (sum lambda)^2 / sum lambda^2."""
    lam = np.maximum(eigenvalues, 0.0)
    s1 = lam.sum()
    s2 = (lam ** 2).sum()
    if s2 < 1e-30:
        return 1.0
    return float(s1 ** 2 / s2)


def _pca_top2_var(eigenvalues: np.ndarray) -> float:
    """Fraction of variance explained by top 2 PCs."""
    lam = np.maximum(eigenvalues, 0.0)
    total = lam.sum()
    if total < 1e-30:
        return 0.0
    return float(lam[:2].sum() / total)


# ===========================================================================
# SEQUENCE-AWARE / TRAJECTORY METRICS
#
# Everything from here to the "Scatter plot helper" banner is new, additive and
# pure: module-level functions over plain CPU-or-GPU tensors and ndarrays, with
# no CUDA assumption, no model, no wandb and no filesystem.  That is deliberate
# -- _probe_forward above is unconditionally device-bound, so any math left
# inside it would be unreachable from a CPU unit test.
# ===========================================================================

#: Namespaces the new metrics are allowed to write into.  ZDynamicsLogger.log
#: filters on this prefix set AND refuses to overwrite an existing key, so a
#: mis-named new metric can never clobber z/eff_rank, z/pca_top2_var,
#: z/mean_norm, z/delta_step_<t>, probe/* or phase/*.  Two independent locks.
_SEQ_KEY_NAMESPACES = ("zseq", "zmi", "ztau", "zperm")

#: Dimension of the PCA subspace the MI decoder sees.  The decoder is CROSS-FIT,
#: so the sample regime that sets the estimator's bias is the FOLD, not the
#: probe: at B=512 each fit sees B/2 = 256 rows, i.e. with C=10 classes only
#: ~25.6 samples per class in 32 dims -- half what the probe size suggests.
#: That is the regime behind the measured -1.10 bit floor documented in the
#: estimator note below, hence r is logged as zmi/subspace_dim.
_MI_SUBSPACE_DIM = 32
#: Label-shuffle repeats for the null.  >1 so the null's own sampling noise is
#: not read as trajectory structure.
_MI_NULL_REPEATS = 4
#: Fixed seed: the fold split and the null shuffles must be identical across
#: evals or the MI trajectory would move for reasons unrelated to the model.
_MI_SEED = 20260806
#: Relative ridge on the pooled within-class covariance (conditioning only).
_MI_RIDGE = 1e-6
#: Shrinkage of the within-class covariance toward the scaled identity.  This is
#: a property of the decoder q, and Barber-Agakov holds for any q, so it affects
#: only how tight the bound is -- never its validity.  See the long note in
#: _heldout_decode_ce_bits.
_MI_SHRINKAGE = 0.10
#: Minimum training rows per retained PCA dimension.  Caps the subspace so the
#: shared covariance is estimable from one fold; at B=512 this leaves r=32.
_MI_ROWS_PER_DIM = 8
#: Largest exponent j searched when auto-detecting k from (inputs, labels).
_K_SEARCH_MAX = 64
#: Belt-and-braces cap on the size of the tau (h, j) agreement grid.  Never
#: fires for n=10 (Landau g(10)=30) but bounds the key count for a future n.
_TAU_MAX_EXPONENT = 64
#: |pooled PR at h=T - logged z/eff_rank| / z/eff_rank above this is warned
#: about.  Logged, never asserted in-process: an assertion here would be
#: precisely the "new code aborts a multi-day run" failure mode.
_LEGACY_PR_RELDIFF_WARN = 1e-3


def _stepkey(h: int) -> str:
    """Zero-padded 1-based ACT-step index.

    Zero padding is not cosmetic: with a bare f"{h}" wandb and every downstream
    extractor sort step_10 before step_2, silently scrambling the trajectory in
    any plot built by iterating sorted key names.
    """
    return f"{h:02d}"


# ---------------------------------------------------------------------------
# Participation ratio, decomposition-free
# ---------------------------------------------------------------------------

def _pr_from_gram(G: torch.Tensor) -> torch.Tensor:
    """
    Participation ratio of a symmetric Gram/covariance matrix, with NO
    eigendecomposition.

    PR = (sum lam)^2 / sum(lam^2).  For symmetric G, sum(lam) = tr(G) and
    sum(lam^2) = tr(G^2) = ||G||_F^2, so

        PR = tr(G)^2 / ||G||_F^2

    exactly.  This matters for two reasons:

    1. COST.  The alternative is np.linalg.eigh on a 512x512 matrix (~30 ms
       each) once per ACT step per PR variant; at halt_max_steps=16 that is
       seconds of stall inside bench.event_span("zprobe"), immediately before
       the phase-transition forced checkpoint.  The trace form is ~0.1 ms and
       removes eigendecomposition (and its LinAlgError) from the training loop.
    2. DIMENSION.  PR depends only on the nonzero spectrum, and the nonzero
       spectra of Xc^T Xc (F x F) and Xc Xc^T (B x B) coincide.  So the EXACT
       full-sequence PR of a (512, 13824) state is a 512x512 problem; the
       13824^2 covariance (~1.5 GB) is never formed.  This is not an
       approximation.

    Note also that PR is invariant to a positive rescaling of G, so whether the
    Gram carries a 1/(B-1) factor does not affect the returned value.

    DEFINITIONAL DIFFERENCE FROM _effective_rank, deliberate and documented:
    _effective_rank clips negative eigenvalues to zero first; the trace form
    cannot, because it never forms eigenvalues.  In exact arithmetic G is PSD
    and the two agree; in float32 round-off can produce eigenvalues of order
    -1e-7 * lam_max, which enter tr(G) linearly and ||G||_F^2 quadratically.
    The observed relative disagreement is ~1e-7.  Do NOT "unify" the two forms:
    the single legacy-parity key below must keep the clipping chain verbatim.
    """
    G = G.double()
    tr = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1)
    fro2 = (G * G).sum(dim=(-2, -1))
    # Mirrors _effective_rank's s2 < 1e-30 guard so an exactly-constant state
    # returns 1.0 rather than NaN.  CAVEAT: that rescue makes PR == 1.0
    # ambiguous -- it means either "genuinely one effective dimension" or "no
    # variance at all".  Read it together with the dispersion keys below.
    out = torch.where(fro2 < 1e-30, torch.ones_like(fro2), tr * tr / fro2)
    return out  # 0-dim for a (B,B) input, shape (...) for a batched one


def _position_grams(Z: torch.Tensor) -> torch.Tensor:
    """
    Z: (B, S, D) -> (S, B, B) batch-centered per-position Gram matrices.

    G_p = Zc[:, p, :] @ Zc[:, p, :].T where Zc is centered over the BATCH axis.
    Two useful identities follow and are used below instead of recomputing:
      * the Gram of the flattened state over any subset P of positions is
        sum_{p in P} G_p  (flattening then centering over the batch is the same
        as centering each position over the batch, then concatenating);
      * therefore pr_joint_all, pr_joint_tok and the per-position PRs all come
        from ONE pass.
    """
    Zc = Z.double()
    Zc = Zc - Zc.mean(dim=0, keepdim=True)
    # (S, B, B); cost S*B*B*D, ~3.6 GFLOP at S=27, B=512, D=512.
    return torch.einsum("bpd,cpd->pbc", Zc, Zc)


def _across_position_stats(Z: torch.Tensor) -> Tuple[float, float]:
    """
    Z: (B, S, D). Returns (pr_across_pos, pos_var_frac).

    pr_across_pos: for each example b, center z[b] over the POSITION axis and
    take the participation ratio of that (S, D) block; average over examples.
    Bounded by S-1.  This is precisely and only the structure that mean-pooling
    annihilates: ~1 means the S positions carry near-identical vectors, so
    z/eff_rank loses nothing by pooling; ~S-1 means pooling is averaging S
    genuinely different states.

    pos_var_frac: mean_b ||z[b] - mean_p z[b]||_F^2 divided by the total
    (grand-mean-centered) energy.  THIS COMPANION IS MANDATORY, not decorative.
    pr_across_pos is 1.0 both when the positions are exactly identical (the
    centered block is zero and the 1e-30 guard rescues it to 1.0) and when they
    vary along a single direction with large norm.  Those are opposite
    conclusions and the PR alone cannot separate them; pos_var_frac ~ 0 marks
    the first case.  Never state "PR ~ 1.0 means mean-pooling is lossless"
    without reading pos_var_frac alongside it.
    """
    Zd = Z.double()
    M = Zd - Zd.mean(dim=1, keepdim=True)          # position-centered, (B,S,D)
    G = torch.einsum("bpd,bqd->bpq", M, M)         # (B,S,S), cheap: S is ~27
    pr = _pr_from_gram(G).mean().item()

    pos_energy = (M * M).sum().item()
    total = ((Zd - Zd.mean(dim=(0, 1), keepdim=True)) ** 2).sum().item()
    frac = float(pos_energy / total) if total > 1e-30 else 0.0
    return float(pr), frac


def _pooled_pr_legacy(z: torch.Tensor) -> float:
    """
    The FROZEN recipe applied to one ACT step: bf16 round-trip -> _mean_pool_z
    -> _pca -> _effective_rank, reusing those three functions verbatim.

    EXEMPTION, READ BEFORE "SIMPLIFYING".  Every other participation ratio in
    this file uses the decomposition-free trace form in _pr_from_gram, which is
    ~200x cheaper.  This one must NOT be converted: _effective_rank clips
    negative eigenvalues to zero and the trace form does not, so the two agree
    only to ~1e-7.  The whole point of this function is that at h = T it
    reproduces the frozen z/eff_rank BITWISE, which is a second independent lock
    on the "do not change the meaning of z/eff_rank" constraint (the first being
    analysis/pr_recompute.py's own validation gate) and is asserted with `==`
    rather than a tolerance in the test suite.  Converting it to the trace form
    silently destroys that lock.

    The .to(bfloat16) is likewise load-bearing: _probe_forward returns
    z_H.to(torch.bfloat16).cpu() and log() pools THAT, so the round-trip has to
    happen here too.  It is exact when z already came from a bf16 carry
    (analysis/pr_recompute.py:220-222 replicates the same round-trip
    deliberately, for the same reason).
    """
    pooled = _mean_pool_z(z.to(torch.bfloat16).cpu())
    eigenvalues, _ = _pca(pooled)
    return _effective_rank(eigenvalues)


# ---------------------------------------------------------------------------
# Mutual information: held-out linear decode (Barber-Agakov) lower bound
# ---------------------------------------------------------------------------
#
# ESTIMAND AND ITS LIMITATIONS -- stated here, and surfaced in the key names and
# in adjacent scalar keys rather than left in a comment.
#
# WHAT IS NOT ESTIMATED.  I(input identity ; z^(h)) as literally pre-registered
# is not informative on this probe.  The probe holds B distinct permutations and
# the encoder is deterministic, so z is GENERICALLY injective in the input and
# the plug-in estimate is a.s. log2(B) = 9 bits at every ACT step, carrying no
# discriminative content.  Note the careful wording: determinism does NOT imply
# injectivity, and the counterexample lives in this very model -- after
# reset_carry (trm.py:190-194) a single H_init vector is broadcast across every
# row and every position, so a collapsed or untrained model really can map
# distinct inputs to coincident (bf16) latents, in which case the raw quantity
# is below 9 bits.  "9 bits exactly" is not a theorem; "generically injective,
# hence uninformative" is the claim.
#
# WHAT IS ESTIMATED.  A per-symbol surrogate I(sigma(i) ; z^(h)), averaged over
# the n token positions i.  The substitution is disclosed in the key name
# ("sym_") and is NOT a bound on the pre-registered quantity in either direction.
#
# WHY THIS ESTIMATOR.  Barber-Agakov: for ANY conditional model q,
#     I(X;Z) >= H(X) - E[-log q(X|Z)].
# The bound holds for any q, so evaluating the cross-entropy on data the model q
# was not fitted on gives an estimate whose SIGN is guaranteed in expectation.
# That is the property a plug-in Gaussian MI does not have: the usual
# 0.5*(logdet Sigma_T - logdet Sigma_W) substitutes a Gaussian for the marginal
# entropy (biasing UP, since the Gaussian maximises entropy at fixed covariance)
# while the homoscedastic conditional biases DOWN -- it is a bound in NEITHER
# direction, and a label-shuffle null cannot repair that because the null only
# measures bias at I = 0.
#
# THE BIAS THAT REMAINS, and how it is surfaced:
#  * It is a LOWER bound, so it under-reports: q is a shared-covariance Gaussian
#    (a linear decoder) over an r=32 PCA subspace and is blind to class structure
#    that is nonlinear or lives outside that subspace.  The value is a
#    linear-decodability floor, not the information content.
#  * The estimate is NOT clipped at zero.  Clipping would hide the floor
#    described next, which is the single most misreadable property of this
#    instrument.
#
# THE INSTRUMENT'S TWO ENDPOINTS ARE BOTH MEASURED, AND NEITHER IS WHERE A
# READER WOULD ASSUME.  This is the honesty-critical paragraph; the naive
# reading interval [0, log2(n)] is wrong at BOTH ends, by a third of the range.
#
# PROVENANCE OF THE NUMBERS BELOW: measured 2026-08-06 at B=512, S=27, D=512,
# n=10 with _MI_SUBSPACE_DIM=32, _MI_ROWS_PER_DIM=8, _MI_SHRINKAGE=0.10.  They
# are illustrative and WILL go stale if any of those constants change -- which
# is precisely why both endpoints are also recomputed every eval and logged
# (sym_decode_null_bits_step_<hh> and sym_decode_saturated_bits).  Trust the
# logged keys over these literals; the literals only say what to expect.
#
#   FLOOR.  The shuffle null is NOT ~0.  Measured at the exact production
#   configuration (B=512, S=27, D=512, n=10) it sits at -1.10 bits, i.e. 33% of
#   the 3.32-bit ceiling BELOW zero, and it is stable to +-0.02 across signal
#   levels, so it is a bias floor and not sampling noise.  Cause: the held-out
#   cross-entropy of an r=32 shared-covariance decoder fitted on B/2=256 rows
#   exceeds H(X) even when the labels carry no information, because the fitted
#   class means are noise that the held-out fold does not share.  CONSEQUENCE
#   FOR READING THE CURVE: a step reporting -1.0 bits is AT THE FLOOR, i.e.
#   indistinguishable from "no linearly decodable information" -- it does NOT
#   mean the latent is worse than the marginal.  Only lb materially ABOVE the
#   null sibling logged at the same wandb step is evidence of anything.  The
#   floor is B-dependent (measured: -1.44 at B=128, -1.11 at B=512, -0.47 at
#   B=1024), which is why it is measured every eval instead of documented as a
#   constant.
#
#   CEILING.  log2(n) = 3.32 is the mathematical ceiling, NOT the attainable
#   one.  A latent from which the symbol is EXACTLY and deterministically
#   recoverable -- true I = log2(10) = 3.3219 -- reads +2.85 at this probe size
#   (measured; held-out nearest-code accuracy 1.000 confirms the truth is the
#   ceiling).  So ~0.47 bits of any gap to 3.32 is pure finite-sample bias and
#   is NOT model shortfall.  Rather than document 2.85 as a constant -- it moves
#   with B, n and r -- the identical estimator is run every eval on exactly such
#   a synthetic latent and the result is logged as
#   zmi/sym_decode_saturated_bits.  READ lb ON THE INTERVAL
#   [sym_decode_null_bits, sym_decode_saturated_bits], never [0, log2(n)].
#
#   Both endpoint keys are emitted unconditionally alongside every lb, so the
#   value cannot be plotted without them.  A materially POSITIVE null remains a
#   live alarm that the fold split leaked and the bound guarantee is void; that
#   alarm is CHECKED IN CODE (see _mi_alarm_check) and warned about, not merely
#   described here.
#
#  * The reported value is dominated by probe size, so it is comparable across
#    ACT steps and across arms ONLY at equal B, n and fold seed.  Measured with
#    an identically informative latent, varying only B: -0.05 bits at B=128
#    against +2.84 at B=512 -- the entire dynamic range, from probe size alone,
#    because r itself is B-dependent (see _mi_step_bits).  B and r are logged as
#    zmi/probe_n and zmi/subspace_dim so the condition is checkable after the
#    fact; z_probe_size is 512 in every current config.
#  * Hard ceilings are logged as static scalars every eval so the estimate is
#    never read without them.  zmi/probe_ceiling_bits = log2(B) = 9.0 is the cap
#    on the identity-MI quantity that is deliberately NOT emitted (see WHAT IS
#    NOT ESTIMATED above) -- it is NOT the ceiling of any logged value, and the
#    ceiling for the logged lb is zmi/sym_ceiling_bits = log2(n) = 3.32.  It is
#    kept because it is the number that makes the pre-registered quantity's
#    infeasibility legible: it sits BELOW the task's own log2(10!) = 21.79 bits
#    (zmi/perm_entropy_bits), so this instrument structurally cannot measure the
#    information content of a permutation at probe_size 512.
#
# REJECTED, and why, so nobody "simplifies" it back:
#  * A binned MI on the leading principal component.  The leading PC ROTATES
#    between ACT steps, so the h-dependence of any PC1-indexed statistic is
#    partly an artefact of which classes happen to align with the dominant
#    variance direction at step h -- corrupting exactly the depth curve the
#    metric exists to reveal.
#  * KSG/Kraskov: the discrete side has one sample per class (inputs are unique),
#    so the discrete-continuous variant is undefined.
#  * Raw CCA on one-hot inputs: canonical correlations saturate at 1.0 as the
#    feature dimension approaches B.


def _mi_scores_from_gram(G: torch.Tensor, r: int) -> np.ndarray:
    """
    Top-r PCA scores from the B x B centered Gram, as float64 (B, r) numpy.

    scores = U[:, :r] * sqrt(lam[:r]) with (lam, U) = eigh(G).  These are the
    true PCA scores up to one global factor of sqrt(B-1) (G here carries no
    1/(B-1)); that factor is an isotropic rescaling and the shared-covariance
    linear decoder below is invariant to any invertible linear map of its
    inputs, so it is left out rather than silently introducing a unit contract.

    This is the ONE place an eigendecomposition is still needed -- the PR family
    uses the trace form and needs no eigenvectors at all.  It is run in float32:
    a 512x512 eigh costs 0.68 s in float64 against 0.28 s in float32 on CPU
    (milliseconds on the probe's GPU either way), and the consumer is a
    shrunk linear decoder that is invariant to any invertible linear map of its
    inputs, so the extra mantissa buys nothing.  The PR family, whose trace form
    IS sensitive to round-off, keeps float64.
    """
    lam, U = torch.linalg.eigh(G.float())
    lam = torch.flip(lam, dims=[0])
    U = torch.flip(U, dims=[1])
    r = int(min(r, U.shape[1]))
    scale = torch.sqrt(torch.clamp(lam[:r], min=0.0))
    return (U[:, :r] * scale).cpu().numpy().astype(np.float64)


def _heldout_decode_ce_bits(scores: np.ndarray, classes: np.ndarray,
                            n_classes: int, folds: np.ndarray,
                            shrinkage: float = _MI_SHRINKAGE) -> float:
    """
    Cross-fit cross-entropy E[-log2 q(x|z)] in bits of a shared-covariance
    Gaussian (LDA) decoder: fit on one fold, score the other, average.

    Returns the held-out CE.  The caller forms the Barber-Agakov bound
    H(X) - CE.  Because q's LABELS are never fitted on the rows it scores, the
    bound's sign is guaranteed in expectation, and averaging the two folds
    averages two individually valid bounds.

    ONE HONEST QUALIFICATION on that guarantee: q is transductive in z.  The
    top-r PCA basis handed in as `scores` is computed in _mi_step_bits from the
    full B x B Gram -- including the held-out rows -- before this function
    splits the folds, so q = (PCA fitted on all B rows) o (LDA fitted on one
    fold).  Only the second stage is held out.  The leak is LABEL-FREE, which is
    why it is tolerable and why the shuffle null empirically controls it (the
    null is negative at every B tested), but "q never sees these rows" would be
    false and the weaker "q never sees these LABELS" is what actually holds.

    REGULARISATION IS FREE HERE, which is worth stating because it looks like a
    fudge factor.  Barber-Agakov holds for ANY conditional model q, so shrinking
    the within-class covariance can only change how TIGHT the bound is, never
    whether it is a bound.  Without shrinkage an r-dimensional shared covariance
    estimated from a few hundred residuals produces confidently-wrong held-out
    predictions and the "bound" collapses to something like -100 bits: still
    valid, still useless.  Shrinking toward the scaled identity keeps it
    informative.  The subspace dimension is likewise capped by the fold size in
    _mi_step_bits for the same reason.
    """
    n, r = scores.shape
    total_ll = 0.0
    total_n = 0
    for held in (0, 1):
        tr_idx = np.nonzero(folds != held)[0]
        te_idx = np.nonzero(folds == held)[0]
        if len(tr_idx) < n_classes + 2 or len(te_idx) == 0:
            continue
        Ztr, ytr = scores[tr_idx], classes[tr_idx]
        counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
        grand = Ztr.mean(axis=0)

        mus = np.empty((n_classes, r), dtype=np.float64)
        resid = np.empty_like(Ztr)
        n_eff = 0
        for c in range(n_classes):
            sel = ytr == c
            if counts[c] >= 1:
                mus[c] = Ztr[sel].mean(axis=0)
                resid[sel] = Ztr[sel] - mus[c]
                n_eff += 1
            else:
                # Class unseen in this fold: fall back to the grand mean so it
                # is still scoreable (with a small Laplace prior) instead of
                # producing a -inf log-likelihood for a held-out row.
                mus[c] = grand
        dof = max(len(tr_idx) - n_eff, 1)
        Sw = (resid.T @ resid) / dof
        avg_var = float(np.trace(Sw)) / max(r, 1)
        Sw *= (1.0 - shrinkage)
        Sw[np.diag_indices(r)] += shrinkage * avg_var + _MI_RIDGE * avg_var + 1e-300

        # Laplace-smoothed priors: a zero prior would make log pi_c = -inf.
        pri = (counts + 0.5) / (counts.sum() + 0.5 * n_classes)
        log_pri = np.log(pri)

        try:
            Sinv = np.linalg.inv(Sw)
        except np.linalg.LinAlgError:
            return float("nan")

        # LINEAR-DISCRIMINANT FORM, not a Mahalanobis distance per class.  With a
        # SHARED covariance the quadratic term z^T Sinv z is class-independent
        # and cancels in the softmax, so
        #     logit_c = log pi_c + (z Sinv) . mu_c - 0.5 mu_c^T Sinv mu_c.
        # This turns each fit into three small matmuls instead of a per-class
        # triangular solve over an (m, C, r) difference tensor.  It matters:
        # this function runs n_pos * (1 + null_repeats) * 2 folds times per ACT
        # step, i.e. ~1600 times per eval at n=10, T=16, all inside
        # bench.event_span("zprobe") on the training critical path.
        Zte = scores[te_idx]
        A = Zte @ Sinv                                       # (m, r)
        term_lin = A @ mus.T                                 # (m, C)
        term_const = 0.5 * ((mus @ Sinv) * mus).sum(axis=1)   # (C,)
        logits = log_pri[None, :] + term_lin - term_const[None, :]

        mx = logits.max(axis=1, keepdims=True)
        logZ = mx[:, 0] + np.log(np.exp(logits - mx).sum(axis=1))
        ll = logits[np.arange(len(te_idx)), classes[te_idx]] - logZ
        total_ll += float(ll.sum())
        total_n += len(te_idx)

    if total_n == 0:
        return float("nan")
    return float(-(total_ll / total_n) / np.log(2.0))


def _entropy_bits(classes: np.ndarray, n_classes: int) -> float:
    """Empirical Shannon entropy of a label vector, in bits."""
    counts = np.bincount(classes, minlength=n_classes).astype(np.float64)
    p = counts / max(counts.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _mi_step_bits(G_tok: torch.Tensor, sigma: np.ndarray,
                  n_classes: int, step_index: int) -> Tuple[float, float]:
    """
    (lower_bound_bits, null_bits) for one ACT step, averaged over token
    positions.  See the long estimator note above for what these mean.
    """
    B = G_tok.shape[0]
    # Cap the subspace by the FOLD size, not the probe size: the decoder's
    # shared covariance is estimated inside one fold (B/2 rows), so keeping
    # r=32 at a small B would produce a valid but vacuous bound.  At B=512 the
    # cap is inactive and r=32, which is the regime zmi/subspace_dim reports.
    r = int(min(_MI_SUBSPACE_DIM, B - 1, max(2, (B // 2) // _MI_ROWS_PER_DIM)))
    if r < 2:
        return float("nan"), float("nan")
    scores = _mi_scores_from_gram(G_tok, r)

    # Fold assignment by a fixed-seed permutation, NOT by row parity: the probe
    # is the first probe_size rows of the file in their stored order, which may
    # carry structure aligned with parity.  Fixed seed => identical split at
    # every eval, so the MI trajectory moves only with the model.
    rng = np.random.RandomState(_MI_SEED)
    folds = np.zeros(B, dtype=np.int64)
    folds[rng.permutation(B)[: B // 2]] = 1

    lbs, nulls = [], []
    n_pos = sigma.shape[1]
    for i in range(n_pos):
        cls = sigma[:, i].astype(np.int64)
        # Shuffling permutes the label vector, so the empirical class multiset
        # and hence H(X) are identical for the real and the null runs.  Computed
        # once and reused for both.
        H = _entropy_bits(cls, n_classes)
        ce = _heldout_decode_ce_bits(scores, cls, n_classes, folds)
        lbs.append(H - ce)
        # Null: identical estimator, labels shuffled, true MI is 0.  On held-out
        # folds this must land at or below 0; a materially positive null means
        # the split leaked and the bound guarantee is void.  Seeded on
        # (step, position, repeat) so it is reproducible and independent across
        # cells without being identical across them.
        for rep in range(_MI_NULL_REPEATS):
            rs = np.random.RandomState(_MI_SEED + 1009 * step_index
                                       + 31 * i + rep)
            perm_cls = cls[rs.permutation(B)]
            ce0 = _heldout_decode_ce_bits(scores, perm_cls, n_classes, folds)
            nulls.append(H - ce0)

    return float(np.nanmean(lbs)), float(np.nanmean(nulls))


def _mi_saturated_bits(sigma: np.ndarray, n_classes: int, hidden_size: int,
                       seed: int = _MI_SEED) -> float:
    """
    The ATTAINABLE ceiling: what this estimator returns on a latent from which
    the symbol is exactly and deterministically recoverable.

    WHY THIS EXISTS.  log2(n) = 3.32 is the mathematical ceiling of the estimand
    but NOT of the instrument.  Position i of the synthetic latent below is
    literally codes[sigma[:, i]] -- a noiseless lookup table, so the true
    I(sigma(i); z) is exactly H(sigma(i)) = log2(n) and a held-out
    nearest-code decode is 100% accurate -- yet the estimator reports ~2.85
    bits at B=512, n=10.  The ~0.47-bit shortfall is entirely finite-sample
    bias of the cross-fit r-dimensional decoder, and without this key it is
    indistinguishable on the dashboard from real model shortfall.

    It is MEASURED rather than documented as a constant because it moves with
    B, n and r (r is itself B-dependent, see _mi_step_bits), so a hardcoded 2.85
    would silently become wrong the moment z_probe_size changed.

    Together with the shuffle null this gives both endpoints of the instrument
    at the exact configuration in play, so lb may be read on
    [null, saturated] instead of the wrong-at-both-ends [0, log2(n)].

    Deliberately reuses the SAME code path as the real measurement (same
    _position_grams -> _mi_scores_from_gram -> _heldout_decode_ce_bits chain,
    same fold seed, same r rule): a calibration computed by a different path
    would not calibrate this estimator.  Costs one extra Gram and n_pos*2
    decoder fits per eval -- ~1/5 of a single ACT step's MI cost, charged once,
    not per step.

    The codes are drawn from a FIXED seed independent of the model, so the key
    is constant for a given (B, n, D) and any movement in it across evals of one
    run is itself a bug signal.
    """
    B, n_pos = sigma.shape
    if B < 4 or n_pos < 1 or hidden_size < 1:
        return float("nan")
    r = int(min(_MI_SUBSPACE_DIM, B - 1, max(2, (B // 2) // _MI_ROWS_PER_DIM)))
    if r < 2:
        return float("nan")

    rng = np.random.RandomState(seed + 7717)
    codes = rng.randn(n_classes, hidden_size).astype(np.float64)
    Z = np.empty((B, n_pos, hidden_size), dtype=np.float64)
    for i in range(n_pos):
        Z[:, i, :] = codes[sigma[:, i].astype(np.int64)]

    G_tok = _position_grams(torch.from_numpy(Z)).sum(dim=0)
    scores = _mi_scores_from_gram(G_tok, r)

    # Same fold construction as _mi_step_bits, so the two numbers are produced
    # under an identical split and are directly comparable.
    frng = np.random.RandomState(_MI_SEED)
    folds = np.zeros(B, dtype=np.int64)
    folds[frng.permutation(B)[: B // 2]] = 1

    lbs = []
    for i in range(n_pos):
        cls = sigma[:, i].astype(np.int64)
        H = _entropy_bits(cls, n_classes)
        lbs.append(H - _heldout_decode_ce_bits(scores, cls, n_classes, folds))
    return float(np.nanmean(lbs))


#: A shuffle null above this many bits is treated as a fold leak.  The null is a
#: bound on ZERO, so any materially positive value voids the sign guarantee.
#: Set above the null's own sampling jitter (measured +-0.02 bits at B=512
#: across signal levels) with a wide margin, so this warns on a real leak rather
#: than on noise.
_MI_NULL_LEAK_WARN_BITS = 0.10


def _mi_alarm_check(nulls_by_step: Dict[str, float], split: str) -> None:
    """
    Actually CHECK the fold-leak alarm the estimator note advertises.

    The null being "a live alarm" is worthless if nothing ever looks at it: a
    positive null in production means the fold split leaked and every lb on that
    curve is no longer a bound in any direction, and silently logging the number
    would leave that discovery to whoever happens to plot the null months later.
    Warn, never raise -- an exception here would abort a multi-day training run
    over a diagnostic, which is exactly the failure mode this module's
    containment design exists to prevent.
    """
    bad = {k: v for k, v in nulls_by_step.items()
           if np.isfinite(v) and v > _MI_NULL_LEAK_WARN_BITS}
    if bad:
        worst = max(bad, key=lambda k: bad[k])
        log.warning(
            "z_logging: MI fold-leak alarm -- zmi/%s shuffle null is positive "
            "at %d/%d ACT step(s) (worst: step %s = %+.3f bits > %.2f). The "
            "labels carry no information there, so the null must be <= 0; a "
            "positive null means the fold split leaked and sym_decode_lb_bits "
            "is NOT a valid bound on this run.",
            split, len(bad), len(nulls_by_step), worst, bad[worst],
            _MI_NULL_LEAK_WARN_BITS)


# ---------------------------------------------------------------------------
# Permutation algebra (numpy, batched over probe rows)
# ---------------------------------------------------------------------------

def _perm_compose(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(A o B)(i) = A[B[i]], batched over rows: (N, n) x (N, n) -> (N, n)."""
    return np.take_along_axis(A, B, axis=1)


def _perm_power(sigma: np.ndarray, j: int) -> np.ndarray:
    """sigma^j by binary exponentiation.  j must be >= 0; j=0 is the identity."""
    if j < 0:
        raise ValueError(f"_perm_power requires j >= 0, got {j}")
    N, n = sigma.shape
    result = np.tile(np.arange(n, dtype=sigma.dtype), (N, 1))
    base = sigma
    while j:
        if j & 1:
            result = _perm_compose(base, result)
        j >>= 1
        if j:
            base = _perm_compose(base, base)
    return result


def _landau_g(n: int) -> int:
    """
    Landau's function g(n): the maximum order of an element of S_n, i.e. the
    largest lcm over all partitions of n.

    Used to bound the tau exponent grid.  g(10) = 30 (cycle type 2+3+5), so the
    contiguous grid j = 0..30 is EXHAUSTIVE over the distinct powers of ANY
    10-element permutation -- no hypothesis can be missed and none is
    duplicated.  The alternative (a doubling grid reaching 2^15 = 32768) is both
    larger and provably redundant, since those columns are aliases of
    j mod ord(sigma) and ord is row-dependent, so the alias cannot even be
    collapsed at the key level.

    The maximising partition uses distinct PRIME POWERS, so the recursion is
    over primes (one exponent per prime), not over arbitrary parts: a naive
    "max over q <= m of q * g(m-q)" overcounts, e.g. it would give g(6) = 9 by
    using 3 twice, when lcm(3,3) = 3 and the true g(6) = 6.
    """
    if n <= 0:
        return 1
    primes = [p for p in range(2, n + 1)
              if all(p % d for d in range(2, int(p ** 0.5) + 1))]

    best = 1

    def rec(idx: int, rem: int, acc: int) -> None:
        nonlocal best
        if acc > best:
            best = acc
        if idx >= len(primes):
            return
        rec(idx + 1, rem, acc)            # skip this prime entirely
        p, q = primes[idx], primes[idx]
        while q <= rem:
            rec(idx + 1, rem - q, acc * q)
            q *= p

    rec(0, n, 1)
    return best


def _detect_permutation_probe(inputs: np.ndarray) -> Optional[int]:
    """
    Return n if every probe row encodes a permutation of {0..n-1} in the
    dataset's convention (PAD=0, inp[:n] = sigma + 1), else None.

    n is taken from the leading run of non-PAD entries in row 0 and then
    VERIFIED on every row -- a silent mis-detection would poison every tau and
    perm number without changing a single shape.
    """
    if inputs.ndim != 2 or inputs.shape[0] == 0:
        return None
    row0 = inputs[0]
    nz = np.nonzero(row0 == 0)[0]
    n = int(nz[0]) if len(nz) else int(inputs.shape[1])
    if n < 2:
        return None
    head = inputs[:, :n].astype(np.int64) - 1
    if head.min() < 0 or head.max() != n - 1:
        return None
    # Every row must be a bijection on {0..n-1}.
    srt = np.sort(head, axis=1)
    if not np.array_equal(srt, np.tile(np.arange(n), (len(head), 1))):
        return None
    # Everything past n must be PAD on every row.
    if inputs.shape[1] > n and np.any(inputs[:, n:] != 0):
        return None
    return n


def _detect_k(sigma: np.ndarray, target: np.ndarray,
              kmax: int = _K_SEARCH_MAX) -> Optional[int]:
    """
    Smallest j in 1..kmax with sigma^j == target on EVERY probe row, else None.

    This is why no config field is needed to interpret the task: k is a property
    of the probe, read off the probe.  Returning None (never a sentinel like -1)
    keeps a failed detection out of the exponent grid -- a negative exponent
    would make _perm_power raise or loop.
    """
    cur = sigma.copy()
    for j in range(1, kmax + 1):
        if np.array_equal(cur, target):
            return j
        cur = _perm_compose(sigma, cur)
    return None


def _agreement(yhat: np.ndarray, target: np.ndarray) -> float:
    """Mean over rows of the fraction of positions where yhat == target."""
    return float((yhat == target).mean())


def _perm_structure(sigma: np.ndarray, yhat: np.ndarray) -> Tuple[float, float]:
    """
    (valid_perm_rate, cycle_consistent_rate) for one decode.

    valid_perm: yhat[:n] is a bijection on {0..n-1}.
    cycle_consistent: sigma[yhat[i]] == yhat[sigma[i]] for all i, i.e. the
    decode commutes with sigma.  Rows containing an out-of-range value (PAD
    decodes map to -1) are counted as failures rather than indexed.

    NECESSARY BUT NEVER SUFFICIENT.  Every genuine power of sigma commutes with
    sigma, so cycle-consistency is a sound screen -- but the centralizer of
    sigma in S_n is strictly larger than <sigma> for essentially every cycle
    type, so a rate of 1.0 does NOT mean "these are powers of sigma".  Cycle
    type [5,5] gives |centralizer| = 50 against |<sigma>| = 5, a factor of ten;
    [6,4] gives 24 vs 12; [4,3,2,1] gives 24 vs 12.  The key that actually
    identifies the power is ztau/train/best_exp_step_<hh>.

    The two rates are also INDEPENDENT, not nested: a constant map yhat(i) = c
    where sigma(c) = c commutes with sigma while being maximally non-bijective.
    """
    N, n = sigma.shape
    in_range = (yhat >= 0) & (yhat < n)
    row_ok = in_range.all(axis=1)

    srt = np.sort(np.where(in_range, yhat, 0), axis=1)
    is_bij = row_ok & np.all(srt == np.arange(n)[None, :], axis=1)

    safe = np.where(in_range, yhat, 0)
    lhs = np.take_along_axis(sigma, safe, axis=1)     # sigma[yhat[i]]
    rhs = np.take_along_axis(yhat, sigma, axis=1)     # yhat[sigma[i]]
    commutes = row_ok & np.all(lhs == rhs, axis=1)

    return float(is_bij.mean()), float(commutes.mean())


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

#: Total cap on ztau/train/agree_h*_j* keys.  Never fires at n=10, T<=16
#: (16 * 31 = 496) but bounds the wandb row width for a future task shape.
_TAU_MAX_AGREE_KEYS = 1024


def _extra_step_metrics(z_traj: list,
                        preds_traj: list,
                        inputs: torch.Tensor,
                        labels: torch.Tensor,
                        *,
                        model_is_training: bool = False,
                        split: str = "train") -> Dict[str, float]:
    """
    Reduce a per-ACT-step latent trajectory and decode trajectory to scalars.

    Pure: no CUDA assumption, no model, no wandb, no filesystem.  Tensors may
    live on any device; ndarray-producing steps move to CPU explicitly.

    Raises rather than swallowing -- containment lives at the single call site
    in _probe_forward, so the unit tests exercise the real failure behaviour
    instead of passing vacuously after a swallowed exception.

    DEGRADATION.  Each family is independent and is simply omitted when its
    inputs are unavailable, never faked:
      * halt_max_steps == 1 -> every family emits exactly its h=1 keys, no
        *_step_02 key is created, and ztau/discriminable is 0.0.  Note the
        deliberate asymmetry with the frozen z/delta_step_<t>, which emits
        NOTHING at T=1: a delta requires two states, whereas a participation
        ratio, an MI and a decode agreement are all well defined at one state.
        Emitting the single honest observation is not a fabricated trajectory.
      * no usable latent (trm_singlez / transformers_baseline / trm_hier6)
        -> zseq/* and zmi/* omitted; ztau/* and zperm/* still emitted, since
        they need only the decode and the probe inputs.
      * "preds" absent from return_keys -> ztau/* and zperm/* omitted; zseq/*
        and zmi/* unaffected (the puzzle-embedding offset is derived from the
        LABELS width, which is always present, never from the decode width).
      * probe not permutation-shaped, or k not found -> ztau/*, zperm/* and the
        symbol-wise zmi/* omitted.
      * model in train mode -> ztau/* and zperm/* omitted (see below).

    MEASURED COST at the largest configuration this can meet (B=512, S=27,
    D=512, n=10, halt_max_steps=16): 3.6 s per eval on GPU, 14 s on CPU,
    producing 750 keys.  At halt_max_steps=1 -- every historical fig1_*
    checkpoint -- it is ~0.2 s.  Charged once per eval, on the train probe
    only, inside bench.event_span("zprobe").  The dominant terms are the ~1600
    small held-out decoder fits (numpy, CPU-bound wherever the latents live)
    and one 512x512 eigh per ACT step; the participation ratios are nearly free
    because they use the trace form rather than an eigendecomposition.
    The MI saturation calibration adds a measured 0.27 s ONCE per eval (not per
    ACT step) at that shape -- about 1/5 of a single step's MI cost -- and is
    independent of halt_max_steps.
    """
    out: Dict[str, float] = {}

    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    inputs_np = inputs.detach().cpu().numpy().astype(np.int64)
    B, seq_len = labels_np.shape

    n = _detect_permutation_probe(inputs_np)
    sigma = (inputs_np[:, :n] - 1) if n is not None else None
    if n is None:
        log.info("z_logging: probe is not permutation-shaped; "
                 "ztau/* and zperm/* will be omitted")

    # ------------------------------------------------------------------ #
    # (A) Full-sequence participation ratio, per ACT step
    # ------------------------------------------------------------------ #
    mi_nulls: Dict[str, float] = {}
    zs = [z for z in z_traj if z is not None]
    if z_traj and len(zs) == len(z_traj) and B >= 3:
        S = z_traj[0].shape[1]
        # The puzzle-embedding offset is DERIVED, never hardcoded: z is
        # (B, puzzle_emb_len + seq_len, D) and lm_head slices [:, P:] before
        # decoding (trm.py:220), so positions 0..P-1 never reach a logit.
        # Deriving it from the labels width (always present) rather than the
        # decode width keeps the PR family alive when "preds" is unavailable.
        # Getting this offset wrong changes no shape and silently poisons every
        # number, which is exactly why it is computed rather than assumed.
        P = S - seq_len
        if P < 0:
            raise ValueError(
                f"latent seq axis {S} is shorter than the label width {seq_len}")

        out[f"zseq/{split}/pr_ceiling"] = float(min(B - 1, seq_len * z_traj[0].shape[2]))
        out[f"zseq/{split}/probe_n"] = float(B)

        for h, Z in enumerate(z_traj, start=1):
            sk = _stepkey(h)
            G = _position_grams(Z)                  # (S, B, B)
            G_all = G.sum(dim=0)
            G_tok = G[P:].sum(dim=0)

            # PRIMARY (A): PR of the UNPOOLED token-slice joint state, i.e. of
            # z[:, P:, :] flattened to (B, seq_len*D).  The decoder reads
            # exactly this slice, and a permutation is a JOINT object across
            # positions (a row is correct only if every position is right at
            # once), so the capacity that matters is the effective dimension of
            # the joint vector -- not of any per-position or pooled summary.
            out[f"zseq/{split}/pr_joint_tok_step_{sk}"] = float(_pr_from_gram(G_tok))
            # Same over all S positions including the P puzzle-embedding slots,
            # so the "those slots dilute z/eff_rank" claim is measurable rather
            # than asserted.
            out[f"zseq/{split}/pr_joint_all_step_{sk}"] = float(_pr_from_gram(G_all))
            # Per-position PR, averaged over token positions.  Kept as a
            # DECOMPOSITION, not a competitor: pr_joint_tok /
            # (seq_len * pr_perpos_mean_tok) lies in (0, 1] by Cauchy-Schwarz
            # and separates "each position independently rich" (ratio ~1) from
            # "all positions share one global code" (ratio ~1/seq_len).  It is
            # derivable downstream from these two keys, so no third key.
            out[f"zseq/{split}/pr_perpos_mean_tok_step_{sk}"] = float(
                _pr_from_gram(G[P:]).mean())

            pr_across, pos_frac = _across_position_stats(Z[:, P:, :])
            out[f"zseq/{split}/pr_across_pos_tok_step_{sk}"] = pr_across
            out[f"zseq/{split}/pos_var_frac_tok_step_{sk}"] = pos_frac

            # The frozen recipe, per step: gives z/eff_rank a trajectory and, at
            # h = T, reproduces it bitwise.  See _pooled_pr_legacy's exemption.
            out[f"zseq/{split}/pr_pooled_all_step_{sk}"] = _pooled_pr_legacy(Z)

            # ---------------------------------------------------------- #
            # (B) Mutual information, per ACT step
            # ---------------------------------------------------------- #
            if sigma is not None:
                lb, null = _mi_step_bits(G_tok, sigma, n, h)
                # Named for the estimator ("decode" = held-out linear decode)
                # and for the estimand ("sym" = per-symbol surrogate, NOT input
                # identity).  Always emitted with its null sibling at the same
                # wandb step so the value cannot be read without its bias floor
                # -- which is materially NEGATIVE (~-1.10 bits at production
                # shape), not ~0.  See the estimator note.
                out[f"zmi/{split}/sym_decode_lb_bits_step_{sk}"] = lb
                out[f"zmi/{split}/sym_decode_null_bits_step_{sk}"] = null
                mi_nulls[sk] = null
                # NO "debiased = lb - null" key, deliberately.  For an UNSIGNED
                # plug-in estimator the shuffle null is the bias floor and the
                # difference is the thing to read; for a signed lower bound it
                # is not.  The null here is <= 0, so subtracting it inflates the
                # value above its own log2(n) ceiling (measured: 4.26 bits
                # against a 3.32-bit ceiling) and the result is no longer a
                # bound in either direction -- a bits-valued key in an MI
                # namespace that overshoots its ceiling is exactly the kind of
                # hidden limitation this metric family exists to avoid.  Read
                # the lb against the two MEASURED endpoints (null below,
                # sym_decode_saturated_bits below); use the null additionally as
                # the fold-leak alarm.

        # ATTAINABLE ceiling, measured once per eval on a latent from which the
        # symbol is exactly recoverable.  Emitted under exactly the condition
        # that produces lb keys, so an lb can never appear without it.
        if sigma is not None:
            out["zmi/sym_decode_saturated_bits"] = _mi_saturated_bits(
                sigma, n, int(z_traj[0].shape[2]))

    # The alarm the estimator note advertises is checked, not just described.
    if mi_nulls:
        _mi_alarm_check(mi_nulls, split)

    # Static MI ceilings.  Emitted whenever the probe shape is known, even if no
    # latent was captured, so the degeneracy of the pre-registered quantity is
    # legible from the dashboard rather than from a design document.
    out["zmi/probe_n"] = float(B)
    out["zmi/subspace_dim"] = float(
        min(_MI_SUBSPACE_DIM, max(B - 1, 0),
            max(2, (B // 2) // _MI_ROWS_PER_DIM)))
    out["zmi/probe_ceiling_bits"] = float(np.log2(B)) if B > 0 else 0.0
    if n is not None:
        out["zmi/sym_ceiling_bits"] = float(np.log2(n))
        # log2(n!) -- the population entropy of the input.  For n=10 this is
        # 21.79 bits, ABOVE zmi/probe_ceiling_bits = 9.0: the instrument cannot
        # reach it at this probe size, by construction.
        out["zmi/perm_entropy_bits"] = float(np.log2(np.arange(1, n + 1)).sum())

    # ------------------------------------------------------------------ #
    # (C) tau(h) recurrence-power alignment and (D) permutation structure
    # ------------------------------------------------------------------ #
    if model_is_training:
        # z comes from loop 1 and preds from loop 2 of _probe_forward, two
        # separate forward passes.  They line up ONLY because eval-mode halting
        # is deterministic and batch-synchronous (trm.py:275 gates adaptive
        # halting plus halt_exploration_prob on self.training).  In train mode
        # the two loops can halt differently per row and every tau/perm number
        # would be silently mispaired, so the families are dropped rather than
        # reported wrong.  The len() check below cannot catch per-row skew.
        log.warning("z_logging: model is in train mode; "
                    "ztau/* and zperm/* omitted (z/preds pairing unsafe)")
        return out

    T = len(preds_traj)
    if T == 0 or sigma is None:
        return out
    if z_traj and T != len(z_traj):
        log.warning("z_logging: decode trajectory (%d) and latent trajectory "
                    "(%d) disagree; ztau/* and zperm/* omitted", T, len(z_traj))
        return out

    # Exhaustive exponent grid.  Landau's g(n) bounds the order of every element
    # of S_n, so j = 0..g(n) covers every DISTINCT power of any probe row: the
    # grid is complete, not heuristic, and contains no aliases.
    jmax = int(min(_landau_g(n), _TAU_MAX_EXPONENT))
    if T > 0 and (jmax + 1) * T > _TAU_MAX_AGREE_KEYS:
        jmax = max(1, _TAU_MAX_AGREE_KEYS // max(T, 1) - 1)
    grid = list(range(0, jmax + 1))
    out["ztau/n_exponents"] = float(len(grid))
    # Sequential (sigma^h) and doubling (sigma^(2^(h-1))) predict the SAME
    # exponent at h=1 (1) and h=2 (2) and first diverge at h=3 (3 vs 4).  A run
    # with T <= 2 -- which includes all 192 historical halt_max_steps=1
    # checkpoints and the halt=2 arm -- cannot distinguish the two hypotheses
    # even in principle.  This key says so numerically so no downstream analyst
    # reads a verdict off a run that cannot produce one.
    out["ztau/discriminable"] = 1.0 if T >= 3 else 0.0
    out["zperm/detected_n"] = float(n)

    powers = [_perm_power(sigma, 0)]
    for _ in range(1, jmax + 1):
        powers.append(_perm_compose(sigma, powers[-1]))

    target = labels_np[:, :n] - 1
    k = _detect_k(sigma, target)
    if k is not None:
        out["zperm/detected_k"] = float(k)

    mask = labels_np != IGNORE_LABEL_ID
    loss_counts = mask.sum(axis=1)
    row_valid = loss_counts > 0
    n_valid = max(int(row_valid.sum()), 1)

    # Doubling hypothesis by repeated squaring, per row.  Exact for any h and
    # immune to the row-dependent order reduction that makes a literal
    # "exponent 2^(h-1)" column meaningless at large h.
    dbl = sigma.copy()

    for h in range(1, T + 1):
        sk = _stepkey(h)
        pr_full = preds_traj[h - 1].detach().cpu().numpy().astype(np.int64)
        # PAD decodes (id 0) map to -1, i.e. out of range, so they fail every
        # equality and every bijection test.  That is the intended behaviour.
        yhat = pr_full[:, :n] - 1

        best_j, best_a = grid[0], -1.0
        for j in grid:
            a = _agreement(yhat, powers[j])
            out[f"ztau/{split}/agree_h{sk}_j{j:02d}"] = a
            if a > best_a:            # ties resolve to the SMALLEST exponent
                best_a, best_j = a, j
        out[f"ztau/{split}/best_exp_step_{sk}"] = float(best_j)
        out[f"ztau/{split}/best_exp_agree_step_{sk}"] = best_a

        seq_pow = powers[h] if h <= jmax else _perm_power(sigma, h)
        out[f"ztau/{split}/agree_seq_step_{sk}"] = _agreement(yhat, seq_pow)
        # Convention 2^(h-1), so h=1 predicts sigma^1.  Using 2^h instead would
        # make the two hypotheses appear to differ at h=1, which is an artefact
        # of the labelling and not a measurement.
        out[f"ztau/{split}/agree_dbl_step_{sk}"] = _agreement(yhat, dbl)
        dbl = _perm_compose(dbl, dbl)

        row_exact = np.all((pr_full == labels_np) | ~mask, axis=1)
        out[f"ztau/{split}/exact_target_step_{sk}"] = float(
            (row_valid & row_exact).sum() / n_valid)

        valid_perm, cycle_ok = _perm_structure(sigma, yhat)
        out[f"zperm/{split}/valid_perm_step_{sk}"] = valid_perm
        out[f"zperm/{split}/cycle_consistent_step_{sk}"] = cycle_ok

        if h == T:
            # Aliases at the final step so a dashboard consumer need not know T.
            out[f"zperm/{split}/valid_perm_final"] = valid_perm
            out[f"zperm/{split}/cycle_consistent_final"] = cycle_ok
            out[f"zperm/{split}/best_exp_final"] = float(best_j)

    return out


# ---------------------------------------------------------------------------
# Scatter plot helper
# ---------------------------------------------------------------------------

def _make_pca_scatter(X_pca: np.ndarray,
                      labels_col0: np.ndarray,
                      correct_mask: np.ndarray,
                      title_prefix: str) -> list:
    """
    Build two wandb.Image scatter plots (label-colored, correct-colored).
    Returns list of (key, wandb.Image) pairs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    images = []
    for (color_arr, color_label, fname_suffix) in [
        (labels_col0, "label[0]", "by_label"),
        (correct_mask.astype(float), "correct", "by_correct"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color_arr,
                        cmap="tab10", s=4, alpha=0.7)
        plt.colorbar(sc, ax=ax, label=color_label)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{title_prefix} PCA ({fname_suffix})")
        fig.tight_layout()
        img = wandb.Image(fig)
        plt.close(fig)
        images.append((f"z/pca_{fname_suffix}_{title_prefix}", img))

    return images


# ---------------------------------------------------------------------------
# Main entry point: ZDynamicsLogger
# ---------------------------------------------------------------------------

class ZDynamicsLogger:
    """
    Constructed once at startup (rank 0 only when log_z_dynamics=True).
    Call .log(model, step, config) after each evaluate() call.
    """

    def __init__(self, data_path: str, probe_size: int,
                 ignore_label_id_in_file: Optional[int],
                 phase_threshold: float, phase_patience: int,
                 checkpoint_path: Optional[str],
                 seq_metrics: bool = False):
        self._checkpoint_path = checkpoint_path
        # Second gate, defaulting OFF and stacking on top of log_z_dynamics.
        # Without it every future run with log_z_dynamics=True would pay the
        # added zprobe cost and emit the new keys unconditionally, so no future
        # run would be cost- or key-comparable with the 74 historical ones and a
        # clean baseline could not be reproduced.  Defaulted so all four existing
        # ZDynamicsLogger call sites across the worktrees keep working unchanged.
        self._seq_metrics = bool(seq_metrics)
        self._phase_tracker = PhaseTracker(phase_threshold, phase_patience)

        # Load probes once; fingerprints logged to verify fixity
        self._train_probe, self._train_fp = _load_probe_tensors(
            data_path, "train", probe_size, ignore_label_id_in_file)
        self._test_probe, self._test_fp = _load_probe_tensors(
            data_path, "test", probe_size, ignore_label_id_in_file)

        log.info("ZDynamicsLogger: train_probe_hash=%s  test_probe_hash=%s",
                 self._train_fp, self._test_fp)
        # Resolved gate state, logged once.  "Keys absent because the feature was
        # off" and "keys absent because the architecture has no usable latent"
        # are otherwise indistinguishable after the fact, and the trajectory
        # cannot be backfilled -- existing snapshots keep only z_history[-1].
        log.info("ZDynamicsLogger: sequence/trajectory metrics %s "
                 "(zseq/, zmi/, ztau/, zperm/)",
                 "ENABLED" if self._seq_metrics else "disabled")

    # ------------------------------------------------------------------

    def log(self, model: torch.nn.Module, step: int,
            save_train_state_fn,
            train_state) -> None:
        """
        Called after evaluate() on rank 0.
        `save_train_state_fn` is pretrain.save_train_state (partial or callable).
        """
        import wandb

        # --- Probe forwards ---
        # Train probe only, matching the train-only convention of every existing
        # z/* key and keeping the per-eval cost from doubling.  The key
        # namespaces carry the split explicitly (zseq/train/...), so adding the
        # test probe later is additive.
        train_result = _probe_forward(model, self._train_probe,
                                      compute_extra=self._seq_metrics)
        test_result = _probe_forward(model, self._test_probe)

        # --- Phase tracking ---
        phase, transitioned = self._phase_tracker.update(
            train_result["exact_acc"], test_result["exact_acc"], step=step
        )

        # --- Build wandb log dict ---
        log_dict: Dict[str, Any] = {
            "probe/train_exact": train_result["exact_acc"],
            "probe/test_exact":  test_result["exact_acc"],
            "phase/index":       phase,
        }

        # Per-step deltas (use train probe deltas as representative)
        for t, delta in enumerate(train_result["step_deltas"], start=1):
            log_dict[f"z/delta_step_{t}"] = delta

        # PCA metrics on final z_H (mean-pool over sequence)
        if train_result["z_H"] is not None:
            z_H_np = _mean_pool_z(train_result["z_H"])  # (B, D)
            eigenvalues, eigenvectors = _pca(z_H_np)

            log_dict["z/eff_rank"]      = _effective_rank(eigenvalues)
            log_dict["z/pca_top2_var"]  = _pca_top2_var(eigenvalues)
            log_dict["z/mean_norm"]     = float(np.linalg.norm(z_H_np, axis=-1).mean())

            # PCA scatter (train probe)
            X_pca = z_H_np @ eigenvectors[:, :2]  # (B, 2)
            label_col0 = train_result["labels"][:, 0].numpy().astype(float)
            correct_np = train_result["correct_mask"].numpy()
            scatter_imgs = _make_pca_scatter(X_pca, label_col0, correct_np, "train")
            for k, img in scatter_imgs:
                log_dict[k] = img

        # --- Additive sequence/trajectory keys ---------------------------- #
        # TWO INDEPENDENT STRUCTURAL LOCKS on the frozen key contract, both
        # cheap, because ordering discipline alone is a convention that any
        # future addition can silently violate:
        #   (1) prefix filter -- only the four new namespaces may enter;
        #   (2) `k not in log_dict` -- an existing key is never overwritten,
        #       so even a metric misnamed "z/eff_rank" cannot clobber it.
        for _k, _v in train_result.get("extra_metrics", {}).items():
            if _k.split("/", 1)[0] not in _SEQ_KEY_NAMESPACES:
                log.warning("z_logging: dropping out-of-namespace key %r", _k)
                continue
            if _k in log_dict:
                log.warning("z_logging: refusing to overwrite existing key %r", _k)
                continue
            log_dict[_k] = _v

        # Live drift detector for the frozen key.  The per-step pooled PR at
        # h = T is computed by the identical _mean_pool_z -> _pca ->
        # _effective_rank chain from the identical tensor, so this is 0.0 unless
        # something upstream changed which latent reaches which consumer.
        # Logged and warned, NEVER asserted in-process: an assertion here would
        # abort a multi-day run at an eval boundary, which is precisely the
        # failure mode the containment elsewhere in this file exists to avoid.
        _n_steps = int(train_result.get("extra_metrics", {})
                       .get("zseq/n_act_steps", 0))
        _pooled_key = f"zseq/train/pr_pooled_all_step_{_stepkey(_n_steps)}"
        if "z/eff_rank" in log_dict and _pooled_key in log_dict:
            _ref = log_dict["z/eff_rank"]
            _rel = abs(log_dict[_pooled_key] - _ref) / max(abs(_ref), 1e-30)
            log_dict["zseq/legacy_pr_reldiff"] = float(_rel)
            if _rel > _LEGACY_PR_RELDIFF_WARN:
                log.warning("z_logging: pooled-PR parity drift %.3e "
                            "(z/eff_rank=%r vs %s=%r)", _rel, _ref,
                            _pooled_key, log_dict[_pooled_key])

        wandb.log(log_dict, step=step)

        # --- Save z snapshot ---
        if self._checkpoint_path is not None:
            snap_dir = os.path.join(self._checkpoint_path, "z_snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            snap_path = os.path.join(snap_dir, f"step_{step}.pt")
            snapshot = {
                "z_H":          train_result["z_H"],   # bfloat16
                "z_L":          train_result["z_L"],   # bfloat16
                "labels":       train_result["labels"],
                "correct_mask": train_result["correct_mask"],
            }
            # Per-ACT-step decode, (T, B, seq_len) int16 -- ~180 KB at T=16,
            # B=512 against the existing snapshot's tens of MB, so it does not
            # threaten the 50 MB budget.  It makes the whole tau/permutation
            # family exactly recomputable offline under conventions that have
            # not been invented yet, which matters because no existing snapshot
            # contains a trajectory at all (only z_history[-1] is stored) and
            # none can be backfilled.  Purely additive: existing consumers such
            # as analysis/pr_recompute.py read by key and ignore extras.
            if train_result.get("preds_traj") is not None:
                snapshot["preds_traj"] = train_result["preds_traj"]
            torch.save(snapshot, snap_path)
            log.info("z snapshot saved: %s", snap_path)

        # --- Phase transition: force checkpoint + extra snapshot ---
        if transitioned:
            log.info("Phase transition → %d at step %d", phase, step)
            if train_state is not None:
                save_train_state_fn(train_state)
                log.info("Forced checkpoint saved at step %d (phase transition)", step)
