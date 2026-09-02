"""
measure_rho.py — Spectral Radius and AGOP Diagnostics for TRM weight-tied blocks.

Implements two diagnostics:
  1. Spectral radius at fixed point (PO-SPECTRAL-WT-01)
  2. AGOP alignment diagnostic (PO-AGOP-SIGMA-K)

Design specs:
  measure_rho_spectral_patch.md  (wave2_resolved)
  measure_rho_agop_patch.md      (wave2_resolved)
  measure_rho_merge_order.md     (canonical layout)

Example usage (GPU 4):
  CUDA_VISIBLE_DEVICES=4 uv run python measure_rho.py \\
      --fixed-point --model-variant trm \\
      --checkpoint checkpoints/Sigma_k/ABC_k3_baseline/step_109863 \\
      --save-spectral /tmp/spec.jsonl

  CUDA_VISIBLE_DEVICES=4 uv run python measure_rho.py \\
      --agop --model-variant trm \\
      --checkpoint checkpoints/Sigma_k/ABC_k3_baseline/step_109863 \\
      --save-agop /tmp/ref.pt

  CUDA_VISIBLE_DEVICES=4 uv run python measure_rho.py \\
      --agop --model-variant trm \\
      --checkpoint checkpoints/Sigma_k/ABC_k3_baseline/step_109863 \\
      --agop-ref /tmp/ref.pt --agop-output /tmp/agop.jsonl \\
      --agop-n-checkpoints 10 --run-id ABC_k3_baseline
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from models.recursive_reasoning.trm_singlez import (
    TinyRecursiveReasoningModel_ACTV1 as TRM_SingleZ,  # noqa: F401 — shared alias, Wave 1.5 locked
)
from models.losses import ACTLossHead

# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_all_config(checkpoint_dir: str) -> dict:
    """Read all_config.yaml from a checkpoint directory."""
    cfg_path = os.path.join(checkpoint_dir, "all_config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"all_config.yaml not found in {checkpoint_dir}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_metadata(data_path: str) -> dict:
    """Read dataset.json from <data_path>/test/dataset.json."""
    ds_path = os.path.join(data_path, "test", "dataset.json")
    if not os.path.isfile(ds_path):
        raise FileNotFoundError(f"dataset.json not found at {ds_path}")
    with open(ds_path, "r") as f:
        return json.load(f)


def build_model_config(
    checkpoint_dir: str,
    batch_size: int = 1,
    h_cycles: int | None = None,
    l_cycles: int | None = None,
) -> dict:
    """
    Build the model config dict from all_config.yaml + dataset metadata.
    Mirrors pretrain.py:create_model() config assembly.

    h_cycles / l_cycles override the recurrence counts recorded in all_config.yaml.
    This is safe because the recurrent block is weight-tied: trm.py reads
    H_cycles/L_cycles fresh on every forward() and loops over a single shared
    L_level module, so no parameter or buffer shape depends on them and a
    checkpoint loads at any cycle count under strict key matching.

    Overriding makes the INFERENCE-TIME DEPTH AXIS measurable without retraining.
    It does not imply accuracy is preserved at the new depth -- that is the open
    question the override exists to measure, not an assumption it encodes.
    """
    all_cfg = load_all_config(checkpoint_dir)
    arch = all_cfg["arch"]

    # data_paths: take first entry, resolve relative to cwd
    data_paths = all_cfg.get("data_paths", [])
    if not data_paths:
        raise ValueError("data_paths is empty in all_config.yaml")
    data_path = data_paths[0]
    meta = load_dataset_metadata(data_path)

    # arch extras (all fields except 'name' and 'loss')
    arch_extras = {k: v for k, v in arch.items() if k not in ("name", "loss")}

    # Depth-axis override. Applied after the trained values are read so the
    # caller can report both. Weight tying is what makes this legal (see docstring).
    for _name, _val in (("H_cycles", h_cycles), ("L_cycles", l_cycles)):
        if _val is None:
            continue
        if not isinstance(_val, int) or _val < 1:
            raise ValueError(f"{_name} override must be a positive int, got {_val!r}")
        arch_extras[_name] = _val

    model_cfg = dict(
        **arch_extras,
        batch_size=batch_size,
        vocab_size=meta["vocab_size"],
        seq_len=meta["seq_len"],
        num_puzzle_identifiers=meta["num_puzzle_identifiers"],
    )
    return model_cfg


def parse_k_from_data_paths(data_paths) -> int:
    """
    Parse the composition depth K from data_paths.
    Expects paths of form .../sigma_k/<K> or .../sigma_k_10/<K>.
    Returns -1 if unable to parse.
    """
    if not data_paths:
        return -1
    try:
        last = data_paths[0].rstrip("/").split("/")[-1]
        return int(last)
    except (ValueError, IndexError):
        return -1


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def _strip_orig_mod(state_dict: dict) -> dict:
    """Strip torch.compile _orig_mod. prefix from state dict keys."""
    if any(k.startswith("_orig_mod.") for k in state_dict):
        print("  [info] _orig_mod. prefix detected — stripping")
        return {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_model(checkpoint_path: str, config: dict, device: str):
    """
    Load standard TRM (trm.py, with z_H carry) from checkpoint.
    Returns wrapped.model.inner (TinyRecursiveReasoningModel_ACTV1_Inner).
    """
    trm = TinyRecursiveReasoningModel_ACTV1(config).to(device)
    wrapped = ACTLossHead(trm, loss_type="stablemax_cross_entropy").to(device)

    sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sd = _strip_orig_mod(sd)

    missing, unexpected = wrapped.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    wrapped.eval()
    return wrapped.model.inner


def load_model_singlez(checkpoint_path: str, config: dict, device: str):
    """
    Load TRM SingleZ (trm_singlez.py, no z_H carry) from checkpoint.
    Returns wrapped.model.inner.

    Shared infrastructure — defined here (spectral patch owns, Wave 1.5).
    Called by both spectral and AGOP diagnostics.
    """
    trm = TRM_SingleZ(config).to(device)
    wrapped = ACTLossHead(trm, loss_type="stablemax_cross_entropy").to(device)

    sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sd = _strip_orig_mod(sd)

    missing, unexpected = wrapped.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    wrapped.eval()
    return wrapped.model.inner


# ─────────────────────────────────────────────────────────────────────────────
# Spectral radius via power iteration (JVP + VJP)
# ─────────────────────────────────────────────────────────────────────────────

def spectral_radius_power_iter(fn, z: torch.Tensor, n_iter: int = 50, seed: int = 42) -> float:
    """
    Estimate spectral_radius(dF_{z}) via power iteration on the Jacobian operator.
    Uses forward-mode (JVP) to apply J, then normalises.

    DEVIATION: flash attention does not support forward AD (JVP). We temporarily
    disable flash SDP and enable math SDP (which supports forward AD) around the
    JVP calls. This may be slower but is numerically equivalent.

    Args:
        fn: callable z -> z_next (the recurrent map F)
        z: point at which to evaluate the Jacobian (q*)
        n_iter: number of power iterations (default 50)
        seed: fixed random seed for reproducibility (Gate 5: rho diff < 1e-3)

    Returns:
        spectral radius estimate (float)
    """
    torch.manual_seed(seed)
    v = torch.randn_like(z)
    v = v / v.norm()

    sigma = 1.0

    # Disable flash SDP: it does not support forward-mode AD.
    # Fall back to math SDP which fully supports JVP.
    # DEVIATION: using torch.nn.attention.sdpa_kernel (preferred API in newer torch)
    # to avoid FutureWarning from the deprecated sdp_kernel context manager.
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        for _ in range(n_iter):
            # JVP: J @ v
            _, jv = torch.func.jvp(fn, (z,), (v,))
            sigma = jv.norm().item()
            if sigma < 1e-30:
                break
            v = jv / jv.norm()

    return sigma


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-point utilities — build the map F per model variant
# ─────────────────────────────────────────────────────────────────────────────

def _make_F_singlez(inner, injection_fp32: torch.Tensor, cos_sin):
    """
    Build the fixed-point map F for trm_singlez.py.

    trm_singlez.py:207: z_L = self.L_level(z_L + input_embeddings, **seq_info)
    F is a single L-level application with injection.
    Weight-tied: L_level uses the SAME weights each call.

    DEVIATION: the spec says to iterate L_cycles times per F call. For a
    fixed-point map we define F as ONE L-level application (matching the
    recurrent driver), not L_cycles applications.  This aligns with spec §1
    "the spectral radius of the single-block Jacobian dF_{q*}" where the
    "single block" is one L-level pass, not the entire K-step composition.

    injection_fp32: fp32 injection tensor (the convergence loop runs in fp32
        to avoid the bf16 resolution floor ~1e-3 preventing tol=1e-4).
    """
    injection_det = injection_fp32.detach()
    L_level = inner.L_level

    def F(z):
        # injection-form: L_level(z + injection)
        # Both z and injection are fp32 for convergence precision.
        return L_level(z + injection_det, cos_sin=cos_sin)

    return F


def _make_F_trm(inner, injection_fp32: torch.Tensor, cos_sin):
    """
    Build the fixed-point map F for trm.py.

    trm.py:211: z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
    L_level.forward(hidden_states, input_injection, **kwargs)

    DEVIATION from patch: the patch was written for singlez (single-carry).
    For trm.py we use the single-step map F(z_L) = L_level(z_L, injection).
    (trm.py L_level adds injection inside: hidden_states + input_injection.)
    The convergence loop runs in fp32 to avoid bf16 precision floor.
    The joint (z_L, z_H) fixed-point structure is not tracked here; we use
    a fixed injection that subsumes z_H + input_embeddings.

    injection_fp32: fp32 injection tensor representing z_H + input_embeddings.
    """
    injection_det = injection_fp32.detach()
    L_level = inner.L_level

    def F(z):
        return L_level(z, injection_det, cos_sin=cos_sin)

    return F


# ─────────────────────────────────────────────────────────────────────────────
# compute_rho_at_fixed_point
# ─────────────────────────────────────────────────────────────────────────────

def compute_rho_at_fixed_point(
    inner,
    injection: torch.Tensor,
    cos_sin,
    device: str = "cuda",
    tol: float = 1e-4,
    max_iter: int = 500,
    n_power_iter: int = 50,
    model_variant: str = "trm",
    k_synth: int | None = None,
) -> dict:
    """
    Compute spectral_radius(dF_{q*}) for the weight-tied TRM block.

    Weight-tied correctness: inner.L_level uses the SAME tied weights each call.
    Fixed-point loop and Jacobian both operate on this single-block F.

    Args:
        inner: Inner model (TinyRecursiveReasoningModel_ACTV1_Inner)
        injection: [B, S, D] input embedding tensor (z_H + x for trm, or x for singlez)
        cos_sin: rotary embedding cache (or None)
        tol: q*-convergence tolerance (spec §6 Step 1: ε = 1e-4)
        max_iter: max fixed-point iterations (spec §6 Step 1: 500)
        n_power_iter: power iteration steps for Jacobian (N_ITER=50)
        model_variant: "trm" or "singlez" (controls which F is built)
        k_synth: synthesis depth k (from data_paths). rho_K = rho**k_synth.
            FIX 1: spec boundary note ("At K=5, ρ=0.8 → 0.8^5") defines the
            exponent K as the *synthesis depth* (task k), NOT L_cycles.
            If None, falls back to L_cycles (legacy behaviour).

    Returns:
        dict with keys: rho, q_star_norm, converged, iters_to_converge, phase, rho_K, K
    """
    # FIX 1: K used for rho_K and reporting is the synthesis depth k_synth
    # (parsed from data_paths upstream), falling back to L_cycles only when
    # k_synth is unavailable. Previously this hardcoded K = L_cycles, which
    # made stdout K=6 and rho_K=rho^6 disagree with the JSONL "K":3.
    K = k_synth if (k_synth is not None and k_synth >= 0) else inner.config.L_cycles
    S = inner.config.seq_len + inner.puzzle_emb_len
    D = inner.config.hidden_size

    # DEVIATION: the convergence loop and spectral radius computation run in
    # fp32 throughout, not the model's native forward_dtype (bfloat16).
    # Rationale: bf16 resolution ~1e-3 prevents reaching tol=1e-4 cleanly
    # (spec §7 Test 6 / audit V8-1). We temporarily cast inner.L_level to fp32,
    # run convergence + spectral radius, then restore to original dtype.
    # This is spec-compliant (spec §3 "promote q to fp32" Option B).
    original_dtype_params = next(inner.L_level.parameters()).dtype
    inner_fp32 = inner.L_level.to(torch.float32)
    injection_fp32 = injection.float()

    # Build fixed-point map F (fp32 weights + fp32 input)
    if model_variant == "singlez":
        F = _make_F_singlez(inner, injection_fp32, cos_sin)
    else:
        F = _make_F_trm(inner, injection_fp32, cos_sin)
    q_seq_len = S  # both variants: single-carry state shape (B, S, D)

    # Step 1: q*-convergence (spec §6 Step 1)
    # Running entirely in fp32 avoids bf16 resolution floor.
    BATCH = injection.shape[0]
    torch.manual_seed(42)  # Deterministic init for Gate 5
    q = torch.randn(BATCH, q_seq_len, D, device=device, dtype=torch.float32)

    converged = False
    iters_to_converge = max_iter
    delta = float("inf")

    with torch.no_grad():
        for i in range(max_iter):
            q_next = F(q)
            delta = (q_next - q).norm().item()
            if delta < tol:
                converged = True
                iters_to_converge = i + 1
                q = q_next
                break
            q = q_next

    q_star = q.detach()  # fp32
    q_star_norm = q_star.norm().item()

    if not converged:
        print(
            f"  [warn] q*-loop did not converge in {max_iter} iterations "
            f"(final Δ={delta:.2e}); using last iterate as q*"
        )

    # Step 2: Spectral radius via power_iter (fp32 F, fp32 q*)
    rho = spectral_radius_power_iter(F, q_star, n_iter=n_power_iter)

    # Restore model to original dtype (bfloat16)
    inner.L_level.to(original_dtype_params)

    # Step 3: Phase classification (spec §6 Step 3, buckets from §6)
    # Note: spec §4 row 4 says >1.0, spec §6 says >1.05; using §6 value
    # for consistency (spec inconsistency V8-2 documented in patch)
    if rho < 0.95:
        phase = "ordered"
    elif rho <= 1.05:
        phase = "edge-of-chaos"
    else:
        phase = "chaotic"

    rho_K = rho ** K

    return {
        "rho": rho,
        "q_star_norm": q_star_norm,
        "converged": converged,
        "iters_to_converge": iters_to_converge,
        "phase": phase,
        "rho_K": rho_K,
        "K": K,
    }


# ─────────────────────────────────────────────────────────────────────────────
# run_fixed_point_sweep — iterate all step_* checkpoints in a directory
# ─────────────────────────────────────────────────────────────────────────────

def run_fixed_point_sweep(
    checkpoints_dir: str,
    config: dict,
    device: str,
    model_variant: str = "trm",
    tol: float = 1e-4,
    max_iter: int = 500,
    n_power_iter: int = 50,
    save_spectral: str | None = None,
    all_config_yaml: dict | None = None,
) -> list:
    """
    Sweep all step_* checkpoints in checkpoints_dir, compute ρ@q* for each.

    Emits JSONL rows with 11 fields:
      run_id, epoch, K, H, d_ff, rho, phase, rho_K,
      converged, iters_to_converge, q_star_norm

    Args:
        checkpoints_dir: directory containing step_* checkpoint files
        config: model config dict
        all_config_yaml: pre-loaded all_config dict (for K parsing); if None,
            loaded from checkpoints_dir
    """
    # Find step_* checkpoint files
    ckpt_files = sorted([
        f for f in os.listdir(checkpoints_dir)
        if f.startswith("step_") and os.path.isfile(os.path.join(checkpoints_dir, f))
    ])

    if not ckpt_files:
        print(f"[warn] No step_* checkpoint files found in {checkpoints_dir}")
        return []

    print(f"[info] Found {len(ckpt_files)} step_* checkpoints in {checkpoints_dir}")

    # Parse K from data_paths
    if all_config_yaml is None:
        try:
            all_config_yaml = load_all_config(checkpoints_dir)
        except FileNotFoundError:
            all_config_yaml = {}
    data_paths = all_config_yaml.get("data_paths", [])
    K_data = parse_k_from_data_paths(data_paths)

    results = []

    for fname in ckpt_files:
        ckpt_path = os.path.join(checkpoints_dir, fname)
        print(f"\nProcessing checkpoint: {fname}")

        try:
            if model_variant == "singlez":
                inner = load_model_singlez(ckpt_path, config, device)
            else:
                inner = load_model(ckpt_path, config, device)
        except Exception as e:
            print(f"  [error] Failed to load {ckpt_path}: {e}")
            continue

        S = inner.config.seq_len + inner.puzzle_emb_len
        D = inner.config.hidden_size
        dtype = inner.forward_dtype
        cos_sin = inner.rotary_emb() if hasattr(inner, "rotary_emb") else None

        BATCH = 1
        torch.manual_seed(42)
        injection = torch.randn(BATCH, S, D, device=device, dtype=dtype)

        result = compute_rho_at_fixed_point(
            inner=inner,
            injection=injection,
            cos_sin=cos_sin,
            device=device,
            tol=tol,
            max_iter=max_iter,
            n_power_iter=n_power_iter,
            model_variant=model_variant,
            k_synth=K_data,  # FIX 1: pass synthesis depth so rho_K = rho^k
        )

        # FIX 1: K is resolved once inside compute_rho_at_fixed_point and is
        # consistent across JSONL "K", stdout, and rho_K's exponent.
        K = result["K"]

        row = {
            "run_id": os.path.basename(checkpoints_dir),
            "epoch": fname,
            "K": K,
            "H": inner.config.H_cycles,  # H_cycles per task spec (overrides num_heads in patch)
            "d_ff": int(inner.config.hidden_size * inner.config.expansion),
            "rho": result["rho"],
            "phase": result["phase"],
            "rho_K": result["rho_K"],
            "converged": result["converged"],
            "iters_to_converge": result["iters_to_converge"],
            "q_star_norm": result["q_star_norm"],
        }
        results.append(row)

        print(
            f"  ρ@q* = {result['rho']:.6f}  phase={result['phase']}  "
            f"converged={result['converged']} ({result['iters_to_converge']} iters)  "
            f"ρ^K={result['rho_K']:.6f}"
        )

    if save_spectral and results:
        with open(save_spectral, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")
        print(f"\n[info] Spectral results written to: {save_spectral}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# measure_rho_fixed_point — single-checkpoint dispatch
# ─────────────────────────────────────────────────────────────────────────────

def measure_rho_fixed_point(
    checkpoint_path: str | None,
    config: dict,
    device: str,
    model_variant: str,
    tol: float,
    max_iter: int,
    n_power_iter: int,
    save_spectral: str | None,
    all_config_yaml: dict | None = None,
) -> dict:
    """
    Single-checkpoint fixed-point ρ measurement.
    Called when --fixed-point flag is set.
    """
    if checkpoint_path is None:
        print("[info] No checkpoint — measuring random-init baseline (epoch 0)")
        if model_variant == "singlez":
            trm = TRM_SingleZ(config).to(device)
            wrapped = ACTLossHead(trm, loss_type="stablemax_cross_entropy").to(device)
            wrapped.eval()
            inner = wrapped.model.inner
        else:
            trm = TinyRecursiveReasoningModel_ACTV1(config).to(device)
            wrapped = ACTLossHead(trm, loss_type="stablemax_cross_entropy").to(device)
            wrapped.eval()
            inner = wrapped.model.inner
    else:
        print(f"Loading checkpoint: {checkpoint_path}")
        if model_variant == "singlez":
            inner = load_model_singlez(checkpoint_path, config, device)
        else:
            inner = load_model(checkpoint_path, config, device)

    S = inner.config.seq_len + inner.puzzle_emb_len
    D = inner.config.hidden_size
    L_cycles = inner.config.L_cycles
    dtype = inner.forward_dtype
    cos_sin = inner.rotary_emb() if hasattr(inner, "rotary_emb") else None

    # FIX 1: resolve canonical K (synthesis depth) ONCE, before the compute call
    # and before stdout. data_paths parse first, fallback to L_cycles.
    K_data = -1
    if all_config_yaml is not None:
        K_data = parse_k_from_data_paths(all_config_yaml.get("data_paths", []))

    BATCH = 1
    torch.manual_seed(42)
    injection = torch.randn(BATCH, S, D, device=device, dtype=dtype)

    print(
        f"Model: seq_len={inner.config.seq_len}, hidden={D}, L_cycles={L_cycles}, "
        f"variant={model_variant}"
    )
    print(f"Fixed-point search: tol={tol}, max_iter={max_iter}, n_power_iter={n_power_iter}")

    result = compute_rho_at_fixed_point(
        inner=inner,
        injection=injection,
        cos_sin=cos_sin,
        device=device,
        tol=tol,
        max_iter=max_iter,
        n_power_iter=n_power_iter,
        model_variant=model_variant,
        k_synth=K_data,  # FIX 1: pass synthesis depth so rho_K = rho^k
    )

    # FIX 1: single K source — result["K"] reflects synthesis depth (or fallback)
    K = result["K"]

    print(f"\n{'='*55}")
    print(f"ρ@q* = {result['rho']:.6f}")
    print(f"Phase classification        = {result['phase']}")
    print(f"q*-convergence              = {'YES' if result['converged'] else 'NO'} ({result['iters_to_converge']} iters)")
    print(f"‖q*‖                        = {result['q_star_norm']:.4f}")
    print(f"K (synthesis depth)         = {K}")
    print(f"ρ^K (informational)         = {result['rho_K']:.6e}")
    print(f"{'='*55}")

    # Decision hint per spec §4
    if result["rho"] < 0.8:
        print("→ ρ < 0.8: ordered/contractive phase (consistent with H-collapse if failed run)")
        print("  Compare against successful-cohort ρ > 0.95 threshold for CONFIRMED criterion.")
    elif result["rho"] < 0.95:
        print("→ 0.8 ≤ ρ < 0.95: partial overlap zone — INCONCLUSIVE per spec §4")
    elif result["rho"] <= 1.05:
        print("→ ρ ≈ 1 (edge-of-chaos): consistent with successful cohort expectation")
    else:
        print("→ ρ > 1.05: chaotic-phase — revise H-023 framing per spec §4")

    # FIX 1: K already resolved above (result["K"]); JSONL uses the same value.
    if save_spectral:
        row = {
            "run_id": os.path.basename(os.path.dirname(checkpoint_path)) if checkpoint_path else "random_init",
            "epoch": os.path.basename(checkpoint_path) if checkpoint_path else "random_init",
            "K": K,
            "H": inner.config.H_cycles,  # H_cycles per task spec (overrides num_heads in patch)
            "d_ff": int(inner.config.hidden_size * inner.config.expansion),
            "rho": result["rho"],
            "phase": result["phase"],
            "rho_K": result["rho_K"],
            "converged": result["converged"],
            "iters_to_converge": result["iters_to_converge"],
            "q_star_norm": result["q_star_norm"],
        }
        with open(save_spectral, "w", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[info] Spectral result written to: {save_spectral}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# AGOP diagnostic functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_agop(
    inner,
    probe_batch: dict,
    device: str,
    n_probe: int = 512,
    warmup: bool = True,
    mask_valid: bool = True,
    scalar: str = "ce",
) -> torch.Tensor:
    """
    Compute AGOP = (1/n) sum_j grad_xemb(j) grad_xemb(j)^T (Frobenius outer product).

    inner: TinyRecursiveReasoningModel_ACTV1_Inner (loaded model, either trm or singlez)
    probe_batch: dict with keys 'inputs' (int tensor, shape (B, seq_len)),
                 'puzzle_identifiers' (int tensor, shape (B,)), and optionally
                 'labels' (int tensor, shape (B, seq_len)) for the CE scalar.
    n_probe: number of probe examples (default 512)
    warmup: FIX 3(a) — if True, run H_cycles-1 warmup under no_grad to reach the
        trained operating point, then take the grad-enabled last cycle. This
        replicates trm.py:208-216 / trm_singlez.py:204-212 EXACTLY. If False,
        uses the legacy single-pass-from-L_init probe (patch A1).
    mask_valid: FIX 3(b) — if True, strip the puzzle-embedding prefix positions
        (the first puzzle_emb_len positions are constant learned puzzle embeddings,
        NOT input-derived) before forming the outer product. This removes
        position contamination and shrinks the AGOP from (S*D)^2 to (seq_len*D)^2.

    Returns: AGOP tensor of shape (d, d) where d = (S or seq_len) * D (float32).

    SCALAR (FIX 3, DEVIATION from patch): the patch's scalar `logits.mean()` is
    a constant-cotangent probe — because lm_head is linear, d(mean)/d(z_out) is
    the SAME vector for every example/position, so every per-example gradient
    g_j = J_jᵀ c probes a single Jacobian direction. The resulting AGOP is
    rank-1-per-example (effective rank collapses to ~5-7 on 512 probes) and is
    hypersensitive to operating-point detail → adjacent checkpoints look
    orthogonal. This makes rho>0.8 unreachable by construction. We replace it
    with a per-example label-based cross-entropy loss (labels from
    all__labels.npy, ignore_index = pad_id = 0), which yields example-specific
    cotangents and a non-degenerate AGOP. If labels are absent (synthetic
    probe), we fall back to logits.mean() with a warning.

    FIX 3(c): the forward is run with L_level cast to fp32 (weights are already
    fp32 on disk; only the activation path was bf16). bf16 activation noise
    (~1e-3) is removed as a confound. Restored to native dtype on exit.
    """
    inner.eval()
    cos_sin = inner.rotary_emb() if hasattr(inner, "rotary_emb") else None
    L_cycles = inner.config.L_cycles
    H_cycles = inner.config.H_cycles
    is_singlez = not hasattr(inner, "H_init")  # variant detection (trm has H_init)

    # FIX 3(c): cast L_level to fp32 for the AGOP forward (remove bf16 confound).
    orig_dtype = next(inner.L_level.parameters()).dtype
    inner.L_level.to(torch.float32)
    fwd_dtype = torch.float32

    try:
        input_ids = probe_batch["inputs"].to(device)
        puzzle_ids = probe_batch["puzzle_identifiers"].to(device)

        # Step 1: float input embedding (detached from integer token ids)
        with torch.no_grad():
            emb = inner._input_embeddings(input_ids, puzzle_ids)  # (B, S, D)

        # Step 2: attach grad to embedding (NOT the integer token ids)
        emb_in = emb.detach().to(torch.float32).requires_grad_(True)  # (B, S, D) fp32
        B, S, D = emb_in.shape

        # Step 3: replicate the model recurrence at the trained operating point.
        # NOTE (FIX1 from audit V2): unsqueeze twice to promote (D,) → (1,1,D).
        z_L = (
            inner.L_init.unsqueeze(0).unsqueeze(0)
            .expand(B, S, -1).to(torch.float32)
        )

        if is_singlez:
            # trm_singlez.py:204-213
            if warmup:
                with torch.no_grad():
                    for _h in range(H_cycles - 1):
                        for _l in range(L_cycles):
                            z_L = inner.L_level((z_L + emb_in).to(fwd_dtype), cos_sin=cos_sin)
                        z_L = inner.L_level(z_L.to(fwd_dtype), cos_sin=cos_sin)
            # grad-enabled last cycle (emb_in graph live here)
            for _l in range(L_cycles):
                z_L = inner.L_level((z_L + emb_in).to(fwd_dtype), cos_sin=cos_sin)
            z_L = inner.L_level(z_L.to(fwd_dtype), cos_sin=cos_sin)
            z_out = z_L  # singlez output is z_L (trm_singlez.py:213,217)
        else:
            # trm.py:208-216 — carries BOTH z_H and z_L
            z_H = (
                inner.H_init.unsqueeze(0).unsqueeze(0)
                .expand(B, S, -1).to(torch.float32)
            )
            if warmup:
                with torch.no_grad():
                    for _h in range(H_cycles - 1):
                        for _l in range(L_cycles):
                            z_L = inner.L_level(z_L.to(fwd_dtype), (z_H + emb_in).to(fwd_dtype), cos_sin=cos_sin)
                        z_H = inner.L_level(z_H.to(fwd_dtype), z_L.to(fwd_dtype), cos_sin=cos_sin)
            # grad-enabled last cycle (emb_in graph live here)
            for _l in range(L_cycles):
                z_L = inner.L_level(z_L.to(fwd_dtype), (z_H + emb_in).to(fwd_dtype), cos_sin=cos_sin)
            z_H = inner.L_level(z_H.to(fwd_dtype), z_L.to(fwd_dtype), cos_sin=cos_sin)
            z_out = z_H  # trm.py:220 output uses z_H

        # Step 4: scalar probe loss.
        logits = inner.lm_head(z_out.to(fwd_dtype)).to(torch.float32)
        logits = logits[:, inner.puzzle_emb_len:]  # (B, seq_len, vocab) — strip puzzle prefix
        labels = probe_batch.get("labels", None)
        if labels is not None:
            labels = labels.to(device).long()  # (B, seq_len)
            V = logits.size(-1)
            valid = labels != 0  # pad_id = 0 → ignore
            if scalar == "onehot":
                # FIX 3 (DEVIATION): checkpoint-INDEPENDENT cotangent. The grad of
                # the true-label logit w.r.t. logits is onehot(label) — the SAME
                # direction for ref and target, so prediction-state drift (p_j-y_j
                # in CE) is removed and only Jacobian/feature drift remains.
                true_logit = logits.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                probe_loss = true_logit[valid].sum()
            else:
                # FIX 3 (DEVIATION): per-example cross-entropy → example-specific
                # cotangents (non-degenerate AGOP). ignore_index = pad_id = 0.
                import torch.nn.functional as F
                probe_loss = F.cross_entropy(
                    logits.reshape(-1, V), labels.reshape(-1),
                    ignore_index=0, reduction="mean",
                )
        else:
            # Fallback (synthetic probe, no labels): degenerate but numerically valid.
            print("  [warn] no labels in probe — using logits.mean() (degenerate AGOP)")
            probe_loss = logits.mean()

        # Step 5: backward → grad w.r.t. emb_in
        probe_loss.backward()
        g = emb_in.grad
        if g is None:
            raise RuntimeError("AGOP: emb_in.grad is None — check requires_grad_ propagation")

        g = g.detach()
        # FIX 3(b): drop puzzle-prefix positions (constant puzzle embeddings,
        # not input-derived) before flattening. Keeps a fixed (seq_len*D) dim.
        if mask_valid:
            g = g[:, inner.puzzle_emb_len:, :]  # (B, seq_len, D)

        # Step 6: flatten spatial dims → (B, d)
        g_flat = g.reshape(g.size(0), -1)  # (B, d) float32

        # Step 7: AGOP = (1/B) * G^T G ∈ R^(d×d)
        agop = torch.einsum("bi,bj->ij", g_flat, g_flat) / g_flat.size(0)
        agop = agop.detach().clone()
    finally:
        # FIX 3(c): restore L_level to native dtype regardless of outcome.
        inner.L_level.to(orig_dtype)

    return agop  # shape (d, d) float32


def agop_alignment(agop_t: torch.Tensor, agop_ref: torch.Tensor) -> float:
    """
    Frobenius cosine similarity between AGOP at step t and reference AGOP.
    Implements: <A, B>_F / (||A||_F * ||B||_F).
    Both tensors must be same shape (d, d) float32.
    Returns scalar in [-1, 1].
    """
    assert agop_t.shape == agop_ref.shape, (
        f"AGOP shape mismatch: {agop_t.shape} vs {agop_ref.shape}"
    )
    num = (agop_t * agop_ref).sum()
    denom = agop_t.norm(p="fro") * agop_ref.norm(p="fro")
    if denom.item() < 1e-12:
        return 0.0
    return (num / denom).item()


def agop_trace(agop: torch.Tensor) -> float:
    """Trace of AGOP — scalar proxy for total feature-learning magnitude."""
    return agop.diagonal().sum().item()


def agop_effective_rank(agop: torch.Tensor) -> float:
    """
    Effective rank = tr(AGOP)^2 / ||AGOP||_F^2.
    Low effective rank in failed runs indicates degenerate feature learning (spec §3).
    """
    tr = agop.diagonal().sum()
    frob_sq = (agop * agop).sum()
    if frob_sq.item() < 1e-12:
        return 0.0
    return (tr * tr / frob_sq).item()


def _probe_signature(probe_batch: dict, source: str, n_probe: int) -> dict:
    """
    FIX 4: build a probe identity signature so ref and target AGOPs can be
    checked for consistency. Mixing a synthetic-probe ref with a real-probe
    target (or different real probe sets) silently produces a meaningless
    alignment; this guard catches that.
    """
    import hashlib
    inp = probe_batch["inputs"].detach().to("cpu").to(torch.int64).contiguous()
    h = hashlib.sha1(inp.numpy().tobytes()).hexdigest()[:16]
    return {
        "source": source,
        "n_used": int(inp.shape[0]),
        "n_probe_arg": int(n_probe),
        "seq_len": int(inp.shape[1]) if inp.ndim > 1 else None,
        "input_sha1": h,
    }


def run_agop_diagnostic(args, config: dict, device: str) -> None:
    """
    Entry point for AGOP diagnostic mode.

    Modes:
      A) --agop --save-agop <out.pt>          : compute + save reference AGOP
      B) --agop --agop-ref <ref.pt>            : compute alignment score against saved reference
      C) --agop --agop-ref <ref.pt>
         --agop-output <out.jsonl>             : append alignment result to JSONL log

    Prints: "AGOP rho = <float>" (acceptance test anchor).
    """
    # 1. Load model (dispatch on --model-variant)
    if args.model_variant == "singlez":
        inner = load_model_singlez(args.checkpoint, config, device)
    else:
        inner = load_model(args.checkpoint, config, device)

    # 2. Build probe batch
    vocab_size = config["vocab_size"]
    seq_len = config["seq_len"]
    n_probe = args.agop_n_probe

    probe_source = "synthetic"  # FIX 4: overwritten below if real data supplied
    if args.agop_probe_data is not None:
        probe_path = args.agop_probe_data
        probe_source = probe_path
        # FIX 2: dispatch on os.path.isdir FIRST, then .npy, then .pt.
        # (Previously the directory branch was nested inside `endswith(".npy")`,
        #  so a bare directory fell through to torch.load → IsADirectoryError.)
        if os.path.isdir(probe_path):
            import numpy as np
            inp_path = os.path.join(probe_path, "all__inputs.npy")
            pid_path = os.path.join(probe_path, "all__puzzle_identifiers.npy")
            lab_path = os.path.join(probe_path, "all__labels.npy")
            if not os.path.exists(inp_path):
                raise FileNotFoundError(
                    f"--agop-probe-data directory '{probe_path}' has no all__inputs.npy"
                )
            inputs_np = np.load(inp_path)
            pids_np = (
                np.load(pid_path) if os.path.exists(pid_path)
                else np.zeros(inputs_np.shape[0], dtype=np.int32)
            )
            labels_np = np.load(lab_path) if os.path.exists(lab_path) else None
            total_probe = inputs_np.shape[0]
            batch_size = min(n_probe, total_probe)
            probe_batch = {
                "inputs": torch.from_numpy(inputs_np[:batch_size]).to(dtype=torch.int32, device=device),
                "puzzle_identifiers": torch.from_numpy(pids_np[:batch_size]).to(dtype=torch.int32, device=device),
            }
            if labels_np is not None:
                probe_batch["labels"] = torch.from_numpy(labels_np[:batch_size]).to(dtype=torch.int64, device=device)
            print(
                f"  [info] Loaded {batch_size} real probe examples from "
                f"{probe_path}/all__inputs.npy (labels={'yes' if labels_np is not None else 'no'})"
            )
        elif probe_path.endswith(".npy"):
            # Single .npy file: load sibling puzzle_identifiers + labels if present
            import numpy as np
            probe_dir = os.path.dirname(probe_path)
            inputs_np = np.load(probe_path)
            pid_path = os.path.join(probe_dir, "all__puzzle_identifiers.npy")
            lab_path = os.path.join(probe_dir, "all__labels.npy")
            pids_np = (
                np.load(pid_path) if os.path.exists(pid_path)
                else np.zeros(inputs_np.shape[0], dtype=np.int32)
            )
            labels_np = np.load(lab_path) if os.path.exists(lab_path) else None
            total_probe = inputs_np.shape[0]
            batch_size = min(n_probe, total_probe)
            probe_batch = {
                "inputs": torch.from_numpy(inputs_np[:batch_size]).to(dtype=torch.int32, device=device),
                "puzzle_identifiers": torch.from_numpy(pids_np[:batch_size]).to(dtype=torch.int32, device=device),
            }
            if labels_np is not None:
                probe_batch["labels"] = torch.from_numpy(labels_np[:batch_size]).to(dtype=torch.int64, device=device)
            print(
                f"  [info] Loaded {batch_size} real probe examples from {probe_path} "
                f"(labels={'yes' if labels_np is not None else 'no'})"
            )
        else:
            # .pt dict format (legacy): {'inputs', 'puzzle_identifiers', [labels]}
            probe_data = torch.load(probe_path, map_location=device)
            total_probe = probe_data["inputs"].shape[0]
            batch_size = min(n_probe, total_probe)
            probe_batch = {
                "inputs": probe_data["inputs"][:batch_size].to(device),
                "puzzle_identifiers": probe_data["puzzle_identifiers"][:batch_size].to(device),
            }
            if "labels" in probe_data:
                probe_batch["labels"] = probe_data["labels"][:batch_size].to(device)
    else:
        # Synthetic fallback — WARNING: for real diagnostic runs, use --agop-probe-data
        # pointing to data/sigma_k_10/<k>/test/ (directory with all__inputs.npy) or
        # data/sigma_k_10/<k>/test/all__inputs.npy (single .npy file).
        # (패치 A1 caveat: synthetic probe yields numerically valid but scientifically
        #  meaningless AGOP; random inputs → random gradients → uninformative outer product)
        print(
            "  [warn] --agop-probe-data not provided; using random synthetic probe "
            "(sanity only, not suitable for real diagnostic runs). "
            "Pass data/sigma_k_10/<k>/test/ to use real data."
        )
        # DEVIATION: synthetic fallback caps at 32 regardless of --agop-n-probe
        # (too many random probes don't improve signal; use real data for large n_probe)
        torch.manual_seed(42)
        probe_batch = {
            "inputs": torch.randint(
                0, vocab_size, (min(n_probe, 32), seq_len),
                dtype=torch.int32, device=device
            ),
            "puzzle_identifiers": torch.zeros(
                min(n_probe, 32), dtype=torch.int32, device=device
            ),
        }

    # 3. Compute AGOP at this checkpoint (FIX 3: operating-point warmup + masking)
    agop_scalar = getattr(args, "agop_scalar", "ce")
    if agop_scalar == "mean":
        probe_batch.pop("labels", None)  # force degenerate logits.mean() path
    agop_t = compute_agop(
        inner, probe_batch, device, n_probe=n_probe,
        warmup=getattr(args, "agop_warmup", True),
        mask_valid=True,
        scalar=agop_scalar,
    )

    # FIX 4: build probe signature for ref/target consistency guarding
    probe_meta = _probe_signature(probe_batch, probe_source, n_probe)

    # 4. Mode A: save reference AGOP and exit
    if args.save_agop is not None:
        # FIX 4: save AGOP + probe metadata in a dict (back-compat handled on load).
        torch.save({"agop": agop_t.cpu(), "probe_meta": probe_meta}, args.save_agop)
        print(f"AGOP saved to: {args.save_agop}  shape={tuple(agop_t.shape)}")
        print(f"  [info] probe_meta: {probe_meta}")
        if args.agop_ref is None:
            print("AGOP rho = N/A (no reference provided; saved reference mode)")
            return

    # 5. Mode B/C: compute alignment against reference
    if args.agop_ref is None:
        print("  [info] No --agop-ref provided; skipping alignment computation.")
        return

    # FIX 4: load ref — handle both new dict format and legacy bare tensor.
    ref_obj = torch.load(args.agop_ref, map_location=device)
    if isinstance(ref_obj, dict):
        agop_ref = ref_obj["agop"].to(device)
        ref_meta = ref_obj.get("probe_meta")
    else:
        agop_ref = ref_obj  # legacy bare tensor
        ref_meta = None

    # FIX 4: probe-set consistency guard. Mismatch (e.g. synthetic ref vs real
    # target, or different real probe sets) makes the alignment meaningless.
    if ref_meta is None:
        print(
            "  [warn] reference has no probe_meta (legacy format) — cannot verify "
            "probe-set consistency; ensure ref and target use the SAME probe set."
        )
    elif ref_meta.get("input_sha1") != probe_meta.get("input_sha1"):
        msg = (
            f"probe-set MISMATCH between reference and target:\n"
            f"    ref:    source={ref_meta.get('source')} "
            f"sha1={ref_meta.get('input_sha1')} n={ref_meta.get('n_used')}\n"
            f"    target: source={probe_meta.get('source')} "
            f"sha1={probe_meta.get('input_sha1')} n={probe_meta.get('n_used')}\n"
            f"    Alignment across different probe sets is NOT comparable."
        )
        if getattr(args, "agop_strict_probe", False):
            raise ValueError(f"[FIX 4] {msg}")
        print(f"  [warn] [FIX 4] {msg}")

    rho = agop_alignment(agop_t, agop_ref)
    tr = agop_trace(agop_t)
    eff_rank = agop_effective_rank(agop_t)

    print(f"AGOP rho = {rho:.6f}")
    print(f"AGOP trace = {tr:.6f}")
    print(f"AGOP effective_rank = {eff_rank:.6f}")

    # Pre-rejection check (spec §4)
    if rho < 0.1:
        print(
            "  [WARN] rho < 0.1 — probe may be defective (wrong layer/probe set). "
            "Do NOT interpret as H0 support."
        )

    # 6. Mode C: append to JSONL log
    if args.agop_output is not None:
        record = {
            "checkpoint": args.checkpoint,
            "run_id": args.run_id,
            "model_variant": args.model_variant,
            "agop_ref": args.agop_ref,
            "n_probe": n_probe,
            "K_total": args.agop_n_checkpoints,
            "agop_rho": rho,
            "agop_trace": tr,
            "agop_effective_rank": eff_rank,
        }
        with open(args.agop_output, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  [info] Result appended to: {args.agop_output}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="measure_rho.py — Spectral Radius and AGOP Diagnostics for TRM",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Core flags ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help=(
            "Path to a single checkpoint file (e.g. step_109863).\n"
            "Used by --fixed-point (single-ckpt mode) and --agop."
        ),
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device (default: cuda). For CUDA, set CUDA_VISIBLE_DEVICES externally.",
    )

    # ── Shared flag (Wave 1.5 locked — spectral defines, AGOP uses) ─────────
    parser.add_argument(
        "--model-variant", type=str, default="trm", choices=["trm", "singlez"],
        dest="model_variant",
        help=(
            "Model variant to load.\n"
            "  'trm'     = standard TRM with z_H carry (trm.py)\n"
            "  'singlez' = no-z cohort TRM (trm_singlez.py)\n"
            "Shared flag — used by both spectral and AGOP diagnostics."
        ),
    )

    # ── Spectral-only flags ─────────────────────────────────────────────────
    parser.add_argument(
        "--fixed-point", action="store_true", default=False,
        dest="fixed_point",
        help=(
            "Run q*-convergence + spectral radius at fixed point (PO-SPECTRAL-WT-01).\n"
            "Use --checkpoint for single-ckpt, or --checkpoints-dir for sweep."
        ),
    )
    parser.add_argument(
        "--q-tol", type=float, default=1e-4,
        dest="q_tol",
        help="q*-convergence tolerance ε (spec §6 Step 1: ε = 1e-4). Default: 1e-4",
    )
    parser.add_argument(
        "--q-max-iter", type=int, default=500,
        dest="q_max_iter",
        help="Max iterations for q*-convergence loop (spec §6: max 500). Default: 500",
    )
    parser.add_argument(
        "--save-spectral", type=str, default=None,
        dest="save_spectral",
        help=(
            "Path to write spectral result as JSONL.\n"
            "Single-ckpt mode: one-line JSONL with 11 fields.\n"
            "Sweep mode (--checkpoints-dir): one line per checkpoint."
        ),
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default=None,
        dest="checkpoints_dir",
        help=(
            "Directory of checkpoints for sweep mode.\n"
            "Iterates all step_* files alphabetically.\n"
            "Example: checkpoints/Sigma_k/ABC_k3_baseline"
        ),
    )
    parser.add_argument(
        "--phase-bucket", action="store_true", default=False,
        dest="phase_bucket",
        help=(
            "Print phase bucket summary after sweep:\n"
            "fraction of checkpoints with ρ<0.95 / [0.95,1.05] / >1.05"
        ),
    )

    # ── AGOP-only flags ─────────────────────────────────────────────────────
    parser.add_argument(
        "--agop", action="store_true", default=False,
        help=(
            "Run AGOP diagnostic (compute AGOP alignment).\n"
            "Requires --checkpoint."
        ),
    )
    parser.add_argument(
        "--agop-ref", type=str, default=None,
        dest="agop_ref",
        help="Path to reference AGOP .pt file (final ckpt of successful cohort).",
    )
    parser.add_argument(
        "--save-agop", type=str, default=None,
        dest="save_agop",
        help="If set, save computed AGOP to this path (.pt).",
    )
    parser.add_argument(
        "--agop-output", type=str, default=None,
        dest="agop_output",
        help="JSONL file to append AGOP alignment result (one record per checkpoint).",
    )
    parser.add_argument(
        "--agop-n-probe", type=int, default=512,
        dest="agop_n_probe",
        help="Number of probe examples for AGOP computation (spec §3: 512). Default: 512",
    )
    parser.add_argument(
        "--agop-probe-data", type=str, default=None,
        dest="agop_probe_data",
        help=(
            "Path to probe data. Supports:\n"
            "  .npy file: path to all__inputs.npy (sibling all__puzzle_identifiers.npy auto-loaded)\n"
            "  directory: path to data/sigma_k_10/<k>/test/ (loads all__inputs.npy + all__puzzle_identifiers.npy)\n"
            "  .pt file: dict with 'inputs' and 'puzzle_identifiers' tensors\n"
            "If absent, falls back to synthetic random probe (sanity use only)."
        ),
    )
    parser.add_argument(
        "--agop-n-checkpoints", type=int, default=None,
        dest="agop_n_checkpoints",
        help=(
            "Total number of checkpoints intended in the sweep (K_total).\n"
            "Written as 'K_total' into the JSONL record for Bonferroni correction (audit V7)."
        ),
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        dest="run_id",
        help=(
            "Run identifier string (e.g. 'K3_success_run1').\n"
            "Written into each JSONL record for N≥3-per-cohort grouping (audit V6)."
        ),
    )
    parser.add_argument(
        "--agop-strict-probe", action="store_true", default=False,
        dest="agop_strict_probe",
        help=(
            "FIX 4: error (instead of warn) if --agop-ref probe set differs from\n"
            "the current probe set (input hash mismatch). Default: warn only."
        ),
    )
    # FIX 3 (a): operating-point AGOP via H_cycles-1 warmup. Default ON.
    parser.add_argument(
        "--h-cycles", dest="h_cycles", type=int, default=None,
        help=(
            "Override H_cycles when rebuilding the model, instead of using the value\n"
            "recorded in the checkpoint's all_config.yaml. Legal because the block is\n"
            "weight-tied: no parameter shape depends on the cycle count, so the\n"
            "checkpoint loads unchanged. This is what makes the inference-time depth\n"
            "axis measurable without retraining. Whether ACCURACY survives the change\n"
            "is the question being measured, not an assumption."
        ),
    )
    parser.add_argument(
        "--l-cycles", dest="l_cycles", type=int, default=None,
        help="Override L_cycles when rebuilding the model. See --h-cycles.",
    )
    parser.add_argument(
        "--agop-warmup", dest="agop_warmup", action="store_true", default=True,
        help=(
            "FIX 3: run H_cycles-1 warmup under no_grad to reach the trained\n"
            "operating point before taking the grad-enabled last cycle (default ON).\n"
            "This matches trm.py:208-216 / trm_singlez forward."
        ),
    )
    parser.add_argument(
        "--no-agop-warmup", dest="agop_warmup", action="store_false",
        help="FIX 3: disable warmup; use legacy single-pass grad from L_init (patch A1).",
    )
    parser.add_argument(
        "--agop-scalar", dest="agop_scalar", default="ce",
        choices=["ce", "onehot", "mean"],
        help=(
            "FIX 3: probe scalar for the AGOP gradient.\n"
            "  'ce'     = per-example cross-entropy (example-specific cotangent; default)\n"
            "  'onehot' = true-label logit sum (checkpoint-INDEPENDENT cotangent;\n"
            "             removes prediction-state drift confound)\n"
            "  'mean'   = legacy logits.mean() (degenerate; patch original)"
        ),
    )

    args = parser.parse_args()

    # ── Validate ─────────────────────────────────────────────────────────────
    if not args.fixed_point and not args.agop:
        parser.error("Specify at least one diagnostic: --fixed-point or --agop")

    if args.agop and args.checkpoint is None:
        parser.error("--agop requires --checkpoint")

    # ── Determine checkpoint directory for config loading ────────────────────
    if args.checkpoints_dir is not None:
        config_source_dir = args.checkpoints_dir
    elif args.checkpoint is not None:
        config_source_dir = os.path.dirname(args.checkpoint)
    else:
        parser.error("Provide --checkpoint or --checkpoints-dir")

    try:
        all_config_yaml = load_all_config(config_source_dir)
        config = build_model_config(
            config_source_dir,
            h_cycles=args.h_cycles,
            l_cycles=args.l_cycles,
        )
    except FileNotFoundError as e:
        print(f"[error] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[error] {e}")
        sys.exit(1)

    # A depth-overridden run must never be mistaken for a native-depth measurement,
    # so say so loudly on stdout next to the numbers it produces.
    if args.h_cycles is not None or args.l_cycles is not None:
        _trained = all_config_yaml.get("arch", {})
        print(
            "[depth-override] H_cycles {} -> {} | L_cycles {} -> {} | "
            "checkpoint was TRAINED at the former; these outputs are an "
            "inference-time depth-axis probe, not a native-depth measurement.".format(
                _trained.get("H_cycles"), config.get("H_cycles"),
                _trained.get("L_cycles"), config.get("L_cycles"),
            )
        )

    device = args.device

    # ── Spectral diagnostic ──────────────────────────────────────────────────
    if args.fixed_point:
        if args.checkpoints_dir is not None:
            # Sweep mode
            results = run_fixed_point_sweep(
                checkpoints_dir=args.checkpoints_dir,
                config=config,
                device=device,
                model_variant=args.model_variant,
                tol=args.q_tol,
                max_iter=args.q_max_iter,
                n_power_iter=50,
                save_spectral=args.save_spectral,
                all_config_yaml=all_config_yaml,
            )

            if args.phase_bucket and results:
                n_ordered = sum(1 for r in results if r["phase"] == "ordered")
                n_edge = sum(1 for r in results if r["phase"] == "edge-of-chaos")
                n_chaotic = sum(1 for r in results if r["phase"] == "chaotic")
                n_total = len(results)
                print(f"\n{'='*55}")
                print(f"Phase bucket summary ({n_total} checkpoints):")
                print(f"  ordered      (ρ < 0.95):   {n_ordered}/{n_total} = {n_ordered/n_total:.1%}")
                print(f"  edge-of-chaos [0.95,1.05]: {n_edge}/{n_total}   = {n_edge/n_total:.1%}")
                print(f"  chaotic      (ρ > 1.05):   {n_chaotic}/{n_total} = {n_chaotic/n_total:.1%}")
                print(f"{'='*55}")

        else:
            # Single-checkpoint mode
            measure_rho_fixed_point(
                checkpoint_path=args.checkpoint,
                config=config,
                device=device,
                model_variant=args.model_variant,
                tol=args.q_tol,
                max_iter=args.q_max_iter,
                n_power_iter=50,
                save_spectral=args.save_spectral,
                all_config_yaml=all_config_yaml,
            )

    # ── AGOP diagnostic ──────────────────────────────────────────────────────
    if args.agop:
        run_agop_diagnostic(args, config, device)
