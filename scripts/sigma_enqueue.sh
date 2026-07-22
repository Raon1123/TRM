#!/usr/bin/env bash
# Declarative grid-search enqueuer for scripts/queue_run.sh.
#
# A sweep is a list of "alias=hydra.param: candidate1 candidate2 ..." lines.
# The cartesian product (grid search) of all candidate lists is enqueued for
# every k in K_LIST. Run names are built automatically from the swept values:
#
#   - 1 candidate   -> fixed value, NOT in the run name
#   - 2+ candidates -> swept, run name gets _<alias><value>
#
#   e.g. TRM_SWEEP below produces run names like  k6_trm_halt16_H3_L6
#
# Usage:
#   scripts/sigma_enqueue.sh [run_prefix]              # write job files
#   scripts/sigma_enqueue.sh --dry-run [run_prefix]    # print grid, write nothing
#
# Re-running appends after existing jobs (sequence numbers continue), so you
# can enqueue more sweeps while the runner is going.

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && { DRY_RUN=1; shift; }

QUEUE_DIR="${QUEUE_DIR:-scripts/queue}"
JOBS_DIR="$QUEUE_DIR/jobs"

RUN_PREFIX="${1:-}"
prefix=""
[[ -n "$RUN_PREFIX" ]] && prefix="${RUN_PREFIX}_"

# ====================  EDIT BELOW: fig1 cohort grid  ====================
# Target = the fig1 accuracy grid (analysis/make_mlp_grid_figure.py + tf_vs_mlp_grid),
# re-run on the ORDER-FILTER-FIXED data (data/sigma_k_10 = ord(σ)>k, EXP-007 fix).
# Legacy fig1 lived in project Sigma_k_fig12 on data/sigma_k_10; this is the clean
# re-run under Sigma_k_new. Run names match legacy so the figure code is unchanged.
#
# 8 cohorts = block {mlp,tf} × z {z,noz} × iter {iter,noiter}:
#   block:  mlp -> arch.mlp_t=True    | tf  -> arch.mlp_t=False
#   z:      z   -> arch=trm (z-carry) | noz -> arch=trm_singlez (no z)
#   iter:   iter-> H_cycles=3 L_cycles=6 | noiter -> H_cycles=1 L_cycles=1
#   fixed across all: arch.L_layers=2, arch.halt_max_steps=1  (matches legacy fig1)
# Single seed (=1) for a fast grid; add 2 3 to SEEDS later for min/max bands.

WANDB_PROJECT="Sigma_k_new"
K_LIST=(3 4 5 6 7 8 10)
SEEDS=(1)
DATA_ROOT="data/sigma_k_10"          # canonical n=10, ord(σ)>k-clean (EXP-007 fixed)

# Protocol-matched to legacy fig1 all_config.yaml (Sigma_k_fig12), verified 2026-07-21:
#   eval_interval=2000 (NOT 5000), log_z_dynamics=True (gates probe/test_exact —
#   the figure's primary metric, z_logging.py:378), z_snapshot=False,
#   checkpoint_every_eval=False (cfg default True would checkpoint every eval).
# (+ prefix: log_z_dynamics / z_snapshot are pretrain.py pydantic fields NOT in
#  cfg_pretrain.yaml — Hydra struct mode rejects bare overrides for absent keys.)
common_args="epochs=100000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 +log_z_dynamics=True +z_snapshot=False checkpoint_every_eval=False"

# tag | arch | mlp_t | H_cycles | L_cycles
#
# mlp_t toggles the token-mixing sublayer only: True swaps self-attention for a
# SwiGLU applied across the sequence axis (MLP-Mixer style); the channel-mixing
# SwiGLU, RMSNorm, recursion, z carry and halting are identical either way
# (models/recursive_reasoning/trm.py:70-104).
COHORTS=(
    "tf_z_iter|trm|False|3|6"
    "tf_z_noiter|trm|False|1|1"
    "tf_noz_iter|trm_singlez|False|3|6"
    "tf_noz_noiter|trm_singlez|False|1|1"
    "mlp_z_iter|trm|True|3|6"
    "mlp_z_noiter|trm|True|1|1"
    "mlp_noz_iter|trm_singlez|True|3|6"
    "mlp_noz_noiter|trm_singlez|True|1|1"
)
# FIFO priority tiers for the fig1 grid (PI 2026-07-22): the primary questions
# are z-presence and iteration-presence, and those are read off the attention
# (tf) half of the grid — so the ENTIRE tf half is emitted, across every k,
# before any mlp cell. Within a tier the loop stays k-major so each k gets its
# complete z × iter 2×2 before moving deeper.
COHORT_TIERS=(tf mlp)

# ---- Module ablations (FIFO priority AFTER the fig1 grid) -----------------
# One-factor-at-a-time around the canonical TRM baseline fig1_tf_z_iter
# (arch=trm, mlp_t=False, H3, L6, L_layers=2, halt=1); axis values taken from
# the pre-fig1 exploratory sweep (halt {8,16}, H {6}, L {3}).  Cells equal to
# the baseline itself are already covered by fig1 — not re-enqueued.
#   tag | extra arch overrides (vs baseline)
TRM_ABLATIONS=(
    "halt8|arch.halt_max_steps=8"
    "halt16|arch.halt_max_steps=16"
    "H6|arch.H_cycles=6"
    "L3|arch.L_cycles=3"
)
# transformers_baseline depth/width ablation (old TRANSFORMER_SWEEP grid).
# arch.halt_max_steps=1 pinned — tfb yaml defaults to 16, which the old sweep
# left in place (protocol mismatch vs fig1); here every run is halt=1.
TFB_LAYERS=(1 2 6)
TFB_CYCLES=(1 6)

# ---- Looped vs multi-layer transformer baselines (STAGES=looped) ---------
# Purpose: give TRM a comparison baseline in the sense the literature uses
# "looped transformer" (Yang et al. ICLR 2024; Fan et al. ICLR 2025) — a
# weight-TIED block applied T times with FULL backprop — against an untied
# multi-layer stack of the same effective depth.
#
# What already exists and is NOT re-run here:
#   - multi-layer arm: `transformers_baseline` IS the untied stack. Its
#     `H_cycles` field is dead code (single pass, transformers_baseline.py
#     Model_ACTV2_Inner.forward), which is why the 18 `abl_tfb_*_cyc6_*`
#     cells were duplicates and got skipped on 10.0.12.93. Pinned by
#     tests/test_looped_transformer.py::test_transformers_baseline_ignores_cycles.
#   - TRM anchor: `fig1_tf_z_iter_k*_s1` (arch=trm, H3 L6, L_layers=2).
#
# Head-count matching: the deep arm is re-run at `arch.num_heads=8` rather
# than reusing the existing abl_tfb_* runs, because tfb's yaml default is 12
# (head_dim = 512//12 = 42) while trm and looped_transformer use 8
# (head_dim = 64). Comparing across that would confound the tying axis.
#
# Grid is built as MATCHED PAIRS, not two independent sweeps — the confound
# to avoid is the one already on record for iter on/off (iter↔recurrent-
# compute): effective depth D = H_layers x H_cycles is stated per cell, and
# every cell holds hidden_size=512, num_heads=8, halt_max_steps=1, seed,
# data and protocol fixed to fig1.
#
#   tag        | arch                  | arch overrides                        | D  | params
#   deep2      | transformers_baseline | H_layers=2                            | 2  | 2 blocks
#   deep12     | transformers_baseline | H_layers=12                           | 12 | 12 blocks
#   loop2x6    | looped_transformer    | H_layers=2  H_cycles=6                | 12 | 2 blocks
#   loop2x21   | looped_transformer    | H_layers=2  H_cycles=21               | 42 | 2 blocks
# deep2/deep12/loop2x6 form the minimal complete triangle: loop2x6 is
# depth-matched to deep12 and parameter-matched to deep2, so "does tying buy
# depth" and "does looping beat just being shallow" are both readable off one
# k-row.
# loop2x21 is the TRM-matched cell and the single most informative one here.
# The fig1 TRM anchor applies its 2-layer block (L_cycles+1) x H_cycles = 7x3
# = 21 times, i.e. effective depth 42 at 6,828,034 params — byte-identical
# param count to loop2x6/loop2x21 (verified by instantiation). So loop2x21 is
# matched to fig1_tf_z_iter on BOTH parameters and effective depth, leaving
# exactly three differences: z-carry, the 1-step gradient, and the injection
# schedule — which tiers C and the trm_singlez cohort isolate one at a time.
# NOTE (memory): loop2x21 is the heaviest cell — full BPTT retains ~42 layer
# activations. If it OOMs, the first fallback is global_batch_size=1024 for
# that cell ONLY, recorded as a protocol deviation.
LOOPED_TIER_A=(
    "deep2|transformers_baseline|arch.H_layers=2"
    "deep12|transformers_baseline|arch.H_layers=12"
    "loop2x6|looped_transformer|arch.H_layers=2 arch.H_cycles=6"
    "loop2x21|looped_transformer|arch.H_layers=2 arch.H_cycles=21"
)
# Tier B — how performance depends on the (block-depth x loop-count)
# factorization at fixed D, and on D itself. Restricted to a k subset.
LOOPED_TIER_B=(
    "loop1x12|looped_transformer|arch.H_layers=1 arch.H_cycles=12"
    "loop3x4|looped_transformer|arch.H_layers=3 arch.H_cycles=4"
    "loop6x2|looped_transformer|arch.H_layers=6 arch.H_cycles=2"
    "loop2x3|looped_transformer|arch.H_layers=2 arch.H_cycles=3"
    "loop2x12|looped_transformer|arch.H_layers=2 arch.H_cycles=12"
    "deep4|transformers_baseline|arch.H_layers=4"
    "deep6|transformers_baseline|arch.H_layers=6"
)
# Tier C — the two knobs that separate a canonical looped transformer from
# TRM's recurrence, ablated at fixed architecture (loop2x6):
#   grad1  = TRM's 1-step-gradient approximation (H_cycles-1 under no_grad)
#            but with no z-carry — isolates the approximation from the z axis.
#   noinj  = input embedding injected only on the first cycle.
LOOPED_TIER_C=(
    "loop2x6_grad1|looped_transformer|arch.H_layers=2 arch.H_cycles=6 arch.loop_grad_cycles=1"
    "loop2x6_noinj|looped_transformer|arch.H_layers=2 arch.H_cycles=6 arch.input_injection_every_cycle=False"
)
LOOPED_K_FULL=(3 4 5 6 7 8 10)   # tier A
LOOPED_K_SUBSET=(4 6 8)          # tiers B, C

# emit_job <run_name> <arch> <k> <seed> <arch_args...>
emit_job() {
    local run_name="$1" arch="$2" k="$3" s="$4"; shift 4
    enqueue "$run_name" <<EOF
uv run pretrain.py arch=${arch} ${common_args} \\
    $* \\
    evaluators="[]" \\
    data_paths="[${DATA_ROOT}/${k}]" \\
    seed=${s} \\
    +k=${k} \\
    +project_name="${WANDB_PROJECT}" \\
    +run_name="${run_name}" \\
    ema=True
EOF
}

# Which stage groups to emit — space-separated subset of "fig1 ablation".
# Lets two machines split the grid cleanly (e.g. STAGES=fig1 on one host,
# STAGES=ablation on another) without touching the grid definitions above.
STAGES="${STAGES:-fig1 ablation}"

main() {
    local tier k spec tag arch arch_args mlp_t Hc Lc s lay cyc
    # -- 1) fig1 grid (highest priority: lowest sequence numbers) --
    #    Tier-major, then k-major: all tf cells first (see COHORT_TIERS).
    if [[ " $STAGES " == *" fig1 "* ]]; then
        for tier in "${COHORT_TIERS[@]}"; do
            for k in "${K_LIST[@]}"; do
                for spec in "${COHORTS[@]}"; do
                    IFS='|' read -r tag arch mlp_t Hc Lc <<< "$spec"
                    [[ "$tag" == "${tier}_"* ]] || continue
                    for s in "${SEEDS[@]}"; do
                        emit_job "${prefix}fig1_${tag}_k${k}_s${s}" "$arch" "$k" "$s" \
                            "arch.mlp_t=${mlp_t} arch.H_cycles=${Hc} arch.L_cycles=${Lc}" \
                            "arch.L_layers=2 arch.halt_max_steps=1"
                    done
                done
            done
        done
    fi
    # -- 2) module ablations (appended: run only after fig1 drains, if same host) --
    if [[ " $STAGES " == *" ablation "* ]]; then
        for k in "${K_LIST[@]}"; do
            for spec in "${TRM_ABLATIONS[@]}"; do
                IFS='|' read -r tag arch_args <<< "$spec"
                for s in "${SEEDS[@]}"; do
                    emit_job "${prefix}abl_${tag}_k${k}_s${s}" "trm" "$k" "$s" \
                        "arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6" \
                        "arch.L_layers=2 arch.halt_max_steps=1 ${arch_args}"
                done
            done
            for lay in "${TFB_LAYERS[@]}"; do
                for cyc in "${TFB_CYCLES[@]}"; do
                    for s in "${SEEDS[@]}"; do
                        emit_job "${prefix}abl_tfb_lay${lay}_cyc${cyc}_k${k}_s${s}" \
                            "transformers_baseline" "$k" "$s" \
                            "arch.H_layers=${lay} arch.H_cycles=${cyc} arch.halt_max_steps=1"
                    done
                done
            done
        done
    fi
    # -- 3) looped vs multi-layer transformer baselines (STAGES=looped) --
    #    Not in the default STAGES: this stage is opt-in so re-running the
    #    script for fig1/ablation never silently adds it.
    if [[ " $STAGES " == *" looped "* ]]; then
        emit_looped_tier LOOPED_TIER_A LOOPED_K_FULL
        emit_looped_tier LOOPED_TIER_B LOOPED_K_SUBSET
        emit_looped_tier LOOPED_TIER_C LOOPED_K_SUBSET
    fi
}

# emit_looped_tier <tier_array_name> <k_array_name>  — k-major within a tier
emit_looped_tier() {
    local -n tier_ref="$1"
    local -n k_ref="$2"
    local k spec tag arch arch_args s
    for k in "${k_ref[@]}"; do
        for spec in "${tier_ref[@]}"; do
            IFS='|' read -r tag arch arch_args <<< "$spec"
            for s in "${SEEDS[@]}"; do
                emit_job "${prefix}lt_${tag}_k${k}_s${s}" "$arch" "$k" "$s" \
                    "${arch_args} arch.num_heads=8 arch.halt_max_steps=1"
            done
        done
    done
}

# =================  machinery below, no need to edit  ====================

trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

# Next sequence number across queued/running/done/failed, so appended jobs
# keep FIFO order even after earlier ones complete.
seq_next() {
    local max=0 f n
    for f in "$JOBS_DIR"/*.job "$QUEUE_DIR"/processing/*.job.gpu* \
             "$QUEUE_DIR"/done/*.job "$QUEUE_DIR"/failed/*.job; do
        [[ -e "$f" ]] || continue
        n="$(basename "$f")"
        n="${n%%_*}"
        [[ "$n" =~ ^[0-9]+$ ]] && (( 10#$n > max )) && max=$(( 10#$n ))
    done
    echo $(( max + 1 ))
}

# enqueue <name>  — job body comes from stdin (heredoc)
# Idempotent: a cell whose run_name already exists anywhere in the queue
# lifecycle (queued / running / done / failed) is skipped, so re-running the
# script after edits only adds the missing cells.
enqueue() {
    if compgen -G "$JOBS_DIR/*_$1.job" >/dev/null \
       || compgen -G "$QUEUE_DIR/processing/*_$1.job.gpu*" >/dev/null \
       || compgen -G "$QUEUE_DIR/done/*_$1.job" >/dev/null \
       || compgen -G "$QUEUE_DIR/failed/*_$1.job" >/dev/null; then
        echo "skip (already in queue lifecycle): $1"
        cat > /dev/null
        return
    fi
    if (( DRY_RUN )); then
        printf '%04d %s\n' "$SEQ" "$1"
        cat > /dev/null
    else
        local file
        file="$(printf '%s/%04d_%s.job' "$JOBS_DIR" "$SEQ" "$1")"
        cat > "$file"
        echo "enqueued: $file"
    fi
    SEQ=$(( SEQ + 1 ))
}

# enqueue_grid <name_prefix> <tag> <base_cmd> <data_path> <sweep_array_name>
# Parses the sweep spec into the GRID_* globals, then recursively enqueues
# the full cartesian product.
enqueue_grid() {
    GRID_NAME_PREFIX="$1" GRID_TAG="$2" GRID_BASE="$3" GRID_DATA="$4"
    local -n spec_ref="$5"

    GRID_ALIASES=() GRID_PARAMS=() GRID_VALUES=()
    local line rest
    for line in "${spec_ref[@]}"; do
        GRID_ALIASES+=("$(trim "${line%%=*}")")
        rest="${line#*=}"
        GRID_PARAMS+=("$(trim "${rest%%:*}")")
        GRID_VALUES+=("$(trim "${rest#*:}")")
    done

    emit_grid 0 "" ""
}

# emit_grid <depth> <name_acc> <args_acc>  — one recursion level per param
emit_grid() {
    local depth="$1" name_acc="$2" args_acc="$3"

    if (( depth == ${#GRID_PARAMS[@]} )); then
        local run_name="${GRID_NAME_PREFIX}_${GRID_TAG}${name_acc}"
        enqueue "$run_name" <<EOF
$GRID_BASE \\
    evaluators="[]" \\
    data_paths="[${GRID_DATA}]" \\
    ${args_acc# } \\
    +project_name="${WANDB_PROJECT}" \\
    +run_name="${run_name}" \\
    ema=True
EOF
        return
    fi

    local vals v suffix
    read -r -a vals <<< "${GRID_VALUES[$depth]}"
    for v in "${vals[@]}"; do
        suffix=""
        (( ${#vals[@]} > 1 )) && suffix="_${GRID_ALIASES[$depth]}${v}"
        emit_grid "$(( depth + 1 ))" \
            "${name_acc}${suffix}" \
            "${args_acc} ${GRID_PARAMS[$depth]}=${v}"
    done
}

mkdir -p "$JOBS_DIR"
SEQ="$(seq_next)"
SEQ_START="$SEQ"

main

echo
echo "jobs: $(( SEQ - SEQ_START ))$( (( DRY_RUN )) && echo ' (dry run, nothing written)' )"
echo "now run:  scripts/queue_run.sh        (GPUS=\"4 5 6 7\" by default)"
echo "status:   scripts/queue_run.sh status"
