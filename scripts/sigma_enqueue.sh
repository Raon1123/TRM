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
DRY_RUN_SHOW_BODY="${DRY_RUN_SHOW_BODY:-0}"

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

# ---- PERF-001 P0.1: profiler-only preset (registered M3) -----------------
# Selected EXCLUSIVELY by run prefix `perf0`; STAGES is ignored in that mode,
# so this can never emit a fig1/ablation cell. Field text is byte-identical
# to the registered M3 command in
# lab/reports/2026-07-26_experiment-speed-profiling.md §4.2.1, with two
# documented deviations: (1) the job body never sets CUDA_VISIBLE_DEVICES —
# the queue worker owns it (queue_run.sh:165); (2) the body uses bare
# `uv run` like every other job body — `rtk` is an operator-shell output
# filter, not a registered Hydra field, and would mangle the queue log.
# perf_benchmark stays at its disabled defaults: M3 is a profiler run.
PERF0_PROFILE_DIR="reports/figures/2026-07-26_experiment-speed-profiling/profiles/m3_tf_z_iter_k6_s1"

emit_perf0_m3_job() {
    local run_name="${prefix}m3_tf_z_iter_k6_s1"
    enqueue "$run_name" <<EOF
uv run pretrain.py arch=trm \\
  global_batch_size=2048 epochs=20 eval_interval=20 min_eval_interval=0 \\
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \\
  ema=True ema_rate=0.999 evaluators="[]" checkpoint_every_eval=False \\
  arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6 arch.L_layers=2 \\
  arch.halt_max_steps=1 arch.forward_dtype=bfloat16 \\
  data_paths="[data/sigma_k_10/6]" seed=1 +k=6 \\
  +log_z_dynamics=True +z_snapshot=False \\
  +project_name=Sigma_k_perf +run_name=${run_name} \\
  perf_profiler.enabled=True perf_profiler.performance_only=True \\
  perf_profiler.output_dir=${PERF0_PROFILE_DIR} \\
  perf_profiler.wait=10 perf_profiler.warmup=10 perf_profiler.active=20 \\
  perf_profiler.repeat=1 perf_profiler.max_steps=40 \\
  perf_profiler.record_shapes=True perf_profiler.profile_memory=True \\
  perf_profiler.with_stack=True perf_profiler.export_chrome_trace=True \\
  perf_profiler.export_tensorboard_trace=True
EOF
}

# ---- PERF-001 P0.1a: unprofiled benchmark preset (registered M2) ---------
# Closes PERF-DEV-14 ("no runnable M2 path ships"): before this, `--dry-run
# perf0` emitted the M3 profiler job only, so the M2 baseline the plan requires
# to run *as a queue job body* had nothing to enqueue.
#
# Field text is byte-identical to the registered M2 command in
# lab/reports/2026-07-26_experiment-speed-action-plan.md §P0.1a (lines 99-109),
# with the same two documented deviations as the M3 emitter above:
# (1) the job body never sets CUDA_VISIBLE_DEVICES — the queue worker owns it
# (queue_run.sh:165); (2) the body drops the operator-shell `rtk` prefix
# (PERF-DEV-16 in the 2026-07-28 ratification ledger of the same document).
#
# The timing contract registers three repeats r1..r3 with "seed=1 and same
# reserved GPU; repeat/seed/hardware differences are provenance, not pooled
# silently". This emitter discharges the repeat/provenance half ONLY: it emits
# three separate jobs whose +run_name=..._r<n> and
# perf_benchmark.output_dir=..._r<n> differ and whose every other field is
# identical, so the three repeats can never be silently pooled into one
# artifact set. The `_r<n>` suffix is the convention utils/perf_benchmark.py
# already parses — derive_condition_id() strips it to the shared condition,
# derive_repeat() reads the repeat index off it (see _REPEAT_SUFFIX and those
# two functions).
#
# The "same reserved GPU" half is NOT discharged here and is NOT recorded as a
# numbered deviation in the 2026-07-28 PERF-DEV ledger. Emitting three jobs
# gives no placement or ordering guarantee whatsoever: queue_run.sh:180-183
# spawns one independent worker PER GPU (default GPUS="4 5 6 7"), each worker
# claims the oldest job it can claim (queue_run.sh:144-146) regardless of which
# condition it belongs to, and runs it on its own device (queue_run.sh:165).
# With the default worker set, r1/r2/r3 are claimable concurrently by three
# workers on three different GPUs — three timing baselines contending for the
# same host CPU/PCIe/power while each measures steady state. A single-worker
# invocation does not fix this either: worker_loop takes the oldest job in the
# WHOLE FIFO, not the oldest perf0 job, so unrelated science jobs (and the M3
# profiler job emitted just above, which precedes r1) would interleave on the
# reserved device. Satisfying "same reserved GPU" is therefore an operator
# obligation at run time, outside anything this file can enforce, and the gap
# belongs to the pre-registration owner to number and rule on.
#
# PERF0_BENCH_DIR_BASE is deliberately UNprefixed (like PERF0_PROFILE_DIR):
# the registered artifact path is .../data/m2_tf_z_iter_k6_s1_r<n>, whereas the
# run name carries the `perf0_` queue prefix. Do not build one from the other.
#
# Caveat carried from PERF-DEV-06: perf_benchmark.max_steps bounds ROW
# COLLECTION, not training length (unlike perf_profiler.max_steps). The real
# training bound of this preset is Hydra `epochs=2000`.
PERF0_BENCH_DIR_BASE="reports/figures/2026-07-26_experiment-speed-profiling/data/m2_tf_z_iter_k6_s1"
PERF0_M2_REPEATS=(1 2 3)

# emit_perf0_m2_job <repeat_index>
emit_perf0_m2_job() {
    local r="$1"
    local run_name="${prefix}m2_tf_z_iter_k6_s1_r${r}"
    enqueue "$run_name" <<EOF
uv run pretrain.py arch=trm \\
  global_batch_size=2048 epochs=2000 eval_interval=2000 min_eval_interval=0 \\
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \\
  ema=True ema_rate=0.999 evaluators="[]" checkpoint_every_eval=False \\
  arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6 arch.L_layers=2 \\
  arch.halt_max_steps=1 arch.forward_dtype=bfloat16 \\
  data_paths="[data/sigma_k_10/6]" seed=1 +k=6 \\
  +log_z_dynamics=True +z_snapshot=False \\
  +project_name=Sigma_k_perf +run_name=${run_name} \\
  perf_profiler.enabled=False \\
  perf_benchmark.enabled=True perf_benchmark.performance_only=True \\
  perf_benchmark.warmup_steps=50 perf_benchmark.measured_steps=200 \\
  perf_benchmark.eval_event_step=2000 perf_benchmark.max_steps=2000 \\
  perf_benchmark.output_dir=${PERF0_BENCH_DIR_BASE}_r${r}
EOF
}

# Which stage groups to emit — space-separated subset of "fig1 ablation".
# Lets two machines split the grid cleanly (e.g. STAGES=fig1 on one host,
# STAGES=ablation on another) without touching the grid definitions above.
STAGES="${STAGES:-fig1 ablation}"

main() {
    local tier k spec tag arch arch_args mlp_t Hc Lc s lay cyc r
    # -- 0) PERF-001 profiler-only preset: prefix-selected and exclusive --
    if [[ "$RUN_PREFIX" == "perf0" ]]; then
        if (( ! DRY_RUN )) && [[ "${PERF0_ALLOW_ENQUEUE:-0}" != "1" ]]; then
            echo "ERROR: perf0 is dry-run-only until the G3 enqueue gate is signed" >&2
            echo "       (profiling doc §4.1: 'no production enqueue in this registration';" >&2
            echo "        §4.2 G3 requires an independent verifier PASS)." >&2
            echo "       Re-run with --dry-run, or set PERF0_ALLOW_ENQUEUE=1 after G3." >&2
            exit 1
        fi
        DRY_RUN_SHOW_BODY=1
        emit_perf0_m3_job
        for r in "${PERF0_M2_REPEATS[@]}"; do
            emit_perf0_m2_job "$r"
        done
        return 0
    fi
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
        if (( DRY_RUN_SHOW_BODY )); then
            while IFS= read -r _line; do printf '    | %s\n' "$_line"; done
        else
            cat > /dev/null
        fi
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
