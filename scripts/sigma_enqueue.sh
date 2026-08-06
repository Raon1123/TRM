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
#   scripts/sigma_enqueue.sh [run_prefix] --dry-run    # same; order does not matter
#
# WARNING: with no --dry-run this writes real job bodies into scripts/queue/jobs/,
# where the runner can claim them immediately.  --dry-run is accepted in any
# position and an unrecognised flag aborts rather than being read as a prefix.
#
# Re-running appends after existing jobs (sequence numbers continue), so you
# can enqueue more sweeps while the runner is going.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing.
#
# --dry-run is honoured in ANY position, and an unrecognised flag is a HARD
# ERROR rather than a run prefix.  Both halves matter, and the reason is an
# incident, not a style preference:
#
#   2026-08-06, `sigma_enqueue.sh pp --dry-run` was run intending a dry run.
#   The old parser tested only "$1" for --dry-run, so "pp" became RUN_PREFIX,
#   "--dry-run" was silently ignored as a stray argument, DRY_RUN stayed 0, and
#   216 real job bodies were written into the live scripts/queue/jobs/ where
#   they sat claimable by the runner for ~39 s before being pulled back out.
#
# A silently-swallowed flag on a script whose default action is "enqueue real
# training jobs" is a one-keystroke path to an unintended production launch, so
# anything starting with "-" that is not understood now stops the script.
# ---------------------------------------------------------------------------
DRY_RUN=0
_positional=()
while (( $# )); do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            # Print the leading comment block by STRUCTURE, not by a hardcoded
            # line range: a range goes stale the moment the header is edited
            # (and silently truncates the queue-writing warning mid-sentence,
            # which is the one line a --help reader most needs to see).
            awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' \
                "$0" >&2
            exit 0
            ;;
        --) shift; _positional+=("$@"); break ;;
        -*)
            echo "ERROR: unknown option '$1'." >&2
            echo "       Refusing to continue: this script ENQUEUES REAL TRAINING" >&2
            echo "       JOBS by default, and treating an unrecognised flag as a run" >&2
            echo "       prefix is how 216 jobs were once enqueued by a command that" >&2
            echo "       was meant to be a dry run.  Did you mean --dry-run?" >&2
            exit 2
            ;;
        *) _positional+=("$1") ;;
    esac
    shift
done
if (( ${#_positional[@]} > 1 )); then
    echo "ERROR: expected at most one run prefix, got: ${_positional[*]}" >&2
    exit 2
fi
set -- "${_positional[@]+"${_positional[@]}"}"
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

# Campaign identity. Every value here is env-overridable so a re-target does not
# need a source edit:  WANDB_PROJECT=... K_LIST="3 4" ./sigma_enqueue.sh
#
# 2026-08-06: default project moved Sigma_k_new -> power_permutation. The old
# name now denotes a SEALED generation (see
# scripts/queue/sealed-20260806-pre-power-permutation/README.md); nothing new
# should be written into it. wandb's `project` is flat, so campaign arms are
# separated by RUN_GROUP (wandb group), not by more project names.
WANDB_PROJECT="${WANDB_PROJECT:-power_permutation}"
RUN_GROUP="${RUN_GROUP:-}"           # per-stage default assigned at emit time

# 2026-08-06: extended to EVERY k directory that exists under D0 (user
# decision "1부터 전부로 D0 확장"). All 14 are content-verified clean --
# min ord(sigma) = the smallest order actually achievable above k, and
# frac(ord<=k) = 0.000 in both train and test for every one of them.
#
# The achievable orders in S_10 are {1,2,...,10,12,14,15,20,21,30}, which
# makes the pools coarser as k grows and produces two facts worth knowing
# before reading any k-sweep as a smooth curve:
#   k=1   -> ord>1  : sigma^1 = sigma, an identity-copy task; near-trivial.
#   k=9   -> ord>9  : {10,12,14,15,20,21,30}
#   k=10  -> ord>10 : {12,14,15,20,21,30}
#   k=11  -> ord>11 : {12,14,15,20,21,30}  <-- IDENTICAL to k=10, because
#           S_10 has NO permutation of order 11 (11 is prime > 10). k=11 is a
#           same-distribution exponent comparison against k=10, not new data.
#   k=12  -> ord>12 : {14,15,20,21,30}
#   k=16  -> ord>16 : {20,21,30}
#   k=20  -> ord>20 : {21,30}      <-- narrowest pool in the sweep
# So k=10/k=11 are a matched pair, and k>=12 progressively narrows the
# distribution rather than merely raising difficulty.
K_LIST=(${K_LIST:-1 2 3 4 5 6 7 8 9 10 11 12 16 20})
SEEDS=(${SEEDS:-1})

# ---- Staged (one-factor-at-a-time) exploration, 2026-08-06 ----------------
# Replaces "enqueue the whole 378-cell cartesian product at once".
#
# Why. The full product spends most of its budget on cells whose answer is
# already implied by cheaper ones: k=1 is an identity-copy task, k=10 and k=11
# draw from an IDENTICAL pool (S_10 has no order-11 element), and the k>=12
# pools narrow monotonically. Meanwhile the phenomena actually under
# investigation -- the H-collapse band and the L3 non-monotonicity -- sit in a
# narrow mid-k window. A grid answers every question at the same low
# resolution; running one factor at a time answers the load-bearing ones first
# and lets each stage choose the next stage's k values.
#
# K_DIAG = the diagnostic k set the ABLATION stages sweep, deliberately much
# smaller than K_LIST. The default brackets the observed transition: 3 is
# reliably learnable, 5/6/7 is where collapse and non-monotonicity appear, 10
# is the far end. Override once stage 1 has told you where the real boundary
# is on THIS data generation -- that is the entire point of staging.
K_DIAG=(${K_DIAG:-3 5 6 7 10})

# The single reference cell every ablation perturbs by exactly one factor.
# Matches the legacy fig1 baseline (tf = attention block, z-carry, iterating).
BASELINE_COHORT="${BASELINE_COHORT:-tf_z_iter}"
DATA_ROOT="${DATA_ROOT:-data/sigma_k_10}"   # canonical n=10, ord(σ)>k-clean (EXP-007 fixed)

# Protocol-matched to legacy fig1 all_config.yaml (Sigma_k_fig12), verified 2026-07-21:
#   eval_interval=2000 (NOT 5000), log_z_dynamics=True (gates probe/test_exact —
#   the figure's primary metric, z_logging.py:378), z_snapshot=False,
#   checkpoint_every_eval=False (cfg default True would checkpoint every eval).
# (+ prefix: log_z_dynamics / z_snapshot are pretrain.py pydantic fields NOT in
#  cfg_pretrain.yaml — Hydra struct mode rejects bare overrides for absent keys.)
# 2026-08-06: z_snapshot flipped False -> True for the power_permutation
# generation. Rationale (design doc §4.2b): the open question is whether an
# H-collapse cell (train_exact~1.0, test_exact~0.0) is ALSO a representational
# collapse (low participation ratio) or a purely accuracy-side failure.
# `z/eff_rank` = PR(z_H) already answers that and is already logged -- the only
# thing missing was that the snapshot path was switched off. This is additive
# telemetry: z_snapshot only controls whether snapshots are written to
# checkpoint_path (pretrain.py `_snap_path`), it does not alter training or any
# existing metric. Legacy fig1 protocol-match is therefore preserved for every
# metric the old figures consume.
# 2026-08-06: per-eval prediction dump ENABLED (eval_save_outputs).
# Requested by intake 20260806T052122Z-c65f6d6d0f, which asks for a raw
# per-eval prediction artifact so that E_comm, valid-permutation rate and the
# d(k) trivial-position split can be computed ENTIRELY DOWNSTREAM -- keeping
# server-side metric code at zero lines. No new code was needed: the machinery
# already exists and was merely inert, gated on this one empty list.
#   models/losses.py:63    preds = argmax(logits)
#   models/losses.py:100   filtered to return_keys
#   pretrain.py:531-535    collects any key named in eval_save_outputs
#   pretrain.py:558-565    torch.save -> checkpoint_path/step_<N>_all_preds.<rank>
# It fires inside evaluate(), so it is independent of checkpoint_every_eval:
# dumps land at EVERY eval interval even though full weights are only written
# at the final iteration.
# Keys chosen: preds/labels/inputs/puzzle_identifiers -- enough to make each
# dump self-describing without a join back to the dataset. Deliberately NOT
# `logits` (vastly larger, and every requested statistic needs only the decoded
# sequence). Cost ~136 KB/eval, ~7 MB/run, ~2.6 GB across the 378-job grid.
# 2026-08-06: epochs 100000 -> 50000 (grid-search budget).
# `epochs` is NOT the optimizer step count. pretrain.py derives
#   total_steps = epochs * total_groups * mean_puzzle_examples / global_batch_size
# and with total_groups=5000, mean_puzzle_examples=1.0 (dataset.json) and
# global_batch_size=2048 (config/cfg_pretrain.yaml):
#   50000 -> 122,070 steps    (100000 -> 244,140;  10000 -> only 24,414)
# NOTE eval_interval is in EPOCHS, not steps (pretrain.py:717), so
# eval_interval=2000 gives epochs/2000 evaluations: 25 here, 50 before.
#
# Why 50000 and not less. Empirically, from the 31 runs in
# reports/figures/2026-08-04_ablation-backlog/ablation_terminal.csv that reached
# peak_test_exact >= 0.99, the step at which they peaked is strongly structured
# BY k -- and truncation therefore is not uniform noise, it is a k-dependent
# bias that manufactures a fake k-curve:
#     budget      k=5    k=6    k=7    total captured
#     24,414 st   0/3    0/3    0/4    15/31  (48%)   <- kills mid-k entirely
#     48,828 st   1/3    2/3    1/4    22/31  (71%)
#     97,656 st   1/3    3/3    3/4    28/31  (90%)
#    122,070 st   3/3    3/3    3/4    30/31  (97%)   <- this budget
# k=5 (median peak 107,404 steps) only becomes whole at this budget, and k=5 is
# precisely the cell where the L3 non-monotone collapse was observed -- a
# shorter budget would kill the one cell the campaign most wants to read.
# The single remaining loss is one k=7 run peaking at 161,106; re-run that cell
# individually at a longer budget if it matters.
# Cost: ~1,142 GPU-h for the 378-job grid (~12 days on 4 GPUs), half of the
# 100000-epoch budget.
# CAVEAT: those 52 abl_* runs are cloud-only, so their data_epoch is `unknown`
# (not confirmed clean). The peak_step distribution is used here only as a
# measure of how long this task/architecture takes to converge, which is
# argued to be usable regardless of the ord-filter contamination question.
# NOTE the `+` on eval_save_outputs. It is a PretrainConfig field
# (pretrain.py `eval_save_outputs: List[str] = []`) but is NOT declared in
# config/cfg_pretrain.yaml, so Hydra's struct mode rejects a bare assignment
# with "Could not override 'eval_save_outputs' ... not in struct" and the job
# dies before the first step. Same reason log_z_dynamics/z_snapshot carry `+`.
# Caught 2026-08-06 by `--cfg job` pre-flight on a real enqueued body; the key
# was added to this line earlier the same day without the prefix, which would
# have failed every job in the campaign at launch.
common_args="${COMMON_ARGS:-epochs=50000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 +log_z_dynamics=True +z_snapshot=True checkpoint_every_eval=False +eval_save_outputs=[inputs,labels,preds,puzzle_identifiers]}"

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
# 2026-08-06: the single TRM_ABLATIONS list was split along the two axes the
# campaign actually reasons about, so each lands in its own wandb group and can
# be enqueued independently (STAGES="ablation_act" vs "ablation_arch").
# Union of the two below == the pre-split list plus the finer ACT rungs.

# --- ACT / halting axis (adaptive computation time: how many halt steps) ------
# The old list only had halt=8 and 16 against the fig1 baseline halt=1, leaving
# a 1 -> 8 gap with nothing in it. halt2/halt4 fill it so a monotone-in-halt
# claim can actually be checked rather than interpolated across one octave.
ACT_ABLATIONS=(
    "halt2|arch.halt_max_steps=2"
    "halt4|arch.halt_max_steps=4"
    "halt8|arch.halt_max_steps=8"
    "halt16|arch.halt_max_steps=16"
)

# --- model-structure axis (recurrence depth / width) --------------------------
# H_cycles = outer recurrence, L_cycles = inner, L_layers = block depth.
# L3 is retained because it is the one cell with a KNOWN non-monotone failure
# (k=5,6 collapsed while k=3,4,7,8,10 reached 1.0) -- see design doc §4.1.
ARCH_ABLATIONS=(
    "H6|arch.H_cycles=6"
    "H1|arch.H_cycles=1"
    "L3|arch.L_cycles=3"
    "L12|arch.L_cycles=12"
    "lay1|arch.L_layers=1"
    "lay4|arch.L_layers=4"
)
# transformers_baseline depth/width ablation (old TRANSFORMER_SWEEP grid).
# arch.halt_max_steps=1 pinned — tfb yaml defaults to 16, which the old sweep
# left in place (protocol mismatch vs fig1); here every run is halt=1.
TFB_LAYERS=(${TFB_LAYERS:-1 2 6})

# 2026-08-06 — TFB_CYCLES REMOVED. `arch.H_cycles` is a NO-OP for
# arch=transformers_baseline: in models/recursive_reasoning/transformers_baseline.py
# the identifier occurs exactly twice, at the docstring (L7, "REMOVED inner
# cycles (no H_cycles/L_cycles loops within reasoning)") and the pydantic field
# declaration (L52). The only range() in the file is over H_layers (L168), and
# Model_ACTV2ReasoningModule.forward (L109-116) applies each layer once with no
# cycle loop. So the old cyc1/cyc6 sweep built IDENTICAL models and burned half
# its budget on duplicates.
#
# What that accident may have measured is worth keeping. `abl_tfb_lay2_cyc1_k3_s1`
# and `abl_tfb_lay2_cyc6_k3_s1` reached final_test_exact 0.980 vs 0.164 (peak 1.0
# both). H_cycles occurs 0 times in pretrain.py / utils/ / models/layers.py /
# models/common.py -- only in the arch YAMLs, where transformers_baseline.yaml:9
# says "H_cycles: 1  # kept for compatibility" -- so parameter count, init path,
# LR schedule and total_steps are all independent of it, and the two runs were
# structurally the same model.
# CAVEAT: both runs are cloud-only (no local wandb dir, no local checkpoint), so
# their all_config.yaml could NOT be diffed directly, and "same seed" rests on
# the ablation CSV's seed column. Treat the 0.980/0.164 gap as strong evidence
# of run-to-run nondeterminism, not as confirmed.
# Either way the cycles axis had to go (it is a no-op), and spending that budget
# on a real seed axis is right under both readings: if the variance is real the
# seeds are required, and if it is not, the repeats explain what actually
# differed.
TFB_SEEDS=(${TFB_SEEDS:-1 2 3})

# CELL_ID is set by each stage before calling emit_job (see the stages below)
# and is emitted into the run config verbatim. It is the machine-readable grid
# coordinate; run_name remains the human-facing label. Intake
# 20260806T052122Z-c65f6d6d0f asked for `k{K}_z{0|1}_it{0|1}_N{N}_s{SEED}`,
# which only spans the old fig1 tau-grid axes and cannot express the halt/H/L/
# lay/tfb ablation arms at all -- so the intent (mechanical run -> cell
# mapping) is preserved here in a form that covers every arm.
CELL_ID=""

# emit_job <run_name> <arch> <k> <seed> <arch_args...>
#
# NOTE the '"..."' nesting on +cell_id, which is NOT cosmetic.  The value is a
# comma-separated key=value string ("arm=baseline,block=tf,z=1,..."), and Hydra
# parses each override's VALUE with its own grammar, in which a bare inner '='
# is a syntax error:
#
#     +cell_id=arm=baseline,block=tf,...
#       -> "mismatched input '=' expecting <EOF>"
#
# A plain "${CELL_ID}" does not help, because the job body is a bash script:
# bash strips those quotes and execs argv WITHOUT them, so Hydra still sees the
# bare form.  The quotes have to survive bash and reach Hydra, hence single
# quotes wrapping double quotes.  Verified with `--cfg job`: the parsed value
# is the clean string 'arm=baseline,block=tf,k=5,s=1', no quotes retained.
# The other +key="..." lines below are safe unquoted (no '=' or ',' in their
# values) and are left as they are so this comment marks the one real hazard.
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
    +run_group="${RUN_GROUP}" \\
    +run_name="${run_name}" \\
    +cell_id='"${CELL_ID}"' \\
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
# 2026-08-06: "ablation" split into ablation_act / ablation_arch / tfb (each its
# own wandb group). The bare legacy token "ablation" is still accepted and
# expands to all three, so any existing caller keeps its old meaning.
# 2026-08-06: the DEFAULT is now stage 1 alone, not the whole product.
# Running `sigma_enqueue.sh pp` used to emit 378 jobs; it now emits 14 -- the
# baseline k-sweep. Each later stage is enqueued deliberately, after reading
# the previous one:
#
#   STAGES=baseline       14 jobs   baseline cohort across the full K_LIST.
#                                   Establishes WHERE the task breaks on this
#                                   data generation. Everything downstream
#                                   picks its k values from this result.
#   STAGES=seedvar        ~10 jobs  seeds 2,3 on K_DIAG for the baseline only.
#                                   Answers "is a single-seed cell readable at
#                                   all?" -- and there is direct reason to
#                                   doubt it: two runs of an identical config
#                                   at the same seed reached final_test_exact
#                                   0.980 vs 0.164 (see the TFB_SEEDS note).
#                                   If dispersion is large, EVERY single-seed
#                                   comparison below is uninterpretable and
#                                   the campaign must widen seeds before
#                                   spending anything on ablations.
#   STAGES=ablation_act   20 jobs   halt in {2,4,8,16} on K_DIAG. The ONLY
#                                   stage that yields a per-ACT-step
#                                   trajectory, so it is the only substrate
#                                   for the tau(h)/PR(h)/MI(h) instrumentation.
#   STAGES=ablation_arch  30 jobs   one architecture factor at a time on K_DIAG.
#   STAGES=cohorts        40 jobs   the 2x2x2 block x z x iter factorial on
#                                   K_DIAG. Genuinely a grid -- kept last
#                                   because it is the only stage whose
#                                   questions need joint variation.
#   STAGES=tfb            45 jobs   transformer baseline, 3 seeds on K_DIAG.
#
# Legacy tokens still work: "fig1" == "cohorts" over the full K_LIST, and
# "ablation" expands to all three ablation stages, so any existing caller
# keeps its old meaning.
STAGES="${STAGES:-baseline}"
[[ " $STAGES " == *" ablation "* ]] && STAGES="$STAGES ablation_act ablation_arch tfb"

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
    # -- 1a) STAGE 1: baseline k-sweep. ONE cohort, every k, one seed. --
    # This is the map the rest of the campaign is read against: it locates the
    # k at which the baseline stops learning. Nothing else should be enqueued
    # until this is read, because every later stage picks its K_DIAG from it.
    if [[ " $STAGES " == *" baseline "* ]]; then
        RUN_GROUP="${RUN_GROUP_BASELINE:-baseline}"
        for k in "${K_LIST[@]}"; do
            for spec in "${COHORTS[@]}"; do
                IFS='|' read -r tag arch mlp_t Hc Lc <<< "$spec"
                [[ "$tag" == "$BASELINE_COHORT" ]] || continue
                local blk zf itf
                blk="${tag%%_*}"
                case "$tag" in *_noz_*) zf=0;; *) zf=1;; esac
                case "$tag" in *_noiter) itf=0;; *) itf=1;; esac
                for s in "${SEEDS[@]}"; do
                    CELL_ID="arm=baseline,block=${blk},z=${zf},it=${itf},n=10,k=${k},s=${s}"
                    emit_job "${prefix}base_${tag}_k${k}_s${s}" "$arch" "$k" "$s" \
                        "arch.mlp_t=${mlp_t} arch.H_cycles=${Hc} arch.L_cycles=${Lc}" \
                        "arch.L_layers=2 arch.halt_max_steps=1"
                done
            done
        done
    fi
    # -- 1b) STAGE 2: seed variance on the baseline, at K_DIAG only. --
    # Gate for everything after it. If identical configs disperse widely, no
    # single-seed ablation cell below can be read as a real effect.
    if [[ " $STAGES " == *" seedvar "* ]]; then
        RUN_GROUP="${RUN_GROUP_SEEDVAR:-seedvar}"
        for k in "${K_DIAG[@]}"; do
            for spec in "${COHORTS[@]}"; do
                IFS='|' read -r tag arch mlp_t Hc Lc <<< "$spec"
                [[ "$tag" == "$BASELINE_COHORT" ]] || continue
                local blk zf itf
                blk="${tag%%_*}"
                case "$tag" in *_noz_*) zf=0;; *) zf=1;; esac
                case "$tag" in *_noiter) itf=0;; *) itf=1;; esac
                for s in ${SEEDVAR_SEEDS:-2 3}; do
                    CELL_ID="arm=seedvar,block=${blk},z=${zf},it=${itf},n=10,k=${k},s=${s}"
                    emit_job "${prefix}base_${tag}_k${k}_s${s}" "$arch" "$k" "$s" \
                        "arch.mlp_t=${mlp_t} arch.H_cycles=${Hc} arch.L_cycles=${Lc}" \
                        "arch.L_layers=2 arch.halt_max_steps=1"
                done
            done
        done
    fi
    # -- 1c) cohort factorial (legacy name: fig1). Genuinely a grid, so it --
    #    runs LAST and only over K_DIAG unless the legacy "fig1" token is used.
    if [[ " $STAGES " == *" cohorts "* || " $STAGES " == *" fig1 "* ]]; then
        RUN_GROUP="${RUN_GROUP_FIG1:-fig1}"
        local -a COHORT_KS
        if [[ " $STAGES " == *" fig1 "* ]]; then COHORT_KS=("${K_LIST[@]}"); else COHORT_KS=("${K_DIAG[@]}"); fi
        for tier in "${COHORT_TIERS[@]}"; do
            for k in "${COHORT_KS[@]}"; do
                for spec in "${COHORTS[@]}"; do
                    IFS='|' read -r tag arch mlp_t Hc Lc <<< "$spec"
                    [[ "$tag" == "${tier}_"* ]] || continue
                    # tag is "<block>_<z>_<iter>", e.g. tf_z_iter / mlp_noz_noiter.
                    # Decode it into the axes the old cell_id convention named,
                    # rather than shipping the tag string and making every
                    # consumer re-learn the grammar.
                    local blk zf itf
                    blk="${tag%%_*}"                       # mlp | tf
                    case "$tag" in *_noz_*) zf=0;; *) zf=1;; esac
                    case "$tag" in *_noiter) itf=0;; *) itf=1;; esac
                    for s in "${SEEDS[@]}"; do
                        CELL_ID="arm=fig1,block=${blk},z=${zf},it=${itf},n=10,k=${k},s=${s}"
                        emit_job "${prefix}fig1_${tag}_k${k}_s${s}" "$arch" "$k" "$s" \
                            "arch.mlp_t=${mlp_t} arch.H_cycles=${Hc} arch.L_cycles=${Lc}" \
                            "arch.L_layers=2 arch.halt_max_steps=1"
                    done
                done
            done
        done
    fi
    # -- 2a) ACT / halting ablation --
    # NOTE the ${arch_args} suffix: it comes AFTER the pinned
    # `arch.halt_max_steps=1` baseline, so a later duplicate key wins under
    # Hydra. That ordering is what makes halt2/4/8/16 actually take effect.
    #
    # z_seq_metrics is set HERE, in the stage body, rather than in common_args
    # or at the call site, because it is a property of the stage and not of the
    # invocation: the depth instrumentation (zseq/ zmi/ ztau/ zperm/) reduces
    # over the per-ACT-step trajectory, and ablation_act is the only stage that
    # produces one.  Every other stage pins arch.halt_max_steps=1, i.e. exactly
    # one ACT step, where those keys would be emitted as flat single-point
    # series -- readable as "the metric is constant" when the truth is "there
    # was nothing to vary".  Setting it stage-locally makes that impossible to
    # get wrong by forgetting an env var.  ACT_SEQ_METRICS=False turns it off
    # without touching this file.
    if [[ " $STAGES " == *" ablation_act "* ]]; then
        RUN_GROUP="${RUN_GROUP_ACT:-ablation_act}"
        for k in "${K_DIAG[@]}"; do
            for spec in "${ACT_ABLATIONS[@]}"; do
                IFS='|' read -r tag arch_args <<< "$spec"
                for s in "${SEEDS[@]}"; do
                    CELL_ID="arm=act,halt=${tag#halt},n=10,k=${k},s=${s}"
                    emit_job "${prefix}abl_${tag}_k${k}_s${s}" "trm" "$k" "$s" \
                        "arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6" \
                        "arch.L_layers=2 arch.halt_max_steps=1 ${arch_args}" \
                        "+z_seq_metrics=${ACT_SEQ_METRICS:-True}"
                done
            done
        done
    fi
    # -- 2b) model-structure ablation --
    if [[ " $STAGES " == *" ablation_arch "* ]]; then
        RUN_GROUP="${RUN_GROUP_ARCH:-ablation_arch}"
        for k in "${K_DIAG[@]}"; do
            for spec in "${ARCH_ABLATIONS[@]}"; do
                IFS='|' read -r tag arch_args <<< "$spec"
                for s in "${SEEDS[@]}"; do
                    # Decode the tag into semantic keys rather than shipping
                    # the tag string -- otherwise a consumer still has to know
                    # the tag grammar, which is exactly what cell_id exists to
                    # avoid (the other three arms already decode).
                    local akey aval
                    case "$tag" in
                        H*)   akey=Hc;     aval="${tag#H}"   ;;
                        L*[0-9]) akey=Lc;  aval="${tag#L}"   ;;
                        lay*) akey=layers; aval="${tag#lay}" ;;
                        *)    akey=axis;   aval="$tag"       ;;
                    esac
                    CELL_ID="arm=arch,${akey}=${aval},n=10,k=${k},s=${s}"
                    emit_job "${prefix}abl_${tag}_k${k}_s${s}" "trm" "$k" "$s" \
                        "arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6" \
                        "arch.L_layers=2 arch.halt_max_steps=1 ${arch_args}"
                done
            done
        done
    fi
    # -- 2c) transformer-baseline depth/width grid --
    if [[ " $STAGES " == *" tfb "* ]]; then
        RUN_GROUP="${RUN_GROUP_TFB:-tfb}"
        for k in "${K_DIAG[@]}"; do
            for lay in "${TFB_LAYERS[@]}"; do
                for s in "${TFB_SEEDS[@]}"; do
                    # arch.H_cycles deliberately NOT passed -- it is a no-op for
                    # this arch (see TFB_SEEDS note above). Passing it would only
                    # write a misleading value into the wandb config.
                    CELL_ID="arm=tfb,lay=${lay},n=10,k=${k},s=${s}"
                    emit_job "${prefix}abl_tfb_lay${lay}_k${k}_s${s}" \
                        "transformers_baseline" "$k" "$s" \
                        "arch.H_layers=${lay} arch.halt_max_steps=1"
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
