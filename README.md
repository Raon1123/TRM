# TRM — Weight-Tied Recurrence: Expressivity & Convergence Dynamics

This is a research fork of **Tiny Recursion Model (TRM)**, repurposed from paper-reproduction
code into an active investigation of *why* a weight-tied recursive block reasons well: what its
latent state `z` actually carries, whether recursion depth alone (without `z`) can substitute for
it, and how the network's fixed-point / spectral behavior tracks its ability to generalize.

The primary experimental domain is **σ^k permutation composition**: given a random permutation
σ on a fixed-size set, predict σ^k (the k-fold composition of σ with itself) for a permutation
size `n=10` held fixed across all runs, with composition depth `k` as the swept variable. This is
a synthetic, exactly-checkable task chosen so that capacity, depth, and convergence behavior can
be measured directly rather than inferred from benchmark accuracy.

This is not the official paper repository — see [Upstream & citation](#upstream--citation) below
for the original work this is built on. The original ARC-AGI / Sudoku / Maze pipeline from that
paper is still present and runnable (see [Upstream paper experiments](#upstream-paper-experiments)),
but it is no longer the focus of active work here.

## How TRM works

<p align="center">
  <img src="https://AlexiaJM.github.io/assets/images/TRM_fig.png" alt="TRM" style="width: 30%;">
</p>

TRM recursively improves a predicted answer `y` with a single tiny weight-tied network. Starting
from the embedded input `x`, an initial answer `y`, and an initial latent `z`, it repeats, for up
to `K` improvement steps: (i) recursively update `z` `n` times given `(x, y, z)`, then (ii) update
`y` given `(y, z)`. The same weights are reused at every step — there is no growth in parameter
count with depth, only more computation.

The `z`-vs-no-`z` and iteration-count axes above are exactly what this fork's experiments sweep:
`arch=trm` keeps the latent carry, `arch=trm_singlez` removes it; `H_cycles`/`L_cycles` control
how much recursive compute each step gets.

## Research focus

- **Architecture variants** (`config/arch/`): `trm` (z-carry), `trm_singlez` (no z — the collapse
  cohort), `transformers_baseline`, `hrm` (the original two-level Hierarchical Reasoning Model).
- **Diagnostics**: `measure_rho.py` computes the spectral radius of the weight-tied block at its
  fixed point and an AGOP (average gradient outer product) alignment measure, from a checkpoint.
- **z dynamics logging**: runs with `+log_z_dynamics=True` emit `z/eff_rank`, `phase/index`, and
  `z/delta_step_*` to Weights & Biases (project `Sigma_k` / `Sigma_k_new`) for tracking how the
  latent evolves across recursive steps and training.
- **Live findings, hypotheses, and experiment reports** live in `lab/` (`lab/reports/`,
  `lab/theory/`), not in this README — start at `lab/INDEX.md` for the current state. Read those
  before treating any specific claim about `z`-removal or collapse as settled; the mechanism is
  under active investigation, not established.

## Repository layout

| Path | Contents |
|---|---|
| `pretrain.py` | Single training run entry point (Hydra config, one process per GPU) |
| `measure_rho.py` | Post-hoc spectral radius / AGOP diagnostics from a checkpoint |
| `config/` | Hydra configs: `cfg_pretrain.yaml` (base) + `arch/*.yaml` (per-variant) |
| `dataset/` | Dataset builders — `build_sigma_k_dataset.py` (σ^k task) plus the original ARC/Sudoku/Maze builders |
| `models/` | Model implementations, incl. `models/recursive_reasoning/{trm,trm_singlez}.py` |
| `scripts/sigma_enqueue.sh` | Declarative grid-search job enqueuer (edit the sweep, `--dry-run` first) |
| `scripts/queue_run.sh` | File-based FIFO GPU job queue runner (see `CONCURRENCY.md`) |
| `analysis/` | Figure generation and metric-extraction scripts |
| `lab/` | Research notebook: hypotheses, experiment reports, figures, daily logs |
| `tests/` | pytest suite covering the queue, figure pipeline, and data tooling |
| `checkpoints/<project>/<run_name>/step_<N>` | Saved model state (EMA weights on disk) |

## Requirements

- Python 3.12 (see `pyproject.toml`)
- CUDA 12.6 (or similar)

Installation should take a few minutes.

## Installation with uv

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Install the package in editable mode
uv pip install -e .

# Login to Weights & Biases (optional)
wandb login YOUR-LOGIN
```

## Dataset preparation

### σ^k permutation composition (primary)

```bash
python dataset/build_sigma_k_dataset.py \
  --output-dir data/sigma_k_10 \
  --n 10 --only-k 3 --train-size 5000 --test-size 1000
```

This writes `data/sigma_k_10/3/{train,test}`. `--output-dir` is the *parent* directory — each
`k` gets its own subdirectory, appended automatically. `n` is fixed at 10 across this project's
experiments; `--only-k` restricts the build to a single composition depth (omit it to build the
default grid `[3,4,5,6,7,8,10]` in one call). `data/sigma_k_10/<k>` is the current, order-filtered
dataset generation — see `dataset/build_sigma_k_dataset.py`'s module docstring for the exact task
format and the `order_filter` provenance note before reusing an older `data/sigma_k/` directory.

### Upstream ARC-AGI / Sudoku / Maze builders (still supported)

```bash
uv run build-arc1
uv run build-arc2
uv run build-sudoku
uv run build-maze
```

## Running σ^k experiments

The experiment pipeline is: define a sweep → enqueue job files → run them on a GPU queue →
measure. Full operational detail (GPU/queue conventions, config keys, report discipline) is in
`CLAUDE.md`; the shape of it:

```bash
# 1. Define the grid by editing the *_SWEEP / K_LIST variables in the script, then:
scripts/sigma_enqueue.sh --dry-run <prefix>    # print the grid — check before spending GPU time
scripts/sigma_enqueue.sh <prefix>              # write job files to scripts/queue/jobs/

# 2. Run: one worker per GPU, single FIFO queue
scripts/queue_run.sh                           # start workers (default GPUs 4 5 6 7)
scripts/queue_run.sh status                    # queued / processing / done / failed
touch scripts/queue/stop                       # drain: finish in-flight jobs, then exit

# 3. Or, a single one-off run without the queue:
uv run pretrain.py arch=trm data_paths="[data/sigma_k_10/3]" +run_name="my_run" ema=True

# 4. Analysis: checkpoint -> spectral radius / AGOP
CUDA_VISIBLE_DEVICES=4 uv run python measure_rho.py --fixed-point --model-variant trm \
    --checkpoint checkpoints/<project>/<run_name>/step_<N> --save-spectral /tmp/spec.jsonl
```

Swap `arch=trm` for `arch=trm_singlez` to run the no-z cohort.

## Tests

```bash
uv run pytest tests/
```

## Upstream paper experiments

The original paper's ARC-AGI-1/2, Sudoku-Extreme, and Maze-Hard training commands still work
unmodified against this codebase (`arch=trm`, no queue involved):

```bash
# ARC-AGI-1 (4 GPUs, ~3 days)
uv run pretrain-arc1

# ARC-AGI-2 (4 GPUs, ~3 days)
uv run pretrain-arc2

# Sudoku-Extreme (1 GPU, <36h)
uv run pretrain-sudoku-mlp-t
uv run pretrain-sudoku-att

# Maze-Hard (4 GPUs, <24h)
uv run pretrain-maze
```

Note: you cannot train on both ARC-AGI-1 and ARC-AGI-2 and evaluate them both, since the
ARC-AGI-2 training split contains some ARC-AGI-1 evaluation data.

## Upstream & citation

This fork is based on ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2510.04871)
(Jolicoeur-Martineau, 2025), which itself is built on the Hierarchical Reasoning Model
[code](https://github.com/sapientinc/HRM) and the [HRM analysis code](https://github.com/arcprize/hierarchical-reasoning-model-analysis).
The upstream repository is [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).

If you build on the original TRM or HRM work, cite the original papers:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```
