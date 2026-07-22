"""Instantiate every cell of the STAGES=looped grid through the real Hydra +
pretrain.create_model path and report effective depth / parameter count.

Run from the repo root:
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. uv run python scripts/verify_looped_grid.py
"""
import json
import re
import subprocess
import sys
import os
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from utils.functions import load_model_class

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (seq_len, vocab_size) from data/sigma_k_10/<k>/train/dataset.json
META = dict(seq_len=11, vocab_size=11, num_puzzle_identifiers=1)


def build(overrides):
    with initialize_config_dir(config_dir=f"{REPO}/config", version_base=None):
        cfg = compose(config_name="cfg_pretrain", overrides=overrides)
    arch = OmegaConf.to_container(cfg.arch, resolve=True)
    name = arch.pop("name")
    arch.pop("loss")
    model_cfg = dict(**arch, batch_size=8, causal=False, **META)
    with torch.device("cpu"):
        model = load_model_class(name)(model_cfg)
    return model, model_cfg


def cells():
    """Parse the grid straight out of sigma_enqueue.sh (single source of truth)."""
    out = subprocess.run(
        ["bash", "-c", f"cd {REPO} && QUEUE_DIR=$(mktemp -d) STAGES=looped scripts/sigma_enqueue.sh --dry-run"],
        capture_output=True, text=True, check=True,
    ).stdout
    seen = {}
    for line in out.splitlines():
        m = re.match(r"^\d{4} (lt_(.+)_k\d+_s\d+)$", line.strip())
        if m:
            seen.setdefault(m.group(2), m.group(1))
    return seen


def arch_for(tag):
    return "looped_transformer" if tag.startswith("loop") else "transformers_baseline"


# Mirror of the sigma_enqueue.sh arrays, keyed by tag -> hydra overrides.
SPECS = {
    "deep2":  ["arch.H_layers=2"],
    "deep4":  ["arch.H_layers=4"],
    "deep6":  ["arch.H_layers=6"],
    "deep12": ["arch.H_layers=12"],
    "loop2x6":  ["arch.H_layers=2", "arch.H_cycles=6"],
    "loop2x21": ["arch.H_layers=2", "arch.H_cycles=21"],
    "loop1x12": ["arch.H_layers=1", "arch.H_cycles=12"],
    "loop3x4":  ["arch.H_layers=3", "arch.H_cycles=4"],
    "loop6x2":  ["arch.H_layers=6", "arch.H_cycles=2"],
    "loop2x3":  ["arch.H_layers=2", "arch.H_cycles=3"],
    "loop2x12": ["arch.H_layers=2", "arch.H_cycles=12"],
    "loop2x6_grad1": ["arch.H_layers=2", "arch.H_cycles=6", "arch.loop_grad_cycles=1"],
    "loop2x6_noinj": ["arch.H_layers=2", "arch.H_cycles=6",
                      "arch.input_injection_every_cycle=False"],
    "loop2x21_pel16": ["arch.H_layers=2", "arch.H_cycles=21", "arch.puzzle_emb_len=16"],
}

rows = []
tags = cells()
missing = set(tags) - set(SPECS)
assert not missing, f"grid tags not covered by this verifier: {missing}"
for tag in SPECS:
    assert tag in tags, f"SPECS has {tag} but the enqueue grid does not emit it"
    ov = [f"arch={arch_for(tag)}"] + SPECS[tag] + ["arch.num_heads=8", "arch.halt_max_steps=1"]
    model, mc = build(ov)
    n = sum(p.numel() for p in model.parameters())
    depth = mc["H_layers"] * (mc.get("H_cycles", 1) if arch_for(tag) == "looped_transformer" else 1)
    rows.append(dict(tag=tag, arch=arch_for(tag), H_layers=mc["H_layers"],
                     H_cycles=mc["H_cycles"], eff_depth=depth, params=n,
                     heads=mc["num_heads"], halt=mc["halt_max_steps"],
                     seq=model.inner.puzzle_emb_len + META["seq_len"]))

# TRM anchor for reference (fig1_tf_z_iter)
trm, mc = build(["arch=trm", "arch.mlp_t=False", "arch.H_cycles=3", "arch.L_cycles=6",
                 "arch.L_layers=2", "arch.halt_max_steps=1"])
rows.append(dict(tag="(anchor) fig1_tf_z_iter", arch="trm", H_layers=mc["L_layers"],
                 H_cycles=mc["H_cycles"], eff_depth=(mc["L_cycles"] + 1) * mc["H_cycles"] * mc["L_layers"],
                 params=sum(p.numel() for p in trm.parameters()),
                 heads=mc["num_heads"], halt=mc["halt_max_steps"],
                 seq=trm.inner.puzzle_emb_len + META["seq_len"]))

w = max(len(r["tag"]) for r in rows)
print(f"{'cell':<{w}}  {'arch':<22} {'blk':>3} {'T':>3} {'D_eff':>6} {'params':>12} {'heads':>5} {'seq':>4}")
for r in rows:
    print(f"{r['tag']:<{w}}  {r['arch']:<22} {r['H_layers']:>3} {r['H_cycles']:>3} "
          f"{r['eff_depth']:>6} {r['params']:>12,} {r['heads']:>5} {r['seq']:>4}")

# Matched-pair assertions the grid's claims rest on.
by = {r["tag"]: r for r in rows}
assert by["loop2x6"]["eff_depth"] == by["deep12"]["eff_depth"] == 12
assert by["loop2x6"]["params"] == by["deep2"]["params"], "param-match arm broken"
assert by["loop2x6"]["params"] < by["deep12"]["params"]
for t in ("loop1x12", "loop3x4", "loop6x2"):
    assert by[t]["eff_depth"] == 12, t
assert by["loop2x6"]["params"] == by["loop2x6_grad1"]["params"] == by["loop2x6_noinj"]["params"]

# Sequence geometry: params CANNOT reveal this axis (puzzle_emb is
# num_puzzle_identifiers x ndim regardless of puzzle_emb_len), so assert it
# directly. The whole grid must share geometry except the one isolation cell.
anchor = by["(anchor) fig1_tf_z_iter"]
grid_seq = {r["seq"] for t, r in by.items() if t.startswith(("deep", "loop")) and t != "loop2x21_pel16"}
assert grid_seq == {12}, f"lt_ grid geometry not uniform: {grid_seq}"
assert by["loop2x21_pel16"]["seq"] == anchor["seq"] == 27, "TRM-geometry cell mismatch"
assert by["loop2x21_pel16"]["params"] == anchor["params"], "TRM-matched cell param mismatch"
assert by["loop2x21_pel16"]["eff_depth"] == anchor["eff_depth"] == 42
print("\nAll matched-pair assertions passed.")
