"""Regression tests for measure_rho.py's inference-time recurrence-count override.

Why this exists (ARG track, P0-1): the stability axis asks whether a trained
checkpoint keeps its accuracy when the recurrence count is changed at inference.
That question was unmeasurable through the analysis path because
build_model_config always rebuilt the model at whatever cycle counts the
checkpoint was trained with, and no CLI flag overrode them.

The override is legal because the recurrent block is weight-tied: trm.py reads
H_cycles/L_cycles fresh on each forward() and loops over one shared L_level
module, so no parameter shape depends on them. These tests pin the config-assembly
half of that contract; they need no checkpoint, no dataset and no GPU.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]

TRAINED_H, TRAINED_L = 3, 6


def _load_measure_rho():
    """Import measure_rho.py by path — it is a top-level script, not a package."""
    spec = importlib.util.spec_from_file_location("measure_rho", REPO / "measure_rho.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["measure_rho"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def checkpoint_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A checkpoint directory whose all_config.yaml records H=3, L=6."""
    data_dir = tmp_path / "data" / "sigma_k_10" / "10"
    (data_dir / "test").mkdir(parents=True)
    (data_dir / "test" / "dataset.json").write_text(
        json.dumps({"vocab_size": 11, "seq_len": 11, "num_puzzle_identifiers": 1})
    )

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "all_config.yaml").write_text(
        yaml.safe_dump(
            {
                "arch": {
                    "name": "trm@TinyRecursiveReasoningModel",
                    "loss": {"name": "softmax_cross_entropy"},
                    "H_cycles": TRAINED_H,
                    "L_cycles": TRAINED_L,
                    "hidden_size": 512,
                },
                "data_paths": [str(data_dir)],
            }
        )
    )
    return ckpt


def test_defaults_to_trained_cycle_counts(checkpoint_dir):
    """With no override the trained values must survive untouched."""
    m = _load_measure_rho()
    cfg = m.build_model_config(str(checkpoint_dir))
    assert cfg["H_cycles"] == TRAINED_H
    assert cfg["L_cycles"] == TRAINED_L


@pytest.mark.parametrize(
    "h, l, expect_h, expect_l",
    [
        (12, None, 12, TRAINED_L),   # H only
        (None, 12, TRAINED_H, 12),   # L only
        (6, 6, 6, 6),                # both
        (1, 1, 1, 1),                # degenerate but legal
    ],
)
def test_override_applies(checkpoint_dir, h, l, expect_h, expect_l):
    m = _load_measure_rho()
    cfg = m.build_model_config(str(checkpoint_dir), h_cycles=h, l_cycles=l)
    assert cfg["H_cycles"] == expect_h
    assert cfg["L_cycles"] == expect_l


def test_override_does_not_disturb_other_arch_fields(checkpoint_dir):
    """Only the two cycle counts may change; everything else is carried verbatim."""
    m = _load_measure_rho()
    base = m.build_model_config(str(checkpoint_dir))
    over = m.build_model_config(str(checkpoint_dir), h_cycles=12, l_cycles=24)
    differing = {k for k in base if base[k] != over.get(k)}
    assert differing == {"H_cycles", "L_cycles"}
    assert over["hidden_size"] == base["hidden_size"]
    assert "name" not in over and "loss" not in over


@pytest.mark.parametrize("bad", [0, -1, -7])
def test_rejects_non_positive(checkpoint_dir, bad):
    m = _load_measure_rho()
    with pytest.raises(ValueError):
        m.build_model_config(str(checkpoint_dir), h_cycles=bad)
    with pytest.raises(ValueError):
        m.build_model_config(str(checkpoint_dir), l_cycles=bad)


def test_rejects_non_int(checkpoint_dir):
    m = _load_measure_rho()
    with pytest.raises(ValueError):
        m.build_model_config(str(checkpoint_dir), h_cycles=3.5)


def test_cli_exposes_both_flags():
    """The flags must exist and default to None, or the override is unreachable."""
    src = (REPO / "measure_rho.py").read_text(encoding="utf-8")
    assert '"--h-cycles"' in src
    assert '"--l-cycles"' in src
    # default None means "use the trained value" — a numeric default would
    # silently rewrite every existing invocation.
    assert 'dest="h_cycles", type=int, default=None' in src
    assert 'dest="l_cycles", type=int, default=None' in src
