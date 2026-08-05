from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from analysis.build_k8_k10_recovery_evidence import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PRESERVATION,
    DEFAULT_TRAJECTORY_ROOT,
    apply_power,
    build,
    cycle_lengths,
    minimum_addition_chain_length,
    permutation_order,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_small_algebra_helpers() -> None:
    permutation = np.array([1, 2, 3, 0], dtype=np.int64)
    assert cycle_lengths(permutation) == (4,)
    assert permutation_order(cycle_lengths(permutation)) == 4
    assert apply_power(permutation, 4).tolist() == [0, 1, 2, 3]
    assert minimum_addition_chain_length(5) == 3
    assert minimum_addition_chain_length(6) == 3
    assert minimum_addition_chain_length(7) == 4
    assert minimum_addition_chain_length(8) == 3
    assert minimum_addition_chain_length(10) == 4


def test_preview_packet_recomputes_frozen_discriminators(tmp_path: Path) -> None:
    packet = build(
        data_root=DEFAULT_DATA_ROOT,
        trajectory_root=DEFAULT_TRAJECTORY_ROOT,
        preservation_path=DEFAULT_PRESERVATION,
        out_dir=tmp_path,
    )
    descriptors = {
        int(row["k"]): row for row in _read_csv(tmp_path / "descriptor_audit.csv")
    }
    acquisition = {
        int(row["k"]): row
        for row in _read_csv(tmp_path / "acquisition_retention.csv")
    }
    overlaps = _read_csv(tmp_path / "successive_pool_overlap.csv")

    assert float(descriptors[6]["preservation_fraction"]) == 0.035
    assert round(float(descriptors[6]["exact_target_entropy_bits"]), 4) == 12.1235
    assert round(float(descriptors[8]["exact_target_entropy_bits"]), 4) == 12.1247
    assert float(descriptors[10]["exact_target_collision_fraction"]) == 0.2106
    assert float(descriptors[10]["test_exact_target_in_train_fraction"]) == 0.293
    assert acquisition[8]["first_consecutive_probe_g_pair_step"] == "126932"
    assert acquisition[8]["terminal_probe_g_streak_evals"] == "21"
    assert acquisition[10]["first_consecutive_probe_g_pair_step"] == "39056"
    assert acquisition[10]["terminal_probe_g_streak_evals"] == "9"
    assert overlaps[-1]["shared_train_inputs"] == "2893"
    assert overlaps[-1]["removed_input_order_counts"] == "9:870;10:1237"

    assert packet["provenance_status"] == "PREVIEW"
    assert packet["figures"][0]["qa_state"] == "UNREVIEWED_PREVIEW"
    assert (tmp_path / "k8_k10_recovery_preview.png").is_file()
    written_packet = json.loads((tmp_path / "evidence_packet.json").read_text())
    assert "the descriptors cause k=8 or k=10 success" in written_packet[
        "forbidden_inferences"
    ]
