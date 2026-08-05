from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from analysis.clean_r1_data_audit import AuditSpec, audit_clean_r1
from analysis.clean_r1_canary_audit import (
    LiveSpec,
    audit_live_canary,
    canonical_probe_identity,
)
from analysis.clean_r1_sealed_eval import (
    METRIC_ID,
    PRIMARY_METRIC,
    _initial_carry_on_batch_device,
    _tensor_devices,
    make_receipt,
    terminal_checkpoint,
    validate_terminal_ema_checkpoint,
)
from dataset.build_clean_r1_dataset import (
    BuildSpec,
    build_clean_r1,
    sample_unique_permutations,
    write_split,
)
from utils.clean_r1_live_log import CleanR1LiveLogger


@dataclass
class _NestedTestCarry:
    inner: dict[str, object]
    steps: torch.Tensor


class _DeviceContextCarryModel:
    def initial_carry(self, batch: dict[str, torch.Tensor]) -> _NestedTestCarry:
        batch_size = batch["inputs"].shape[0]
        return _NestedTestCarry(
            inner={
                "latents": [
                    torch.empty((batch_size, 2)),
                    {"current_inputs": torch.empty_like(batch["inputs"])},
                ]
            },
            steps=torch.zeros((batch_size,), dtype=torch.int32),
        )


def _build_tiny_contract(tmp_path: Path) -> tuple[BuildSpec, Path]:
    legacy = tmp_path / "legacy_d0"
    k = 4
    seen: set[bytes] = set()
    sigmas = sample_unique_permutations(
        n=10, total=5, rng=np.random.default_rng(91), seen=seen, k=k
    )
    write_split(legacy, k, "train", sigmas[:3], 10)
    write_split(legacy, k, "test", sigmas[3:], 10)
    spec = BuildSpec(
        existing_d0_root=legacy,
        d1_root=tmp_path / "d1",
        d0_sealed_root=tmp_path / "d0_sealed",
        k_values=(k,),
        train_size=3,
        monitor_size=2,
        sealed_size=2,
        seed=20_260_729,
    )
    build_clean_r1(spec)
    return spec, spec.d1_root / str(k)


def test_clean_r1_builder_and_audit_are_disjoint_and_reproducible(tmp_path: Path) -> None:
    spec, _ = _build_tiny_contract(tmp_path)
    manifest = json.loads((spec.d1_root / "clean_r1_manifest.json").read_text())
    assert manifest["spec"]["seed"] == 20_260_729
    assert manifest["per_k"]["4"]["counts"] == {
        "d0_sealed": 2,
        "d1_monitor_test": 2,
        "d1_sealed": 2,
        "d1_train": 3,
        "existing_d0_train_test": 5,
    }
    receipt = audit_clean_r1(AuditSpec(
        existing_d0_root=spec.existing_d0_root,
        d1_root=spec.d1_root,
        d0_sealed_root=spec.d0_sealed_root,
        k_values=spec.k_values,
    ))
    assert receipt["audit_passed"], receipt["failures"]
    within = receipt["per_k"]["4"]["within_k_disjointness"]
    assert within["complete_d0_vs_complete_d1"]["overlap_count"] == 0
    assert within["d1"]["train_sealed"]["overlap_count"] == 0
    assert receipt["per_k"]["4"]["d1"]["sealed"]["order_filter"]["contamination_count"] == 0


def test_clean_r1_audit_catches_tampered_labels(tmp_path: Path) -> None:
    spec, _ = _build_tiny_contract(tmp_path)
    label_path = spec.d1_root / "4" / "sealed" / "all__labels.npy"
    labels = np.load(label_path)
    labels[0, 0] = (labels[0, 0] % 10) + 1
    np.save(label_path, labels)
    receipt = audit_clean_r1(AuditSpec(
        existing_d0_root=spec.existing_d0_root,
        d1_root=spec.d1_root,
        d0_sealed_root=spec.d0_sealed_root,
        k_values=spec.k_values,
    ))
    assert not receipt["audit_passed"]
    sealed = receipt["per_k"]["4"]["d1"]["sealed"]
    assert sealed["label_regeneration"]["mismatch_rows"] == 1


def test_terminal_ema_preflight_is_cpu_safe_and_hashes_explicit_seal(tmp_path: Path) -> None:
    spec, sealed_dataset_path = _build_tiny_contract(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "all_config.yaml").write_text(
        "ema: true\nema_rate: 0.999\ndata_paths:\n  - " + str(sealed_dataset_path) + "\n",
        encoding="utf-8",
    )
    old_step = run_dir / "step_10"
    old_step.write_bytes(b"old")
    terminal_step = run_dir / "step_20"
    terminal_step.write_bytes(b"terminal")
    assert terminal_checkpoint(run_dir) == terminal_step
    validated_run_dir, config = validate_terminal_ema_checkpoint(terminal_step)
    assert validated_run_dir == run_dir
    assert config["ema"] is True
    receipt = make_receipt(
        checkpoint=terminal_step,
        sealed_path=sealed_dataset_path,
        sealed_split="sealed",
        batch_size=2,
        dry_run=True,
    )
    assert receipt["dry_run"] is True
    assert receipt["evaluation_status"] == "dry_run_preflight_only"
    assert receipt["metrics"] is None
    assert receipt["metric_id"] == METRIC_ID == "sealed-test-exact"
    assert receipt["metric_contract"]["metric_id"] == METRIC_ID
    assert receipt["metric_contract"]["primary_metric"] == PRIMARY_METRIC
    assert receipt["sealed_input"]["hashes"]["arrays"]["inputs"]["shape"][0] == 2


def test_sealed_eval_carry_device_check_traverses_nested_structure_on_cpu() -> None:
    batch = {"inputs": torch.zeros((2, 3), dtype=torch.int64)}
    carry = _initial_carry_on_batch_device(_DeviceContextCarryModel(), batch)
    assert _tensor_devices(carry) == {torch.device("cpu")}

    class MixedDeviceCarryModel:
        def initial_carry(self, _batch: dict[str, torch.Tensor]) -> _NestedTestCarry:
            return _NestedTestCarry(
                inner={"latents": [torch.empty(1), {"bad": torch.empty(1, device="meta")}]},
                steps=torch.zeros(1, dtype=torch.int32),
            )

    with pytest.raises(RuntimeError, match="carry spans devices"):
        _initial_carry_on_batch_device(MixedDeviceCarryModel(), batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device regression")
def test_sealed_eval_initial_carry_uses_cuda_batch_device() -> None:
    batch = {"inputs": torch.zeros((2, 3), dtype=torch.int64, device="cuda")}
    carry = _initial_carry_on_batch_device(_DeviceContextCarryModel(), batch)
    assert _tensor_devices(carry) == {batch["inputs"].device}


def _build_canary_contract(tmp_path: Path) -> tuple[BuildSpec, Path, Path, Path]:
    """Minimal data that still supports the fixed 512-example z probe."""
    legacy = tmp_path / "legacy_d0"
    k = 4
    seen: set[bytes] = set()
    sigmas = sample_unique_permutations(
        n=10, total=517, rng=np.random.default_rng(223), seen=seen, k=k
    )
    write_split(legacy, k, "train", sigmas[:5], 10)
    write_split(legacy, k, "test", sigmas[5:], 10)
    spec = BuildSpec(
        existing_d0_root=legacy,
        d1_root=tmp_path / "d1",
        d0_sealed_root=tmp_path / "d0_sealed",
        k_values=(k,),
        train_size=5,
        monitor_size=512,
        sealed_size=2,
        seed=20_260_729,
    )
    build_clean_r1(spec)
    return spec, legacy / "4", spec.d1_root / "4", spec.d0_sealed_root / "4"


def test_live_canary_audit_passes_only_with_bound_material_artifacts(tmp_path: Path) -> None:
    spec, d0_path, d1_path, d0_sealed_path = _build_canary_contract(tmp_path)
    run_name = "cleanr1_d1_k4_sm1"
    probe = canonical_probe_identity(d1_path)
    prelaunch_path = tmp_path / "prelaunch.json"
    prelaunch_path.write_text(json.dumps({
        "status": "PRELAUNCH_CONTRACT_PASS",
        "jobs": [{
            "run_name": run_name,
            "passed": True,
            "expected_first_eval_step": 4,
            "expected_eval_interval_epochs": 2000,
            "test_probe_identity": probe,
            "live_log_contract": {
                "enabled": True,
                "enablement": "run_name_prefix:cleanr1_",
                "contract_id": "CLEAN-R1-LIVE-v1",
                "sidecar_name": "clean_r1_live.jsonl",
            },
        }],
    }), encoding="utf-8")
    job_path = tmp_path / "canary.job"
    job_path.write_text(
        "uv run pretrain.py arch=trm_singlez checkpoint_every_eval=True "
        "eval_interval=2000 min_eval_interval=0 ema=True "
        "+log_z_dynamics=True "
        f"data_paths=\"[{d1_path}]\" +run_name=\"{run_name}\"\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    all_config = {
        "run_name": run_name,
        "project_name": "Sigma_k_new",
        "ema": True,
        "ema_rate": 0.999,
        "checkpoint_every_eval": True,
        "eval_interval": 2000,
        "min_eval_interval": 0,
        "log_z_dynamics": True,
        "z_probe_size": 512,
        "clean_r1_live_log": False,
        "data_paths": [str(d1_path)],
        "arch": {"name": "trm_singlez", "H_cycles": 3},
        "seed": 1,
        "k": 4,
    }
    (run_dir / "all_config.yaml").write_text(
        yaml.safe_dump(all_config, sort_keys=True),
        encoding="utf-8",
    )
    checkpoint = run_dir / "step_4"
    torch.save({"weight": torch.ones(1)}, checkpoint)
    structured = CleanR1LiveLogger(
        checkpoint_dir=run_dir,
        resolved_config=all_config,
        eval_cadence_steps=4,
        enablement="run_name_prefix:cleanr1_",
    )
    structured.observe_train_metrics(
        step=4, metrics={"train/lm_loss": 0.5}, grad_norm=1.25
    )
    structured.emit_eval_checkpoint(
        step=4,
        eval_index=1,
        eval_metrics={"all": {"lm_loss": 0.25}},
        checkpoint_path=checkpoint,
        checkpoint_role="nonterminal",
        ema_enabled=True,
        ema_rate=0.999,
    )
    live = LiveSpec(
        run_name=run_name,
        job_path=job_path,
        run_dir=run_dir,
        checkpoint=checkpoint,
        log_path=structured.path,
        d0_data_path=d0_path,
        d1_data_path=d1_path,
        d0_sealed_path=d0_sealed_path,
        d1_manifest=spec.d1_root / "clean_r1_manifest.json",
        prelaunch_receipt=prelaunch_path,
        expected_first_eval_step=4,
        expected_eval_cadence_epochs=2000,
        checkpoint_role="nonterminal",
        expected_config=(("arch.name", "trm_singlez"), ("arch.H_cycles", 3), ("seed", 1), ("k", 4)),
    )
    receipt = audit_live_canary(live)
    assert receipt["status"] == "LIVE_CANARY_PASS", receipt["failures"]
    assert receipt["metric_contract"]["sealed_metric_id"] == "sealed-test-exact"
    assert receipt["checks"]["checkpoint_reloadability"]["state_dict_key_count"] == 1
    assert receipt["checks"]["log_scan"]["first_eval_event"]["grad_norm"] == 1.25

    absent_log = LiveSpec(**{**live.__dict__, "log_path": tmp_path / "missing.log"})
    absent_receipt = audit_live_canary(absent_log)
    assert absent_receipt["status"] == "LIVE_CANARY_FAIL"
    assert "required live artifact absent: log" in absent_receipt["failures"]


def test_clean_r1_structured_logger_fails_closed_on_nonfinite(tmp_path: Path) -> None:
    _spec, _d0_path, d1_path, _d0_sealed_path = _build_canary_contract(tmp_path)
    config = {
        "run_name": "cleanr1_d1_k4_sm1",
        "project_name": "Sigma_k_new",
        "data_paths": [str(d1_path)],
        "seed": 1,
        "k": 4,
        "ema": True,
        "ema_rate": 0.999,
    }
    logger = CleanR1LiveLogger(
        checkpoint_dir=tmp_path / "nan_run",
        resolved_config=config,
        eval_cadence_steps=4,
        enablement="run_name_prefix:cleanr1_",
    )
    with pytest.raises(FloatingPointError, match="non-finite"):
        logger.observe_train_metrics(
            step=1, metrics={"train/lm_loss": float("nan")}, grad_norm=None
        )
    event = json.loads(logger.path.read_text().strip())
    assert event["event_type"] == "nonfinite_abort"
    assert event["nan_inf_flag"] is True
