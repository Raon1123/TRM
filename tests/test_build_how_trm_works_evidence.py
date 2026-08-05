from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from analysis import build_how_trm_works_evidence as evidence_builder
from analysis.build_how_trm_works_evidence import (
    CANONICAL_DATA_MANIFEST,
    CANONICAL_EVALUATOR_SOURCE,
    EXP_ID,
    SEALED_HALTED_METRIC,
    SEALED_METRIC_ID,
    SEALED_PRIMARY_METRIC,
    build_status_rows,
    classify_probe,
    discover_sealed_receipts,
    evaluator_source_binding,
    expected_cells,
    load_post_order_ablation,
    load_preservation,
    parse_exp_contract,
    preservation_outcomes,
    sealed_full_sha256,
    sealed_identity_sha256,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "lab/experiments/EXP-012_clean-r1-mechanism-scaling-gates.md"
PERCELL = ROOT / "lab/figure_pipeline/outputs/fig1-full-20260728/percell_table.csv"
PRESERVATION = (
    ROOT
    / "lab/reports/2026-07-30_success-mechanism-briefs/preservation_clean.json"
)


def _approve_hypothetical_evaluator(monkeypatch: pytest.MonkeyPatch) -> str:
    approved_sha = sha256_file(CANONICAL_EVALUATOR_SOURCE)
    monkeypatch.setattr(
        evidence_builder, "APPROVED_EVALUATOR_SOURCE_SHA256", approved_sha
    )
    monkeypatch.setattr(
        evidence_builder,
        "APPROVED_EVALUATOR_AMENDMENT_ID",
        "TEST-ONLY-INDEPENDENTLY-REVIEWED-EXP-012-10.7",
    )
    return approved_sha


@pytest.fixture
def hypothetical_v2_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _approve_hypothetical_evaluator(monkeypatch)
    monkeypatch.setattr(
        evidence_builder,
        "CANONICAL_CHECKPOINT_ROOT",
        (tmp_path / "checkpoints" / "Sigma_k_new").resolve(),
    )


def test_exp_contract_and_registered_grid() -> None:
    contract = parse_exp_contract(EXP)
    assert EXP_ID == "EXP-012"
    assert contract.failure_k == (5, 6, 7)
    assert contract.control_k == (4, 8, 10)
    assert contract.d0_new_seeds == (2, 3)
    assert contract.d1_new_seeds == (1, 2, 3)
    assert contract.terminal_step == 244_100
    assert contract.cadence == 4_882
    assert contract.probe_n == 512
    assert contract.sealed_n == 1_000
    cells = expected_cells(contract)
    assert len(cells) == contract.registered_new_runs == 30
    assert len({cell.base_run_name for cell in cells}) == 30


def test_probe_class_thresholds() -> None:
    contract = parse_exp_contract(EXP)
    assert classify_probe(1.0, 0.95, contract) == "G"
    assert classify_probe(0.95, 0.05, contract) == "H"
    assert classify_probe(0.89, 0.05, contract) == "A"
    assert classify_probe(1.0, 0.50, contract) == "A"


def test_current_post_order_and_preservation_merge() -> None:
    contract = parse_exp_contract(EXP)
    ablation = load_post_order_ablation(PERCELL, contract)
    meta, preservation = load_preservation(PRESERVATION)
    merged, signature = preservation_outcomes(preservation, ablation, PRESERVATION)
    assert meta["n"] == 10
    assert len(ablation) == 28
    assert len(merged) == 7
    assert signature["k"] == 6
    assert signature["run_id"] == "fig1_tf_noz_iter_k6_s1"
    assert signature["train_preservation_fraction"] == 0.035
    assert signature["final_test_exact"] == 0.023438
    assert signature["probe_class"] == "H"


def _complete_summary(run_name: str) -> dict:
    return {
        "base_run_name": run_name.removesuffix("_a2"),
        "run_name": run_name,
        "training_complete": True,
        "wandb_run_id": "testrun1",
        "n_eval_rows": 50,
        "final_step": 244_100,
        "final_probe_train_exact": 1.0,
        "final_probe_test_exact": 1.0,
        "final_probe_class": "G",
    }


def _v2_fixture(tmp_path: Path, run_name: str = "cleanr1_d0_k10_sm2") -> tuple[dict, list[dict]]:
    pool = "D0" if "_d0_" in run_name else "D1"
    seed = 2 if pool == "D0" else 1
    run_dir = tmp_path / "checkpoints" / "Sigma_k_new" / run_name
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "step_244100"
    checkpoint.write_bytes(b"realistic terminal EMA checkpoint fixture\n")

    training_family = "sigma_k_10" if pool == "D0" else "sigma_k_10_clean_r1_d1"
    training_path = ROOT / "data" / training_family / "10"
    assert training_path.is_dir()
    config_path = run_dir / "all_config.yaml"
    config_path.write_text(
        "\n".join([
            f"run_name: {run_name}",
            "k: 10",
            f"seed: {seed}",
            "ema: true",
            "data_paths:",
            f"- {training_path}",
            "",
        ]),
        encoding="utf-8",
    )

    sealed_family = (
        "sigma_k_10_clean_r1_d0_sealed"
        if pool == "D0"
        else "sigma_k_10_clean_r1_d1"
    )
    sealed_root = ROOT / "data" / sealed_family / "10"
    sealed_split = sealed_root / "sealed"
    assert sealed_split.is_dir()
    file_hashes = {
        path.relative_to(sealed_split).as_posix(): sha256_file(path)
        for path in sorted(sealed_split.rglob("*"))
        if path.is_file()
    }
    manifest_path = CANONICAL_DATA_MANIFEST
    manifest_sha = sha256_file(manifest_path)
    full_sha = sealed_full_sha256(sealed_root, "sealed")

    receipt = {
        "schema_version": 2,
        "run_name": run_name,
        "protocol": "CLEAN-R1-terminal-EMA-sealed-evaluation",
        "metric_id": SEALED_METRIC_ID,
        "metric_contract": {
            "metric_id": SEALED_METRIC_ID,
            "primary_metric": SEALED_PRIMARY_METRIC,
        },
        "evaluation_status": "completed",
        "dry_run": False,
        "evaluator_source": {
            "path": str(CANONICAL_EVALUATOR_SOURCE),
            "sha256_file": sha256_file(CANONICAL_EVALUATOR_SOURCE),
            "protocol": "CLEAN-R1-terminal-EMA-sealed-evaluation",
            "source_binding": evaluator_source_binding(),
        },
        "checkpoint": {
            "run_dir": str(run_dir),
            "path": str(checkpoint),
            "terminal_step": 244_100,
            "weight_kind": "EMA",
            "ema_required_and_verified": True,
            "sha256_file": sha256_file(checkpoint),
            "config_path": str(config_path),
            "config_sha256_file": sha256_file(config_path),
            "data_manifest_path": str(manifest_path),
            "data_manifest_sha256_file": manifest_sha,
        },
        "sealed_input": {
            "path": str(sealed_root),
            "split": "sealed",
            "n_examples": 1_000,
            "file_hashes": file_hashes,
            "full_sha256": full_sha,
            "identity_sha256": sealed_identity_sha256(
                root=sealed_root,
                split="sealed",
                full_sha256=full_sha,
                data_manifest_sha256=manifest_sha,
                pool=pool,
                k=10,
            ),
            "manifest_binding": {
                "pool": pool,
                "k": 10,
                "branch": (
                    "per_k.10.generated_d0_sealed"
                    if pool == "D0"
                    else "per_k.10.generated_d1.sealed"
                ),
                "matches": True,
            },
        },
        "metrics": {
            SEALED_PRIMARY_METRIC: 0.99,
            SEALED_HALTED_METRIC: 0.99,
            "sealed/terminal_ema/n_examples": 1_000,
            "sealed/terminal_ema/n_halted_examples": 1_000,
            "sealed/terminal_ema/n_exact_and_halted_examples": 990,
            "sealed/terminal_ema/n_batches": 4,
        },
    }
    return receipt, [_complete_summary(run_name)]


def _write_receipt(tmp_path: Path, receipt: dict, name: str = "receipt.json") -> Path:
    receipt_path = tmp_path / name
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt_path


def _relocate_data_tree(
    receipt: dict, tmp_path: Path, *, manifest_mutation: str | None = None
) -> None:
    checkpoint = receipt["checkpoint"]
    sealed = receipt["sealed_input"]
    pool = sealed["manifest_binding"]["pool"]
    family = (
        "sigma_k_10_clean_r1_d0_sealed"
        if pool == "D0"
        else "sigma_k_10_clean_r1_d1"
    )
    relocated_data = tmp_path / "relocated" / "data"
    relocated_root = relocated_data / family / "10"
    shutil.copytree(Path(sealed["path"]) / "sealed", relocated_root / "sealed")
    relocated_training = relocated_data / (
        "sigma_k_10" if pool == "D0" else "sigma_k_10_clean_r1_d1"
    ) / "10"
    relocated_training.mkdir(parents=True, exist_ok=True)
    config_path = Path(checkpoint["config_path"])
    config_text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        config_text.replace(str(ROOT / "data" / ("sigma_k_10" if pool == "D0" else "sigma_k_10_clean_r1_d1") / "10"), str(relocated_training)),
        encoding="utf-8",
    )
    checkpoint["config_sha256_file"] = sha256_file(config_path)

    master = json.loads(CANONICAL_DATA_MANIFEST.read_text(encoding="utf-8"))
    branch = (
        master["per_k"]["10"]["generated_d0_sealed"]
        if pool == "D0"
        else master["per_k"]["10"]["generated_d1"]["sealed"]
    )
    for entry in [*branch["arrays"].values(), branch["dataset_json"]]:
        entry["path"] = str(relocated_root / "sealed" / Path(entry["path"]).name)
    if manifest_mutation == "protocol":
        master["protocol"] = "WRONG-PROTOCOL"
    elif manifest_mutation == "branch":
        branch["arrays"]["inputs"]["sha256_file"] = "0" * 64
    manifest_path = relocated_data / "sigma_k_10_clean_r1_d1" / "clean_r1_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(master) + "\n", encoding="utf-8")
    manifest_sha = sha256_file(manifest_path)
    checkpoint["data_manifest_path"] = str(manifest_path)
    checkpoint["data_manifest_sha256_file"] = manifest_sha
    if pool == "D0":
        pointer_path = (
            manifest_path.parent.parent
            / "sigma_k_10_clean_r1_d0_sealed"
            / "clean_r1_manifest.json"
        )
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        pointer = {
            "schema_version": 1,
            "manifest_path": str(manifest_path),
            "manifest_sha256_file": manifest_sha,
        }
        pointer_path.write_text(json.dumps(pointer) + "\n", encoding="utf-8")
    sealed["path"] = str(relocated_root)
    relocated_split = relocated_root / "sealed"
    sealed["file_hashes"] = {
        path.relative_to(relocated_split).as_posix(): sha256_file(path)
        for path in sorted(relocated_split.rglob("*"))
        if path.is_file()
    }
    sealed["full_sha256"] = sealed_full_sha256(relocated_root, "sealed")
    sealed["identity_sha256"] = sealed_identity_sha256(
        root=relocated_root,
        split=sealed["split"],
        full_sha256=sealed["full_sha256"],
        data_manifest_sha256=manifest_sha,
        pool=pool,
        k=sealed["manifest_binding"]["k"],
    )


@pytest.mark.parametrize(
    "valid_name",
    ["cleanr1_d0_k10_sm2", "cleanr1_d1_k10_sm1"],
)
def test_receipt_inventory_accepts_fully_bound_schema_v2(
    tmp_path: Path, valid_name: str, hypothetical_v2_contract: None
) -> None:
    contract = parse_exp_contract(EXP)
    receipt, run_summaries = _v2_fixture(tmp_path, valid_name)
    _write_receipt(tmp_path, receipt)
    inventory, admissible, paths = discover_sealed_receipts(
        tmp_path, contract, run_summaries
    )
    assert len(inventory) == len(paths) == 1
    assert admissible[valid_name]["sealed_test_exact"] == 0.99
    assert inventory[0]["validation_error_count"] == 0
    assert inventory[0]["hashes_all_match"] is True
    assert inventory[0]["metric_arithmetic_match"] is True

    status = build_status_rows(contract, run_summaries, inventory, admissible)
    valid = next(row for row in status if row["base_run_name"] == valid_name)
    assert valid["admissible_sealed_endpoint"] is True


def test_relocated_content_identical_data_tree_is_not_canonical(
    tmp_path: Path, hypothetical_v2_contract: None,
) -> None:
    contract = parse_exp_contract(EXP)
    receipt, run_summaries = _v2_fixture(tmp_path)
    _relocate_data_tree(receipt, tmp_path)
    _write_receipt(tmp_path, receipt)
    inventory, admissible, _ = discover_sealed_receipts(
        tmp_path, contract, run_summaries
    )
    assert admissible == {}
    errors = inventory[0]["validation_errors"]
    assert "config.data_paths" in errors
    assert "checkpoint.data_manifest_path" in errors
    assert "sealed_input.path" in errors


def test_d1_generated_sealed_branch_mismatch_is_rejected(
    tmp_path: Path, hypothetical_v2_contract: None,
) -> None:
    run_name = "cleanr1_d1_k10_sm1"
    receipt, run_summaries = _v2_fixture(tmp_path, run_name)
    _relocate_data_tree(receipt, tmp_path, manifest_mutation="branch")
    _write_receipt(tmp_path, receipt)
    inventory, admissible, _ = discover_sealed_receipts(
        tmp_path, parse_exp_contract(EXP), run_summaries
    )
    assert admissible == {}
    assert inventory[0]["manifest_branch_match"] is False
    assert "data_manifest.per_k.branch" in inventory[0]["validation_errors"]


def test_production_gate_rejects_unfrozen_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert evidence_builder.APPROVED_EVALUATOR_SOURCE_SHA256 is None
    assert evidence_builder.APPROVED_EVALUATOR_AMENDMENT_ID is None
    monkeypatch.setattr(
        evidence_builder,
        "CANONICAL_CHECKPOINT_ROOT",
        (tmp_path / "checkpoints" / "Sigma_k_new").resolve(),
    )
    receipt, run_summaries = _v2_fixture(tmp_path)
    _write_receipt(tmp_path, receipt)
    inventory, admissible, _ = discover_sealed_receipts(
        tmp_path, parse_exp_contract(EXP), run_summaries
    )
    assert admissible == {}
    assert inventory[0]["evaluator_approval_match"] is False
    assert "evaluator_source.approved_sha256" in inventory[0]["validation_errors"]


def test_relocated_synthetic_checkpoint_is_not_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _approve_hypothetical_evaluator(monkeypatch)
    receipt, run_summaries = _v2_fixture(tmp_path)
    _write_receipt(tmp_path, receipt)
    inventory, admissible, _ = discover_sealed_receipts(
        tmp_path, parse_exp_contract(EXP), run_summaries
    )
    assert admissible == {}
    assert inventory[0]["checkpoint_identity_match"] is False
    assert "registered canonical run directory" in inventory[0]["validation_errors"]


@pytest.mark.parametrize(
    ("case", "error_field"),
    [
        ("schema_v1", "schema_version"),
        ("missing_run_name", "run_name"),
        ("missing_evaluator", "evaluator_source"),
        ("evaluator_path", "evaluator_source.path"),
        ("evaluator_hash", "evaluator_source.sha256_file"),
        ("evaluator_protocol", "evaluator_source.protocol"),
        ("evaluator_source_binding", "evaluator_source.source_binding"),
        ("evaluator_source_changed", "evaluator_source.approved_sha256"),
        ("missing_run_dir", "checkpoint.run_dir"),
        ("missing_checkpoint_path", "checkpoint.path"),
        ("terminal_step", "checkpoint.terminal_step"),
        ("weight_kind", "checkpoint.weight_kind"),
        ("checkpoint_hash", "checkpoint.sha256_file"),
        ("missing_config_path", "checkpoint.config.path"),
        ("config_hash", "checkpoint.config.sha256_file"),
        ("config_run", "config.run_name"),
        ("config_k", "config.k"),
        ("config_seed", "config.seed"),
        ("config_data", "config.data_paths"),
        ("missing_manifest_path", "checkpoint.data_manifest.path"),
        ("manifest_hash", "checkpoint.data_manifest.sha256_file"),
        ("manifest_path_suffix", "checkpoint.data_manifest_path"),
        ("manifest_protocol", "data_manifest.protocol"),
        ("manifest_branch", "data_manifest.per_k.branch"),
        ("d0_pointer", "data_manifest.d0_pointer"),
        ("sealed_path", "sealed_input.path"),
        ("sealed_split", "sealed_input.split"),
        ("sealed_n", "sealed_input.n_examples"),
        ("sealed_file_hash", "sealed_input.file_hashes"),
        ("missing_sealed_full_hash", "sealed_input.full_sha256"),
        ("sealed_full_hash", "sealed_input.full_sha256"),
        ("sealed_identity", "sealed_input.identity_sha256"),
        ("manifest_binding", "sealed_input.manifest_binding.matches"),
        ("metric_arithmetic", SEALED_PRIMARY_METRIC),
        ("metric_count", "metrics.counts"),
        ("trajectory", "trajectory"),
    ],
)
def test_completed_receipt_rejects_missing_or_mismatched_binding(
    tmp_path: Path,
    case: str,
    error_field: str,
    hypothetical_v2_contract: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = parse_exp_contract(EXP)
    receipt, run_summaries = _v2_fixture(tmp_path)
    checkpoint = receipt["checkpoint"]

    if case == "schema_v1":
        receipt["schema_version"] = 1
    elif case == "missing_run_name":
        receipt.pop("run_name")
    elif case == "missing_evaluator":
        receipt.pop("evaluator_source")
    elif case == "evaluator_path":
        copied_source = tmp_path / "analysis" / "clean_r1_sealed_eval.py"
        copied_source.parent.mkdir(parents=True)
        copied_source.write_bytes(CANONICAL_EVALUATOR_SOURCE.read_bytes())
        receipt["evaluator_source"]["path"] = str(copied_source)
        receipt["evaluator_source"]["sha256_file"] = sha256_file(copied_source)
    elif case == "evaluator_hash":
        receipt["evaluator_source"]["sha256_file"] = "0" * 64
    elif case == "evaluator_protocol":
        receipt["evaluator_source"]["protocol"] = "WRONG-PROTOCOL"
    elif case == "evaluator_source_binding":
        receipt["evaluator_source"]["source_binding"]["gate"] = "WRONG-GATE"
    elif case == "evaluator_source_changed":
        changed_source = tmp_path / "reviewed" / "clean_r1_sealed_eval.py"
        changed_source.parent.mkdir(parents=True)
        changed_source.write_bytes(
            CANONICAL_EVALUATOR_SOURCE.read_bytes() + b"\n# unreviewed source drift\n"
        )
        monkeypatch.setattr(
            evidence_builder, "CANONICAL_EVALUATOR_SOURCE", changed_source.resolve()
        )
        receipt["evaluator_source"]["path"] = str(changed_source)
        receipt["evaluator_source"]["sha256_file"] = sha256_file(changed_source)
    elif case == "missing_run_dir":
        checkpoint.pop("run_dir")
    elif case == "missing_checkpoint_path":
        checkpoint.pop("path")
    elif case == "terminal_step":
        checkpoint["terminal_step"] = 244_099
    elif case == "weight_kind":
        checkpoint["weight_kind"] = "RAW"
    elif case == "checkpoint_hash":
        checkpoint["sha256_file"] = "0" * 64
    elif case == "missing_config_path":
        checkpoint.pop("config_path")
    elif case == "config_hash":
        checkpoint["config_sha256_file"] = "0" * 64
    elif case in {"config_run", "config_k", "config_seed", "config_data"}:
        config_path = Path(checkpoint["config_path"])
        text = config_path.read_text(encoding="utf-8")
        replacements = {
            "config_run": ("run_name: cleanr1_d0_k10_sm2", "run_name: cleanr1_d0_k8_sm2"),
            "config_k": ("k: 10", "k: 8"),
            "config_seed": ("seed: 2", "seed: 3"),
            "config_data": ("sigma_k_10/10", "sigma_k_10/8"),
        }
        old, new = replacements[case]
        config_path.write_text(text.replace(old, new), encoding="utf-8")
        checkpoint["config_sha256_file"] = sha256_file(config_path)
    elif case == "manifest_hash":
        checkpoint["data_manifest_sha256_file"] = "0" * 64
    elif case == "missing_manifest_path":
        checkpoint.pop("data_manifest_path")
    elif case == "manifest_path_suffix":
        old_path = Path(checkpoint["data_manifest_path"])
        wrong_path = tmp_path / "manifests" / "clean_r1_manifest.json"
        wrong_path.parent.mkdir(parents=True)
        wrong_path.write_bytes(old_path.read_bytes())
        checkpoint["data_manifest_path"] = str(wrong_path)
        checkpoint["data_manifest_sha256_file"] = sha256_file(wrong_path)
        receipt["sealed_input"]["identity_sha256"] = sealed_identity_sha256(
            root=Path(receipt["sealed_input"]["path"]),
            split="sealed",
            full_sha256=receipt["sealed_input"]["full_sha256"],
            data_manifest_sha256=sha256_file(wrong_path),
            pool="D0",
            k=10,
        )
    elif case in {"manifest_protocol", "manifest_branch"}:
        _relocate_data_tree(
            receipt,
            tmp_path,
            manifest_mutation="protocol" if case == "manifest_protocol" else "branch",
        )
    elif case == "d0_pointer":
        _relocate_data_tree(receipt, tmp_path)
    elif case == "sealed_path":
        receipt["sealed_input"]["path"] = str(tmp_path / "wrong" / "10")
    elif case == "sealed_split":
        receipt["sealed_input"]["split"] = "test"
    elif case == "sealed_n":
        receipt["sealed_input"]["n_examples"] = 999
    elif case == "sealed_file_hash":
        first = next(iter(receipt["sealed_input"]["file_hashes"]))
        receipt["sealed_input"]["file_hashes"][first] = "0" * 64
    elif case == "sealed_full_hash":
        receipt["sealed_input"]["full_sha256"] = "0" * 64
    elif case == "missing_sealed_full_hash":
        receipt["sealed_input"].pop("full_sha256")
    elif case == "sealed_identity":
        receipt["sealed_input"]["identity_sha256"] = "0" * 64
    elif case == "manifest_binding":
        receipt["sealed_input"]["manifest_binding"]["matches"] = False
    elif case == "metric_arithmetic":
        receipt["metrics"][SEALED_PRIMARY_METRIC] = 0.98
    elif case == "metric_count":
        receipt["metrics"]["sealed/terminal_ema/n_halted_examples"] = 989
    elif case == "trajectory":
        run_summaries[0]["training_complete"] = False
    else:  # pragma: no cover - the parametrization is exhaustive
        raise AssertionError(case)

    _write_receipt(tmp_path, receipt)
    inventory, admissible, _ = discover_sealed_receipts(
        tmp_path, contract, run_summaries
    )
    assert admissible == {}
    assert inventory[0]["admissible"] is False
    assert inventory[0]["validation_error_count"] >= 1
    assert error_field in inventory[0]["validation_errors"]


def test_legacy_dry_run_remains_excluded(tmp_path: Path) -> None:
    contract = parse_exp_contract(EXP)
    run_name = "cleanr1_d0_k8_sm2"
    receipt = {
        "schema_version": 1,
        "protocol": "CLEAN-R1-terminal-EMA-sealed-evaluation",
        "metric_id": SEALED_METRIC_ID,
        "evaluation_status": "dry_run_preflight_only",
        "dry_run": True,
        "checkpoint": {
            "run_dir": str(tmp_path / "checkpoints" / run_name),
            "path": str(tmp_path / "checkpoints" / run_name / "step_244100"),
            "terminal_step": 244_100,
            "ema_required_and_verified": True,
        },
        "metrics": None,
    }
    _write_receipt(tmp_path, receipt)
    inventory, admissible, _ = discover_sealed_receipts(
        tmp_path, contract, [_complete_summary(run_name)]
    )
    assert admissible == {}
    assert inventory[0]["admissible"] is False
    assert inventory[0]["schema_version_match"] is False
    assert inventory[0]["explicit_run_name_present"] is False
    assert "dry-run/preflight" in inventory[0]["exclusion_reason"]
