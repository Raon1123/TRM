from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from shutil import copyfile

import pytest

from analysis.clean_r1_incident_requeue import IncidentError, _tree_inventory, _verify_tree_inventory


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "analysis" / "clean_r1_incident_requeue.py"
RUNNER = REPO_ROOT / "scripts" / "queue_run.sh"
TARGETS = (
    ("0223", "cleanr1_d0_k4_sm2", 6),
    ("0224", "cleanr1_d0_k4_sm3", 4),
    ("0225", "cleanr1_d0_k5_sm2", 3),
    ("0226", "cleanr1_d0_k5_sm3", 2),
    ("0227", "cleanr1_d0_k6_sm2", 7),
    ("0228", "cleanr1_d0_k6_sm3", 5),
    ("0229", "cleanr1_d0_k7_sm2", 3),
)
K_VALUES = (4, 5, 6, 7, 8, 10)
BASE_NAMES = tuple(
    [f"cleanr1_d0_k{k}_sm{s}" for k in K_VALUES for s in (2, 3)]
    + [f"cleanr1_d1_k{k}_sm{s}" for k in K_VALUES for s in (1, 2, 3)]
)
TARGET_NAMES = {name for _, name, _ in TARGETS}


def test_wandb_leaf_symlink_inventory_is_opaque_and_deterministic(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    logs = repo / "wandb" / "run-20260729_124349-eiyimzpj" / "logs"
    logs.mkdir(parents=True)
    (logs / "debug-internal.log").write_text("ordinary log\n", encoding="utf-8")
    outside = tmp_path / "outside-cache" / "core-debug-20260729_124349.log"
    outside.parent.mkdir()
    outside.write_text("TARGET CONTENT MUST NOT BE HASHED v1\n", encoding="utf-8")
    link = logs / "debug-core.log"
    target_text = os.path.relpath(outside, link.parent)
    link.symlink_to(target_text)

    first = _tree_inventory(logs.parent, repo)
    record = next(item for item in first["files"] if item["path"].endswith("logs/debug-core.log"))
    assert record == {
        "path": "wandb/run-20260729_124349-eiyimzpj/logs/debug-core.log",
        "kind": "symlink",
        "target": target_text,
        "target_text_sha256": hashlib.sha256(os.fsencode(target_text)).hexdigest(),
        "lstat_mode": 0o777,
        "lstat_size_bytes": len(os.fsencode(target_text)),
    }

    # Changing the outside target cannot change inventory: only the link entry
    # and exact target text are sealed.  This also proves no target-file hash.
    outside.write_text("TARGET CONTENT MUST NOT BE HASHED v2, much longer\n", encoding="utf-8")
    second = _tree_inventory(logs.parent, repo)
    assert second == first
    _verify_tree_inventory(first, repo)


def test_inventory_rejects_symlink_root_parent_and_known_directory_link(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    artifact = repo / "artifacts" / "run"
    artifact.mkdir(parents=True)
    (artifact / "payload.log").write_text("payload\n", encoding="utf-8")

    root_link = repo / "artifact-root-link"
    root_link.symlink_to(artifact, target_is_directory=True)
    with pytest.raises(IncidentError, match="symlink"):
        _tree_inventory(root_link, repo)

    parent_link = repo / "parent-link"
    parent_link.symlink_to(artifact, target_is_directory=True)
    with pytest.raises(IncidentError, match="symlink"):
        _tree_inventory(parent_link / "payload.log", repo)

    real_directory = repo / "directory-target"
    real_directory.mkdir()
    directory_link = artifact / "linked-directory"
    directory_link.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(IncidentError, match="directory symlink"):
        _tree_inventory(artifact, repo)

    # A leaf link cannot be repurposed as a capture root/traversed file.
    outside_leaf = tmp_path / "outside.log"
    outside_leaf.write_text("opaque\n", encoding="utf-8")
    leaf = artifact / "leaf.log"
    leaf.symlink_to(outside_leaf)
    with pytest.raises(IncidentError, match="symlink"):
        _tree_inventory(leaf, repo)


def _normalise(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.replace("\\\n", " ").replace("\n", " ")).strip()


def _sha(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def _cell(name: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"cleanr1_d([01])_k(\d+)_sm([123])", name)
    assert match is not None
    return tuple(map(int, match.groups()))


def _body(name: str) -> str:
    data_id, k, seed = _cell(name.removesuffix("_a2"))
    root = "data/sigma_k_10" if data_id == 0 else "data/sigma_k_10_clean_r1_d1"
    return (
        "uv run pretrain.py arch=trm_singlez global_batch_size=2048 epochs=100000 \\\n"
        "  eval_interval=2000 min_eval_interval=0 lr=1e-4 puzzle_emb_lr=1e-4 \\\n"
        "  weight_decay=1.0 puzzle_emb_weight_decay=1.0 +log_z_dynamics=True +z_snapshot=False \\\n"
        "  checkpoint_every_eval=True arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6 \\\n"
        "  arch.L_layers=2 arch.halt_max_steps=1 evaluators=\"[]\" \\\n"
        f"  data_paths=\"[{root}/{k}]\" seed={seed} +k={k} +project_name=\"Sigma_k_new\" \\\n"
        f"  +run_name=\"{name}\" ema=True\n"
    )


def _tool(repo: Path, queue: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if env:
        environment.update(env)
    return subprocess.run(
        [sys.executable, str(TOOL), "--repo-root", str(repo), "--queue-dir", str(queue), *args],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    queue = repo / "scripts" / "queue"
    for path in (queue / "jobs", queue / "processing", queue / "done", queue / "failed", repo / "analysis"):
        path.mkdir(parents=True, exist_ok=True)
    # The tool records and verifies the *synthetic repository* source suite.
    (repo / "analysis" / "clean_r1_incident_requeue.py").write_bytes(TOOL.read_bytes())
    (repo / "scripts" / "queue_run.sh").write_bytes(RUNNER.read_bytes())
    (repo / "scripts" / "sigma_enqueue.sh").write_text("#!/usr/bin/env bash\n# synthetic emitter\n", encoding="utf-8")
    for name in ("verification_report.md", "job_semantics.json", "job_semantics.csv"):
        source = REPO_ROOT / "lab" / "audits" / "2026-07-30_clean-r1-job-semantics" / name
        target = repo / "lab" / "audits" / "2026-07-30_clean-r1-job-semantics" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, target)
    (queue / "stop").write_text("incident safety hold\n", encoding="utf-8")
    records = [{"run_name": name, "body_sha256": _sha(_normalise(_body(name))), "passed": True} for name in BASE_NAMES]
    base = {"schema_version": 1, "protocol": "CLEAN-R1-prelaunch-contract", "status": "PRELAUNCH_CONTRACT_PASS",
            "passed": True, "failures": [], "jobs": records}
    base_path = repo / "base.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    for index, name in enumerate(name for name in BASE_NAMES if name not in TARGET_NAMES):
        (queue / "jobs" / f"{300 + index:04d}_{name}.job").write_text(_body(name), encoding="utf-8")
    # Real incident shape: five stale claims and two already-terminalized
    # attempt-1 failures.  Terminal records must be preserved, not recycled.
    for sequence, name, gpu in TARGETS:
        if sequence in {"0225", "0226"}:
            (queue / "failed" / f"{sequence}_{name}.job").write_text(_body(name), encoding="utf-8")
        else:
            (queue / "processing" / f"{sequence}_{name}.job.gpu{gpu}").write_text(_body(name), encoding="utf-8")
    # Inventory-only partial artifacts prove no artifact rewrite/deletion.
    old_name = TARGETS[0][1]
    log = repo / "logs" / "queue" / f"0223_{old_name}.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"partial queue output\n")
    checkpoint = repo / "checkpoints" / "Sigma_k_new" / old_name
    checkpoint.mkdir(parents=True)
    (checkpoint / "step_7").write_bytes(b"partial checkpoint")
    (checkpoint / "all_config.yaml").write_text(f"run_name: {old_name}\n", encoding="utf-8")
    metadata = repo / "wandb" / "run-20260729_000000-a2id" / "files" / "wandb-metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"id": "a2id", "args": [f"+run_name={old_name}"]}), encoding="utf-8")
    return repo, queue, base_path, log


def _capture_and_plan(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    repo, queue, base, log = _fixture(tmp_path)
    manifest = repo / "audits" / "manifest.json"
    capture = _tool(repo, queue, "capture", "--base-receipt", str(base), "--output", str(manifest))
    assert capture.returncode == 0, capture.stderr
    receipt = repo / "audits" / "amended.json"
    plan = repo / "audits" / "plan.json"
    result = _tool(repo, queue, "plan", "--manifest", str(manifest), "--base-receipt", str(base),
                   "--output", str(plan), "--receipt-output", str(receipt))
    assert result.returncode == 0, result.stderr
    return repo, queue, base, log, manifest, plan


def test_capture_plan_apply_models_seven_attempts_and_preserves_artifacts(tmp_path: Path) -> None:
    repo, queue, _base, log, manifest, plan = _capture_and_plan(tmp_path)
    receipt = repo / "audits" / "amended.json"
    captured = json.loads(manifest.read_text())
    assert [claim["run_name"] for claim in captured["claims"]] == [name for _, name, _ in TARGETS]
    assert [claim["gpu_claim"] for claim in captured["claims"]] == [6, 4, None, None, 7, 5, 3]
    assert [claim["source_lifecycle"] for claim in captured["claims"]] == [
        "processing_claim", "processing_claim", "runner_failed_job_pending_invalid_infra_classification",
        "runner_failed_job_pending_invalid_infra_classification", "processing_claim", "processing_claim", "processing_claim",
    ]
    assert captured["policy"]["data_paths"] == {"d0": "data/sigma_k_10", "d1": "data/sigma_k_10_clean_r1_d1"}
    assert captured["policy"]["sealed_eval_paths"]["d0"].endswith("{k}/sealed")
    preserved_before = {
        path.name: path.read_bytes()
        for path in (queue / "failed").glob("022[56]_cleanr1_*.job")
    }
    assert len(preserved_before) == 2
    original_log = log.read_bytes()
    original_checkpoint = (repo / "checkpoints" / "Sigma_k_new" / TARGETS[0][1] / "step_7").read_bytes()
    applied = _tool(repo, queue, "apply", "--plan", str(plan), "--receipt", str(receipt), "--confirm-apply")
    assert applied.returncode == 0, applied.stderr
    assert log.read_bytes() == original_log
    assert (repo / "checkpoints" / "Sigma_k_new" / TARGETS[0][1] / "step_7").read_bytes() == original_checkpoint
    failed = sorted((queue / "failed").glob("*.attempt1-invalid-infra"))
    assert len(failed) == len(TARGETS)
    assert {
        name.removesuffix(".job") + ".job.attempt1-invalid-infra":
        (queue / "failed" / (name.removesuffix(".job") + ".job.attempt1-invalid-infra")).read_bytes()
        for name in preserved_before
    } == {
        name.removesuffix(".job") + ".job.attempt1-invalid-infra": body
        for name, body in preserved_before.items()
    }
    assert not list((queue / "failed").glob("022[56]_cleanr1_*.job"))
    assert not list((queue / "processing").glob("*_cleanr1_*.job.gpu*"))
    retries = sorted((queue / "jobs").glob("025*_cleanr1_*_a2.job"))
    assert [path.name.split("_", 1)[0] for path in retries] == [f"{253 + i:04d}" for i in range(7)]
    assert len(list((queue / "jobs").glob("*_cleanr1_*.job"))) == 30
    assert json.loads((queue / ".cleanr1-incident-requeue.journal.json").read_text())["phase"] == "committed"
    assert not (queue / ".cleanr1-incident-requeue.fence").exists()
    second = _tool(repo, queue, "apply", "--plan", str(plan), "--receipt", str(receipt), "--confirm-apply")
    assert second.returncode != 0


def test_plan_requires_all_three_capture_source_hashes_current(tmp_path: Path) -> None:
    for relative in ("analysis/clean_r1_incident_requeue.py", "scripts/sigma_enqueue.sh", "scripts/queue_run.sh"):
        repo, queue, base, _log = _fixture(tmp_path / relative.replace("/", "_"))
        manifest = repo / "audits" / "manifest.json"
        assert _tool(repo, queue, "capture", "--base-receipt", str(base), "--output", str(manifest)).returncode == 0
        changed = repo / relative
        changed.write_bytes(changed.read_bytes() + b"# stale\n")
        result = _tool(repo, queue, "plan", "--manifest", str(manifest), "--base-receipt", str(base),
                       "--output", str(repo / "audits" / "plan.json"), "--receipt-output", str(repo / "audits" / "receipt.json"))
        assert result.returncode != 0
        assert "changed since capture" in result.stderr


def test_plan_rejects_retry_collision_and_wrong_d1_training_root(tmp_path: Path) -> None:
    repo, queue, base, _log = _fixture(tmp_path / "collision")
    manifest = repo / "audits" / "manifest.json"
    assert _tool(repo, queue, "capture", "--base-receipt", str(base), "--output", str(manifest)).returncode == 0
    collision = queue / "jobs" / "0253_cleanr1_d0_k4_sm2_a2.job"
    collision.write_text("collision\n", encoding="utf-8")
    collided = _tool(repo, queue, "plan", "--manifest", str(manifest), "--base-receipt", str(base),
                     "--output", str(repo / "audits" / "plan.json"), "--receipt-output", str(repo / "audits" / "receipt.json"))
    assert collided.returncode != 0
    assert "retry queue collision" in collided.stderr

    repo, queue, base, _log = _fixture(tmp_path / "semantic")
    # D1 training is the D1 k-root.  Its /sealed split belongs only to the
    # sealed evaluator and must not silently enter a pretrain body.
    bad = next((queue / "jobs").glob("*_cleanr1_d1_k4_sm1.job"))
    bad.write_text(_body("cleanr1_d1_k4_sm1").replace("data/sigma_k_10_clean_r1_d1/4", "data/sigma_k_10_clean_r1_d1/4/sealed"), encoding="utf-8")
    bad_manifest = repo / "audits" / "manifest.json"
    captured = _tool(repo, queue, "capture", "--base-receipt", str(base), "--output", str(bad_manifest))
    assert captured.returncode == 0, captured.stderr
    semantic = _tool(repo, queue, "plan", "--manifest", str(bad_manifest), "--base-receipt", str(base),
                     "--output", str(repo / "audits" / "plan.json"), "--receipt-output", str(repo / "audits" / "receipt.json"))
    assert semantic.returncode != 0
    assert "data_paths must be" in semantic.stderr


def test_capture_requires_one_exact_allowed_source_per_incident_target(tmp_path: Path) -> None:
    repo, queue, base, _log = _fixture(tmp_path / "duplicate")
    # A second lifecycle entry is ambiguous even if it looks terminal.
    duplicate = queue / "failed" / "0223_cleanr1_d0_k4_sm2.job.attempt1-invalid-infra"
    duplicate.write_text(_body("cleanr1_d0_k4_sm2"), encoding="utf-8")
    result = _tool(repo, queue, "capture", "--base-receipt", str(base), "--output", str(repo / "audits" / "manifest.json"))
    assert result.returncode != 0
    assert "exactly one allowed incident source" in result.stderr

    repo, queue, base, _log = _fixture(tmp_path / "wrong-failed-suffix")
    source = queue / "failed" / "0225_cleanr1_d0_k5_sm2.job"
    source.rename(source.with_name(source.name + ".attempt1-invalid-infra"))
    result = _tool(repo, queue, "capture", "--base-receipt", str(base), "--output", str(repo / "audits" / "manifest.json"))
    assert result.returncode != 0
    assert "not an allowed processing or registered runner-failed .job state" in result.stderr

    repo, queue, base, _log = _fixture(tmp_path / "missing")
    (queue / "failed" / "0225_cleanr1_d0_k5_sm2.job").unlink()
    result = _tool(repo, queue, "capture", "--base-receipt", str(base), "--output", str(repo / "audits" / "manifest.json"))
    assert result.returncode != 0
    assert "exactly one allowed incident source" in result.stderr

    repo, queue, base, _log = _fixture(tmp_path / "wrong-body")
    bad = queue / "processing" / "0223_cleanr1_d0_k4_sm2.job.gpu6"
    bad.write_text(_body("cleanr1_d0_k4_sm2").replace("seed=2", "seed=1"), encoding="utf-8")
    result = _tool(repo, queue, "capture", "--base-receipt", str(base), "--output", str(repo / "audits" / "manifest.json"))
    assert result.returncode != 0
    assert "seed must be" in result.stderr


def test_fault_rollback_and_crash_recover_leave_no_unsealed_queue_mutation(tmp_path: Path) -> None:
    repo, queue, _base, _log, _manifest, plan = _capture_and_plan(tmp_path / "rollback")
    receipt = repo / "audits" / "amended.json"
    rollback = _tool(repo, queue, "apply", "--plan", str(plan), "--receipt", str(receipt), "--confirm-apply",
                     "--test-fail-after", "3", env={"CLEANR1_INCIDENT_TEST_MODE": "1"})
    assert rollback.returncode != 0
    assert len(list((queue / "failed").glob("*.job"))) == 2
    assert not list((queue / "jobs").glob("025*_cleanr1_*_a2.job"))
    assert not list((queue / "jobs").glob(".*.incident-staging"))
    assert not (queue / ".cleanr1-incident-requeue.fence").exists()
    assert json.loads((queue / ".cleanr1-incident-requeue.journal.json").read_text())["phase"] == "aborted"

    repo, queue, _base, _log, _manifest, plan = _capture_and_plan(tmp_path / "crash")
    receipt = repo / "audits" / "amended.json"
    crashed = _tool(repo, queue, "apply", "--plan", str(plan), "--receipt", str(receipt), "--confirm-apply",
                    "--test-hard-crash-after", "3", env={"CLEANR1_INCIDENT_TEST_MODE": "1"})
    assert crashed.returncode != 0
    assert (queue / ".cleanr1-incident-requeue.fence").exists()
    assert len(list((queue / "failed").glob("*.attempt1-invalid-infra"))) == 3
    assert len(list((queue / "failed").glob("*.job"))) == 1
    recovered = _tool(repo, queue, "recover", "--confirm-recover")
    assert recovered.returncode == 0, recovered.stderr
    assert len(list((queue / "failed").glob("*.attempt1-invalid-infra"))) == 0
    assert len(list((queue / "failed").glob("*.job"))) == 2
    assert not list((queue / "jobs").glob(".*.incident-staging"))
    assert not (queue / ".cleanr1-incident-requeue.fence").exists()


def test_runner_revalidates_amended_manifest_and_rejects_gpu_outside_4_to_7(tmp_path: Path) -> None:
    repo, queue, _base, _log, manifest, plan = _capture_and_plan(tmp_path)
    receipt = repo / "audits" / "amended.json"
    assert _tool(repo, queue, "apply", "--plan", str(plan), "--receipt", str(receipt), "--confirm-apply").returncode == 0
    (queue / "stop").unlink()
    env = os.environ | {"QUEUE_DIR": str(queue), "GPUS": "7 4 6 5", "CLEANR1_LAUNCH_RECEIPT": str(receipt)}
    allowed = subprocess.run(["bash", str(repo / "scripts" / "queue_run.sh"), "dry-run"], cwd=repo, env=env, text=True, capture_output=True, check=False)
    assert allowed.returncode == 0, allowed.stderr
    assert "CLEAN-R1 launch gate PASS" in allowed.stdout
    forbidden = subprocess.run(["bash", str(repo / "scripts" / "queue_run.sh"), "dry-run"], cwd=repo,
                                env=env | {"GPUS": "2 4 5 6"}, text=True, capture_output=True, check=False)
    assert forbidden.returncode != 0
    assert "outside the fixed allowed set" in forbidden.stderr
    # A changed manifest must fail the runner before it starts/claims anything.
    manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")
    stale = subprocess.run(["bash", str(repo / "scripts" / "queue_run.sh"), "dry-run"], cwd=repo, env=env, text=True, capture_output=True, check=False)
    assert stale.returncode != 0
    assert "binding is stale" in stale.stderr
