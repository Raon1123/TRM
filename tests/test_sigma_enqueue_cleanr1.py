from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "sigma_enqueue.sh"
ROW_RE = re.compile(r"^(?P<seq>\d{4}) (?P<name>cleanr1_d(?P<data>[01])_k(?P<k>\d+)_sm(?P<seed>[123]))$")
K_VALUES = {4, 5, 6, 7, 8, 10}
FAILURE_K = {5, 6, 7}
CONTROL_K = {4, 8, 10}


def _run_cleanr1(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["QUEUE_DIR"] = str(tmp_path / "queue")
    env["CLEANR1_TEST_QUEUE_DIR_ALLOW"] = "1"
    env["HOME"] = "/dev/null/cleanr1-home-unavailable"
    env["UV_CACHE_DIR"] = "/dev/null/uv-cache-unavailable"
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _normalised_records(stdout: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_name: str | None = None
    body_lines: list[str] = []
    for line in stdout.splitlines():
        row = ROW_RE.match(line)
        if row:
            if current_name is not None:
                body = re.sub(r"\s+", " ", "\n".join(body_lines).replace("\\\n", " ")).strip()
                records.append((current_name, body))
            current_name, body_lines = row.group("name"), []
        elif current_name is not None and line.startswith("    | "):
            body_lines.append(line.removeprefix("    | "))
    if current_name is not None:
        body = re.sub(r"\s+", " ", "\n".join(body_lines).replace("\\\n", " ")).strip()
        records.append((current_name, body))
    return records


def _passing_receipt(tmp_path: Path) -> Path:
    dry = _run_cleanr1(tmp_path, "--dry-run", "cleanr1")
    assert dry.returncode == 0, dry.stderr
    records = _normalised_records(dry.stdout)
    script_sha = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    jobs = []
    for name, body in records:
        match = ROW_RE.match(f"0001 {name}")
        assert match is not None
        jobs.append({
            "run_name": name,
            "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "checkpoint_every_eval": True,
            "data_id": int(match.group("data")),
            "ema": True,
            "expected_eval_cadence_steps": 4882,
            "expected_eval_interval_epochs": 2000,
            "expected_first_eval_step": 4882,
            "failures": [],
            "k": int(match.group("k")),
            "live_log_contract": {
                "contract_id": "CLEAN-R1-LIVE-v1",
                "enabled": True,
                "enablement": "run_name_prefix:cleanr1_",
                "schema_version": 1,
                "sidecar_name": "clean_r1_live.jsonl",
            },
            "metric_contract": {
                "monitor_metric_path": "probe/test_exact",
                "monitor_probe_size": 512,
                "sealed_metric_id": "sealed-test-exact",
                "sealed_metric_path": "sealed/terminal_ema/exact_sequence_accuracy_all_examples",
            },
            "passed": True,
            "seed": int(match.group("seed")),
            "test_probe_identity": {
                "inputs_file_sha256": "a" * 64,
                "n_examples": 512,
                "semantic_sha256": "b" * 64,
                "split": "test",
                "z_logging_input_md5_8": "c" * 8,
            },
        })
    receipt = {
        "schema_version": 1,
        "protocol": "CLEAN-R1-prelaunch-contract",
        "status": "PRELAUNCH_CONTRACT_PASS",
        "passed": True,
        "failures": [],
        "queue_emitter": {"path": str(SCRIPT), "sha256_file": script_sha},
        "source_hashes": {"scripts/sigma_enqueue.sh": script_sha},
        "command": {"returncode": 0},
        "contract": {
            "job_count": 30,
            "checkpoint_every_eval": True,
            "monitor_metric_path": "probe/test_exact",
            "monitor_probe_size": 512,
            "eval_interval_epochs": 2000,
            "live_log_enabled_by": "run_name_prefix:cleanr1_",
            "live_log_schema_version": 1,
            "live_log_contract_id": "CLEAN-R1-LIVE-v1",
            "live_log_sidecar_name": "clean_r1_live.jsonl",
            "sealed_metric_id": "sealed-test-exact",
            "sealed_metric_path": "sealed/terminal_ema/exact_sequence_accuracy_all_examples",
        },
        "jobs": jobs,
    }
    path = tmp_path / "prelaunch_contract.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def _body_by_name(stdout: str) -> dict[str, str]:
    bodies: dict[str, list[str]] = {}
    current_name: str | None = None
    for line in stdout.splitlines():
        row = ROW_RE.match(line)
        if row:
            current_name = row.group("name")
            bodies[current_name] = []
        elif current_name is not None and line.startswith("    | "):
            bodies[current_name].append(line.removeprefix("    | "))
    return {name: "\n".join(lines) for name, lines in bodies.items()}


def test_cleanr1_dry_run_expands_exact_contract_without_queue_writes(tmp_path: Path) -> None:
    result = _run_cleanr1(tmp_path, "--dry-run", "cleanr1")

    assert result.returncode == 0, result.stderr
    assert "jobs: 30 (dry run, nothing written)" in result.stdout
    assert not (tmp_path / "queue").exists()

    rows = [match for line in result.stdout.splitlines() if (match := ROW_RE.match(line))]
    assert len(rows) == 30
    assert len({row.group("name") for row in rows}) == 30
    assert [int(row.group("seq")) for row in rows] == list(range(1, 31))

    cells = {(int(row.group("data")), int(row.group("k")), int(row.group("seed"))) for row in rows}
    expected = {
        (0, k, seed) for k in K_VALUES for seed in (2, 3)
    } | {
        (1, k, seed) for k in K_VALUES for seed in (1, 2, 3)
    }
    assert cells == expected
    assert sum(k in FAILURE_K for _, k, _ in cells) == 15
    assert sum(k in CONTROL_K for _, k, _ in cells) == 15

    bodies = _body_by_name(result.stdout)
    assert set(bodies) == {row.group("name") for row in rows}
    for data_id, k, seed in cells:
        name = f"cleanr1_d{data_id}_k{k}_sm{seed}"
        body = bodies[name]
        data_root = "data/sigma_k_10" if data_id == 0 else "data/sigma_k_10_clean_r1_d1"
        assert f"arch=trm_singlez" in body
        assert "arch.mlp_t=False" in body
        assert "arch.H_cycles=3" in body
        assert "arch.L_cycles=6" in body
        assert "arch.L_layers=2" in body
        assert "arch.halt_max_steps=1" in body
        assert "global_batch_size=2048" in body
        assert "epochs=100000" in body
        assert "eval_interval=2000" in body
        assert "checkpoint_every_eval=True" in body
        assert f'data_paths="[{data_root}/{k}]"' in body
        assert f"seed={seed}" in body
        assert f"+k={k}" in body
        assert '+project_name="Sigma_k_new"' in body
        assert f'+run_name="{name}"' in body
        assert "ema=True" in body
        assert 'data_paths="[data/sigma_k/' not in body
        assert "CUDA_VISIBLE_DEVICES" not in body


def test_cleanr1_default_is_dry_run_without_creating_queue(tmp_path: Path) -> None:
    result = _run_cleanr1(tmp_path, "cleanr1")

    assert result.returncode == 0, result.stderr
    assert "jobs: 30 (dry run, nothing written)" in result.stdout
    assert not (tmp_path / "queue").exists()


def test_cleanr1_apply_fails_closed_for_missing_unparseable_and_stale_receipts(tmp_path: Path) -> None:
    missing = _run_cleanr1(tmp_path, "--apply", "cleanr1")
    assert missing.returncode != 0
    assert "requires --receipt" in missing.stderr

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-json", encoding="utf-8")
    bad = _run_cleanr1(tmp_path, "--apply", "cleanr1", "--receipt", str(bad_json))
    assert bad.returncode != 0
    assert "cannot parse JSON" in bad.stderr

    stale_path = _passing_receipt(tmp_path)
    stale = json.loads(stale_path.read_text())
    stale["queue_emitter"]["sha256_file"] = "0" * 64
    stale_path.write_text(json.dumps(stale), encoding="utf-8")
    stale_result = _run_cleanr1(tmp_path, "--apply", "cleanr1", "--receipt", str(stale_path))
    assert stale_result.returncode != 0
    assert "stale queue-emitter" in stale_result.stderr

    bad_contract_path = _passing_receipt(tmp_path)
    bad_contract = json.loads(bad_contract_path.read_text())
    bad_contract["jobs"].pop()
    bad_contract_path.write_text(json.dumps(bad_contract), encoding="utf-8")
    bad_contract_result = _run_cleanr1(
        tmp_path, "--apply", "cleanr1", "--receipt", str(bad_contract_path)
    )
    assert bad_contract_result.returncode != 0
    assert "job count is not exactly 30" in bad_contract_result.stderr
    assert not (tmp_path / "queue").exists()


def test_cleanr1_apply_materializes_exact_jobs_and_rejects_second_apply(tmp_path: Path) -> None:
    receipt = _passing_receipt(tmp_path)
    result = _run_cleanr1(tmp_path, "--apply", "cleanr1", "--receipt", str(receipt))

    assert result.returncode == 0, result.stderr
    assert not Path("/dev/null/cleanr1-home-unavailable").exists()
    assert "receipt validated:" in result.stdout
    assert "jobs: 30" in result.stdout
    job_paths = sorted((tmp_path / "queue" / "jobs").glob("*.job"))
    assert len(job_paths) == 30
    assert [int(path.name.split("_", 1)[0]) for path in job_paths] == list(range(1, 31))
    names = [path.stem.split("_", 1)[1] for path in job_paths]
    assert names == [name for name, _ in _normalised_records(_run_cleanr1(tmp_path / "fresh", "--dry-run", "cleanr1").stdout)]
    for path in job_paths:
        body = path.read_text()
        assert "arch=trm_singlez" in body
        assert "checkpoint_every_eval=True" in body
        assert "+log_z_dynamics=True" in body
        assert '+project_name="Sigma_k_new"' in body
        assert "ema=True" in body
        assert 'data_paths="[data/sigma_k_10/' in body or 'data_paths="[data/sigma_k_10_clean_r1_d1/' in body
        assert 'data_paths="[data/sigma_k/' not in body

    before = {path.name: path.read_bytes() for path in job_paths}
    second = _run_cleanr1(tmp_path, "--apply", "cleanr1", "--receipt", str(receipt))
    assert second.returncode != 0
    assert "collision in queue lifecycle" in second.stderr
    after_paths = sorted((tmp_path / "queue" / "jobs").glob("*.job"))
    assert {path.name: path.read_bytes() for path in after_paths} == before
