from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "queue_run.sh"


def _run(tmp_path: Path, gpus: str, *, cleanr1: bool = False) -> subprocess.CompletedProcess[str]:
    queue = tmp_path / "queue"
    jobs = queue / "jobs"
    jobs.mkdir(parents=True)
    name = "0001_cleanr1_d0_k4_sm2.job" if cleanr1 else "0001_generic.job"
    (jobs / name).write_text("echo never-executed\n", encoding="utf-8")
    env = os.environ | {"QUEUE_DIR": str(queue), "GPUS": gpus}
    return subprocess.run(["bash", str(RUNNER), "dry-run"], cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)


def test_runner_has_non_overridable_gpu_4_to_7_allowlist(tmp_path: Path) -> None:
    rejected = _run(tmp_path, "2 3 4 5")
    assert rejected.returncode != 0
    assert "outside the fixed allowed set {4 5 6 7}" in rejected.stderr

    # Even an environment attempt to widen the old allowlist cannot re-enable
    # GPUs 2/3; the runner now ignores ALLOWED_GPUS/FORCE_GPUS entirely.
    queue = tmp_path / "override" / "queue"
    (queue / "jobs").mkdir(parents=True)
    (queue / "jobs" / "0001_generic.job").write_text("echo never-executed\n", encoding="utf-8")
    env = os.environ | {"QUEUE_DIR": str(queue), "GPUS": "2 4 5 6", "ALLOWED_GPUS": "2 3 4 5 6 7", "FORCE_GPUS": "1"}
    bypass = subprocess.run(["bash", str(RUNNER), "dry-run"], cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
    assert bypass.returncode != 0
    assert "outside the fixed allowed set" in bypass.stderr


def test_cleanr1_without_amended_receipt_claims_zero(tmp_path: Path) -> None:
    result = _run(tmp_path, "4 5 6 7", cleanr1=True)
    assert result.returncode != 0
    assert "requires CLEANR1_LAUNCH_RECEIPT=<amended-prelaunch-contract.json>" in result.stderr
    assert (tmp_path / "queue" / "jobs" / "0001_cleanr1_d0_k4_sm2.job").is_file()
    assert not list((tmp_path / "queue" / "processing").glob("*.job.gpu*"))
