from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / ".agents/skills/goal-graph-orchestrator/scripts/ledger.py"


def load_ledger_module():
    spec = importlib.util.spec_from_file_location("goal_graph_ledger_test_module", LEDGER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trusted_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = load_ledger_module()
    mount = tmp_path / "mnt"
    user_root = mount / "ayp"
    parent = user_root / "agents"
    parent.mkdir(parents=True)
    for anchor in (mount, user_root, parent):
        anchor.chmod(0o755)
    trusted = parent / "trm"
    monkeypatch.setattr(module, "TRUSTED_EXTERNAL_ANCHORS", (mount, user_root, parent))
    monkeypatch.setattr(module, "TRUSTED_EXTERNAL_ROOT", trusted)
    return module, trusted


def run(*args: object, ok: bool = True) -> dict:
    completed = subprocess.run(
        [sys.executable, str(LEDGER), *map(str, args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if ok:
        assert completed.returncode == 0, completed.stdout + completed.stderr
    else:
        assert completed.returncode != 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def scaffold(
    tmp_path: Path, *, goal_effort: int = 100, terra_effort: int = 60, terra_reserve: int = 12,
) -> tuple[Path, Path]:
    state = tmp_path / "state"
    run(
        "init", "--state-root", state, "--goal-id", "G-20260802-test",
        "--objective", "exercise engine", "--wall-minutes", 60,
        "--effort-minutes", goal_effort, "--verification-reserve", 20,
        "--write-scope", "outputs/goal-graph-test",
    )
    goal = state / "G-20260802-test"
    run(
        "add-terra", goal, "--subproject-id", "SP-01", "--owner", "terra-author",
        "--objective", "own intermediate project", "--wall-minutes", 30,
        "--effort-minutes", terra_effort, "--verification-reserve", terra_reserve,
        "--concurrency", 2, "--write-scope", "outputs/goal-graph-test",
    )
    return goal, goal / "subprojects/SP-01/ledger.yaml"


def add_task(terra: Path, task_id: str, output: str, *dependencies: str) -> None:
    args: list[object] = [
        "add-task", terra, "--task-id", task_id, "--objective", f"produce {output}",
        "--parent-criterion", f"{output} is valid", "--criterion", "artifact is valid",
        "--evidence-path", output, "--timebox", 15, "--write-scope", output,
        "--output", output,
    ]
    for dependency in dependencies:
        args += ["--depends-on", dependency]
    run(*args)


def start(terra: Path, task_id: str = "SP-01-L01", *, agent: str = "worker-1", model: str = "gpt-5.6-terra") -> Path:
    return Path(run("start-attempt", terra, "--task-id", task_id, "--agent-id", agent, "--actual-model", model)["attempt_dir"])


def pass_review(attempt: Path, *, review_id: str = "fresh", agent: str = "verifier-1") -> dict:
    """A PASS review must carry direct evidence for every frozen criterion."""
    return run(
        "record-review", attempt, "--review-id", review_id, "--seat", "luna-verifier",
        "--agent-id", agent, "--actual-model", "gpt-5.6-terra", "--verdict", "PASS",
        "--criterion-pass", f"C1={attempt / 'result.yaml'}", "--form-check", "schema",
        "--intent-check", "outcome", "--evidence", attempt / "result.yaml",
    )


def record_feedback(attempt: Path) -> dict:
    return run(
        "record-feedback", attempt, "--failed-item", "C1", "--observed", "artifact failed",
        "--expected", "artifact passes", "--evidence", attempt / "result.yaml",
        "--likely-cause", "implementation defect", "--minimal-repair", "repair artifact",
        "--rerun-scope", "outputs/goal-graph-test/a.txt", "--rerun-command", "rtk uv run pytest",
        "--owner", "terra-author",
    )


def test_happy_path_requires_independent_review_and_resumes(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    add_task(terra, "SP-01-L02", "outputs/goal-graph-test/b.txt", "SP-01-L01")
    assert run("ready", terra)["ready"] == ["SP-01-L01"]
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    attempt = start(terra)
    evidence = attempt.parent.parent / "orders/001.yaml"
    run("record-result", attempt, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra", "--verdict", "PASS", "--elapsed-minutes", 1, "--evidence", evidence, "--form-check", "schema", "--intent-check", "outcome")
    run("transition", terra, "--node", "SP-01-L01", "--to", "VERIFYING")
    failed = run("record-review", attempt, "--review-id", "self", "--seat", "luna-verifier", "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra", "--verdict", "PASS", "--form-check", "schema", "--intent-check", "outcome", "--evidence", attempt / "result.yaml", ok=False)
    assert "differ" in failed["error"]
    review = pass_review(attempt)
    run("transition", terra, "--node", "SP-01-L01", "--to", "PASS", "--evidence", review["review"])
    assert run("ready", terra)["ready"] == ["SP-01-L02"]
    assert run("validate", goal)["error_count"] == 0
    assert any(node["id"] == "SP-01-L01" and node["status"] == "PASS" for node in run("status", goal)["nodes"])


def test_pass_without_evidence_and_illegal_transition_are_rejected(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    assert "illegal transition" in run("transition", terra, "--node", "SP-01-L01", "--to", "PASS", ok=False)["error"]
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    assert "illegal transition" in run("transition", terra, "--node", "SP-01-L01", "--to", "PASS", ok=False)["error"]


def test_validate_rejects_cycle_and_active_scope_overlap(tmp_path: Path) -> None:
    goal, terra_path = scaffold(tmp_path)
    add_task(terra_path, "SP-01-L01", "outputs/goal-graph-test/shared")
    add_task(terra_path, "SP-01-L02", "outputs/goal-graph-test/shared/child")
    terra = yaml.safe_load(terra_path.read_text())
    terra["children"][0]["dependencies"] = ["SP-01-L02"]
    terra["children"][1]["dependencies"] = ["SP-01-L01"]
    terra["children"][0]["status"] = "READY"
    terra["children"][1]["status"] = "READY"
    terra_path.write_text(yaml.safe_dump(terra, sort_keys=False), encoding="utf-8")
    result = run("validate", goal, ok=False)
    assert any("cycle" in error for error in result["errors"])
    assert any("write-scope overlap" in error for error in result["errors"])


def test_validate_rejects_budget_reserve_consumption(tmp_path: Path) -> None:
    goal, _ = scaffold(tmp_path)
    goal_path = goal / "goal-ledger.yaml"
    data = yaml.safe_load(goal_path.read_text())
    data["graph"]["nodes"][0]["allocation"]["agent_effort_budget_minutes"] = 81
    goal_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    result = run("validate", goal, ok=False)
    assert any("consume verification reserve" in error for error in result["errors"])


def test_immutable_order_and_optimistic_revision(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    duplicate = run("add-task", terra, "--task-id", "SP-01-L01", "--objective", "different", "--parent-criterion", "same", "--criterion", "same", "--evidence-path", "outputs/goal-graph-test/a.txt", "--write-scope", "outputs/goal-graph-test/a.txt", ok=False)
    assert "duplicate task" in duplicate["error"]
    stale = run("transition", terra, "--node", "SP-01-L01", "--to", "READY", "--expected-revision", 0, ok=False)
    assert "revision mismatch" in stale["error"]


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    result = run("init", "--state-root", tmp_path, "--goal-id", "G-20260802-escape", "--objective", "escape", "--effort-minutes", 100, "--verification-reserve", 20, "--write-scope", "/etc", ok=False)
    assert "escapes workspace" in result["error"]


def test_trusted_external_scope_is_canonical_and_stable_when_root_appears(trusted_scope) -> None:
    module, trusted = trusted_scope
    requested = trusted / "future/artifact"
    before = module.safe_scope(str(requested))
    trusted.mkdir(mode=0o700)
    after = module.safe_scope(str(requested))
    assert before == after == requested.as_posix()

    target = trusted / "canonical"
    target.mkdir()
    (trusted / "alias").symlink_to(target, target_is_directory=True)
    assert module.safe_scope(str(trusted / "alias/artifact")) == (target / "artifact").as_posix()


def test_trusted_external_scope_rejects_broad_prefixes_traversal_and_relative_alias(trusted_scope) -> None:
    module, trusted = trusted_scope
    rejected = [
        trusted.parent,
        trusted.parent.parent,
        trusted.parent.parent / "agents-not-approved",
        trusted.parent.parent / "trm-not-approved",
        trusted.with_name("trm2"),
        trusted / ".." / "trm2",
    ]
    for path in rejected:
        with pytest.raises(module.LedgerError, match="escapes workspace"):
            module.safe_scope(str(path))

    relative_alias = os.path.relpath(trusted / "artifact", module.WORKSPACE)
    with pytest.raises(module.LedgerError, match="must be absolute"):
        module.safe_scope(relative_alias)


def test_workspace_root_does_not_contain_the_external_trusted_root(trusted_scope) -> None:
    module, trusted = trusted_scope
    assert not module.contained(str(trusted / "artifact"), ".")
    assert not module.overlaps(str(trusted / "artifact"), ".")
    with pytest.raises(module.LedgerError, match="exceeds parent scope"):
        module.validate_delegated_scope([str(trusted / "artifact")], ["."], [])


def test_trusted_external_anchor_rejects_unsafe_permissions_and_symlinks(trusted_scope) -> None:
    module, trusted = trusted_scope
    parent = module.TRUSTED_EXTERNAL_ANCHORS[-1]
    parent.chmod(0o777)
    with pytest.raises(module.LedgerError, match=r"parent permissions are unsafe \(0777\)"):
        module.safe_scope(str(trusted / "artifact"))

    parent.chmod(0o755)
    outside = parent.parent / "outside"
    outside.mkdir()
    trusted.symlink_to(outside, target_is_directory=True)
    with pytest.raises(module.LedgerError, match="trusted root must be a real directory"):
        module.safe_scope(str(trusted / "artifact"))

    trusted.unlink()
    trusted.mkdir(mode=0o700)
    trusted.chmod(0o777)
    with pytest.raises(module.LedgerError, match=r"trusted root permissions are unsafe \(0777\)"):
        module.safe_scope(str(trusted / "artifact"))
    trusted.chmod(0o700)
    (trusted / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(module.LedgerError, match="symlink escapes trusted root"):
        module.safe_scope(str(trusted / "escape/artifact"))


def test_trusted_external_anchor_rejects_a_symlinked_parent(trusted_scope, monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = trusted_scope
    mount, user_root, _ = module.TRUSTED_EXTERNAL_ANCHORS
    real_parent = user_root / "real-agents"
    real_parent.mkdir()
    linked_parent = user_root / "agents-link"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    monkeypatch.setattr(module, "TRUSTED_EXTERNAL_ANCHORS", (mount, user_root, linked_parent))
    monkeypatch.setattr(module, "TRUSTED_EXTERNAL_ROOT", linked_parent / "trm")
    with pytest.raises(module.LedgerError, match="anchor must be a real directory"):
        module.safe_scope(str(linked_parent / "trm/artifact"))


def test_trusted_external_anchor_rejects_missing_or_wrong_owner(trusted_scope, monkeypatch: pytest.MonkeyPatch) -> None:
    module, trusted = trusted_scope
    mount, user_root, parent = module.TRUSTED_EXTERNAL_ANCHORS
    missing_parent = user_root / "missing-agents"
    monkeypatch.setattr(module, "TRUSTED_EXTERNAL_ANCHORS", (mount, user_root, missing_parent))
    monkeypatch.setattr(module, "TRUSTED_EXTERNAL_ROOT", missing_parent / "trm")
    with pytest.raises(module.LedgerError, match="anchor is missing"):
        module.safe_scope(str(missing_parent / "trm/artifact"))

    monkeypatch.setattr(module, "TRUSTED_EXTERNAL_ANCHORS", (mount, user_root, parent))
    monkeypatch.setattr(module, "TRUSTED_EXTERNAL_ROOT", trusted)
    actual_uid = os.geteuid()
    monkeypatch.setattr(module.os, "geteuid", lambda: actual_uid + 1)
    with pytest.raises(module.LedgerError, match="parent must be owned by uid"):
        module.safe_scope(str(trusted / "artifact"))


def test_missing_dependency_is_rejected(tmp_path: Path) -> None:
    goal, terra_path = scaffold(tmp_path)
    add_task(terra_path, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    terra = yaml.safe_load(terra_path.read_text())
    terra["children"][0]["dependencies"] = ["SP-01-L99"]
    terra_path.write_text(yaml.safe_dump(terra, sort_keys=False), encoding="utf-8")
    result = run("validate", goal, ok=False)
    assert any("depends on missing SP-01-L99" in error for error in result["errors"])


def test_tampered_ready_node_cannot_bypass_dependency_gate(tmp_path: Path) -> None:
    goal, terra_path = scaffold(tmp_path)
    add_task(terra_path, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    add_task(terra_path, "SP-01-L02", "outputs/goal-graph-test/b.txt", "SP-01-L01")
    terra = yaml.safe_load(terra_path.read_text())
    terra["children"][1]["status"] = "READY"
    terra_path.write_text(yaml.safe_dump(terra, sort_keys=False), encoding="utf-8")
    result = run("validate", goal, ok=False)
    assert any("before dependency SP-01-L01 PASS" in error for error in result["errors"])
    rejected = run(
        "start-attempt", terra_path, "--task-id", "SP-01-L02", "--agent-id", "worker-2",
        "--actual-model", "gpt-5.6-terra", ok=False,
    )
    assert "dependencies must PASS" in rejected["error"]


def test_failure_local_retry_preserves_order_and_prior_attempt(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    order = terra.parent / "tasks/SP-01-L01/orders/001.yaml"
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    first = Path(run("start-attempt", terra, "--task-id", "SP-01-L01", "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra")["attempt_dir"])
    run("record-result", first, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra", "--verdict", "FAIL", "--elapsed-minutes", 1, "--form-check", "schema", "--intent-check", "failed predicate")
    record_feedback(first)
    run("transition", terra, "--node", "SP-01-L01", "--to", "FAIL")
    frozen_hash = hashlib.sha256((first / "result.yaml").read_bytes() + order.read_bytes()).hexdigest()
    run("reissue-order", terra, "--task-id", "SP-01-L01", "--feedback", first / "feedback.yaml")
    second = Path(run("start-attempt", terra, "--task-id", "SP-01-L01", "--agent-id", "worker-2", "--actual-model", "gpt-5.6-terra")["attempt_dir"])
    assert second.name == "002"
    assert hashlib.sha256((first / "result.yaml").read_bytes() + order.read_bytes()).hexdigest() == frozen_hash


def test_missing_immutable_order_is_rejected(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    order = terra.parent / "tasks/SP-01-L01/orders/001.yaml"
    order.rename(order.with_suffix(".missing"))
    result = run("validate", goal, ok=False)
    assert any("immutable order is missing" in error for error in result["errors"])


def test_active_concurrency_quota_is_enforced(tmp_path: Path) -> None:
    goal, terra_path = scaffold(tmp_path)
    add_task(terra_path, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    add_task(terra_path, "SP-01-L02", "outputs/goal-graph-test/b.txt")
    terra = yaml.safe_load(terra_path.read_text())
    terra["schedule"]["concurrency_quota"] = 1
    for child in terra["children"]:
        child["status"] = "READY"
    terra_path.write_text(yaml.safe_dump(terra, sort_keys=False), encoding="utf-8")
    result = run("validate", goal, ok=False)
    assert any("active tasks exceed concurrency quota" in error for error in result["errors"])


def test_result_identity_must_match_frozen_assignment(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    attempt = start(terra, agent="worker-1", model="gpt-5.6-terra")
    mismatch = run(
        "record-result", attempt, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-sol",
        "--verdict", "PASS", "--elapsed-minutes", 1, ok=False,
    )
    assert "agent/model" in mismatch["error"]


def test_fail_result_cannot_be_promoted_by_a_pass_review(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    attempt = start(terra)
    run(
        "record-result", attempt, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra",
        "--verdict", "FAIL", "--elapsed-minutes", 1, "--form-check", "schema",
        "--intent-check", "failed predicate",
    )
    rejected = run(
        "record-review", attempt, "--review-id", "fresh", "--seat", "luna-verifier",
        "--agent-id", "verifier-1", "--actual-model", "gpt-5.6-terra", "--verdict", "PASS",
        "--criterion-pass", f"C1={attempt / 'result.yaml'}", "--form-check", "schema",
        "--intent-check", "outcome", "--evidence", attempt / "result.yaml", ok=False,
    )
    assert "non-PASS" in rejected["error"]


def test_unsafe_ids_and_delegated_scope_violations_are_rejected(tmp_path: Path) -> None:
    unsafe_goal = run(
        "init", "--state-root", tmp_path, "--goal-id", "G-20260802/bad", "--objective", "bad",
        "--effort-minutes", 100, "--verification-reserve", 20, "--write-scope", "outputs/goal-graph-test", ok=False,
    )
    assert "unsafe goal" in unsafe_goal["error"]
    goal, terra = scaffold(tmp_path)
    unsafe_task = run(
        "add-task", terra, "--task-id", "SP-01-L01/escape", "--objective", "bad",
        "--parent-criterion", "bad", "--criterion", "bad", "--evidence-path", "proof",
        "--write-scope", "outputs/goal-graph-test/a.txt", ok=False,
    )
    assert "unsafe task" in unsafe_task["error"]
    escaped_scope = run(
        "add-terra", goal, "--subproject-id", "SP-02", "--owner", "terra-author",
        "--objective", "outside", "--wall-minutes", 10, "--effort-minutes", 10,
        "--verification-reserve", 2, "--write-scope", "outputs/outside", ok=False,
    )
    assert "exceeds parent scope" in escaped_scope["error"]
    data = yaml.safe_load(terra.read_text())
    data["contract"]["forbidden_paths"] = ["outputs/goal-graph-test/denied"]
    terra.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    forbidden = run(
        "add-task", terra, "--task-id", "SP-01-L01", "--objective", "bad",
        "--parent-criterion", "bad", "--criterion", "bad", "--evidence-path", "proof",
        "--write-scope", "outputs/goal-graph-test/denied", ok=False,
    )
    assert "forbidden" in forbidden["error"]


def test_validate_rejects_tampered_goal_external_scope(tmp_path: Path, trusted_scope) -> None:
    module, trusted = trusted_scope
    goal, _ = scaffold(tmp_path)
    goal_path = goal / "goal-ledger.yaml"
    data = yaml.safe_load(goal_path.read_text())
    data["contract"]["write_scope"] = [str(trusted.with_name("trm2"))]
    goal_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    errors = module.validate_root(goal)
    assert any("goal contract has unsafe scope" in error and "escapes workspace" in error for error in errors)


def test_validate_rejects_tampered_terra_external_scope(tmp_path: Path, trusted_scope) -> None:
    module, trusted = trusted_scope
    goal, terra_path = scaffold(tmp_path)
    terra = yaml.safe_load(terra_path.read_text())
    terra["contract"]["write_scope"] = [str(trusted / "terra")]
    terra_path.write_text(yaml.safe_dump(terra, sort_keys=False), encoding="utf-8")
    errors = module.validate_root(goal)
    assert any("SP-01 has unsafe contract scope" in error and "exceeds parent scope" in error for error in errors)


def test_tampered_order_external_scope_fails_validation_and_start(tmp_path: Path, trusted_scope) -> None:
    module, trusted = trusted_scope
    goal, terra_path = scaffold(tmp_path)
    add_task(terra_path, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    order_path = terra_path.parent / "tasks/SP-01-L01/orders/001.yaml"
    order = yaml.safe_load(order_path.read_text())
    order["scope"]["write_scope"] = [str(trusted / "task")]
    order_path.write_text(yaml.safe_dump(order, sort_keys=False), encoding="utf-8")
    terra = yaml.safe_load(terra_path.read_text())
    terra["children"][0]["order_sha256"] = hashlib.sha256(order_path.read_bytes()).hexdigest()
    terra_path.write_text(yaml.safe_dump(terra, sort_keys=False), encoding="utf-8")

    errors = module.validate_root(goal)
    assert any("unsafe order scope" in error and "exceeds parent scope" in error for error in errors)
    run("transition", terra_path, "--node", "SP-01-L01", "--to", "READY")
    args = module.parser().parse_args([
        "start-attempt", str(terra_path), "--task-id", "SP-01-L01",
        "--agent-id", "worker-1", "--actual-model", "test-model",
    ])
    with pytest.raises(module.LedgerError, match="exceeds parent scope"):
        args.func(args)


def test_tampered_feedback_external_scope_fails_validation_and_reissue(tmp_path: Path, trusted_scope) -> None:
    module, trusted = trusted_scope
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    attempt = start(terra)
    run(
        "record-result", attempt, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra",
        "--verdict", "FAIL", "--elapsed-minutes", 1, "--form-check", "schema",
        "--intent-check", "failed predicate",
    )
    feedback = Path(record_feedback(attempt)["feedback"])
    run("transition", terra, "--node", "SP-01-L01", "--to", "FAIL")
    packet = yaml.safe_load(feedback.read_text())
    packet["rerun_scope"] = [str(trusted / "rerun")]
    feedback.write_text(yaml.safe_dump(packet, sort_keys=False), encoding="utf-8")

    errors = module.validate_root(goal)
    assert any("unsafe feedback scope" in error and "exceeds parent scope" in error for error in errors)
    args = module.parser().parse_args([
        "reissue-order", str(terra), "--task-id", "SP-01-L01", "--feedback", str(feedback),
    ])
    with pytest.raises(module.LedgerError, match="exceeds parent scope"):
        args.func(args)


def test_reissue_rejects_tampered_root_or_terra_external_scope(tmp_path: Path, trusted_scope) -> None:
    module, trusted = trusted_scope
    for tamper in ("root", "terra"):
        goal, terra_path = scaffold(tmp_path / tamper)
        add_task(terra_path, "SP-01-L01", "outputs/goal-graph-test/a.txt")
        run("transition", terra_path, "--node", "SP-01-L01", "--to", "READY")
        attempt = start(terra_path)
        run(
            "record-result", attempt, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra",
            "--verdict", "FAIL", "--elapsed-minutes", 1, "--form-check", "schema",
            "--intent-check", "failed predicate",
        )
        feedback = Path(record_feedback(attempt)["feedback"])
        run("transition", terra_path, "--node", "SP-01-L01", "--to", "FAIL")
        target = goal / "goal-ledger.yaml" if tamper == "root" else terra_path
        data = yaml.safe_load(target.read_text())
        data["contract"]["write_scope"] = [
            str(trusted.with_name("trm2")) if tamper == "root" else str(trusted / "terra")
        ]
        target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        args = module.parser().parse_args([
            "reissue-order", str(terra_path), "--task-id", "SP-01-L01", "--feedback", str(feedback),
        ])
        expected = "escapes workspace" if tamper == "root" else "exceeds parent scope"
        with pytest.raises(module.LedgerError, match=expected):
            args.func(args)


def test_reissue_requires_immutable_feedback_and_preserves_prior_order(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path, terra_effort=50)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    first = start(terra)
    run(
        "record-result", first, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra",
        "--verdict", "FAIL", "--elapsed-minutes", 1, "--form-check", "schema",
        "--intent-check", "failed predicate",
    )
    missing = run("transition", terra, "--node", "SP-01-L01", "--to", "FAIL", ok=False)
    assert "feedback" in missing["error"]
    prior_order = terra.parent / "tasks/SP-01-L01/orders/001.yaml"
    frozen = prior_order.read_bytes()
    missing = run("reissue-order", terra, "--task-id", "SP-01-L01", ok=False)
    assert "feedback" in missing["error"]
    feedback = record_feedback(first)
    run("transition", terra, "--node", "SP-01-L01", "--to", "FAIL")
    reissued = run("reissue-order", terra, "--task-id", "SP-01-L01", "--feedback", feedback["feedback"])
    assert Path(reissued["order"]).name == "002.yaml"
    assert prior_order.read_bytes() == frozen
    terra_data = yaml.safe_load(terra.read_text())
    child = terra_data["children"][0]
    assert child["order_version"] == 2
    assert child["current_attempt"] == 1


def test_retry_cannot_consume_terra_verification_reserve(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path, terra_effort=30, terra_reserve=12)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    first = start(terra)
    run(
        "record-result", first, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra",
        "--verdict", "FAIL", "--elapsed-minutes", 1, "--form-check", "schema",
        "--intent-check", "failed predicate",
    )
    feedback = record_feedback(first)
    run("transition", terra, "--node", "SP-01-L01", "--to", "FAIL")
    rejected = run("reissue-order", terra, "--task-id", "SP-01-L01", "--feedback", feedback["feedback"], ok=False)
    assert "reserve" in rejected["error"]


def test_ordinary_first_failure_cannot_fall_back_to_sol(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    first = start(terra)
    run(
        "record-result", first, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra",
        "--verdict", "FAIL", "--elapsed-minutes", 1, "--form-check", "schema",
        "--intent-check", "local defect",
    )
    record_feedback(first)
    run("transition", terra, "--node", "SP-01-L01", "--to", "FAIL")
    rejected = run("transition", terra, "--node", "SP-01-L01", "--to", "ESCALATED", ok=False)
    assert "escalation packet" in rejected["error"]
    packet = run(
        "record-escalation", first, "--root-contract-field", "authorization",
        "--observed", "required operation is not authorized", "--decision-required", "approve or deny operation",
        "--evidence", first / "result.yaml", "--requested-by", "terra-author",
    )
    run("transition", terra, "--node", "SP-01-L01", "--to", "ESCALATED", "--escalation", packet["escalation"])


def test_nonpass_review_must_score_every_frozen_criterion(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    attempt = start(terra)
    run(
        "record-result", attempt, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra",
        "--verdict", "FAIL", "--elapsed-minutes", 1, "--form-check", "schema", "--intent-check", "defect",
    )
    rejected = run(
        "record-review", attempt, "--review-id", "negative-empty", "--seat", "luna-verifier",
        "--agent-id", "verifier-1", "--actual-model", "gpt-5.6-terra", "--verdict", "FAIL", ok=False,
    )
    assert "score every frozen criterion" in rejected["error"]
    run(
        "record-review", attempt, "--review-id", "negative-complete", "--seat", "luna-verifier",
        "--agent-id", "verifier-1", "--actual-model", "gpt-5.6-terra", "--verdict", "FAIL",
        "--criterion", f"C1=FAIL={attempt / 'result.yaml'}",
    )


def test_terra_review_author_is_bound_to_frozen_owner(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    rejected = run(
        "record-terra-review", terra.parent, "--review-id", "forged", "--author-agent-id", "fake-author",
        "--agent-id", "terra-author", "--actual-model", "gpt-5.6-sol", "--verdict", "PASS",
        "--form-check", "schema", "--intent-check", "outcome", "--evidence", terra, ok=False,
    )
    assert "frozen owner" in rejected["error"]


def test_terra_fallback_requires_root_contract_packet(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    goal_ledger = goal / "goal-ledger.yaml"
    run("transition", goal_ledger, "--node", "SP-01", "--to", "BLOCKED")
    rejected = run("transition", goal_ledger, "--node", "SP-01", "--to", "ESCALATED", ok=False)
    assert "Terra Sol fallback" in rejected["error"]
    packet = run(
        "record-terra-escalation", terra.parent, "--root-contract-field", "shared_file_owner",
        "--observed", "shared file has no owner", "--decision-required", "assign one owner",
        "--evidence", terra, "--requested-by", "terra-author",
    )
    run("transition", goal_ledger, "--node", "SP-01", "--to", "ESCALATED", "--escalation", packet["escalation"])
    assert run("validate", goal)["error_count"] == 0


def test_recursive_child_terra_is_structurally_valid_and_scope_bounded(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    child = run(
        "add-child-terra", terra, "--subproject-id", "SP-01-C01", "--owner", "terra-child",
        "--objective", "independent child criterion", "--wall-minutes", 10, "--effort-minutes", 10,
        "--verification-reserve", 2, "--concurrency", 1,
        "--write-scope", "outputs/goal-graph-test/child",
    )
    assert Path(child["ledger"]).is_file()
    assert run("validate", goal)["error_count"] == 0


def test_finish_requires_every_global_gate_to_pass(tmp_path: Path) -> None:
    state = tmp_path / "state"
    run(
        "init", "--state-root", state, "--goal-id", "G-20260802-gate", "--objective", "gate",
        "--effort-minutes", 100, "--verification-reserve", 20, "--write-scope", "outputs/goal-graph-test",
    )
    goal = state / "G-20260802-gate"
    run("add-gate", goal, "--gate-id", "GATE-01", "--criterion", "final audit")
    rejected = run("finish", goal, ok=False)
    assert "global gate" in rejected["error"]
    passed = run("pass-gate", goal, "--gate-id", "GATE-01", "--evidence", goal / "goal-ledger.yaml")
    assert passed["status"] == "PASS"
    assert run("finish", goal)["status"] == "PASS"


def test_usage_recording_preserves_the_verification_reserve(tmp_path: Path) -> None:
    goal, _ = scaffold(tmp_path, goal_effort=100)
    rejected = run(
        "record-usage", goal / "goal-ledger.yaml", "--agent-id", "worker-1", "--seat", "luna-worker",
        "--category", "execution", "--wall-minutes", 1, "--effort-minutes", 81,
        "--evidence", goal / "goal-ledger.yaml", ok=False,
    )
    assert "verification reserve" in rejected["error"]


def test_usage_category_cannot_impersonate_a_verifier(tmp_path: Path) -> None:
    goal, _ = scaffold(tmp_path, goal_effort=100)
    rejected = run(
        "record-usage", goal / "goal-ledger.yaml", "--agent-id", "worker-1", "--seat", "luna-worker",
        "--category", "verification", "--wall-minutes", 1, "--effort-minutes", 95,
        "--evidence", goal / "goal-ledger.yaml", ok=False,
    )
    assert "cannot record verification" in rejected["error"]


def test_validate_rejects_phantom_global_gate_evidence(tmp_path: Path) -> None:
    state = tmp_path / "state"
    run(
        "init", "--state-root", state, "--goal-id", "G-20260802-phantom", "--objective", "gate",
        "--effort-minutes", 100, "--verification-reserve", 20, "--write-scope", "outputs/goal-graph-test",
    )
    goal = state / "G-20260802-phantom"
    run("add-gate", goal, "--gate-id", "GATE-01", "--criterion", "final audit")
    goal_path = goal / "goal-ledger.yaml"
    data = yaml.safe_load(goal_path.read_text())
    data["global_gates"][0]["status"] = "PASS"
    data["global_gates"][0]["evidence"] = ["does/not/exist.txt"]
    goal_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    result = run("validate", goal, ok=False)
    assert any("evidence path does not exist" in error for error in result["errors"])
    assert "validation failed" in run("finish", goal, ok=False)["error"]
