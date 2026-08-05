"""Fast, network-free CLI contracts for the staged figure pipeline."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
PIPE = REPO / "lab/monitoring/dashboard/run_figure_pipeline.py"
LOCAL = REPO / "lab/monitoring/dashboard/extract_evals.py"
DEFAULT_RUN_ROOT = REPO / "lab/figures/pipeline-runs"


def call(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *map(str, args)], cwd=REPO, text=True, capture_output=True)


def test_pipeline_dry_run_is_a_plan_and_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    result = call(PIPE, "--source", "web", "--output-dir", out, "--dry-run")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["stages"] == ["extract", "peak", "final", "dashboard", "smoke"]
    assert "extract_evals_web.py" in " ".join(plan["commands"]["extract"])
    assert not out.exists()


def test_default_output_is_timestamped_under_lab_and_dry_run_writes_nothing() -> None:
    result = call(PIPE, "--source", "web", "--dry-run")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    out = Path(plan["output_dir"])
    assert out.parent == DEFAULT_RUN_ROOT
    assert re.fullmatch(r"\d{8}T\d{12}Z", out.name)
    assert not out.exists()


def test_explicit_pipeline_run_child_under_lab_is_allowed_but_root_is_not() -> None:
    out = DEFAULT_RUN_ROOT / "explicit-dry-run"
    result = call(PIPE, "--source", "local", "--output-dir", out, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert not out.exists()
    refused = call(PIPE, "--source", "local", "--output-dir", DEFAULT_RUN_ROOT, "--dry-run")
    assert refused.returncode == 2


def test_reachability_stability_family_uses_same_manifested_pipeline() -> None:
    result = call(PIPE, "--figures", "reachability-stability", "--dry-run")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["figure_set"] == "reachability-stability"
    assert plan["source"] is None
    assert plan["stages"] == [
        "rs-tables", "rs-r1", "rs-r2", "rs-r3", "rs-r4", "rs-interactive",
    ]
    commands = plan["commands"]
    assert "rs_build_tables.py" in " ".join(commands["rs-tables"])
    assert "rs_make_R1.py" in " ".join(commands["rs-r1"])
    assert "rs_make_R4.py" in " ".join(commands["rs-r4"])
    assert "rs_make_interactive.py" in " ".join(commands["rs-interactive"])
    assert not Path(plan["output_dir"]).exists()


def test_all_figure_families_share_one_run_and_rs_dependencies_are_enforced() -> None:
    result = call(PIPE, "--figures", "all", "--source", "web", "--dry-run")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["stages"][:3] == ["extract", "peak", "final"]
    assert plan["stages"][3:9] == [
        "rs-tables", "rs-r1", "rs-r2", "rs-r3", "rs-r4", "rs-interactive",
    ]
    assert plan["stages"][-2:] == ["dashboard", "smoke"]
    invalid = call(
        PIPE, "--figures", "reachability-stability", "--stages", "rs-r1", "--dry-run",
    )
    assert invalid.returncode == 2
    assert "require rs-tables" in invalid.stderr


def test_pipeline_rejects_invalid_dag_and_repo_root(tmp_path: Path) -> None:
    bad_dag = call(PIPE, "--source", "local", "--output-dir", tmp_path / "x", "--stages", "peak")
    assert bad_dag.returncode == 2
    unsafe = call(PIPE, "--source", "local", "--output-dir", REPO, "--dry-run")
    assert unsafe.returncode == 2


@pytest.mark.parametrize(
    ("stages", "policy"),
    [
        ("extract", "both"),
        ("extract,peak", "peak"),
        ("extract,final", "final"),
        ("extract,peak,final,dashboard,smoke", "both"),
    ],
)
def test_valid_canonical_stage_subsets(tmp_path: Path, stages: str, policy: str) -> None:
    result = call(PIPE, "--source", "local", "--output-dir", tmp_path / stages.replace(",", "-"),
                  "--stages", stages, "--selection-policy", policy, "--dry-run")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("stages", ["peak,extract", "smoke,dashboard", "extract,final,peak"])
def test_reversed_stage_order_is_rejected(tmp_path: Path, stages: str) -> None:
    result = call(PIPE, "--source", "local", "--output-dir", tmp_path / "bad",
                  "--stages", stages, "--dry-run")
    assert result.returncode == 2
    assert "canonical order" in result.stderr


@pytest.mark.parametrize("descendant", ["reports/x", "lab/monitoring/dashboard/x", ".git/x",
                                        "scripts/queue/x", "data/x", "checkpoints/x", "wandb/x"])
def test_pipeline_rejects_every_repository_descendant(descendant: str) -> None:
    result = call(PIPE, "--source", "local", "--output-dir", REPO / descendant, "--dry-run")
    assert result.returncode == 2
    assert "protected root" in result.stderr


def test_pipeline_rejects_symlink_escape(tmp_path: Path) -> None:
    link = tmp_path / "repo-link"
    link.symlink_to(REPO, target_is_directory=True)
    result = call(PIPE, "--source", "local", "--output-dir", link / "bundle", "--dry-run")
    assert result.returncode == 2


def test_pipeline_rejects_symlink_inside_existing_staging_layout(tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    out.mkdir()
    (out / "pipeline_manifest.json").write_text("{}")
    (out / "figures").symlink_to(REPO / "reports", target_is_directory=True)
    result = call(PIPE, "--source", "local", "--output-dir", out, "--dry-run")
    assert result.returncode == 2
    assert "escapes output" in result.stderr or "symlink inside staging" in result.stderr


@pytest.mark.parametrize("broad", [Path("/"), Path("/tmp"), REPO.parent])
def test_pipeline_rejects_broad_or_repo_ancestor_output(broad: Path) -> None:
    result = call(PIPE, "--source", "local", "--output-dir", broad, "--dry-run")
    assert result.returncode == 2


def normalized_local_fixture(path: Path) -> None:
    path.write_text(json.dumps({
        "schema_version": 1,
        "source": "wandb-local-datastore",
        "project": "Sigma_k_new",
        "generated_utc": "2026-07-26T00:00:00Z",
        "metrics": ["probe/train_exact", "probe/test_exact"],
        "runs": {},
        "warnings": [],
    }), encoding="utf-8")


def test_resume_dry_run_validates_and_hashes_without_writes(tmp_path: Path) -> None:
    fixture = tmp_path / "source-evals.json"
    normalized_local_fixture(fixture)
    out = tmp_path / "dry-resume"
    result = call(
        PIPE, "--source", "local", "--input-evals", fixture, "--output-dir", out,
        "--stages", "peak", "--selection-policy", "peak", "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["supplied_input"]["sha256"]
    assert not out.exists()
    mismatch = call(
        PIPE, "--source", "web", "--input-evals", fixture, "--output-dir", out,
        "--stages", "peak", "--selection-policy", "peak", "--dry-run",
    )
    assert mismatch.returncode == 2


def test_staged_resume_integration_and_require_node(tmp_path: Path) -> None:
    fixture = tmp_path / "source-evals.json"
    normalized_local_fixture(fixture)
    out = tmp_path / "bundle"
    result = call(
        PIPE, "--source", "local", "--input-evals", fixture, "--output-dir", out,
        "--stages", "peak,final,dashboard,smoke", "--selection-policy", "both",
        "--skip-preflight", "--stage-timeout-seconds", "60",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((out / "pipeline_manifest.json").read_text())
    assert manifest["status"] == "ok"
    assert [stage["name"] for stage in manifest["stages_detail"]] == [
        "peak", "final", "dashboard", "smoke",
    ]
    assert manifest["supplied_input"]["sha256"]
    assert (out / "figures/clean_fig1_peak_preview.png").is_file()
    assert (out / "figures/clean_fig1_final_preview.png").is_file()
    assert (out / "dashboard/trm-sigmak-live.html").is_file()
    assert (out / "dashboard/trm-rs-pipeline.html").is_file()

    missing_node = subprocess.run(
        [sys.executable, REPO / "lab/monitoring/dashboard/smoke_render.py",
         "--html-dir", out / "dashboard", "--require-node"],
        cwd=REPO, text=True, capture_output=True, env={"PATH": "/definitely-missing"},
    )
    assert missing_node.returncode != 0
    assert "node not available" in missing_node.stdout + missing_node.stderr


def test_timeout_is_manifested_and_fail_fast_stops_later_stage(tmp_path: Path) -> None:
    out = tmp_path / "timeout"
    result = call(
        PIPE, "--source", "local", "--output-dir", out,
        "--stages", "extract,peak", "--selection-policy", "peak",
        "--stage-timeout-seconds", "0.001", "--skip-preflight",
    )
    assert result.returncode != 0
    manifest = json.loads((out / "pipeline_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert [stage["name"] for stage in manifest["stages_detail"]] == ["extract"]
    assert manifest["stages_detail"][0]["exit_code"] == 124


def test_manifest_redaction_helpers_cover_inline_and_split_secrets() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("figure_pipeline", PIPE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    cleaned = module.redact(["cmd", "--token", "super-secret", "--password=hunter2",
                             "api_key=abc", "WANDB_API_KEY=wandb-secret", "ordinary"])
    joined = " ".join(cleaned)
    assert all(secret not in joined for secret in ("super-secret", "hunter2", "abc", "wandb-secret"))
    assert joined.endswith("ordinary")
    detail = module.run_stage(
        "redaction",
        [sys.executable, "-c",
         "import sys; print('token=abc'); print('password=hunter2 Authorization: Bearer xyz', file=sys.stderr)"],
        [], [], 5,
    )
    serialized = json.dumps(detail)
    assert all(secret not in serialized for secret in ("abc", "hunter2", "xyz"))


def test_local_extractor_dry_run_has_no_output_mutation(tmp_path: Path) -> None:
    output = tmp_path / "evals.json"
    result = call(LOCAL, "--wandb-root", REPO / "wandb", "--output", output, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["source"] == "wandb-local-datastore"
    assert not output.exists()
