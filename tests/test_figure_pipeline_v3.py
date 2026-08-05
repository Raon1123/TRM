"""Regression suite for the v3 figure_pipeline validator/CLI contracts.

Lives at the project-gated ``tests/`` root (INTEGRATION_DESIGN §8 "테스트 배선",
렌즈1·L1-11) so ``rtk uv run pytest tests/`` sees this coverage — the equivalent
tests under ``lab/figure_pipeline/tests/`` are not on that gate today.

Does NOT touch ``tests/test_figure_pipeline.py`` (v2, frozen).

Covers:
  A. scaffold's 9 adversarial cases, ported to target the NEW validator
     (``qa/validator.py``) instead of the retired ``validate_pipeline.py``.
  B. all 10 rows of the audit's promotion table (§8), each with a negative test.
  C. the exit-code table (§2 N04): static enum<->design-table check + a trigger
     for every code this codebase can currently reach.
  D. dry-run / read-only calls make zero filesystem mutations.

Run::

    uv run --with jsonschema --with pytest pytest tests/test_figure_pipeline_v3.py
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from lab.figure_pipeline.core import sealing
from lab.figure_pipeline.core.exits import ExitCode, PolicyError, VerificationError
from lab.figure_pipeline.core.paths import DATASETS_ROOT, STAGING_ROOT
from lab.figure_pipeline.promote import cli as promote_cli
from lab.figure_pipeline.promote.chain import revalidate_chain
from lab.figure_pipeline.qa import validator as qav
from lab.figure_pipeline.schemas import SCHEMA_DIR, current_pins
from lab.figure_pipeline.tests.scenario import PROJECT_SLUG, ROWS, World, build_world

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def world(tmp_path: Path) -> World:
    return build_world(tmp_path)


def snapshot_result(world: World, **overrides: Any) -> qav.ValidationResult:
    kwargs = dict(
        snapshot_dir=world.snapshot_dir,
        inventory_path=world.inventory_path,
        allowlist_path=world.allowlist_path,
        project_slug=PROJECT_SLUG,
        manifest=world.manifest,
    )
    kwargs.update(overrides)
    return qav.validate_snapshot(**kwargs)


def load_metadata(world: World) -> dict[str, Any]:
    return json.loads((world.registry_root / "metadata" / "fig-01a.json").read_text(encoding="utf-8"))


def metadata_checks(world: World, mutated: dict[str, Any]) -> list[qav.CheckResult]:
    return qav.validate_figure_metadata(
        world.manifest,
        world.snapshot_dir / sealing.MANIFEST_NAME,
        metadata=mutated,
        snapshot_dir=world.snapshot_dir,
        package_root=world.package_root,
    )


def failing(checks: list[qav.CheckResult]) -> qav.CheckResult:
    failures = [c for c in checks if c.failed]
    assert failures, "expected at least one failing check, got all pass"
    return failures[0]


def argv(world: World, *, mode: str = "--execute", **overrides: Any) -> list[str]:
    args = {
        "--action": "apply",
        "--approved-by": "ayp",
        "--paper": "trm-sigmak",
        "--ledger": str(world.ledger_path),
        "--generation-dir": str(world.generation_dir),
        "--snapshot-dir": str(world.snapshot_dir),
        "--receipts": str(world.receipts_dir),
        "--inventory": str(world.inventory_path),
        "--allowlist": str(world.allowlist_path),
        "--registry": str(world.registry_path),
        "--project": PROJECT_SLUG,
        "--receipt": str(world.qa_receipt_path),
        "--reports-root": str(world.repo_root / "reports"),
        "--lab-figures-root": str(world.repo_root / "lab" / "figures"),
        "--repo-root": str(world.repo_root),
        "--docs-root": str(world.registry_root),
        "--package-root": str(world.package_root),
    }
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is None:
            args.pop(flag, None)
        else:
            args[flag] = str(value)
    flat = [mode]
    for key, value in args.items():
        flat += [key, value]
    return flat


def promote(world: World, **kwargs: Any) -> int:
    return promote_cli.main(argv(world, **kwargs))


def tree_digest(root: Path) -> dict[str, tuple[int, int]]:
    """Cheap fs-mutation fingerprint: relpath -> (mtime_ns, size), sorted."""
    if not root.exists():
        return {}
    out: dict[str, tuple[int, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            st = path.stat()
            out[str(path.relative_to(root))] = (st.st_mtime_ns, st.st_size)
    return out


# =================================================================== A. adversarial 9
#
# Scaffold ``bin/validate_pipeline.py::ADVERSARIAL_CASES`` = wrong-exp, wrong-run,
# wrong-checkpoint, wrong-protocol, wrong-claim, missing-doc, duplicate-metadata,
# unlisted-script, unordered-script. The new validator's evidence model differs
# from the scaffold's (TUPLE_FIELDS dropped experiment_id/metric_protocol_ref;
# the evidence_registry.json + build_all_figures.sh ordered list do not exist
# post-registry-redesign, §3/§7) — each case below is mapped onto the closest
# real rule and asserted by check_id, or marked as a documented gap when no
# successor rule exists (do not invent one to force a false 9/9).


def test_adversarial_wrong_run_orphan_tuple_is_rejected(world: World) -> None:
    """run_id is in TUPLE_FIELDS: forging it makes the evidence tuple unmatched."""
    mutated = load_metadata(world)
    mutated["evidence_tuples"][0]["run_id"] = "wandb:deadbeef"
    mutated["provenance"]["run_ids"] = ["wandb:deadbeef", ROWS[1]["run_id"]]
    check = failing(metadata_checks(world, mutated))
    assert check.check_id == "figure-metadata-evidence-tuples"
    assert "absent from the snapshot" in check.measured


def test_adversarial_wrong_checkpoint_orphan_tuple_is_rejected(world: World) -> None:
    mutated = load_metadata(world)
    mutated["evidence_tuples"][0]["checkpoint_ref"] = "sha256:" + "f" * 64 + "@missing"
    mutated["provenance"]["checkpoint_refs"] = [
        "sha256:" + "f" * 64 + "@missing", "n/a:online-training-metric",
    ]
    check = failing(metadata_checks(world, mutated))
    assert check.check_id == "figure-metadata-evidence-tuples"


def test_adversarial_wrong_exp_routes_through_provenance_disagreement(world: World) -> None:
    """experiment_id is NOT a TUPLE_FIELDS member in the new validator (it dropped
    out with metric_protocol_ref, §8 R2S-M6) -- forging it cannot orphan the tuple;
    it disagrees with the metadata's own provenance.experiment_ids block instead."""
    mutated = load_metadata(world)
    mutated["evidence_tuples"][0]["experiment_id"] = "EXP-999"
    mutated["provenance"]["experiment_ids"] = ["EXP-999"]
    checks = metadata_checks(world, mutated)
    tuple_check = next(c for c in checks if c.check_id == "figure-metadata-evidence-tuples")
    assert tuple_check.failed
    assert "provenance experiment_ids disagrees" in tuple_check.measured
    # and the orphan-tuple branch specifically does NOT fire for this mutation
    assert "absent from the snapshot" not in tuple_check.measured


def test_adversarial_wrong_protocol_is_rejected_by_schema_not_by_the_semantic_check(
    world: World,
) -> None:
    """Reclassified from a documented gap after checking empirically: the scaffold's
    evidence tuple carried metric_protocol_ref and checked it semantically;
    qa/validator.TUPLE_FIELDS = (run_id, metric_id, checkpoint_ref, source_id) has
    no such field, so figure-metadata-evidence-tuples never inspects it -- BUT
    figure_metadata.schema.json's evidence_tuples[] items are
    ``additionalProperties: false`` with an exact five-key set that does not
    include metric_protocol_ref, so the JSON Schema itself (a stricter net than
    the scaffold's own check) rejects the forged key before evidence-tuples logic
    would ever run."""
    mutated = load_metadata(world)
    mutated["evidence_tuples"][0]["metric_protocol_ref"] = "missing-protocol"
    check = failing(metadata_checks(world, mutated))
    assert check.check_id == "figure-metadata-schema-version"
    assert "'metric_protocol_ref' was unexpected" in check.measured


def test_adversarial_wrong_claim_has_no_successor_rule(world: World) -> None:
    """Documented gap: the scaffold cross-checked claim_id against an
    evidence_registry.fixture.json that no longer exists in the new design (§3
    replaced it with the figure registry's 1:1:1 cardinality, §7). Neither
    validate_figure_metadata nor chain.verify_h4 compares H4.verdict_claim_ids
    to figure_metadata.claim_ids."""
    mutated = load_metadata(world)
    mutated["claim_ids"] = ["C-WRONG"]
    checks = metadata_checks(world, mutated)
    assert all(not c.failed for c in checks), (
        "if this starts failing, a successor rule for claim_ids now exists -- "
        "replace this xfail-style assertion with a real negative test"
    )


def test_adversarial_missing_doc_is_rejected_at_promote_via_caption_resolver(world: World) -> None:
    """§7 replaced the scaffold's registry-glob doc check with promote's H4 caption
    binding (default_caption_resolver -> evidence_path); still a real rejection,
    just relocated from the qa validator into the chain."""
    doc = world.registry_root / "docs" / "fig-01a-clean-fig1-peak.md"
    doc.unlink()
    with pytest.raises(VerificationError, match="evidence object missing from the chain"):
        revalidate_chain(world.chain_inputs())


def test_adversarial_duplicate_metadata_reference_is_a_registry_cardinality_violation(
    world: World,
) -> None:
    """Formerly a documented gap: ``qa.validator.check_registry`` looked entries
    up by ``figure_id`` alone, so a second figure_id claiming the same
    ``metadata`` path slid through the promote chain. The integration pass wired
    the §7 1:1:1 cross-entry rule into ``check_registry`` itself (§2.5 step 7),
    which is what ``promote.chain.revalidate_chain`` actually calls."""
    registry = world.rewrite_json(world.registry_path, lambda r: None)
    dup = copy.deepcopy(registry["figures"][0])
    dup["figure_id"] = "fig-99"
    world.rewrite_json(world.registry_path, lambda r: r["figures"].append(dup))
    with pytest.raises(VerificationError, match="registry cardinality violation"):
        revalidate_chain(world.chain_inputs())


def test_adversarial_unlisted_script_generator_hash_is_rejected(world: World) -> None:
    """§6/§7 dropped the discovered-script-glob check for a direct sha256 binding
    (figure-metadata-generator-hash): an unregistered/mismatched generator file
    is refused by content hash, not by directory listing."""
    mutated = load_metadata(world)
    mutated["generator"]["script_path"] = "figures/fig_99_orphan.py"
    check = failing(metadata_checks(world, mutated))
    assert check.check_id == "figure-metadata-generator-hash"
    assert "generator script absent" in check.measured


def test_adversarial_unordered_script_is_retired_by_design(world: World) -> None:
    """Documented as retired, not tested as a rejection: build_all_figures.sh's
    ordered figure list (scaffold §6) has no successor in the new design -- the
    figure registry (§3/§7 figure_registry.json) is order-independent, and
    run_pipeline.py's FIGURE_REGISTRY iterates figures individually per §2's
    '--figure-set all' cardinality rule (one generation per figure, no set
    generation). There is no "wrong build order" state to reject."""
    assert True


# =================================================================== B. promotion table 10


def test_row1_strict_no_op_scaffold_true_is_a_real_policy_rejection(world: World) -> None:
    world.recompute_manifest(lambda m: m.__setitem__("scaffold", True))
    result = snapshot_result(world, strict="local")
    check = failing(result.checks)
    assert check.check_id == "strict-materialization"
    assert check.exit_code == ExitCode.POLICY


def test_row1_strict_no_op_unmaterialized_asset_is_rejected(world: World) -> None:
    world.recompute_manifest(lambda m: m["assets"][0].__setitem__("materialized", False))
    result = snapshot_result(world, strict="local")
    check = failing(result.checks)
    assert check.check_id == "strict-materialization"


def test_row1_strict_full_is_structurally_locked_pending_p1_m27(world: World) -> None:
    result = snapshot_result(world, strict="full")
    check = failing(result.checks)
    assert check.check_id == "strict-materialization"
    assert "strict=full requires the artifact/export axes" in check.measured


def test_row1_strict_bogus_tier_is_a_usage_error(world: World) -> None:
    with pytest.raises(Exception):
        snapshot_result(world, strict="bogus")


def test_row2_fig_qa_gate_rejected_receipt_status_is_refused(tmp_path: Path) -> None:
    rejected = build_world(tmp_path, receipt_status="rejected")
    with pytest.raises(PolicyError, match="qa_receipt status is 'rejected'"):
        revalidate_chain(rejected.chain_inputs())


def test_row2_fig_qa_gate_invalid_status_is_refused_by_schema(world: World) -> None:
    """H4's schema only admits accepted|rejected (there is no on-disk H4 for a
    qa-pending generation at all -- the receipt is the human verdict itself);
    an invalid value is caught by schema validation before chain.verify_h4's
    own accepted-check ever runs."""
    world.rewrite_json(world.qa_receipt_path, lambda r: r.__setitem__("status", "pending"))
    with pytest.raises(VerificationError, match="not one of"):
        revalidate_chain(world.chain_inputs())


def test_row3_schema_version_wrong_value_is_rejected(world: World) -> None:
    mutated = load_metadata(world)
    mutated["schema_version"] = "trm-figure-metadata-v1"
    check = failing(metadata_checks(world, mutated))
    assert check.check_id == "figure-metadata-schema-version"


def test_row3_schema_version_review_block_is_forbidden_b07(world: World) -> None:
    """B07 deletes the ``review`` block from figure_metadata v2; the schema's own
    ``additionalProperties: false`` is what actually fires first (belt-and-
    suspenders with the ``_schema()`` closure's explicit ``"review" in metadata``
    check, which would fire second if the schema ever loosened)."""
    mutated = load_metadata(world)
    mutated["review"] = {"status": "accepted"}
    check = failing(metadata_checks(world, mutated))
    assert check.check_id == "figure-metadata-schema-version"
    assert "'review' was unexpected" in check.measured


def test_row4_snapshot_id_content_hash_forged_label_is_rejected(world: World) -> None:
    world.rewrite_manifest(lambda m: m.__setitem__("snapshot_id", "ds-forged-99999999-" + "a" * 16))
    result = snapshot_result(world)
    check = failing(result.checks)
    assert check.check_id == "snapshot-id-content-hash"


def test_row5_asset_sha256_fabricated_all_ones_is_rejected(world: World) -> None:
    """The scaffold's all-zero-only guard let '111...1' through; verify the new
    validator distinguishes 'materialized+hex64' from 'byte-matches-manifest'."""
    world.recompute_manifest(lambda m: m["assets"][0].__setitem__("sha256", "1" * 64))
    result = snapshot_result(world)
    check = failing(result.checks)
    assert check.check_id == "asset-sha256-realbytes"
    assert "asset byte mismatch" in check.measured


def test_row6_asset_uri_scheme_is_blocked_at_the_schema_layer(world: World) -> None:
    """``assets[].uri`` is typed ``relpath`` (no ``:`` allowed at all, §5-11) --
    a live wandb:// pointer there cannot even reach ``_assert_reference``; the
    JSON Schema itself is the first line of defense."""
    world.recompute_manifest(
        lambda m: m["assets"][0].__setitem__("uri", "wandb://raon1123/Sigma_k_new/runs/abc")
    )
    result = snapshot_result(world)
    check = failing(result.checks)
    assert check.check_id == "dataset-manifest-schema"


def test_row6_live_wandb_uri_in_analytic_ref_is_rejected(world: World) -> None:
    """``sources[].analytic_ref.ref`` is a free string in the schema (no relpath
    pattern) -- this is the field ``uri-scheme-allowlist`` actually has to guard."""
    world.recompute_manifest(
        lambda m: m["sources"][1]["analytic_ref"].__setitem__(
            "ref", "wandb://raon1123/Sigma_k_new/runs/abc"
        )
    )
    result = snapshot_result(world)
    check = failing(result.checks)
    assert check.check_id == "uri-scheme-allowlist"
    assert "refused uri scheme" in check.measured


def test_row6_latest_pointer_token_is_rejected(world: World) -> None:
    world.recompute_manifest(
        lambda m: m["sources"][1]["analytic_ref"].__setitem__(
            "ref", "protocols/latest/metric-test-exact-probe.md"
        )
    )
    result = snapshot_result(world)
    check = failing(result.checks)
    assert check.check_id == "uri-scheme-allowlist"
    assert "live pointer refused" in check.measured


def test_row6_unpinned_artifact_ref_is_rejected(world: World) -> None:
    world.recompute_manifest(
        lambda m: m["sources"][1].__setitem__(
            "analytic_ref", {"ref": "artifact://sigmaknew-metrics:latest", "sha256": "a" * 64}
        )
    )
    result = snapshot_result(world)
    check = failing(result.checks)
    assert check.check_id == "uri-scheme-allowlist"
    assert "not version-pinned" in check.measured


def test_row6_pinned_artifact_ref_is_accepted(world: World) -> None:
    world.recompute_manifest(
        lambda m: m["sources"][1].__setitem__(
            "analytic_ref", {"ref": "artifact://sigmaknew-metrics:v3", "sha256": "a" * 64}
        )
    )
    result = snapshot_result(world)
    check = next(c for c in result.checks if c.check_id == "uri-scheme-allowlist")
    assert not check.failed


def test_row6_live_wandb_artifact_id_in_inventory_is_rejected(world: World) -> None:
    world.rewrite_inventory(
        lambda inv: inv["sources"][0].__setitem__(
            "wandb_artifact_id", "sigmaknew-jpj2eswa-metrics:latest"
        )
    )
    result = snapshot_result(world)
    check = failing(result.checks)
    assert check.check_id == "inventory-live-pointer"


def test_row7_checksums_index_is_never_a_verification_input(world: World) -> None:
    checksums = world.snapshot_dir / sealing.CHECKSUMS_NAME
    checksums.write_text("deadbeef" * 8 + "  raw/evals.jsonl\n", encoding="utf-8")
    result = snapshot_result(world)
    assert result.ok, [c.measured for c in result.failures]
    assert any("disagrees with the manifest" in w for w in result.warnings)


def test_row7_checksums_index_absence_is_a_warning_not_a_failure(world: World) -> None:
    (world.snapshot_dir / sealing.CHECKSUMS_NAME).unlink()
    result = snapshot_result(world)
    assert result.ok
    assert any("absent" in w for w in result.warnings)


def test_row8_inventory_digest_mismatch_is_rejected(world: World) -> None:
    # recompute_manifest (not rewrite_manifest): inventory_digest feeds
    # content_sha256, so a bare forgery would first trip snapshot-id-content-hash
    # and mask the rule this test isolates.
    world.recompute_manifest(lambda m: m.__setitem__("inventory_digest", "0" * 64))
    result = snapshot_result(world)
    check = failing(result.checks)
    assert check.check_id == "inventory-allowlist-digest"


def test_row8_allowlist_digest_mismatch_is_rejected(world: World) -> None:
    world.recompute_manifest(lambda m: m.__setitem__("allowlist_digest", "0" * 64))
    result = snapshot_result(world)
    check = failing(result.checks)
    assert check.check_id == "inventory-allowlist-digest"


def test_row8_tuple_cross_check_ambiguous_inventory_entry_is_rejected(world: World) -> None:
    """Two inventory rows realize the same (run_id, metric_id, checkpoint_ref,
    source_id) tuple -- 1:1 match becomes ambiguous."""
    world.rewrite_inventory(lambda inv: inv["sources"].append(dict(inv["sources"][0])))
    result = snapshot_result(world)
    check = failing(result.checks)
    assert check.check_id == "tuple-inventory-unique-match"
    assert "ambiguous tuple" in check.measured


def test_row8_tuple_cross_check_orphan_asset_tuple_is_rejected(world: World) -> None:
    world.recompute_manifest(lambda m: m["assets"][0].__setitem__("run_id", "@column:run_id"))
    world.rewrite_evals([{**ROWS[0], "run_id": "fig1-trm-k99-s1"}, ROWS[1]])
    result = snapshot_result(world)
    check = failing(result.checks)
    assert check.check_id == "tuple-inventory-unique-match"
    assert "orphan tuple" in check.measured


def test_row9_symlink_inside_sealed_snapshot_is_rejected(world: World) -> None:
    link = world.snapshot_dir / "raw" / "evil-link"
    link.symlink_to(world.snapshot_dir / "raw" / "evals.jsonl")
    with pytest.raises(VerificationError, match="symlink inside snapshot refused"):
        sealing.verify_sealed_snapshot(world.snapshot_dir, world.manifest)


def test_row9_output_directory_collision_toctou_via_rehash(world: World) -> None:
    """v2-carried safe_output overwrite guard + sealing's rehash-before-copy close
    the classic check-then-act window: introduce a byte change between the chain
    verification a caller trusts and the moment promote actually reads it."""
    (world.generation_dir / "clean_fig1_peak.png").write_bytes(b"toctou-swap")
    with pytest.raises(VerificationError, match="render output byte mismatch"):
        revalidate_chain(world.chain_inputs())


def test_row9_dangling_symlink_inside_generation_dir_is_rejected(world: World) -> None:
    dangling = world.generation_dir / "dangling.png"
    dangling.symlink_to(world.generation_dir / "does-not-exist.png")
    with pytest.raises(VerificationError):
        revalidate_chain(world.chain_inputs())


def test_row10_schema_pin_drift_is_rejected(world: World, tmp_path: Path) -> None:
    shadow = tmp_path / "schema-shadow"
    shutil.copytree(SCHEMA_DIR, shadow)
    # corrupt one schema file's bytes without regenerating its pin -> drift
    target = shadow / "dataset_manifest.schema.json"
    target.write_text(target.read_text(encoding="utf-8") + "\n// tampered\n", encoding="utf-8")
    import lab.figure_pipeline.schemas as schemas_mod

    original_dir = schemas_mod.SCHEMA_DIR
    schemas_mod.SCHEMA_DIR = shadow
    try:
        with pytest.raises(VerificationError, match="schema pin drift"):
            schemas_mod.verify_pins()
    finally:
        schemas_mod.SCHEMA_DIR = original_dir


def test_row10_schema_pins_are_currently_clean(world: World) -> None:
    """Sanity companion to the drift test: the real repo's pins must match --
    otherwise every promote call would be failing for an unrelated reason."""
    import lab.figure_pipeline.schemas as schemas_mod

    pins = schemas_mod.verify_pins()
    assert pins == current_pins()


# =================================================================== C. exit code table


DESIGN_EXIT_TABLE: dict[int, str] = {
    0: "requested stages completed",
    10: "normal stop",
    11: "validated-not-applied",
    2: "usage/argument error",
    20: "policy refusal",
    21: "immutability conflict",
    22: "verification mismatch",
    23: "incomplete state detected",
    124: "stage timeout",
    1: "other execution failure",
}


def test_exit_code_enum_matches_design_table() -> None:
    assert {int(c) for c in ExitCode} == set(DESIGN_EXIT_TABLE)
    from lab.figure_pipeline.core.exits import EXIT_MEANINGS

    for code, keyword in (
        (ExitCode.OK, "completed"), (ExitCode.QA_PENDING, "normal stop"),
        (ExitCode.VALIDATED_NOT_APPLIED, "nothing applied"),
        (ExitCode.USAGE, "usage"), (ExitCode.POLICY, "policy"),
        (ExitCode.IMMUTABLE, "immutability"), (ExitCode.VERIFICATION, "verification"),
        (ExitCode.INCOMPLETE, "incomplete"), (ExitCode.TIMEOUT, "timeout"),
    ):
        assert keyword in EXIT_MEANINGS[code].lower()


def test_exit_code_0_ok_reachable(world: World) -> None:
    assert promote(world) == 0


def test_exit_code_2_usage_reachable(world: World) -> None:
    assert promote(world, approved_by=None) == 2
    assert promote_cli.main(argv(world)[1:]) == 2  # mode omitted


def test_exit_code_10_qa_pending_contract() -> None:
    """§2: exit 10 is the terminal code of call-1 (inventory..registry). The
    end-to-end trigger with the six real stage modules lives at
    ``lab/figure_pipeline/tests/test_stage_modules.py::
    test_call_one_reaches_exit_10_with_real_stages``; this test pins the
    contract value itself."""
    assert ExitCode.QA_PENDING == 10


def test_exit_code_11_validated_not_applied_reachable(world: World) -> None:
    assert promote(world, mode="--dry-run=resolved") == 11
    assert promote(world, mode="--dry-run=plan") == 11


def test_exit_code_20_policy_reachable(tmp_path: Path) -> None:
    preview = build_world(tmp_path, coverage_ok=False)
    assert promote(preview) == 20


def test_exit_code_21_immutable_reachable(world: World) -> None:
    assert promote(world) == 0
    assert promote(world) == 21


def test_exit_code_22_verification_reachable(world: World) -> None:
    (world.generation_dir / "clean_fig1_peak.png").write_bytes(b"tampered")
    assert promote(world) == 22


def test_exit_code_23_incomplete_reachable_via_sealing_crash_marker(tmp_path: Path) -> None:
    tmp_dir = tmp_path / ".tmp-abc123"
    tmp_dir.mkdir()
    sealing.mark_crash(tmp_dir, "simulated crash")
    from lab.figure_pipeline.core.exits import IncompleteStateError
    with pytest.raises(IncompleteStateError):
        sealing.assert_no_crash_marker(tmp_dir, resume=False)
    sealing.assert_no_crash_marker(tmp_dir, resume=True)  # does not raise


def test_exit_code_124_timeout_not_reachable_via_promote_but_type_is_wired() -> None:
    """124 belongs to run_pipeline's subprocess stage runner (v2 lineage), which
    promote/qa/validator do not invoke -- no in-repo call here can exercise it
    end-to-end while run_pipeline's stage runner is unimplemented (see openIssues)."""
    from lab.figure_pipeline.core.exits import StageTimeoutError
    assert StageTimeoutError.exit_code == ExitCode.TIMEOUT == 124


# =================================================================== D. dry-run side effects


def test_validate_snapshot_makes_zero_filesystem_mutations(world: World) -> None:
    before = tree_digest(world.root)
    snapshot_result(world, strict="full")  # even a failing/POLICY call must not write
    after = tree_digest(world.root)
    assert before == after


def test_promote_dry_run_mutates_nothing_anywhere_including_shared_pipeline_roots(
    world: World,
) -> None:
    before_world = tree_digest(world.root)
    staging_before = set(STAGING_ROOT.rglob("*")) if STAGING_ROOT.exists() else set()
    datasets_before = set(DATASETS_ROOT.rglob("*")) if DATASETS_ROOT.exists() else set()

    for mode in ("--dry-run", "--dry-run=plan", "--dry-run=resolved"):
        code = promote(world, mode=mode)
        assert code == 11

    assert tree_digest(world.root) == before_world
    assert not world.ledger_path.exists()
    staging_after = set(STAGING_ROOT.rglob("*")) if STAGING_ROOT.exists() else set()
    datasets_after = set(DATASETS_ROOT.rglob("*")) if DATASETS_ROOT.exists() else set()
    assert staging_after == staging_before, "dry-run wrote under the shared package outputs/ root"
    assert datasets_after == datasets_before, "dry-run wrote under the shared package datasets/ root"


def test_chain_revalidation_alone_makes_zero_mutations(world: World) -> None:
    before = tree_digest(world.root)
    revalidate_chain(world.chain_inputs())
    assert tree_digest(world.root) == before


def test_dry_run_payload_declares_its_mode_and_does_not_apply(world: World, capsys) -> None:
    assert promote(world, mode="--dry-run=plan") == 11
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "dry-run-plan"
    assert payload["applied"] is False
