"""Chain/ledger/adapter contract tests for the v3 figure_pipeline (project-gated).

Companion to ``test_figure_pipeline_v3.py`` (validator/CLI-level). This file
covers the parts of INTEGRATION_DESIGN §8's "테스트 배선" that are about
cross-object wiring rather than single-rule checks:

  1. H1->H2->H3->QR->H4->H5 chain: happy path + one negative test per link,
     table-driven, asserting the *first* chain step that fails.
  2. selected_figures ledger: hash-chain integrity, CAS rejection, concurrent
     append under flock, supersede/retract terminal-state transitions.
  3. Sealing TOCTOU: tamper a sealed snapshot after seal() succeeds, in the
     three structurally distinct ways (byte swap / added file / symlink), and
     confirm promote's independent re-verification (not sealing itself) catches
     each.
  4. Web-channel acquisition: FakeApi/FakeRun mock pattern (item 6).
  5. NAS-channel acquisition: tmp-fixture pattern, never writing the real NAS
     root (item 7).
  6. Provenance grade derivation: weakest-link aggregation + declared/derived
     mismatch rejection (item 8).

Never touches the repository's real ``reports/``, ``lab/figures/``, or the NAS
mount; NAS-channel writes go through the same ``STAGING_ROOT``-scoped,
teardown-cleaned pattern as ``lab/figure_pipeline/tests/test_nas_export.py``.

Run::

    uv run --with jsonschema --with pytest pytest tests/test_figpipe_contracts.py
"""
from __future__ import annotations

import copy
import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import pytest

from lab.figure_pipeline import run_pipeline as rp
from lab.figure_pipeline.acquisition import nas_export as ne
from lab.figure_pipeline.acquisition import web_api as web
from lab.figure_pipeline.core import canonical
from lab.figure_pipeline.core import identity as identity_mod
from lab.figure_pipeline.core import ledger as ledger_mod
from lab.figure_pipeline.core import paths, sealing, states
from lab.figure_pipeline.core.exits import (
    ExitCode,
    PipelineError,
    PolicyError,
    UsageError,
    VerificationError,
)
from lab.figure_pipeline.core.hashing import sha256_file
from lab.figure_pipeline.promote import apply as promote_apply
from lab.figure_pipeline.promote.chain import revalidate_chain
from lab.figure_pipeline.qa import receipt as receipt_mod
from lab.figure_pipeline.qa import validator as qa_validator
from lab.figure_pipeline.tests.scenario import PAPER_ID, ROWS, World, build_world
from lab.figure_pipeline.tests.test_acquisition_web_api import FakeApi, FakeRun


@pytest.fixture
def world(tmp_path: Path) -> World:
    return build_world(tmp_path)


# ============================================================ 1. H1..H5+QR chain


def test_happy_path_chain_passes_every_step_in_order(world: World) -> None:
    result = revalidate_chain(world.chain_inputs())
    assert [s["step"] for s in result.summary()["steps"]] == [
        "H1", "H2", "H3", "QR", "figure-metadata", "H4", "provenance",
        "registry", "ledger-chain",
    ]
    assert all(s["result"] == "pass" for s in result.summary()["steps"])
    assert result.provenance == "CANONICAL"


def _tamper_h1_channel(world: World) -> None:
    world.rewrite_json(
        world.receipts_dir / "acquisition_receipt_local.json",
        lambda r: r.__setitem__("channel", "nas"),
    )


def _tamper_h1_missing(world: World) -> None:
    (world.receipts_dir / "acquisition_receipt_local.json").unlink()


def _tamper_h2_asset_byte(world: World) -> None:
    """Isolates the H2 asset-sha256-realbytes rule specifically: also patch H1's
    cross-referenced raw_files sha256 (and re-bind manifest.acquisition_receipts[0]
    to the now-edited receipt file's new sha256) so H1's independent
    receipt<->manifest cross-check does not catch the forgery first -- otherwise
    it does, which is a real and correct defense-in-depth property but the wrong
    row to demonstrate it on (see the companion test right below)."""
    world.recompute_manifest(lambda m: m["assets"][0].__setitem__("sha256", "1" * 64))
    receipt_path = world.receipts_dir / "acquisition_receipt_local.json"
    world.rewrite_json(
        receipt_path,
        lambda r: next(f for f in r["raw_files"] if f["path"] == "raw/evals.jsonl").__setitem__(
            "sha256", "1" * 64
        ),
    )
    def _rebind(m: dict[str, Any]) -> None:
        new_sha = sha256_file(receipt_path)
        m["acquisition_receipts"][0]["sha256"] = new_sha
        for source in m["sources"]:
            if source.get("kind") == "acquired":
                source["acquisition_receipt_sha256"] = new_sha

    world.recompute_manifest(_rebind)


def test_h1_catches_manifest_asset_forgery_before_h2_when_receipt_is_untouched(world: World) -> None:
    """Defense-in-depth companion to _tamper_h2_asset_byte: with H1 left alone,
    the SAME manifest forgery is caught one step earlier, at H1's
    receipt<->manifest raw-file cross-check (chain.verify_h1), not at H2."""
    world.recompute_manifest(lambda m: m["assets"][0].__setitem__("sha256", "1" * 64))
    with pytest.raises(VerificationError, match="sha256 disagrees with manifest"):
        revalidate_chain(world.chain_inputs())


def _tamper_h2_provenance_label(world: World) -> None:
    world.recompute_manifest(lambda m: m.__setitem__("provenance", "LEGACY-CONTAMINATED"))


def _tamper_h3_output_byte(world: World) -> None:
    (world.generation_dir / "clean_fig1_peak.png").write_bytes(b"corrupted-after-render")


def _tamper_h3_snapshot_binding(world: World) -> None:
    world.rewrite_json(
        world.generation_dir / sealing.RENDER_MANIFEST_NAME,
        lambda r: r.__setitem__("snapshot_id", "ds-forged-20260728-" + "b" * 16),
    )


def _tamper_qr_overall(world: World) -> None:
    world.rewrite_json(
        world.generation_dir / sealing.QA_REPORT_NAME, lambda r: r.__setitem__("overall", "fail")
    )


def _tamper_qr_smoke(world: World) -> None:
    world.rewrite_json(
        world.generation_dir / sealing.QA_REPORT_NAME,
        lambda r: r["smoke"][0].__setitem__("opened", False),
    )


def _tamper_figure_metadata_title(world: World) -> None:
    world.rewrite_json(
        world.registry_root / "metadata" / "fig-01a.json",
        lambda m: m.__setitem__("title", "retitled after render, unbound from H3"),
    )


def _tamper_h4_status(world: World) -> None:
    world.rewrite_json(world.qa_receipt_path, lambda r: r.__setitem__("status", "rejected"))


def _tamper_h4_caption_binding(world: World) -> None:
    doc = world.registry_root / "docs" / "fig-01a-clean-fig1-peak.md"
    doc.write_text(doc.read_text(encoding="utf-8") + "\nchanged post-accept\n", encoding="utf-8")


def _tamper_registry_status_reserved(world: World) -> None:
    world.rewrite_json(
        world.registry_path,
        lambda r: next(f for f in r["figures"] if f["figure_id"] == "fig-01a").__setitem__(
            "status", "reserved"
        ),
    )


def _tamper_ledger_broken_chain(world: World) -> None:
    ledger_mod.append_entry(
        world.ledger_path,
        {"event": "promote", "generation_id": "planted"},
        paper_id="trm-sigmak",
    )
    ledger = json.loads(world.ledger_path.read_text(encoding="utf-8"))
    ledger["entries"][0]["prev_entry_sha256"] = "f" * 64
    world.ledger_path.write_text(json.dumps(ledger), encoding="utf-8")


#: (label, mutation, step this row targets, expected exit code, a message
#: substring unique to *that step's* failure). The substring column is the
#: difference between "some link caught it" and "this link caught it" -- a
#: mutation that is silently caught one step earlier (as H2-asset-byte-forged
#: originally was, at H1) fails this table's assertion instead of passing green
#: for the wrong reason.
CHAIN_TAMPER_TABLE: list[tuple[str, Callable[[World], None], str, ExitCode, str]] = [
    ("H1-channel-mismatch", _tamper_h1_channel, "H1", ExitCode.VERIFICATION, "absent from"),
    ("H1-receipt-missing", _tamper_h1_missing, "H1", ExitCode.VERIFICATION, "no acquisition receipts found"),
    ("H2-asset-byte-forged", _tamper_h2_asset_byte, "H2", ExitCode.VERIFICATION, "asset byte mismatch"),
    ("H2-provenance-label-mismatch", _tamper_h2_provenance_label, "H2", ExitCode.VERIFICATION,
     "provenance mismatch"),
    ("H3-output-byte-tampered", _tamper_h3_output_byte, "H3", ExitCode.VERIFICATION,
     "render output byte mismatch"),
    ("H3-snapshot-binding-forged", _tamper_h3_snapshot_binding, "H3", ExitCode.VERIFICATION,
     "snapshot_id != sealed snapshot_id"),
    ("QR-overall-fail", _tamper_qr_overall, "QR", ExitCode.VERIFICATION, "qa_report overall"),
    ("QR-smoke-not-opened", _tamper_qr_smoke, "QR", ExitCode.VERIFICATION, "figure output smoke failed"),
    ("figure-metadata-content-drift", _tamper_figure_metadata_title, "figure-metadata",
     ExitCode.VERIFICATION, "figure_metadata_sha256"),
    ("H4-status-not-accepted", _tamper_h4_status, "H4", ExitCode.POLICY, "qa_receipt status is 'rejected'"),
    ("H4-caption-bytes-changed", _tamper_h4_caption_binding, "H4", ExitCode.VERIFICATION, "caption_sha256"),
    ("registry-status-reserved", _tamper_registry_status_reserved, "registry", ExitCode.POLICY,
     "registry status 'reserved'"),
]


@pytest.mark.parametrize(
    "label,mutate,failing_step,expected_exit_code,message_substring",
    CHAIN_TAMPER_TABLE,
    ids=[row[0] for row in CHAIN_TAMPER_TABLE],
)
def test_each_chain_link_tamper_is_independently_caught(
    world: World, label: str, mutate: Callable[[World], None], failing_step: str,
    expected_exit_code: ExitCode, message_substring: str,
) -> None:
    """Judged by exit code (design's own authority, §2 N04) AND by a
    step-specific message substring -- exit code alone cannot distinguish "H2
    caught it" from "H1 caught it one step earlier" when both raise
    exit_code=VERIFICATION, which is exactly the failure mode this table hit
    once already (H2-asset-byte-forged first drafted, silently passing green
    while actually failing at H1; see the companion test above it).

    Exception *type* is deliberately not asserted: chain.verify_h2 funnels
    every validate_snapshot rule failure through
    ValidationResult.raise_for_status(), which raises the base PipelineError
    with the failing rule's exit_code attached rather than the typed
    VerificationError/PolicyError subclass -- so PipelineError is the one type
    every row in this table is guaranteed to raise, regardless of which step."""
    mutate(world)
    with pytest.raises(PipelineError) as excinfo:
        revalidate_chain(world.chain_inputs())
    message = str(excinfo.value)
    assert excinfo.value.exit_code == expected_exit_code, (
        f"{label} (targets {failing_step}): expected exit {int(expected_exit_code)}, "
        f"got {int(excinfo.value.exit_code)}: {message}"
    )
    assert message_substring in message, (
        f"{label}: expected the {failing_step} step's signature ({message_substring!r}) "
        f"in the failure, got: {message!r} -- likely caught by an earlier step instead"
    )


def test_ledger_chain_broken_prev_hash_is_detected_by_verify_chain(world: World) -> None:
    _tamper_ledger_broken_chain(world)
    with pytest.raises(VerificationError, match="ledger chain broken"):
        ledger_mod.verify_chain(ledger_mod.read_ledger(world.ledger_path))


def test_receipt_replay_across_snapshots_is_a_chain_break(world: World) -> None:
    """A render_manifest that claims a *different* dataset_manifest_sha256 than
    the sealed snapshot it sits next to (receipt replay/cherry-pick) is refused
    at H3, not silently accepted because H1/H2 alone still check out."""
    world.rewrite_json(
        world.generation_dir / sealing.RENDER_MANIFEST_NAME,
        lambda r: r.__setitem__("dataset_manifest_sha256", "d" * 64),
    )
    with pytest.raises(VerificationError, match="dataset_manifest_sha256"):
        revalidate_chain(world.chain_inputs())


# ============================================================ 2. ledger


def test_ledger_empty_read_and_append_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "selected_figures.json"
    empty = ledger_mod.read_ledger(path, paper_id="trm-sigmak")
    assert empty["entries"] == []
    assert ledger_mod.head_sha256(empty) is None

    head1 = ledger_mod.append_entry(path, {"event": "promote", "generation_id": "g1"}, paper_id="trm-sigmak")
    assert len(head1) == 64
    ledger = ledger_mod.read_ledger(path)
    assert len(ledger["entries"]) == 1
    assert ledger["entries"][0]["prev_entry_sha256"] is None
    assert ledger_mod.verify_chain(ledger) == head1


def test_ledger_cas_guard_rejects_stale_expected_prev(tmp_path: Path) -> None:
    path = tmp_path / "selected_figures.json"
    ledger_mod.append_entry(path, {"event": "promote", "generation_id": "g1"}, paper_id="trm-sigmak")
    with pytest.raises(VerificationError, match="ledger CAS failure"):
        ledger_mod.append_entry(
            path, {"event": "promote", "generation_id": "g2"},
            paper_id="trm-sigmak", expected_prev_sha256="0" * 64,
        )


def test_ledger_declared_prev_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "selected_figures.json"
    ledger_mod.append_entry(path, {"event": "promote", "generation_id": "g1"}, paper_id="trm-sigmak")
    with pytest.raises(VerificationError, match="prev_entry_sha256"):
        ledger_mod.append_entry(
            path, {"event": "promote", "generation_id": "g2", "prev_entry_sha256": "a" * 64},
            paper_id="trm-sigmak",
        )


def test_ledger_wrong_schema_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "selected_figures.json"
    path.write_text(json.dumps({"schema_version": "wrong-v0", "paper_id": "p", "entries": []}),
                     encoding="utf-8")
    with pytest.raises(VerificationError, match="unsupported ledger schema_version"):
        ledger_mod.read_ledger(path)


def test_ledger_concurrent_append_serializes_under_flock(tmp_path: Path) -> None:
    """Threads share the process's fd table only by name, not by inode -- each
    thread's own ``open()`` call gets a distinct open-file-description, so
    ``fcntl.flock`` genuinely contends within one process (it would not if we
    tried to prove this with a single shared file handle)."""
    path = tmp_path / "selected_figures.json"
    ledger_mod.append_entry(path, {"event": "promote", "generation_id": "seed"}, paper_id="trm-sigmak")

    n_threads = 12
    errors: list[BaseException] = []
    barrier = threading.Barrier(n_threads)

    def worker(index: int) -> None:
        barrier.wait()
        for attempt in range(50):
            try:
                current = ledger_mod.read_ledger(path)
                prev = ledger_mod.verify_chain(current)
                ledger_mod.append_entry(
                    path, {"event": "promote", "generation_id": f"g-{index}"},
                    paper_id="trm-sigmak", expected_prev_sha256=prev,
                )
                return
            except VerificationError:
                time.sleep(0.001 * (attempt + 1))
        errors.append(RuntimeError(f"worker {index} never won the CAS race"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    ledger = ledger_mod.read_ledger(path)
    assert len(ledger["entries"]) == 1 + n_threads
    assert ledger_mod.verify_chain(ledger)  # full chain re-walks clean, no interleaving corruption
    generation_ids = {e["generation_id"] for e in ledger["entries"]}
    assert generation_ids == {"seed"} | {f"g-{i}" for i in range(n_threads)}


def test_supersede_and_retract_are_ledger_events_not_byte_mutations(world: World) -> None:
    from lab.figure_pipeline.promote import apply as promote_apply

    promote_apply.promote_generation(
        world.chain_inputs(), destinations=world.destinations(), approved_by="ayp", dry_run=None,
    )
    outcome = promote_apply.supersede_or_retract(
        ledger_path=world.ledger_path, paper_id="trm-sigmak", event="retract",
        generation_id=world.generation_id, approved_by="ayp", reason="ledger contract test",
    )
    assert outcome.applied and outcome.event == "retract"
    ledger = ledger_mod.read_ledger(world.ledger_path)
    assert [e["event"] for e in ledger["entries"]] == ["promote", "retract"]
    assert ledger_mod.verify_chain(ledger)
    assert states.GenerationState.RETRACTED not in states.GENERATION_TRANSITIONS[states.GenerationState.RETRACTED]


def test_generation_state_machine_forbids_retract_before_promote() -> None:
    with pytest.raises(PolicyError):
        states.assert_generation_transition(states.GenerationState.ACCEPTED, states.GenerationState.RETRACTED)
    states.assert_generation_transition(states.GenerationState.ACCEPTED, states.GenerationState.PROMOTED)


def test_dataset_state_machine_has_no_path_back_from_sealed() -> None:
    assert states.DATASET_TRANSITIONS[states.DatasetState.SEALED] == frozenset()
    with pytest.raises(PolicyError):
        states.assert_dataset_transition(states.DatasetState.SEALED, states.DatasetState.ACQUIRED)


# ============================================================ 3. sealing TOCTOU
#
# All three tamper the already-*sealed* ``world.snapshot_dir`` -- i.e. after
# ``sealing.seal()`` succeeded and returned -- then prove the tamper is caught
# at promote-time re-verification (chain.verify_h2 -> sealing.verify_sealed_snapshot),
# not merely by re-running seal() again.


def test_toctou_registered_byte_swapped_after_seal_is_caught_at_reverify(world: World) -> None:
    assert sealing.is_sealed(world.snapshot_dir)
    (world.snapshot_dir / "raw" / "evals.jsonl").write_bytes(b"swapped after sealing succeeded\n")
    with pytest.raises(VerificationError, match="asset byte mismatch"):
        sealing.verify_sealed_snapshot(world.snapshot_dir, world.manifest)
    # promote's re-verification wraps the same sealing-level rule inside
    # ValidationResult.raise_for_status(), which raises the base PipelineError
    # (exit_code=VERIFICATION) rather than the VerificationError subclass.
    with pytest.raises(PipelineError) as excinfo:
        revalidate_chain(world.chain_inputs())
    assert excinfo.value.exit_code == ExitCode.VERIFICATION
    assert "asset byte mismatch" in str(excinfo.value)


def test_toctou_unregistered_file_added_after_seal_is_caught_fail_closed(world: World) -> None:
    (world.snapshot_dir / "raw" / "sneaked-in.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(VerificationError, match="unregistered files inside snapshot"):
        sealing.verify_sealed_snapshot(world.snapshot_dir, world.manifest)
    with pytest.raises(PipelineError) as excinfo:
        revalidate_chain(world.chain_inputs())
    assert excinfo.value.exit_code == ExitCode.VERIFICATION
    assert "unregistered files inside snapshot" in str(excinfo.value)


def test_toctou_symlink_introduced_after_seal_is_caught(world: World) -> None:
    link = world.snapshot_dir / "raw" / "sneaky-link.jsonl"
    link.symlink_to(world.snapshot_dir / "raw" / "evals.jsonl")
    with pytest.raises(VerificationError, match="symlink inside snapshot refused"):
        sealing.verify_sealed_snapshot(world.snapshot_dir, world.manifest)
    with pytest.raises(PipelineError) as excinfo:
        revalidate_chain(world.chain_inputs())
    assert excinfo.value.exit_code == ExitCode.VERIFICATION
    assert "symlink inside snapshot refused" in str(excinfo.value)


def test_seal_refuses_to_rematerialize_an_existing_identity(tmp_path: Path) -> None:
    from lab.figure_pipeline.core.exits import ImmutabilityError

    final_dir = tmp_path / "already-there"
    final_dir.mkdir()
    with pytest.raises(ImmutabilityError):
        sealing.seal(tmp_path / "staging-never-used", final_dir, assets=[])


def test_seal_verifies_every_byte_before_dropping_the_sealed_marker(tmp_path: Path) -> None:
    staging = tmp_path / ".tmp-seal-test"
    (staging).mkdir()
    payload = staging / "a.txt"
    payload.write_text("hello\n", encoding="utf-8")
    wrong_asset = [{"uri": "a.txt", "sha256": "0" * 64}]
    with pytest.raises(VerificationError, match="asset byte mismatch"):
        sealing.seal(staging, tmp_path / "final", assets=wrong_asset)
    assert not (tmp_path / "final").exists()
    assert not (staging / sealing.SEALED_MARKER).exists()


# ============================================================ 4. web adapter mock (FakeApi)


def test_web_channel_happy_path_snapshot_via_fake_api() -> None:
    payload = web.extract(FakeApi([FakeRun()]), sleep=lambda _: None, rand=lambda: 0.0)
    run = payload["runs"]["fig1_mlp_noz_noiter_k3_s1"]
    assert run["history_status"] == "verified"
    assert run["dir"].startswith("wandb://raon1123/Sigma_k_new/")


def test_web_channel_refuses_unallowlisted_entity_project() -> None:
    with pytest.raises(PolicyError, match="outside allowlist"):
        web.extract(FakeApi([FakeRun()]), entity="someone-else", sleep=lambda _: None, rand=lambda: 0.0)


def test_web_channel_forbidden_covariate_denylist_unit_and_structural_defense() -> None:
    """Two layers, checked separately: (a) qa.denylist.assert_no_forbidden_fields
    is a real, working guard against a poisoned raw payload; (b) web_api's own
    candidate_record builds ``config`` from a fixed ALLOWLIST_CONFIG projection
    (web_api.py ~line 297) rather than passing the raw run.config through, so a
    forbidden key on the *raw* wandb run config structurally never reaches the
    extracted payload in the first place -- a stronger guarantee than the scan,
    demonstrated here rather than assumed."""
    from lab.figure_pipeline.qa.denylist import assert_no_forbidden_fields

    with pytest.raises(PolicyError, match="forbidden order/cycle-derived field"):
        assert_no_forbidden_fields({"cycle_type": "3-cycle"}, label="unit-test-payload")

    poisoned = FakeRun(config={
        "project_name": "Sigma_k_new", "run_name": "fig1_mlp_noz_noiter_k3_s1",
        "k": 3, "seed": 1, "data_paths": "data/sigma_k_10/3", "cycle_type": "3-cycle",
    })
    payload = web.extract(FakeApi([poisoned]), sleep=lambda _: None, rand=lambda: 0.0)
    run = payload["runs"]["fig1_mlp_noz_noiter_k3_s1"]
    assert "cycle_type" not in json.dumps(run["config"])
    assert_no_forbidden_fields(run["config"], label="web-run-config")  # does not raise


# ============================================================ 5. NAS adapter tmp fixture


@pytest.fixture
def nas_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never touches the real /mnt NAS mount -- redirects the adapter's root
    constant to a throwaway tree, exactly like lab/figure_pipeline/tests/test_nas_export.py."""
    fake_base = tmp_path / "figure-exports"
    monkeypatch.setattr(ne, "NAS_EXPORT_BASE", fake_base)
    return fake_base


@pytest.fixture
def staging_dest():
    """acquire()'s dest_dir must resolve under a WRITABLE_ROOTS member
    (core.paths.safe_output default allowlist) -- a bare tmp_path is refused.
    Mirrors the existing sibling fixture; always cleaned up in teardown."""
    dest = paths.STAGING_ROOT / f"test-figpipe-contracts-nas-{uuid.uuid4().hex}"
    yield dest
    shutil.rmtree(dest, ignore_errors=True)


def _write_export_bundle(root: Path, host_alias: str) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    payload = root / "evals.jsonl"
    payload.write_bytes(b'{"run_id":"fig1-trm-k5-s1","value":0.91}\n')
    manifest = {
        "schema_version": ne.EXPORT_MANIFEST_SCHEMA_VERSION,
        "host_alias": host_alias,
        "export_revision": "rev-contracts-1",
        "created_utc": "20260728T120000Z",
        "config_digest": "c" * 64,
        "source_commit": {"commit_sha": "a" * 40, "kind": "commit", "branch": "main", "dirty": False},
        "files": [{
            "path": "evals.jsonl", "sha256": sha256_file(payload),
            "bytes": payload.stat().st_size, "run_id": "fig1-trm-k5-s1", "kind": "jsonl",
        }],
    }
    (root / ne.EXPORT_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_nas_channel_happy_path_acquire_builds_a_valid_h1_receipt(
    nas_base: Path, staging_dest: Path,
) -> None:
    _write_export_bundle(nas_base / "gpu-a", "gpu-a")
    result = ne.acquire(
        "gpu-a", staging_dest,
        inventory_digest="d" * 64, allowlist_digest="e" * 64,
        expected_run_ids=["fig1-trm-k5-s1"],
    )
    assert result is not None
    assert result.receipt["channel"] == "nas"
    assert result.receipt["missing_runs"] == []
    assert len(result.staged_files) == 1
    staged_path = staging_dest / "raw" / "evals.jsonl"
    assert staged_path.is_file()
    assert sha256_file(staged_path) == result.staged_files[0]["sha256"]


def test_nas_channel_expected_run_id_not_present_is_recorded_as_missing(
    nas_base: Path, staging_dest: Path,
) -> None:
    _write_export_bundle(nas_base / "gpu-a", "gpu-a")
    result = ne.acquire(
        "gpu-a", staging_dest,
        inventory_digest="d" * 64, allowlist_digest="e" * 64,
        expected_run_ids=["fig1-trm-k5-s1", "fig1-trm-k9-s1"],
    )
    assert result.receipt["missing_runs"] == ["fig1-trm-k9-s1"]


def test_nas_channel_tampered_bytes_are_caught_before_staging(nas_base: Path, staging_dest: Path) -> None:
    manifest = _write_export_bundle(nas_base / "gpu-a", "gpu-a")
    (nas_base / "gpu-a" / "evals.jsonl").write_bytes(b'{"run_id":"fig1-trm-k5-s1","value":0.42}\n')
    with pytest.raises(VerificationError, match="do not match the NAS payload"):
        ne.acquire(
            "gpu-a", staging_dest,
            inventory_digest="d" * 64, allowlist_digest="e" * 64,
            expected_run_ids=["fig1-trm-k5-s1"],
        )
    assert not staging_dest.exists() or not any(staging_dest.rglob("*"))


def test_nas_channel_fails_closed_without_expected_run_ids(nas_base: Path, staging_dest: Path) -> None:
    """§5-24(d): a channel must not self-derive its own coverage expectation."""
    _write_export_bundle(nas_base / "gpu-a", "gpu-a")
    from lab.figure_pipeline.core.exits import UsageError

    with pytest.raises(UsageError, match="expected_run_ids is required"):
        ne.acquire(
            "gpu-a", staging_dest,
            inventory_digest="d" * 64, allowlist_digest="e" * 64,
            expected_run_ids=None,
        )


def test_nas_channel_soft_empty_when_host_export_absent(nas_base: Path, staging_dest: Path) -> None:
    assert ne.acquire(
        "host-never-exported", staging_dest,
        inventory_digest="d" * 64, allowlist_digest="e" * 64, expected_run_ids=["x"],
    ) is None


def test_nas_channel_never_writes_the_real_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one test in this section that deliberately does NOT monkeypatch
    NAS_EXPORT_BASE -- proves the real mount prefix is refused as a write target."""
    with pytest.raises(PolicyError, match="read-only mount prefix"):
        ne.stage_export_files(
            "gpu-a", {"files": [], "export_revision": "rev-1"}, ne.NAS_EXPORT_BASE / "gpu-a",
        )


# ============================================================ 6. provenance grade derivation


def test_weakest_provenance_mixed_canonical_and_preview_is_preview() -> None:
    assert states.weakest_provenance(
        [states.Provenance.CANONICAL, states.Provenance.PREVIEW]
    ) is states.Provenance.PREVIEW


def test_weakest_provenance_any_legacy_dominates() -> None:
    assert states.weakest_provenance(
        [states.Provenance.CANONICAL, states.Provenance.PREVIEW, states.Provenance.LEGACY_CONTAMINATED]
    ) is states.Provenance.LEGACY_CONTAMINATED


def test_weakest_provenance_all_canonical_is_canonical() -> None:
    assert states.weakest_provenance(
        [states.Provenance.CANONICAL, states.Provenance.CANONICAL]
    ) is states.Provenance.CANONICAL


def test_weakest_provenance_requires_at_least_one_value() -> None:
    with pytest.raises(VerificationError, match="requires at least one value"):
        states.weakest_provenance([])


@pytest.mark.parametrize(
    "pre_cutoff,project_ok,contam_ok,coverage_complete,any_dirty,expected",
    [
        (False, True, True, True, False, states.Provenance.CANONICAL),
        (True, True, True, True, False, states.Provenance.LEGACY_CONTAMINATED),
        (False, False, True, True, False, states.Provenance.LEGACY_CONTAMINATED),
        (False, True, False, True, False, states.Provenance.LEGACY_CONTAMINATED),
        (False, True, True, False, False, states.Provenance.PREVIEW),
        (False, True, True, True, True, states.Provenance.PREVIEW),
        # LEGACY inputs outrank PREVIEW inputs when both would apply
        (True, True, True, False, True, states.Provenance.LEGACY_CONTAMINATED),
    ],
)
def test_derive_provenance_table(
    pre_cutoff: bool, project_ok: bool, contam_ok: bool, coverage_complete: bool,
    any_dirty: bool, expected: states.Provenance,
) -> None:
    assert states.derive_provenance(
        pre_cutoff_runs=pre_cutoff, project_filter_ok=project_ok,
        contamination_gate_ok=contam_ok, coverage_complete=coverage_complete,
        any_source_dirty=any_dirty,
    ) is expected


def test_declared_provenance_label_mismatch_is_rejected() -> None:
    with pytest.raises(VerificationError, match="provenance mismatch"):
        states.assert_declared_provenance("CANONICAL", states.Provenance.PREVIEW)
    assert states.assert_declared_provenance("PREVIEW", states.Provenance.PREVIEW) is states.Provenance.PREVIEW


def test_assert_promotable_hard_vetoes_everything_but_canonical() -> None:
    assert states.assert_promotable("CANONICAL") is states.Provenance.CANONICAL
    with pytest.raises(PolicyError, match="promotion refused"):
        states.assert_promotable("PREVIEW")
    with pytest.raises(PolicyError, match="promotion refused"):
        states.assert_promotable("LEGACY-CONTAMINATED")


def test_manifest_level_provenance_derivation_matches_the_declared_label(world: World) -> None:
    """Validator-level version of the same rule (qa.validator.derive_manifest_provenance),
    exercised through the world's real gate_results/sources rather than the bare enum API."""
    from lab.figure_pipeline.qa.validator import derive_manifest_provenance

    assert derive_manifest_provenance(world.manifest) is states.Provenance.CANONICAL

    dirty_manifest = json.loads(json.dumps(world.manifest))
    dirty_manifest["sources"][0]["source_commit"]["dirty"] = True
    assert derive_manifest_provenance(dirty_manifest) is states.Provenance.PREVIEW


def test_coverage_incomplete_manifest_derives_to_preview_and_matches_world_label(
    tmp_path: Path,
) -> None:
    preview_world = build_world(tmp_path, coverage_ok=False)
    from lab.figure_pipeline.qa.validator import derive_manifest_provenance

    assert derive_manifest_provenance(preview_world.manifest) is states.Provenance.PREVIEW
    assert preview_world.manifest["provenance"] == "PREVIEW"


# ============================================================ 7. D3-2 CRITICAL+HIGH negative regressions
#
# Each test isolates exactly one finding from the D3-2 core/gate audit summary and
# proves the *fixed* behaviour actually bites -- a rule that is declared but not
# wired would pass a happy-path suite while still being forgeable.


def _qa_report_path(world: World) -> Path:
    return world.generation_dir / sealing.QA_REPORT_NAME


# --- (1) C1: sealed manifest replaced in place without re-sealing --------------------


def test_sealed_manifest_replaced_in_place_without_resealing_is_rejected(world: World) -> None:
    """Edits ``dataset_manifest.json`` directly on disk and re-writes it through
    ``world.rewrite_json`` (which -- unlike ``recompute_manifest``/``reseal`` --
    does *not* re-issue ``_SEALED``), models an operator who owns the directory
    hand-editing the sealed manifest after ``seal()`` returned. The mutated field
    (``source_commit.branch``) is deliberately a NON_IDENTITY_SUBFIELD so
    ``content_sha256``/``snapshot_id`` still recompute clean and the earlier
    ``snapshot-id-content-hash`` check does not mask this one -- the only thing
    that must catch it is C1's manifest-sha256-vs-marker binding."""
    manifest_path = world.snapshot_dir / sealing.MANIFEST_NAME
    tampered = world.rewrite_json(
        manifest_path,
        lambda m: m["sources"][0]["source_commit"].__setitem__("branch", "feature/tamper"),
    )
    with pytest.raises(VerificationError, match="changed since sealing"):
        sealing.verify_sealed_snapshot(world.snapshot_dir, tampered)
    with pytest.raises(PipelineError) as excinfo:
        revalidate_chain(world.chain_inputs())
    assert excinfo.value.exit_code == ExitCode.VERIFICATION
    assert "changed since sealing" in str(excinfo.value)


# --- (2) C2: paper_id/generation_id path-escape strings ------------------------------


PATH_ESCAPE_IDS = [
    "..", "../../../etc/passwd", "trm/../etc", "/etc/passwd", "trm sigmak",
    "trm\\sigmak", "trm\x00sigmak", ".hidden", "", "-leading-dash",
]


@pytest.mark.parametrize("bad_id", PATH_ESCAPE_IDS)
def test_paper_id_path_escape_strings_are_rejected_at_every_ingress(
    bad_id: str, tmp_path: Path
) -> None:
    with pytest.raises(UsageError):
        identity_mod.assert_paper_id(bad_id)
    # run_pipeline.generation_dir is the one place every stage-side caller composes
    # outputs/generations/<paper>/<figure>/<generation>/ -- the ingress guard must
    # fire before the path is ever joined.
    with pytest.raises(UsageError):
        rp.generation_dir(tmp_path / "output", bad_id, "fig-01a", "20260728T130000Z_0000000000000000")


@pytest.mark.parametrize("bad_id", PATH_ESCAPE_IDS)
def test_generation_id_path_escape_strings_are_rejected_at_every_ingress(
    bad_id: str, tmp_path: Path
) -> None:
    with pytest.raises(UsageError):
        identity_mod.assert_generation_id_syntax(bad_id)
    with pytest.raises(UsageError):
        rp.generation_dir(tmp_path / "output", "trm-sigmak", "fig-01a", bad_id)


def test_generation_id_path_escape_is_also_rejected_by_promote_destination_composition(
    world: World,
) -> None:
    """The second path-composition ingress: promote's own publication paths."""
    with pytest.raises(UsageError):
        promote_apply.authorized_destinations(
            world.destinations(), paper_id=PAPER_ID, figure_id="fig-01a",
            generation_id="../../../etc/passwd", timestamp="20260728T130000Z",
        )


# --- (3) S2: qa_receipt replaced after chain re-verification, before the copy --------


def test_qa_receipt_swapped_after_revalidation_blocks_promotion(world: World) -> None:
    chain = revalidate_chain(world.chain_inputs())
    world.rewrite_json(
        world.qa_receipt_path, lambda r: r.__setitem__("session_ref", "swapped-after-verify"),
    )
    with pytest.raises(VerificationError, match="changed since chain re-verification"):
        promote_apply._expected_tree(chain, world.generation_dir)


# --- (4) S3(a): forbidden covariate only in the real payload, not the declaration ----


def _rebind_h1_receipt_sha(world: World, receipt_path: Path) -> None:
    """After editing the H1 receipt directly, re-bind manifest.acquisition_receipts[]
    (and the matching sources[].acquisition_receipt_sha256) to the receipt file's new
    sha256 -- otherwise H1's own receipt<->manifest binding check fires first on the
    receipt-file-changed fact alone, masking whatever H2-level rule the test targets.
    Mirrors the existing ``_tamper_h2_asset_byte`` isolation technique above."""
    def _rebind(m: dict[str, Any]) -> None:
        new_sha = sha256_file(receipt_path)
        m["acquisition_receipts"][0]["sha256"] = new_sha
        for source in m["sources"]:
            if source.get("kind") == "acquired":
                source["acquisition_receipt_sha256"] = new_sha

    world.recompute_manifest(_rebind)


def test_forbidden_covariate_in_real_payload_blocks_promotion_even_when_undeclared(
    world: World,
) -> None:
    """The manifest's declared column list is untouched by ``rewrite_evals`` --
    only the real on-disk jsonl payload gains ``cycle_type``. A denylist that only
    scanned the declared schema would pass this generation straight through the
    whole H1..H5+QR chain to promotion.

    H1's raw-file cross-check is kept in sync with the new bytes first (receipt
    sha256 + rebind) so it isolates the S3(a) rule under test rather than catching
    the payload edit one step earlier for an unrelated (also correct) reason."""
    rows = [{**r, "cycle_type": "5^2"} for r in ROWS]
    world.rewrite_evals(rows)
    evals_path = world.snapshot_dir / "raw" / "evals.jsonl"
    receipt_path = world.receipts_dir / "acquisition_receipt_local.json"
    world.rewrite_json(
        receipt_path,
        lambda r: next(f for f in r["raw_files"] if f["path"] == "raw/evals.jsonl").__setitem__(
            "sha256", sha256_file(evals_path)
        ),
    )
    _rebind_h1_receipt_sha(world, receipt_path)

    with pytest.raises(PipelineError) as excinfo:
        revalidate_chain(world.chain_inputs())
    assert excinfo.value.exit_code == ExitCode.POLICY
    assert "cycle_type" in str(excinfo.value)


# --- (5) S3(b): gate_results boolean forged pass vs. recomputed disagreement ---------


def test_gate_results_forged_pass_disagrees_with_recompute_and_blocks_promotion(
    world: World,
) -> None:
    """``gate_results.project_filter.passed`` stays declared ``True`` while the
    inventory is edited to reference a foreign project -- exactly the "trust the
    boolean" gap S3(b) closes by recomputing the gate from ``sources[]`` instead.

    H1's inventory_digest<->receipt cross-check is re-synced first, for the same
    isolation reason as the forbidden-covariate test above."""
    world.rewrite_inventory(
        lambda inv: inv["sources"][0].__setitem__("wandb_project", "Sigma_k_old")
    )
    receipt_path = world.receipts_dir / "acquisition_receipt_local.json"
    world.rewrite_json(
        receipt_path, lambda r: r.__setitem__("inventory_digest", world.manifest["inventory_digest"])
    )
    _rebind_h1_receipt_sha(world, receipt_path)

    with pytest.raises(PipelineError) as excinfo:
        revalidate_chain(world.chain_inputs())
    assert excinfo.value.exit_code == ExitCode.VERIFICATION
    assert "gate-recompute-equivalence" in str(excinfo.value) or "project_filter" in str(
        excinfo.value
    )


# --- (6) S4: empty smoke record and a validator_sha256 from a foreign build ----------


def test_empty_smoke_record_and_forged_validator_sha_are_both_rejected(tmp_path: Path) -> None:
    world_a = build_world(tmp_path / "empty-smoke")
    world_a.rewrite_json(_qa_report_path(world_a), lambda r: r.__setitem__("smoke", []))
    with pytest.raises(VerificationError, match="smoke"):
        revalidate_chain(world_a.chain_inputs())

    world_b = build_world(tmp_path / "forged-validator-sha")
    world_b.rewrite_json(
        _qa_report_path(world_b),
        lambda r: [c.__setitem__("validator_sha256", "f" * 64) for c in r["checks"]],
    )
    with pytest.raises(VerificationError, match="different qa/validator.py"):
        revalidate_chain(world_b.chain_inputs())


# --- (7) S5: a raw asset no receipt covers -------------------------------------------


def test_raw_asset_not_covered_by_any_receipt_is_rejected(world: World) -> None:
    extra = world.snapshot_dir / "raw" / "smuggled-contracts.jsonl"
    extra.write_text(json.dumps({"run_id": "x"}) + "\n", encoding="utf-8")
    sha = sha256_file(extra)

    def mutate(manifest: dict[str, Any]) -> None:
        template = copy.deepcopy(manifest["assets"][1])
        template.update({
            "asset_id": "raw-smuggled-contracts", "format": "jsonl",
            "uri": "raw/smuggled-contracts.jsonl", "sha256": sha, "source_sha256": sha,
            "source_id": "src-local",
        })
        template.pop("analytic_ref", None)
        manifest["assets"].append(template)

    world.recompute_manifest(mutate)
    with pytest.raises(VerificationError, match="bidirectional"):
        revalidate_chain(world.chain_inputs())


# --- (8) S6: a reject event is durable and survives a forged re-accept --------------


def test_reject_event_is_durable_and_blocks_any_later_promotion(world: World) -> None:
    world.qa_receipt_path.unlink()
    caption_path = world.registry_root / "docs" / "fig-01a-clean-fig1-peak.md"
    outcome = receipt_mod.issue_receipt(
        world.generation_dir, caption_path=caption_path, reviewer="ayp",
        session_ref="/fig-qa contracts-test", caption_version="v2",
        verdict_claim_ids=["C-H023.3"], status="rejected",
        ledger_path=world.ledger_path,
        reason="contracts regression: forged accept must not override reject",
    )
    assert outcome["ledger_entry_sha256"]

    # Deleting the receipt and re-accepting must not clear the ledger's refusal --
    # REJECTED is terminal in states.GENERATION_TRANSITIONS.
    world.qa_receipt_path.unlink()
    receipt_mod.issue_receipt(
        world.generation_dir, caption_path=caption_path, reviewer="ayp",
        session_ref="/fig-qa contracts-test", caption_version="v2",
        verdict_claim_ids=["C-H023.3"], status="accepted",
    )
    with pytest.raises(PolicyError, match="rejected"):
        promote_apply.promote_generation(
            world.chain_inputs(), destinations=world.destinations(), approved_by="ayp",
        )


# --- (9) H10: the dry-run envelope refuses to execute anything ----------------------


def test_dry_run_subprocess_runner_refuses_to_execute(tmp_path: Path) -> None:
    runner = rp.make_dry_run_subprocess_runner()
    with pytest.raises(PolicyError, match="dry run"):
        runner("contracts-test-stage", ["true"], timeout=1, cwd=tmp_path, allow_read_roots=[])


# --- (10) H11: duplicate JSON object keys are refused, not silently last-wins -------


def test_strict_loads_rejects_duplicate_object_keys() -> None:
    with pytest.raises(VerificationError, match="duplicate JSON object key"):
        canonical.strict_loads('{"sha256":"a","sha256":"b"}')


def test_render_manifest_with_a_duplicated_json_key_is_rejected_end_to_end(world: World) -> None:
    """Wires H11 through the real evidence-reading path used by chain re-verification
    (schemas.io.read_evidence -> core.canonical.strict_load_path), not just the
    bare ``strict_loads`` unit above -- a brand-new key is duplicated at the front
    of the file so the corruption is independent of the manifest's real fields."""
    path = world.generation_dir / sealing.RENDER_MANIFEST_NAME
    text = path.read_text(encoding="utf-8")
    assert text.startswith("{")
    corrupted = text[:1] + '"__dup_probe__":1,"__dup_probe__":2,' + text[1:]
    path.write_text(corrupted, encoding="utf-8")
    with pytest.raises(VerificationError, match="duplicate JSON object key"):
        revalidate_chain(world.chain_inputs())


# ============================================================ 8. D3-3 final hardening negative regressions
#
# Each test below isolates exactly one closed gap from the D3-3 final hardening
# summary (qa/validator.py G1, promote/apply.py+chain.py G2/G3, core/sealing.py+
# core/paths.py+core/hashing.py S1/S2/S3) and proves the *fixed* behaviour actually
# bites, following the same table's pattern as section 7 above.


# --- (1) G1(a): top-level JSON array payload scanned element-wise -------------------


def test_toplevel_json_array_payload_with_forbidden_key_is_rejected(world: World) -> None:
    """The earlier ``isinstance(payload, Mapping)`` guard let ``[{"cycle_type": ...}]``
    sail through unscanned. Real bytes written into the world's real sealed
    snapshot_dir; the containment-checked open (snapshot_asset_path) and the scan
    itself run for real, isolated from schema/H1 concerns via a synthetic
    single-asset manifest handed directly to the function under test."""
    poisoned = world.snapshot_dir / "raw" / "array-poison.json"
    poisoned.write_text(
        json.dumps([{"run_id": "fig1-trm-k5-s1", "cycle_type": "5^2"}]), encoding="utf-8"
    )
    sha = sha256_file(poisoned)
    asset = {
        "asset_id": "raw-array-poison", "format": "json", "uri": "raw/array-poison.json",
        "sha256": sha, "materialized": True,
    }
    with pytest.raises(PolicyError, match="forbidden order/cycle-derived field"):
        qa_validator.assert_payload_covariates_clean({"assets": [asset]}, world.snapshot_dir)


# --- (2) G1(b): safetensors header tensor keys / __metadata__ scanned ---------------


def test_safetensors_forbidden_tensor_key_is_rejected(world: World) -> None:
    """safetensors headers are opened for their tensor keys (and the free-form
    ``__metadata__`` map) via a size-bounded header parse -- no new dependency, and
    tensor bytes are never loaded. A forbidden tensor name is caught by the same
    central denylist as every other scanned format."""
    import struct

    header = {
        "cycle_type": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "__metadata__": {"note": "clean"},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    payload = struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00\x00\x00\x00"
    path = world.snapshot_dir / "raw" / "poison.safetensors"
    path.write_bytes(payload)
    sha = sha256_file(path)
    asset = {
        "asset_id": "raw-poison-safetensors", "format": "safetensors",
        "uri": "raw/poison.safetensors", "sha256": sha, "materialized": True,
    }
    with pytest.raises(PolicyError, match="forbidden order/cycle-derived field"):
        qa_validator.assert_payload_covariates_clean({"assets": [asset]}, world.snapshot_dir)


# --- (3) G1(c): a materialized asset with nothing scanned fails closed --------------


def test_materialized_json_bare_scalar_fails_closed_nothing_scanned(world: World) -> None:
    """``NO_PAYLOAD_FORMATS = {"md"}`` is the only declared no-payload exemption. A
    materialized json whose body is a bare scalar (neither object nor array) is not
    covered by it and must fail closed rather than sail through with ``scanned``
    left at 0."""
    path = world.snapshot_dir / "raw" / "scalar.json"
    path.write_text("42", encoding="utf-8")
    sha = sha256_file(path)
    asset = {
        "asset_id": "raw-bare-scalar", "format": "json", "uri": "raw/scalar.json",
        "sha256": sha, "materialized": True,
    }
    with pytest.raises(PolicyError, match="declared clean but nothing scanned"):
        qa_validator.assert_payload_covariates_clean({"assets": [asset]}, world.snapshot_dir)


# --- (4) G2(a): _publish_generation_dir is internal, not a public entrypoint --------


def test_publish_generation_dir_is_internal_not_exported() -> None:
    """Publication is a step of the ``promote_generation`` transaction only -- the
    standalone primitive is renamed internal (``_`` prefix) and removed from both
    ``promote.apply.__all__`` and the ``promote`` package's public export, so a
    caller cannot reach it without going through the module's private surface."""
    import lab.figure_pipeline.promote as promote_pkg

    assert "_publish_generation_dir" not in promote_apply.__all__
    assert not hasattr(promote_pkg, "_publish_generation_dir")
    assert "_publish_generation_dir" not in promote_pkg.__all__
    assert not hasattr(promote_pkg, "publish_generation_dir")
    assert hasattr(promote_apply, "_publish_generation_dir")  # still exists, module-internal


# --- (5) G2(b): reconcile_orphan_publications quarantines an H5-less publication ---


def test_direct_internal_publish_produces_orphan_that_reconcile_quarantines(world: World) -> None:
    """A caller that reaches around ``promote_generation`` straight to the internal
    publish primitive (no ledger lock, no H5 append) leaves exactly the state the
    reconciliation sweep exists to find: bytes at a canonical publication path with
    no promote entry. ``reconcile_orphan_publications`` must move it aside (never
    delete it)."""
    destinations = world.destinations()
    authorized = promote_apply.authorized_destinations(
        destinations, paper_id=PAPER_ID, figure_id="fig-01a",
        generation_id=world.generation_id, timestamp=world.render_manifest["timestamp"],
    )
    target = authorized[0]
    expected = sealing.rehash_tree(world.generation_dir)
    copied = promote_apply._publish_generation_dir(
        world.generation_dir, target.path, generation_id=world.generation_id,
        expected=expected, root=target.root,
    )
    assert copied is True
    assert target.path.is_dir()

    ledger = ledger_mod.read_ledger(world.ledger_path, paper_id=PAPER_ID)
    assert promote_apply.generation_state(ledger, world.generation_id) is None  # no H5 entry at all

    findings = promote_apply.reconcile_orphan_publications(
        [destinations.reports_root / "figures"], ledger=ledger,
    )
    matches = [f for f in findings if f["generation_id"] == world.generation_id]
    assert len(matches) == 1
    assert matches[0]["state"] == "quarantined-orphan"
    assert not target.path.exists()  # moved aside, not left publicly reachable
    quarantined = Path(matches[0]["path"])
    assert quarantined.is_dir()
    assert quarantined.name.startswith(promote_apply.ORPHAN_PREFIX)


# --- (6) G2(c): retry after a crash between H5 append and final rename -------------


def test_crash_after_h5_append_before_rename_then_retry_completes(
    world: World, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``recover_promotion_residue`` runs BEFORE ``_assert_promotable_state`` inside
    the ledger lock. A crash between the H5 append and the final rename leaves the
    generation PROMOTED in the ledger with staged bytes still on disk; the retry
    must reach recover's 'completed' branch and finish the rename -- not trip the
    immutability veto that (correctly) refuses a genuinely fresh second promotion,
    which is checked at the end of this test too."""
    from lab.figure_pipeline.core.exits import ImmutabilityError

    calls = {"n": 0}
    real_publish_staged = promote_apply._publish_staged

    def _crash_once(partial: Path, destination: Any, expected: Mapping[str, str]) -> Path:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated crash: after H5 append, before final rename")
        return real_publish_staged(partial, destination, expected)

    monkeypatch.setattr(promote_apply, "_publish_staged", _crash_once)
    with pytest.raises(RuntimeError, match="simulated crash"):
        promote_apply.promote_generation(
            world.chain_inputs(), destinations=world.destinations(), approved_by="ayp", dry_run=None,
        )

    ledger = ledger_mod.read_ledger(world.ledger_path, paper_id=PAPER_ID)
    assert promote_apply.generation_state(ledger, world.generation_id) is states.GenerationState.PROMOTED

    authorized = promote_apply.authorized_destinations(
        world.destinations(), paper_id=PAPER_ID, figure_id="fig-01a",
        generation_id=world.generation_id, timestamp=world.render_manifest["timestamp"],
    )
    assert any(promote_apply.staging_residue(d.path, world.generation_id) for d in authorized)

    monkeypatch.setattr(promote_apply, "_publish_staged", real_publish_staged)
    outcome = promote_apply.promote_generation(
        world.chain_inputs(), destinations=world.destinations(), approved_by="ayp", dry_run=None,
    )
    assert outcome.applied is True
    assert "completed" in outcome.detail
    for d in authorized:
        assert d.path.is_dir()
        assert sealing.rehash_tree(d.path) == sealing.rehash_tree(world.generation_dir)

    # A genuinely fresh re-promote attempt, with no residue left to recover, still
    # falls through to _assert_promotable_state -> ImmutabilityError (exit 21).
    with pytest.raises(ImmutabilityError):
        promote_apply.promote_generation(
            world.chain_inputs(), destinations=world.destinations(), approved_by="ayp", dry_run=None,
        )


# --- (7) G3(a): caption_version is bound against the registry's declared value -----


def test_caption_version_forged_is_rejected_by_verify_h4(world: World) -> None:
    """``caption_version`` used to be checked only non-blank at issue time -- a
    free-floating label. Once the registry entry declares an expected value,
    ``verify_h4`` now binds the receipt's value to it, so a mislabeled version on an
    otherwise byte-identical (``caption_sha256``-matching) caption is refused."""
    world.rewrite_json(world.qa_receipt_path, lambda r: r.__setitem__("caption_version", "v999"))
    with pytest.raises(PolicyError, match="caption_version"):
        revalidate_chain(world.chain_inputs())


# --- (8) G3(b): H5 approved_by is recorded canonical (stripped) --------------------


def test_h5_approved_by_is_recorded_stripped_matching_receipt_reviewer(world: World) -> None:
    """H5 entries now record ``str(approved_by).strip()`` for both promote and
    supersede/retract events, so the durable ledger's ``approved_by`` exactly
    matches the receipt's stripped ``reviewer`` -- previously a raw ``'  ayp  '``
    could sit in the ledger next to a receipt signed ``'ayp'``."""
    outcome = promote_apply.promote_generation(
        world.chain_inputs(), destinations=world.destinations(), approved_by="  ayp  ", dry_run=None,
    )
    assert outcome.applied
    ledger = ledger_mod.read_ledger(world.ledger_path, paper_id=PAPER_ID)
    promote_entry = next(e for e in ledger["entries"] if e["event"] == promote_apply.EVENT_PROMOTE)
    assert promote_entry["approved_by"] == "ayp"

    retract_outcome = promote_apply.supersede_or_retract(
        ledger_path=world.ledger_path, paper_id=PAPER_ID, event=promote_apply.EVENT_RETRACT,
        generation_id=world.generation_id, approved_by="  ayp  ",
        reason="contracts regression: G3(b) canonical approved_by",
    )
    assert retract_outcome.applied
    ledger2 = ledger_mod.read_ledger(world.ledger_path, paper_id=PAPER_ID)
    retract_entry = next(e for e in ledger2["entries"] if e["event"] == promote_apply.EVENT_RETRACT)
    assert retract_entry["approved_by"] == "ayp"


# --- (9a) S1: fd-anchored rehash_tree refuses a symlink (static case) --------------


def test_symlink_in_generation_dir_is_refused_by_fd_anchored_rehash(world: World) -> None:
    """``rehash_tree``/``iter_payload_files`` now walk on a shared fd-anchored handle
    (``os.listdir(dir_fd)`` + ``fstatat`` + ``O_NOFOLLOW`` opens), so a symlink is
    refused rather than followed. Exercised on the *generation* directory -- a
    locus chain re-verification hashes directly at H3 (``verify_h3`` ->
    ``_generation_tree``), distinct from the dataset-snapshot TOCTOU coverage
    already in section 3 above. Deterministic TOCTOU is not attempted here (a
    statically-planted symlink is the documented fallback for this axis)."""
    (world.generation_dir / "sneaky-output-link.png").symlink_to(
        world.generation_dir / "clean_fig1_peak.png"
    )
    with pytest.raises(VerificationError, match="symlink inside snapshot refused"):
        sealing.rehash_tree(world.generation_dir)
    with pytest.raises(PipelineError) as excinfo:
        revalidate_chain(world.chain_inputs())
    assert excinfo.value.exit_code == ExitCode.VERIFICATION
    assert "symlink inside snapshot refused" in str(excinfo.value)


# --- (9b) S2: refuse_symlinked_descent blocks a symlinked QR write path ------------


def test_refuse_symlinked_descent_blocks_symlinked_qa_report_write_path(world: World) -> None:
    """``qa/auto.py`` calls ``core.paths.refuse_symlinked_descent(ctx.output_root,
    qr_path)`` before the QR write-through -- ``assert_within`` alone is lexical and
    would pass a symlinked *component* inside the output root. Static symlink case:
    a symlinked directory planted directly under the output root is refused by
    ``lstat``, never followed, for any path composed underneath it."""
    output_root = world.generation_dir.parent  # outputs/ -- generation dirs are direct children
    link = output_root / "evil-component"
    link.symlink_to(world.generation_dir)
    forged_qr_path = link / sealing.QA_REPORT_NAME
    with pytest.raises(PolicyError, match="symlink path component refused"):
        paths.refuse_symlinked_descent(output_root, forged_qr_path, label="qa_report path")


# --- (9c) S3: verify_sealed_snapshot threads one manifest read through both checks -


def test_verify_sealed_snapshot_reads_manifest_bytes_exactly_once(
    world: World, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest file bytes are read once and threaded (``manifest_bytes=``) into
    both ``assert_sealed_tree`` and ``assert_supplied_manifest_is_the_sealed_one``,
    closing the two-separate-opens window a sealed->other->sealed swap could split
    between them. A deterministic race is not attempted; this guards the invariant
    the fix establishes directly: exactly one ``Path.read_bytes()`` call against the
    manifest file per ``verify_sealed_snapshot`` call (``rehash_tree``'s own re-read
    of the manifest as part of the whole-tree digest is fd/os-level, not
    ``Path.read_bytes``, and is separately self-checking against the tree digest)."""
    manifest_path = world.snapshot_dir / sealing.MANIFEST_NAME
    real_read_bytes = Path.read_bytes
    calls: list[int] = []

    def _spy(self: Path, *a: Any, **kw: Any) -> bytes:
        if self == manifest_path:
            calls.append(1)
        return real_read_bytes(self, *a, **kw)

    monkeypatch.setattr(Path, "read_bytes", _spy, raising=True)
    sealing.verify_sealed_snapshot(world.snapshot_dir, world.manifest)
    assert len(calls) == 1, (
        f"expected exactly one manifest read inside verify_sealed_snapshot (S3 "
        f"threading), got {len(calls)} -- a re-split reopens the sealed->other->"
        "sealed swap window between the two manifest-binding checks"
    )
