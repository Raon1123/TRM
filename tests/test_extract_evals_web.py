"""Mock-only tests for the W&B Public API extractor."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import time

import pytest


MODULE_PATH = Path(__file__).parents[1] / "lab/monitoring/dashboard/extract_evals_web.py"
SPEC = importlib.util.spec_from_file_location("extract_evals_web", MODULE_PATH)
web = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(web)


class FakeRun:
    def __init__(self, *, name="fig1_mlp_noz_noiter_k3_s1", run_id="abc", state="finished", config=None,
                 rows=None, created_at="2026-07-21T00:00:00Z", history_error=None):
        self.name, self.id, self.state = name, run_id, state
        self.path = ["raon1123", "Sigma_k_new", run_id]
        self.created_at = created_at
        self.updated_at = created_at
        self.lastHistoryStep = 244100
        self.args = {"project_name": "Sigma_k_new"}
        self.config = config if config is not None else {"project_name": "Sigma_k_new", "run_name": name, "k": 3, "seed": 1, "data_paths": "data/sigma_k_10/3"}
        self.rows, self.history_error, self.scan_calls = rows if rows is not None else full_rows(), history_error, 0

    def scan_history(self, *, keys, page_size):
        self.scan_calls += 1
        assert keys == ["_step", "probe/train_exact", "probe/test_exact"]
        assert page_size == 25000
        if self.history_error:
            raise self.history_error
        return iter(self.rows)


class FakeApi:
    def __init__(self, runs, error=None): self._runs, self.error, self.calls = runs, error, 0
    def runs(self, path, *, order, per_page, filters=None):
        self.calls += 1
        assert path == "raon1123/Sigma_k_new" and order == "+created_at" and per_page == 50
        if self.error:
            error, self.error = self.error, None
            raise error
        return iter(self._runs)


def full_rows():
    return [{"_step": 4882 * (index + 1), "probe/train_exact": 1.0, "probe/test_exact": 0.5} for index in range(50)]


def extract(api, **kwargs):
    return web.extract(api, entity="raon1123", project="Sigma_k_new", sleep=lambda _: None, rand=lambda: 0.0, **kwargs)


def test_finished_snapshot_and_pagination_contract():
    payload = extract(FakeApi([FakeRun()]))
    run = payload["runs"]["fig1_mlp_noz_noiter_k3_s1"]
    assert run["history_status"] == "verified" and len(run["rows"]) == 50
    assert run["dir"] == "wandb://raon1123/Sigma_k_new/abc"


def test_run_name_uses_server_display_name_filter_and_keeps_retry_candidates():
    class FilteringApi(FakeApi):
        def __init__(self, runs): super().__init__(runs); self.filters = []
        def runs(self, path, *, order, per_page, filters=None):
            self.filters.append(filters)
            return super().runs(path, order=order, per_page=per_page, filters=filters)
    api = FilteringApi([FakeRun(run_id="old", state="crashed"), FakeRun(run_id="new")])
    payload = extract(api, run_names={"fig1_mlp_noz_noiter_k3_s1"})
    assert api.filters == [{"display_name": "fig1_mlp_noz_noiter_k3_s1"}]
    assert payload["runs"]["fig1_mlp_noz_noiter_k3_s1"]["candidate_total"] == 2


def test_crash_retry_selects_only_valid_nonfailed():
    crashed = FakeRun(run_id="old", state="crashed")
    retry = FakeRun(run_id="new")
    payload = extract(FakeApi([crashed, retry]))
    run = payload["runs"][retry.name]
    assert run["id"] == "new" and run["n_candidates"] == 2
    assert run["selection_status"] == "unique_valid_nonfailed_retry"
    assert crashed.scan_calls == 0


def test_two_valid_candidates_are_quarantined():
    a, b = FakeRun(run_id="a"), FakeRun(run_id="b")
    payload = extract(FakeApi([a, b]))
    assert payload["runs"] == {} and a.scan_calls == b.scan_calls == 0
    with pytest.raises(web.StrictDuplicateError): extract(FakeApi([a, b]), strict_duplicates=True)


def test_config_mismatch_is_quarantined():
    bad = FakeRun(config={"project_name": "wrong", "run_name": "fig1_mlp_noz_noiter_k3_s1", "k": 3, "seed": 1, "data_paths": "data/sigma_k_10/3"})
    payload = extract(FakeApi([bad]))
    assert payload["runs"] == {}
    assert payload["candidate_ledger"][bad.name][0]["config_valid"] is False


def test_allowlist_prevents_secret_args_or_config_from_reaching_serialized_output():
    sentinel = "NEVER_SERIALIZE_THIS_SECRET_7bb8"
    run = FakeRun(config={"project_name": "Sigma_k_new", "run_name": "fig1_mlp_noz_noiter_k3_s1", "k": 3,
                          "seed": 1, "data_paths": "data/sigma_k_10/3", "api_key": sentinel,
                          "nested": {"token": sentinel}})
    run.args = ["--api-key", sentinel, f"token={sentinel}", f"Authorization: Bearer {sentinel}"]
    encoded = json.dumps(extract(FakeApi([run])), sort_keys=True)
    record = next(iter(json.loads(encoded)["candidate_ledger"].values()))[0]
    assert sentinel not in encoded
    assert set(record["config"]) == {"project_name", "run_name", "k", "seed", "data_paths"}
    assert record["args"] == record["config"]
    assert web._redact(run.args) == ["<redacted>", "<redacted>", "<redacted>", "<redacted>"]


@pytest.mark.parametrize("rows", [
    [{"_step": 2, "probe/train_exact": 1}, {"_step": 1, "probe/test_exact": 1}],
    [{"_step": 1, "probe/train_exact": math.nan}],
    [{"_step": 1}],
])
def test_invalid_or_missing_or_nonterminal_history_is_metric_unverified(rows):
    payload = extract(FakeApi([FakeRun(rows=rows)]))
    run = next(iter(payload["runs"].values()))
    assert run["history_status"] == "metric_unverified" and run["rows"] in ([], [[1, None, None]])


def test_retry_success_and_exhaustion_does_not_publish_existing(tmp_path):
    transient = type("Transient", (Exception,), {"status_code": 503})
    api = FakeApi([FakeRun()], error=transient())
    assert extract(api)["fetch_stats"]["retry_count"] == 1
    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="utf-8")
    failed_api = FakeApi([], error=transient())
    original_make_api = web._make_api
    web._make_api = lambda timeout: failed_api
    try:
        assert web.main(["--output", str(output), "--max-retries", "0"]) == 2
    finally:
        web._make_api = original_make_api
    assert output.read_text(encoding="utf-8") == "preserve"


def test_dry_run_never_scans_or_writes(tmp_path, monkeypatch, capsys):
    run = FakeRun()
    monkeypatch.setattr(web, "_make_api", lambda timeout: FakeApi([run]))
    output = tmp_path / "no-write.json"
    assert web.main(["--dry-run", "--output", str(output)]) == 0
    assert not output.exists() and run.scan_calls == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True


def test_atomic_publish_replaces_only_after_complete_write(tmp_path):
    output = tmp_path / "payload.json"
    output.write_text("old", encoding="utf-8")
    web.atomic_publish({"ok": True}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}


def test_deadline_interrupts_blocking_sdk_call_and_preserves_existing_output(tmp_path, monkeypatch):
    class BlockingApi:
        def runs(self, path, **kwargs):
            time.sleep(0.2)
            return iter([])
    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(web, "_make_api", lambda timeout: BlockingApi())
    started = time.monotonic()
    assert web.main(["--output", str(output), "--total-timeout", "0.02", "--max-retries", "0"]) == 2
    assert time.monotonic() - started < 0.15
    assert output.read_text(encoding="utf-8") == "preserve"


def test_nonofficial_wandb_base_url_is_rejected(monkeypatch):
    monkeypatch.setenv("WANDB_BASE_URL", "https://example.invalid")
    with pytest.raises(web.FetchError):
        web._make_api(1)
