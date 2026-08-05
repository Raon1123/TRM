"""Low-overhead structured live evidence for CLEAN-R1 runs.

The logger is inert unless explicitly enabled or the resolved run name starts
with ``cleanr1_``.  It does not perform model work.  Callers supply the
already-reduced scalar metrics, a boundary-only gradient norm, and the
checkpoint that was just durably saved.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = 1
CONTRACT_ID = "CLEAN-R1-LIVE-v1"
SIDECAR_NAME = "clean_r1_live.jsonl"
MONITOR_METRIC_ID = "monitor-test-exact"
MONITOR_METRIC_PATH = "probe/test_exact"
MONITOR_PROBE_SIZE = 512
SEALED_METRIC_ID = "sealed-test-exact"
SEALED_METRIC_PATH = "sealed/terminal_ema/exact_sequence_accuracy_all_examples"

REQUIRED_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "event_type",
        "run_name",
        "project_name",
        "seed",
        "k",
        "data_paths",
        "data_identity",
        "config_identity_sha256",
        "step",
        "eval_index",
        "wall_clock_seconds",
        "loss",
        "loss_source",
        "nan_inf_flag",
        "grad_norm",
        "checkpoint_ref",
        "checkpoint_sha256",
        "checkpoint_durable_ref",
        "checkpoint_role",
        "ema_enabled",
        "ema_rate",
        "monitor_metric_id",
        "monitor_metric_path",
        "probe_identity",
        "sealed_metric_id",
        "sealed_metric_path",
        "instrumentation_enablement",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def instrumentation_enablement(
    *, run_name: str | None, config_enabled: bool = False, environ: Mapping[str, str] | None = None
) -> str | None:
    environment = os.environ if environ is None else environ
    if config_enabled:
        return "config:clean_r1_live_log"
    if environment.get("CLEAN_R1_LIVE_LOG") == "1":
        return "env:CLEAN_R1_LIVE_LOG=1"
    if (run_name or "").startswith("cleanr1_"):
        return "run_name_prefix:cleanr1_"
    return None


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _probe_identity(data_path: Path) -> dict[str, Any]:
    inputs_path = data_path / "test" / "all__inputs.npy"
    inputs = np.load(inputs_path, mmap_mode="r")
    if len(inputs) < MONITOR_PROBE_SIZE:
        raise ValueError(
            f"{inputs_path} has {len(inputs)} examples; CLEAN-R1 requires the first "
            f"{MONITOR_PROBE_SIZE}"
        )
    first = np.asarray(inputs[:MONITOR_PROBE_SIZE], dtype=np.int32)
    return {
        "split": "test",
        "n_examples": MONITOR_PROBE_SIZE,
        "z_logging_input_md5_8": hashlib.md5(first.tobytes()).hexdigest()[:8],
        "semantic_sha256": _semantic_array_sha256(first),
        "inputs_file_sha256": sha256_file(inputs_path),
    }


def _data_identity(data_paths: list[str]) -> dict[str, Any]:
    records = []
    for raw_path in data_paths:
        path = Path(raw_path)
        metadata = path / "train" / "dataset.json"
        inputs = path / "train" / "all__inputs.npy"
        if not metadata.is_file() or not inputs.is_file():
            raise FileNotFoundError(f"CLEAN-R1 data identity files missing under {path}")
        records.append(
            {
                "path": str(path.resolve()),
                "train_dataset_json_sha256": sha256_file(metadata),
                "train_inputs_sha256": sha256_file(inputs),
            }
        )
    return {"paths": records, "identity_sha256": _canonical_json_sha256(records)}


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def _flatten_numeric(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            result.extend(_flatten_numeric(item, name))
    elif isinstance(value, (int, float, np.number)):
        result.append((prefix, float(value)))
    return result


def _select_loss(metrics: Mapping[str, Any]) -> tuple[float | None, str | None]:
    candidates = (
        "train/lm_loss",
        "all/lm_loss",
        "test/lm_loss",
        "train/q_halt_loss",
    )
    flattened = dict(_flatten_numeric(metrics))
    for key in candidates:
        if key in flattened:
            return flattened[key], key
    for key, value in flattened.items():
        if key.endswith("loss"):
            return value, key
    return None, None


class CleanR1LiveLogger:
    """Stateful JSONL writer. Normal records are emitted only at eval boundaries."""

    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        resolved_config: Mapping[str, Any],
        eval_cadence_steps: int,
        enablement: str,
    ):
        if eval_cadence_steps < 1:
            raise ValueError("CLEAN-R1 eval cadence in steps must be positive")
        self.path = checkpoint_dir / SIDECAR_NAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._config = dict(resolved_config)
        self._eval_cadence_steps = eval_cadence_steps
        self._enablement = enablement
        self._started = time.monotonic()
        self._last_wall = -1.0
        self._last_loss: float | None = None
        self._last_loss_source: str | None = None
        self._boundary_grad_norm: dict[int, float] = {}
        self._data_paths = [str(path) for path in self._config.get("data_paths", [])]
        if not self._data_paths:
            raise ValueError("CLEAN-R1 structured logging requires data_paths")
        self._data_identity = _data_identity(self._data_paths)
        self._probe_identity = _probe_identity(Path(self._data_paths[0]))
        self._config_identity = _canonical_json_sha256(self._config)

    @property
    def probe_identity(self) -> dict[str, Any]:
        return dict(self._probe_identity)

    def should_capture_grad_norm(self, step: int) -> bool:
        return step > 0 and step % self._eval_cadence_steps == 0

    def observe_train_metrics(
        self, *, step: int, metrics: Mapping[str, Any], grad_norm: float | None
    ) -> None:
        numeric = _flatten_numeric(metrics)
        nonfinite = [(key, value) for key, value in numeric if not math.isfinite(value)]
        loss, loss_source = _select_loss(metrics)
        if loss is not None:
            self._last_loss = loss
            self._last_loss_source = loss_source
        if grad_norm is not None:
            if not math.isfinite(float(grad_norm)) or float(grad_norm) < 0:
                nonfinite.append(("grad_norm", float(grad_norm)))
            else:
                self._boundary_grad_norm[step] = float(grad_norm)
        if nonfinite:
            self._emit(
                {
                    **self._base_record(step=step, eval_index=None),
                    "event_type": "nonfinite_abort",
                    "loss": _finite_float(loss),
                    "loss_source": loss_source,
                    "nan_inf_flag": True,
                    "grad_norm": _finite_float(grad_norm),
                    "checkpoint_ref": None,
                    "checkpoint_sha256": None,
                    "checkpoint_durable_ref": None,
                    "checkpoint_role": None,
                    "ema_enabled": bool(self._config.get("ema")),
                    "ema_rate": self._config.get("ema_rate"),
                    "nonfinite_fields": [
                        {"field": key, "value": repr(value)} for key, value in nonfinite
                    ],
                }
            )
            raise FloatingPointError(f"CLEAN-R1 non-finite training metric at step {step}: {nonfinite}")

    def emit_eval_checkpoint(
        self,
        *,
        step: int,
        eval_index: int,
        eval_metrics: Mapping[str, Any] | None,
        checkpoint_path: Path,
        checkpoint_role: str,
        ema_enabled: bool,
        ema_rate: float,
    ) -> dict[str, Any]:
        if checkpoint_role not in {"nonterminal", "terminal"}:
            raise ValueError(f"invalid checkpoint role {checkpoint_role!r}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"CLEAN-R1 eval checkpoint absent: {checkpoint_path}")
        metrics = {} if eval_metrics is None else eval_metrics
        numeric = _flatten_numeric(metrics)
        nonfinite = [(key, value) for key, value in numeric if not math.isfinite(value)]
        eval_loss, eval_loss_source = _select_loss(metrics)
        loss = eval_loss if eval_loss is not None else self._last_loss
        loss_source = eval_loss_source if eval_loss_source is not None else self._last_loss_source
        grad_norm = self._boundary_grad_norm.pop(step, None)
        if loss is None or loss_source is None:
            nonfinite.append(("loss", float("nan")))
        if grad_norm is None:
            nonfinite.append(("grad_norm", float("nan")))
        if not ema_enabled:
            nonfinite.append(("ema_enabled", float("nan")))
        stat = checkpoint_path.stat()
        durable_ref = {
            "path": str(checkpoint_path.resolve()),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        # Hash the first checkpoint (the live-canary artifact) exactly. Later
        # events retain a durable stat binding without repeatedly rereading a
        # large checkpoint.
        checkpoint_sha256 = sha256_file(checkpoint_path) if eval_index == 1 else None
        record = {
            **self._base_record(step=step, eval_index=eval_index),
            "event_type": "eval_checkpoint",
            "loss": _finite_float(loss),
            "loss_source": loss_source,
            "nan_inf_flag": bool(nonfinite),
            "grad_norm": _finite_float(grad_norm),
            "checkpoint_ref": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_durable_ref": durable_ref,
            "checkpoint_role": checkpoint_role,
            "ema_enabled": ema_enabled,
            "ema_rate": ema_rate,
        }
        if nonfinite:
            record["nonfinite_fields"] = [
                {"field": key, "value": repr(value)} for key, value in nonfinite
            ]
        self._emit(record)
        if nonfinite:
            raise FloatingPointError(
                f"CLEAN-R1 non-finite/incomplete eval evidence at step {step}: {nonfinite}"
            )
        return record

    def _base_record(self, *, step: int, eval_index: int | None) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "contract_id": CONTRACT_ID,
            "run_name": self._config.get("run_name"),
            "project_name": self._config.get("project_name"),
            "seed": self._config.get("seed"),
            "k": self._config.get("k"),
            "data_paths": list(self._data_paths),
            "data_identity": self._data_identity,
            "config_identity_sha256": self._config_identity,
            "step": step,
            "eval_index": eval_index,
            "wall_clock_seconds": self._wall_clock_seconds(),
            "monitor_metric_id": MONITOR_METRIC_ID,
            "monitor_metric_path": MONITOR_METRIC_PATH,
            "probe_identity": self._probe_identity,
            "sealed_metric_id": SEALED_METRIC_ID,
            "sealed_metric_path": SEALED_METRIC_PATH,
            "instrumentation_enablement": self._enablement,
        }

    def _wall_clock_seconds(self) -> float:
        elapsed = max(0.0, time.monotonic() - self._started)
        if elapsed < self._last_wall:
            raise RuntimeError("monotonic clock regressed")
        self._last_wall = elapsed
        return elapsed

    def _emit(self, record: Mapping[str, Any]) -> None:
        missing = REQUIRED_EVENT_FIELDS - record.keys()
        if missing:
            raise ValueError(f"CLEAN-R1 event missing required fields: {sorted(missing)}")
        rendered = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
