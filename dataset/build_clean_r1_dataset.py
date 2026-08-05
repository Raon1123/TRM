#!/usr/bin/env python3
"""Construct the isolated CLEAN-R1 sigma^k data contract.

The ordinary ``data/sigma_k_10/<k>`` trees are D0.  This builder leaves them
unchanged and creates two new trees:

* ``sigma_k_10_clean_r1_d1/<k>``: D1 ``train`` (5,000), monitoring ``test``
  (1,000), and a non-training ``sealed`` split (1,000).
* ``sigma_k_10_clean_r1_d0_sealed/<k>``: a fresh D0 ``sealed`` split (1,000).

Every new permutation is excluded from *all* existing D0 train/test examples
for that k.  The fresh D0 seal is additionally excluded from every D1 split.
Thus the complete D0 family and D1 family are structurally disjoint within k.
``analysis/clean_r1_data_audit.py`` independently checks that claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# The legacy generator is intentionally reused for the task definition and
# Format-D encoding; adding this directory makes its historic ``from common``
# import work both as a script and as a module imported by tests.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from build_sigma_k_dataset import (  # noqa: E402
    MAX_N,
    SEQ_LEN,
    VOCAB_SIZE,
    apply_sigma_k,
    make_example,
    perm_order,
)
from common import PuzzleDatasetMetadata  # noqa: E402


DEFAULT_K_VALUES = (4, 5, 6, 7, 8, 10)
ARRAY_FIELDS = (
    "inputs",
    "labels",
    "puzzle_identifiers",
    "puzzle_indices",
    "group_indices",
)
MANIFEST_NAME = "clean_r1_manifest.json"
NON_USE_CONTRACT_NAME = "SEALED_NON_USE_CONTRACT.json"


@dataclass(frozen=True)
class BuildSpec:
    """All material inputs to a reproducible CLEAN-R1 construction."""

    existing_d0_root: Path = Path("data/sigma_k_10")
    d1_root: Path = Path("data/sigma_k_10_clean_r1_d1")
    d0_sealed_root: Path = Path("data/sigma_k_10_clean_r1_d0_sealed")
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    n: int = 10
    train_size: int = 5_000
    monitor_size: int = 1_000
    sealed_size: int = 1_000
    seed: int = 20_260_729


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash dtype, shape, and C-order values (not a NumPy-header accident)."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _array_path(root: Path, k: int, split: str, field: str) -> Path:
    return root / str(k) / split / f"all__{field}.npy"


def load_split_arrays(root: Path, k: int, split: str) -> dict[str, np.ndarray]:
    """Read a Format-D split and fail loudly if its standard fields are absent."""
    split_dir = root / str(k) / split
    if not (split_dir / "dataset.json").is_file():
        raise FileNotFoundError(f"missing dataset metadata: {split_dir / 'dataset.json'}")
    arrays: dict[str, np.ndarray] = {}
    for field in ARRAY_FIELDS:
        path = _array_path(root, k, split, field)
        if not path.is_file():
            raise FileNotFoundError(f"missing array: {path}")
        arrays[field] = np.load(path, mmap_mode="r")
    return arrays


def split_hashes(root: Path, k: int, split: str) -> dict[str, Any]:
    """Return file and semantic hashes for every array consumed/generated."""
    arrays = load_split_arrays(root, k, split)
    records: dict[str, Any] = {}
    for field, array in arrays.items():
        path = _array_path(root, k, split, field)
        records[field] = {
            "path": str(path),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256_array": sha256_array(array),
            "sha256_file": sha256_file(path),
        }
    meta_path = root / str(k) / split / "dataset.json"
    return {
        "arrays": records,
        "dataset_json": {"path": str(meta_path), "sha256_file": sha256_file(meta_path)},
    }


def permutation_key(sigma: np.ndarray) -> bytes:
    sigma = np.asarray(sigma)
    if sigma.ndim != 1:
        raise ValueError(f"expected a one-dimensional permutation, got {sigma.shape}")
    return np.ascontiguousarray(sigma, dtype=np.uint8).tobytes()


def sigmas_from_inputs(inputs: np.ndarray, n: int) -> list[np.ndarray]:
    """Decode the 1-indexed sigma encoding and validate it is a permutation."""
    if inputs.ndim != 2 or inputs.shape[1] != SEQ_LEN:
        raise ValueError(f"inputs must have shape [N, {SEQ_LEN}], got {inputs.shape}")
    sigmas: list[np.ndarray] = []
    expected = np.arange(n, dtype=np.int64)
    for row_index, row in enumerate(inputs):
        sigma = np.asarray(row[:n], dtype=np.int64) - 1
        if not np.array_equal(np.sort(sigma), expected):
            raise ValueError(f"row {row_index} is not an n={n} permutation")
        if not np.all(row[n:] == 0):
            raise ValueError(f"row {row_index} has non-PAD input suffix")
        sigmas.append(sigma)
    return sigmas


def sample_unique_permutations(
    *, n: int, total: int, rng: np.random.Generator, seen: set[bytes], k: int
) -> list[np.ndarray]:
    """Sample unique σ with ord(σ)>k, respecting all caller supplied exclusions."""
    result: list[np.ndarray] = []
    while len(result) < total:
        sigma = rng.permutation(n)
        if perm_order(sigma) <= k:
            continue
        key = permutation_key(sigma)
        if key in seen:
            continue
        seen.add(key)
        result.append(sigma.copy())
    return result


def seed_for(seed: int, k: int, namespace: int) -> int:
    """Stable per-k/per-family seed, explicitly derived from the public seed."""
    sequence = np.random.SeedSequence([seed, k, namespace])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _metadata(n_examples: int) -> PuzzleDatasetMetadata:
    return PuzzleDatasetMetadata(
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=n_examples,
        mean_puzzle_examples=1.0,
        total_puzzles=n_examples,
        sets=["all"],
    )


def write_split(root: Path, k: int, split: str, sigmas: Iterable[np.ndarray], n: int) -> None:
    """Write one ordinary Format-D split; no training reader changes are needed."""
    sigma_list = list(sigmas)
    if not sigma_list:
        raise ValueError("refusing to write an empty split")
    inputs, labels = zip(*(make_example(n, k, sigma) for sigma in sigma_list), strict=True)
    inputs_array = np.stack(inputs).astype(np.int32, copy=False)
    labels_array = np.stack(labels).astype(np.int32, copy=False)
    count = len(sigma_list)
    arrays = {
        "inputs": inputs_array,
        "labels": labels_array,
        "puzzle_identifiers": np.zeros(count, dtype=np.int32),
        "puzzle_indices": np.arange(count + 1, dtype=np.int32),
        "group_indices": np.arange(count + 1, dtype=np.int32),
    }
    split_dir = root / str(k) / split
    split_dir.mkdir(parents=True, exist_ok=False)
    (split_dir / "dataset.json").write_text(
        json.dumps(_metadata(count).model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for field, array in arrays.items():
        np.save(split_dir / f"all__{field}.npy", array)


def _existing_d0_sigmas(spec: BuildSpec, k: int) -> tuple[set[bytes], dict[str, Any]]:
    keys: set[bytes] = set()
    hashes: dict[str, Any] = {}
    for split in ("train", "test"):
        arrays = load_split_arrays(spec.existing_d0_root, k, split)
        sigmas = sigmas_from_inputs(arrays["inputs"], spec.n)
        before = len(keys)
        keys.update(permutation_key(sigma) for sigma in sigmas)
        if len(keys) - before != len(sigmas):
            raise ValueError(f"existing D0 k={k}/{split} contains an overlap")
        hashes[split] = split_hashes(spec.existing_d0_root, k, split)
    return keys, hashes


def sealed_non_use_contract(spec: BuildSpec) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract": "CLEAN-R1 sealed non-use",
        "rules": [
            "D1/sealed and D0-sealed are never training or monitoring inputs.",
            "Training receives only D1/train; monitoring receives only D1/test.",
            "A sealed result must use analysis/clean_r1_sealed_eval.py with an explicit --sealed-path and --sealed-split.",
            "The sealed evaluator must receive a terminal checkpoint whose all_config.yaml records ema: true.",
            "Any prior access, configuration, or tuning against either sealed split invalidates its confirmatory use.",
        ],
        "d1_train_path_pattern": str(spec.d1_root / "<k>" / "train"),
        "d1_monitor_path_pattern": str(spec.d1_root / "<k>" / "test"),
        "d1_sealed_path_pattern": str(spec.d1_root / "<k>" / "sealed"),
        "d0_sealed_path_pattern": str(spec.d0_sealed_root / "<k>" / "sealed"),
    }


def _assert_target_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing CLEAN-R1 root: {path}; choose new roots or remove it deliberately"
        )


def build_clean_r1(spec: BuildSpec, *, dry_run: bool = False) -> dict[str, Any]:
    """Build D1 and the new D0 sealed family, returning the durable manifest."""
    if spec.n != MAX_N:
        raise ValueError(f"CLEAN-R1 is fixed to n={MAX_N}, got n={spec.n}")
    if not spec.k_values or any(k < 1 for k in spec.k_values):
        raise ValueError(f"invalid k values: {spec.k_values}")
    if len(set(spec.k_values)) != len(spec.k_values):
        raise ValueError(f"duplicate k values: {spec.k_values}")
    if min(spec.train_size, spec.monitor_size, spec.sealed_size) < 1:
        raise ValueError("all split sizes must be positive")
    if dry_run:
        # A dry run intentionally reads the source arrays and therefore catches
        # wrong source paths before anyone creates an irreversible data root.
        consumed = {str(k): _existing_d0_sigmas(spec, k)[1] for k in spec.k_values}
        return {
            "dry_run": True,
            "spec": _jsonable_spec(spec),
            "consumed_d0_hashes": consumed,
            "would_create": [str(spec.d1_root), str(spec.d0_sealed_root)],
        }

    _assert_target_absent(spec.d1_root)
    _assert_target_absent(spec.d0_sealed_root)
    spec.d1_root.mkdir(parents=True, exist_ok=False)
    spec.d0_sealed_root.mkdir(parents=True, exist_ok=False)
    for root in (spec.d1_root, spec.d0_sealed_root):
        (root / "identifiers.json").write_text('["<blank>"]\n', encoding="utf-8")
        (root / NON_USE_CONTRACT_NAME).write_text(
            json.dumps(sealed_non_use_contract(spec), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    per_k: dict[str, Any] = {}
    for k in spec.k_values:
        existing_d0_keys, consumed_hashes = _existing_d0_sigmas(spec, k)
        # D1 is sampled first after excluding all pre-existing D0.  D0 sealed
        # then excludes both of those sets, giving a bidirectional structural
        # D0<->D1 separation without relying on post-hoc luck.
        d1_seen = set(existing_d0_keys)
        d1_rng_seed = seed_for(spec.seed, k, 1)
        d1_sigmas = sample_unique_permutations(
            n=spec.n,
            total=spec.train_size + spec.monitor_size + spec.sealed_size,
            rng=np.random.default_rng(d1_rng_seed),
            seen=d1_seen,
            k=k,
        )
        d1_train = d1_sigmas[:spec.train_size]
        d1_test = d1_sigmas[spec.train_size : spec.train_size + spec.monitor_size]
        d1_sealed = d1_sigmas[spec.train_size + spec.monitor_size :]
        write_split(spec.d1_root, k, "train", d1_train, spec.n)
        write_split(spec.d1_root, k, "test", d1_test, spec.n)
        write_split(spec.d1_root, k, "sealed", d1_sealed, spec.n)

        d0_seen = set(d1_seen)  # existing D0 + every D1 permutation
        d0_rng_seed = seed_for(spec.seed, k, 2)
        d0_sealed = sample_unique_permutations(
            n=spec.n,
            total=spec.sealed_size,
            rng=np.random.default_rng(d0_rng_seed),
            seen=d0_seen,
            k=k,
        )
        write_split(spec.d0_sealed_root, k, "sealed", d0_sealed, spec.n)

        per_k[str(k)] = {
            "order_filter": "ord(sigma) > k",
            "derived_rng_seeds": {"d1": d1_rng_seed, "d0_sealed": d0_rng_seed},
            "counts": {
                "existing_d0_train_test": len(existing_d0_keys),
                "d1_train": len(d1_train),
                "d1_monitor_test": len(d1_test),
                "d1_sealed": len(d1_sealed),
                "d0_sealed": len(d0_sealed),
            },
            "consumed_existing_d0": consumed_hashes,
            "generated_d1": {
                split: split_hashes(spec.d1_root, k, split)
                for split in ("train", "test", "sealed")
            },
            "generated_d0_sealed": split_hashes(spec.d0_sealed_root, k, "sealed"),
        }

    manifest = {
        "schema_version": 1,
        "protocol": "CLEAN-R1-G0",
        "generated_at_utc": utc_now(),
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256_file": sha256_file(Path(__file__).resolve()),
            "legacy_generator_path": str((THIS_DIR / "build_sigma_k_dataset.py").resolve()),
            "legacy_generator_sha256_file": sha256_file(THIS_DIR / "build_sigma_k_dataset.py"),
        },
        "spec": _jsonable_spec(spec),
        "layout": {
            "d0_existing": str(spec.existing_d0_root),
            "d1": str(spec.d1_root),
            "d0_sealed": str(spec.d0_sealed_root),
            "monitor_split_name": "test",
            "sealed_split_name": "sealed",
        },
        "sealed_non_use_contract": sealed_non_use_contract(spec),
        "per_k": per_k,
    }
    manifest_path = spec.d1_root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    d0_pointer = {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "manifest_sha256_file": sha256_file(manifest_path),
        "sealed_non_use_contract": sealed_non_use_contract(spec),
    }
    (spec.d0_sealed_root / MANIFEST_NAME).write_text(
        json.dumps(d0_pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _jsonable_spec(spec: BuildSpec) -> dict[str, Any]:
    result = asdict(spec)
    return {
        key: (str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value)
        for key, value in result.items()
    }


def parse_k_values(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--k-values must be comma-separated integers") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-d0-root", type=Path, default=Path("data/sigma_k_10"))
    parser.add_argument("--d1-root", type=Path, default=Path("data/sigma_k_10_clean_r1_d1"))
    parser.add_argument("--d0-sealed-root", type=Path, default=Path("data/sigma_k_10_clean_r1_d0_sealed"))
    parser.add_argument("--k-values", type=parse_k_values, default=DEFAULT_K_VALUES)
    parser.add_argument("--seed", type=int, default=20_260_729)
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument("--monitor-size", type=int, default=1_000)
    parser.add_argument("--sealed-size", type=int, default=1_000)
    parser.add_argument("--dry-run", action="store_true", help="hash D0 inputs and print the planned roots; write nothing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = BuildSpec(
        existing_d0_root=args.existing_d0_root,
        d1_root=args.d1_root,
        d0_sealed_root=args.d0_sealed_root,
        k_values=args.k_values,
        train_size=args.train_size,
        monitor_size=args.monitor_size,
        sealed_size=args.sealed_size,
        seed=args.seed,
    )
    result = build_clean_r1(spec, dry_run=args.dry_run)
    print(json.dumps(result if args.dry_run else {
        "status": "built",
        "manifest": str(spec.d1_root / MANIFEST_NAME),
        "d0_sealed_manifest": str(spec.d0_sealed_root / MANIFEST_NAME),
        "k_values": list(spec.k_values),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
