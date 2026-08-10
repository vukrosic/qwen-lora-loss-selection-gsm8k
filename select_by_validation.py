#!/usr/bin/env python3
"""Seal one endpoint choice from a common validation metric before test access."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


METRIC = "token_weighted_nll"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_validation(path: Path) -> dict:
    record = json.loads(path.read_text())
    if record.get("split") != "valid":
        raise ValueError(f"Expected validation evidence, got {record.get('split')}: {path}")
    value = record.get("summary", {}).get(METRIC)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric {METRIC}: {path}")
    return record


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--token-validation", type=Path, required=True)
    parser.add_argument("--example-validation", type=Path, required=True)
    parser.add_argument("--forbid-test-path", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite sealed selection: {args.output}")
    existing_test = [str(path) for path in args.forbid_test_path if path.exists()]
    if existing_test:
        raise RuntimeError(f"Test evidence already exists before selection: {existing_test}")

    records = {
        "token": load_validation(args.token_validation),
        "example": load_validation(args.example_validation),
    }
    for condition, record in records.items():
        if record.get("seed") != args.seed or record.get("condition") != condition:
            raise ValueError(f"Validation identity mismatch for {condition}")
    split_hashes = {record.get("split_sha256") for record in records.values()}
    if len(split_hashes) != 1 or None in split_hashes:
        raise ValueError("Validation arms do not share one explicit split hash")
    protocol_hashes = {record.get("protocol_sha256") for record in records.values()}
    lock_hashes = {record.get("data_lock_sha256") for record in records.values()}
    if len(protocol_hashes) != 1 or None in protocol_hashes:
        raise ValueError("Validation arms do not share one protocol hash")
    if len(lock_hashes) != 1 or None in lock_hashes:
        raise ValueError("Validation arms do not share one data-lock hash")
    values = {condition: record["summary"][METRIC] for condition, record in records.items()}
    selected = "token" if values["token"] <= values["example"] else "example"
    output = {
        "seed": args.seed,
        "rule": f"lower common validation {METRIC}; exact tie chooses token",
        "metric": METRIC,
        "validation_metric": values,
        "selected_condition": selected,
        "test_evaluation_status_at_seal": "NOT_STARTED",
        "sealed_at": timestamp(),
        "split_sha256": next(iter(split_hashes)),
        "protocol_sha256": next(iter(protocol_hashes)),
        "data_lock_sha256": next(iter(lock_hashes)),
        "adapter_sha256": {
            condition: record["fingerprints"]["adapters"]
            for condition, record in records.items()
        },
        "fingerprints": {
            condition: sha256(path)
            for condition, path in {
                "token_validation": args.token_validation,
                "example_validation": args.example_validation,
            }.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
