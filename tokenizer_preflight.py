#!/usr/bin/env python3
"""Tokenizer-only GSM8K length audit and immutable dataset lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from natural_data import NaturalInstructionDataset


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentiles(values: list[int]) -> dict:
    return {
        name: float(np.percentile(values, quantile))
        for name, quantile in (("p10", 10), ("p50", 50), ("p90", 90))
    }


def inspect_split(path: Path, tokenizer, max_length: int) -> dict:
    dataset = NaturalInstructionDataset(path, tokenizer)
    lengths = []
    completions = []
    by_stratum = {"short": [], "long": []}
    over = []
    for index in range(len(dataset)):
        tokens, offset, stratum, record_id = dataset.processed(index)
        length = len(tokens)
        completion = length - offset
        lengths.append(length)
        completions.append(completion)
        by_stratum[stratum].append(completion)
        if length > max_length:
            over.append({"id": record_id, "tokens": length})
    return {
        "records": len(dataset),
        "sequence_tokens": {"min": min(lengths), "max": max(lengths), **percentiles(lengths)},
        "completion_tokens": {
            "min": min(completions),
            "max": max(completions),
            **percentiles(completions),
        },
        "completion_tokens_by_stratum": {
            name: {"records": len(values), **percentiles(values)}
            for name, values in by_stratum.items()
        },
        "over_max_records": len(over),
        "over_max_fraction": len(over) / len(dataset),
        "over_max_examples": sorted(over, key=lambda row: row["tokens"], reverse=True)[:20],
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset lock: {args.output}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    tokenizer_hashes = {
        name: sha256(args.model / name)
        for name in TOKENIZER_FILES
        if (args.model / name).exists()
    }
    splits = {
        split: inspect_split(
            args.data / f"{split}.jsonl", tokenizer, args.max_seq_length
        )
        for split in ("train", "valid", "test")
    }
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ratio = (
        splits["test"]["completion_tokens"]["p90"]
        / splits["test"]["completion_tokens"]["p10"]
    )
    gates = {
        "manifest_materialized": manifest.get("status") == "MATERIALIZED",
        "counts_match_frozen_design": [
            splits[name]["records"] for name in ("train", "valid", "test")
        ]
        == [1000, 256, 512],
        "each_split_balanced": all(
            details["completion_tokens_by_stratum"]["short"]["records"]
            == details["completion_tokens_by_stratum"]["long"]["records"]
            for details in splits.values()
        ),
        "no_records_over_max_each_split": all(
            details["over_max_records"] == 0 for details in splits.values()
        ),
        "test_length_ratio_at_least_1_5": ratio >= 1.5,
        "no_cross_partition_near_duplicates": manifest.get(
            "cross_partition_near_duplicates"
        )
        == 0,
    }
    lock = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "max_seq_length": args.max_seq_length,
        "model_path_read_only": str(args.model),
        "model_weights_loaded": False,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_hashes": tokenizer_hashes,
        "tokenizer_fingerprint": hashlib.sha256(
            json.dumps(tokenizer_hashes, sort_keys=True).encode()
        ).hexdigest(),
        "data_manifest_sha256": sha256(manifest_path),
        "data_design_sha256": sha256(Path(__file__).with_name("DATA-DESIGN.md")),
        "preprocessing_script_sha256": sha256(
            Path(__file__).with_name("prepare_gsm8k.py")
        ),
        "test_completion_p90_p10_ratio": ratio,
        "splits": splits,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(lock, indent=2, sort_keys=True))
    if lock["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
