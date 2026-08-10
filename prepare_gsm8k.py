#!/usr/bin/env python3
"""Materialize the frozen source-aware GSM8K subset and evidence manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from transformers import AutoTokenizer


COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
SEED = 20260810
FINAL_RE = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def normalized_target(answer: str) -> str:
    match = FINAL_RE.search(answer)
    if not match:
        raise ValueError("Answer lacks required #### numeric target")
    raw = match.group(1).replace(",", "")
    try:
        value = Decimal(raw)
    except InvalidOperation as error:
        raise ValueError(f"Invalid GSM8K target: {raw}") from error
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"-0", ""} else rendered


def source_records(path: Path, source_split: str) -> list[dict]:
    records = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        raw = json.loads(line)
        question = raw["question"].strip()
        answer = raw["answer"].strip()
        if not question or not answer:
            raise ValueError(f"Empty source record {source_split}:{index}")
        records.append(
            {
                "source_id": f"{source_split}-{index:05d}",
                "source_split": source_split,
                "source_index": index,
                "prompt": question,
                "completion": answer,
                "target": normalized_target(answer),
                "answer_whitespace_tokens": len(answer.split()),
            }
        )
    return records


def remove_exact_duplicates(records: list[dict]):
    seen = {}
    kept = []
    excluded = []
    for record in records:
        key = sha256_bytes(
            (canonical(record["prompt"]) + "\n" + canonical(record["completion"])).encode()
        )
        if key in seen:
            excluded.append(
                {
                    "excluded_source_id": record["source_id"],
                    "kept_source_id": seen[key],
                    "canonical_sha256": key,
                }
            )
        else:
            seen[key] = record["source_id"]
            record["canonical_sha256"] = key
            kept.append(record)
    return kept, excluded


def filter_tokenizer_eligible(
    records: list[dict], tokenizer, max_seq_length: int
) -> tuple[list[dict], list[dict]]:
    kept = []
    excluded = []
    for record in records:
        user = [{"role": "user", "content": record["prompt"]}]
        messages = user + [
            {"role": "assistant", "content": record["completion"]}
        ]
        tokens = tokenizer.apply_chat_template(
            messages, return_dict=False, enable_thinking=False
        )
        record["formatted_sequence_tokens"] = len(tokens)
        if len(tokens) <= max_seq_length:
            kept.append(record)
        else:
            excluded.append(
                {
                    "source_id": record["source_id"],
                    "source_split": record["source_split"],
                    "formatted_sequence_tokens": len(tokens),
                }
            )
    return kept, excluded


def assign_strata(records: list[dict]) -> None:
    ordered = sorted(
        records, key=lambda row: (row["answer_whitespace_tokens"], row["source_id"])
    )
    boundary = len(ordered) // 2
    for position, record in enumerate(ordered):
        record["stratum"] = "short" if position < boundary else "long"


def selection_key(partition: str, record: dict) -> str:
    return sha256_bytes(f"{SEED}:{partition}:{record['source_id']}".encode())


def choose(records: list[dict], per_stratum: int, partition: str) -> list[dict]:
    chosen = []
    for stratum in ("short", "long"):
        candidates = [row for row in records if row["stratum"] == stratum]
        ordered = sorted(candidates, key=lambda row: selection_key(partition, row))
        if len(ordered) < per_stratum:
            raise ValueError(f"Not enough {stratum} records for {partition}")
        chosen.extend(ordered[:per_stratum])
    return sorted(chosen, key=lambda row: selection_key(partition, row))


def choose_test_partitions(records: list[dict]):
    validation = []
    final = []
    for stratum in ("short", "long"):
        candidates = [row for row in records if row["stratum"] == stratum]
        ordered = sorted(candidates, key=lambda row: selection_key("test-pool", row))
        validation.extend(ordered[:128])
        final.extend(ordered[128:384])
    return (
        sorted(validation, key=lambda row: selection_key("validation", row)),
        sorted(final, key=lambda row: selection_key("final-test", row)),
    )


def shingles(record: dict, size: int = 5) -> set[tuple[str, ...]]:
    tokens = canonical(record["prompt"] + " " + record["completion"]).split()
    if len(tokens) < size:
        return {tuple(tokens)}
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def cross_partition_near_duplicates(partitions: dict[str, list[dict]]) -> list[dict]:
    cached = {
        record["source_id"]: shingles(record)
        for records in partitions.values()
        for record in records
    }
    names = list(partitions)
    found = []
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            for left in partitions[left_name]:
                a = cached[left["source_id"]]
                for right in partitions[right_name]:
                    b = cached[right["source_id"]]
                    similarity = len(a & b) / len(a | b)
                    if similarity >= 0.90:
                        found.append(
                            {
                                "left_partition": left_name,
                                "left_id": left["source_id"],
                                "right_partition": right_name,
                                "right_id": right["source_id"],
                                "jaccard": similarity,
                            }
                        )
    return found


def public_record(record: dict) -> dict:
    return {
        "id": "gsm8k-" + record["source_id"],
        "source_split": record["source_split"],
        "source_index": record["source_index"],
        "prompt": record["prompt"],
        "completion": record["completion"],
        "stratum": record["stratum"],
        "target": record["target"],
        "formatted_sequence_tokens": record.get("formatted_sequence_tokens"),
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(public_record(record), sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-test", type=Path, required=True)
    parser.add_argument("--source-license", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    train_source = source_records(args.source_train, "train")
    test_source = source_records(args.source_test, "test")
    all_deduplicated, exclusions = remove_exact_duplicates(train_source + test_source)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model, local_files_only=True, trust_remote_code=False
    )
    all_kept, length_exclusions = filter_tokenizer_eligible(
        all_deduplicated, tokenizer, args.max_seq_length
    )
    kept_by_split = defaultdict(list)
    for record in all_kept:
        kept_by_split[record["source_split"]].append(record)
    assign_strata(kept_by_split["train"])
    assign_strata(kept_by_split["test"])

    partitions = {"train": choose(kept_by_split["train"], 500, "train")}
    partitions["valid"], partitions["test"] = choose_test_partitions(
        kept_by_split["test"]
    )
    near = cross_partition_near_duplicates(partitions)
    if near:
        (args.output / "near-duplicate-blocker.json").write_text(
            json.dumps(near, indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError(f"Found {len(near)} cross-partition near duplicates")

    split_evidence = {}
    for name, records in partitions.items():
        path = args.output / f"{name}.jsonl"
        write_jsonl(path, records)
        split_evidence[name] = {
            "records": len(records),
            "short": sum(row["stratum"] == "short" for row in records),
            "long": sum(row["stratum"] == "long" for row in records),
            "source_splits": sorted({row["source_split"] for row in records}),
            "sha256": sha256(path),
            "answer_whitespace_tokens_min": min(
                row["answer_whitespace_tokens"] for row in records
            ),
            "answer_whitespace_tokens_max": max(
                row["answer_whitespace_tokens"] for row in records
            ),
        }

    exclusion_path = args.output / "exact-duplicate-exclusions.json"
    exclusion_path.write_text(json.dumps(exclusions, indent=2, sort_keys=True) + "\n")
    length_exclusion_path = args.output / "length-exclusions.json"
    length_exclusion_path.write_text(
        json.dumps(length_exclusions, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "status": "MATERIALIZED",
        "corpus": "GSM8K main",
        "upstream_commit": COMMIT,
        "license": "MIT",
        "selection_seed": SEED,
        "design": "official-train-only training; disjoint official-test validation/final",
        "source": {
            "train": {"records": len(train_source), "sha256": sha256(args.source_train)},
            "test": {"records": len(test_source), "sha256": sha256(args.source_test)},
            "license_sha256": sha256(args.source_license),
        },
        "exact_duplicate_exclusions": len(exclusions),
        "max_seq_length": args.max_seq_length,
        "tokenizer_eligible_filter": True,
        "length_exclusions": {
            "total": len(length_exclusions),
            "train": sum(
                row["source_split"] == "train" for row in length_exclusions
            ),
            "test": sum(row["source_split"] == "test" for row in length_exclusions),
            "ledger_sha256": sha256(length_exclusion_path),
        },
        "cross_partition_near_duplicates": len(near),
        "near_duplicate_rule": "canonical prompt+completion token 5-shingle Jaccard >= 0.90",
        "target_rule": "first numeric value following ####; commas removed; Decimal equality",
        "splits": split_evidence,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
