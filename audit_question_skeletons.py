#!/usr/bin/env python3
"""Hash GSM8K question skeletons and report cross-partition overlaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from natural_data import load_records


NUMBER_RE = re.compile(
    r"(?<!\w)(?:[$£€¥]\s*)?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
)


def skeleton(question: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", question).lower().split())
    return " ".join(NUMBER_RE.sub("<NUM>", normalized).split())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite skeleton audit: {args.output}")

    groups = defaultdict(list)
    split_hashes = {}
    for split in ("train", "valid", "test"):
        path = args.data / f"{split}.jsonl"
        split_hashes[split] = sha256(path)
        for record in load_records(path):
            value = skeleton(record["prompt"])
            groups[sha256_text(value)].append(
                {"split": split, "id": record["id"], "skeleton": value}
            )
    overlaps = []
    for skeleton_hash, rows in groups.items():
        splits = sorted({row["split"] for row in rows})
        if len(splits) > 1:
            overlaps.append(
                {
                    "skeleton_sha256": skeleton_hash,
                    "splits": splits,
                    "records": rows,
                }
            )
    report = {
        "status": "PASS" if not overlaps else "OVERLAP_PRESENT",
        "rule": "NFKC + lowercase + whitespace collapse + signed integer/decimal/percent/currency replacement with <NUM>",
        "question_only": True,
        "split_sha256": split_hashes,
        "unique_skeletons": len(groups),
        "cross_partition_overlap_groups": len(overlaps),
        "overlaps": overlaps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
