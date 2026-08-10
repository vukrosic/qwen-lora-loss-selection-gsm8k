#!/usr/bin/env python3
"""Evaluate one N1-GSM8K arm on validation or sealed final-test data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import time
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from natural_data import NaturalInstructionDataset


PROJECT = Path(__file__).resolve().parent
PROTOCOL = PROJECT / "CANONICAL-PROTOCOL.md"
FINAL_RE = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_number(raw: str) -> str | None:
    try:
        rendered = format(Decimal(raw.replace(",", "")), "f")
    except InvalidOperation:
        return None
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"-0", ""} else rendered


def extract_final(text: str) -> str | None:
    match = FINAL_RE.search(text)
    return normalize_number(match.group(1)) if match else None


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def collate(dataset, indices: list[int], max_seq_length: int):
    items = [dataset.processed(index) for index in indices]
    if any(len(item[0]) > max_seq_length for item in items):
        raise ValueError("Evaluation record would be truncated")
    padded = min(
        1 + 32 * ((max(len(item[0]) for item in items) + 31) // 32),
        max_seq_length,
    )
    batch = np.zeros((len(items), padded), dtype=np.int32)
    spans = []
    metadata = []
    for row, (tokens, offset, stratum, record_id) in enumerate(items):
        batch[row, : len(tokens)] = tokens
        spans.append((offset, len(tokens)))
        metadata.append((stratum, record_id))
    return mx.array(batch), mx.array(spans), metadata


def aggregate_teacher(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["stratum"]].append(record)
    by_stratum = {}
    for stratum, rows in sorted(grouped.items()):
        tokens = sum(row["supervised_tokens"] for row in rows)
        by_stratum[stratum] = {
            "records": len(rows),
            "macro_example_nll": sum(row["example_nll"] for row in rows) / len(rows),
            "token_weighted_nll": sum(row["loss_sum"] for row in rows) / tokens,
            "target_token_accuracy": sum(row["correct_tokens"] for row in rows) / tokens,
        }
    total_tokens = sum(row["supervised_tokens"] for row in records)
    return {
        "macro_example_nll": sum(row["example_nll"] for row in records) / len(records),
        "token_weighted_nll": sum(row["loss_sum"] for row in records) / total_tokens,
        "target_token_accuracy": sum(row["correct_tokens"] for row in records)
        / total_tokens,
        "by_stratum": by_stratum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=("token", "example"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--max-generation-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    if args.batch_size != 4 or args.max_seq_length != 256:
        raise ValueError("Frozen evaluation uses batch 4 and max sequence 256")
    protocol_hash = sha256(PROTOCOL)
    lock_hash = sha256(args.data_lock)
    split_path = args.data / f"{args.split}.jsonl"
    split_hash = sha256(split_path)
    selection_record = None
    if args.split == "test":
        if args.selection is None or not args.selection.exists():
            raise RuntimeError("Final-test evaluation requires a sealed selection")
        selection_record = json.loads(args.selection.read_text())
        if selection_record.get("seed") != args.seed:
            raise ValueError("Selection seed mismatch")
        if selection_record.get("protocol_sha256") != protocol_hash:
            raise ValueError("Selection protocol hash mismatch")
        if selection_record.get("data_lock_sha256") != lock_hash:
            raise ValueError("Selection data-lock hash mismatch")
    elif args.selection is not None:
        raise ValueError("Validation evaluation must not receive a selection")

    adapter_config = json.loads((args.adapter / "adapter_config.json").read_text())
    if adapter_config["condition"] != args.condition or adapter_config["seed"] != args.seed:
        raise ValueError("Adapter identity does not match requested arm")
    if adapter_config["fingerprints"]["canonical_protocol"] != protocol_hash:
        raise ValueError("Adapter protocol hash mismatch")
    if adapter_config["fingerprints"]["data_lock"] != lock_hash:
        raise ValueError("Adapter data-lock hash mismatch")
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.time()
    mx.reset_peak_memory()
    model, tokenizer = load(str(args.model), adapter_path=str(args.adapter))
    model.eval()
    dataset = NaturalInstructionDataset(split_path, tokenizer)
    expected = 256 if args.split == "valid" else 512
    if len(dataset) != expected:
        raise ValueError(f"Expected {expected} {args.split} records, got {len(dataset)}")

    teacher = []
    for start in range(0, len(dataset), args.batch_size):
        indices = list(range(start, min(start + args.batch_size, len(dataset))))
        batch, spans, metadata = collate(dataset, indices, args.max_seq_length)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= spans[:, 0:1], steps < spans[:, 1:2])
        losses = nn.losses.cross_entropy(logits, targets).astype(mx.float32) * mask
        counts = mask.sum(axis=1)
        loss_sums = losses.sum(axis=1)
        correct_counts = ((mx.argmax(logits, axis=-1) == targets) * mask).sum(axis=1)
        mx.eval(counts, loss_sums, correct_counts)
        for row, (stratum, record_id) in enumerate(metadata):
            count = int(counts[row])
            loss_sum = float(loss_sums[row])
            teacher.append(
                {
                    "id": record_id,
                    "stratum": stratum,
                    "supervised_tokens": count,
                    "loss_sum": loss_sum,
                    "example_nll": loss_sum / count,
                    "correct_tokens": int(correct_counts[row]),
                }
            )

    summary = aggregate_teacher(teacher)
    generations = []
    if args.split == "test":
        sampler = make_sampler(temp=0.0)
        for record in dataset.records:
            user = [{"role": "user", "content": record["prompt"]}]
            prompt_tokens = tokenizer.apply_chat_template(
                user,
                add_generation_prompt=True,
                return_dict=False,
                enable_thinking=False,
            )
            generated = generate(
                model,
                tokenizer,
                prompt_tokens,
                max_tokens=args.max_generation_tokens,
                sampler=sampler,
                verbose=False,
            )
            prediction = extract_final(generated)
            generations.append(
                {
                    "id": record["id"],
                    "stratum": record["stratum"],
                    "target": record["target"],
                    "generated": generated,
                    "predicted_target": prediction,
                    "marker_present": prediction is not None,
                    "exact_answer": prediction == record["target"],
                }
            )
        summary["exact_answer"] = sum(row["exact_answer"] for row in generations) / len(generations)
        summary["marker_rate"] = sum(row["marker_present"] for row in generations) / len(generations)
        summary["exact_answer_by_stratum"] = {
            stratum: sum(row["exact_answer"] for row in generations if row["stratum"] == stratum)
            / sum(row["stratum"] == stratum for row in generations)
            for stratum in ("short", "long")
        }

    metrics = {
        "protocol": "N1-GSM8K-A1",
        "protocol_sha256": protocol_hash,
        "data_lock_sha256": lock_hash,
        "split": args.split,
        "split_sha256": split_hash,
        "condition": args.condition,
        "seed": args.seed,
        "records": len(dataset),
        "summary": summary,
        "fingerprints": {
            "evaluate_condition": sha256(Path(__file__)),
            "adapter_config": sha256(args.adapter / "adapter_config.json"),
            "adapters": sha256(args.adapter / "adapters.safetensors"),
            **(
                {"selection": sha256(args.selection)}
                if args.selection is not None
                else {}
            ),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "mlx": importlib.metadata.version("mlx"),
            "mlx-lm": importlib.metadata.version("mlx-lm"),
        },
        "resource": {
            "wall_seconds": time.time() - started,
            "peak_mlx_memory_gb": mx.get_peak_memory() / 1e9,
            "active_mlx_memory_gb": mx.get_active_memory() / 1e9,
            "cache_mlx_memory_gb": mx.get_cache_memory() / 1e9,
        },
    }
    write_jsonl(args.output / "teacher-forced.jsonl", teacher)
    if generations:
        write_jsonl(args.output / "generations.jsonl", generations)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
