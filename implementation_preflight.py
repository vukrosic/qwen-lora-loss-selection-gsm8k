#!/usr/bin/env python3
"""No-model preflight for frozen data, compiled losses, batching, and scripts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from losses import example_mean_loss, mlx_masked_losses, token_mean_loss
from natural_data import (
    DeterministicBatchIterator,
    NaturalInstructionDataset,
    validate_tokenized_dataset,
)


PROJECT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite preflight: {args.output}")
    lock = json.loads(args.data_lock.read_text())
    if lock.get("status") != "PASS":
        raise ValueError("Dataset lock did not pass")
    if lock["data_manifest_sha256"] != sha256(args.data / "manifest.json"):
        raise ValueError("Manifest hash mismatch")

    for module in (
        "mlx",
        "mlx_lm",
        "train_condition",
        "evaluate_condition",
        "select_by_validation",
    ):
        importlib.import_module(module)

    mx.random.seed(20260810)
    model = nn.Embedding(32, 16)
    batch = mx.array([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]], dtype=mx.int32)
    spans = mx.array([[2, 3], [2, 4]], dtype=mx.int32)
    compiled = {}
    for name, loss in (("token", token_mean_loss), ("example", example_mean_loss)):
        value_and_grad = nn.value_and_grad(model, loss)
        state = [model.state, mx.random.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(batch, spans):
            return value_and_grad(model, batch, spans)

        (value, tokens), gradient = step(batch, spans)
        mx.eval(value, tokens, gradient)
        compiled[name] = {"loss": float(value), "supervised_tokens": int(tokens)}
    counts = mlx_masked_losses(model, batch, spans)[1]
    mx.eval(counts)
    if counts.tolist() != [1, 2]:
        raise AssertionError(f"Completion mask counts were {counts.tolist()}")
    if any(row["supervised_tokens"] != 3 for row in compiled.values()):
        raise AssertionError("Compiled loss token count mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    data_checks = {}
    for split, expected in (("train", 1000), ("valid", 256), ("test", 512)):
        dataset = NaturalInstructionDataset(args.data / f"{split}.jsonl", tokenizer)
        evidence = validate_tokenized_dataset(dataset, 256)
        if evidence["records"] != expected or evidence["sequence_tokens_max"] > 256:
            raise AssertionError({split: evidence})
        data_checks[split] = evidence
    train_dataset = NaturalInstructionDataset(args.data / "train.jsonl", tokenizer)
    iterator = DeterministicBatchIterator(order_seed=20260820 + 1000)
    epoch_zero = iterator.ordered_indices(train_dataset, 4, 0)
    repeated_check = DeterministicBatchIterator(
        order_seed=20260820 + 1000
    ).ordered_indices(train_dataset, 4, 0)
    if epoch_zero != repeated_check:
        raise AssertionError("Same-seed record order is not identical")
    if len(epoch_zero) != 1000 or len(set(epoch_zero)) != 1000:
        raise AssertionError("Epoch order oversamples or omits a locked record")
    first_batch, first_spans = next(iterator(train_dataset, 4, 256, loop=False))
    mx.eval(first_batch, first_spans)
    first_counts = (first_spans[:, 1] - first_spans[:, 0]).tolist()
    if any(count <= 0 for count in first_counts):
        raise AssertionError("First frozen batch contains an empty completion")

    report = {
        "status": "PASS",
        "model_weights_loaded": False,
        "compiled_losses": compiled,
        "corrected_mask_counts": counts.tolist(),
        "data_checks": data_checks,
        "first_seed_20260820_batch": {
            "first_record_indices": epoch_zero[:4],
            "shape": list(first_batch.shape),
            "completion_tokens": first_counts,
        },
        "record_order": {
            "epoch_zero_records": len(epoch_zero),
            "epoch_zero_unique": len(set(epoch_zero)),
            "same_seed_repeat_identical": epoch_zero == repeated_check,
            "sampling_with_replacement": False,
            "length_balanced_batches": False,
        },
        "fingerprints": {
            "canonical_protocol": sha256(PROJECT / "CANONICAL-PROTOCOL.md"),
            "data_lock": sha256(args.data_lock),
            "manifest": sha256(args.data / "manifest.json"),
            "train_condition": sha256(PROJECT / "train_condition.py"),
            "evaluate_condition": sha256(PROJECT / "evaluate_condition.py"),
            "select_by_validation": sha256(PROJECT / "select_by_validation.py"),
            "natural_data": sha256(PROJECT / "natural_data.py"),
            "losses": sha256(PROJECT / "losses.py"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
