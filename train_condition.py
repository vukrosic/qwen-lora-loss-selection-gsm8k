#!/usr/bin/env python3
"""Train one frozen N1-GSM8K token-mean or example-mean LoRA arm."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

from losses import example_mean_loss, token_mean_loss
from natural_data import (
    DeterministicBatchIterator,
    NaturalInstructionDataset,
    validate_tokenized_dataset,
)


PROJECT = Path(__file__).resolve().parent
PROTOCOL = PROJECT / "CANONICAL-PROTOCOL.md"
LOSSES = {"token": token_mean_loss, "example": example_mean_loss}
FROZEN_SEEDS = {20260820, 20260821, 20260822}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(LOSSES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iters", type=int, default=576)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-seq-length", type=int, default=256)
    args = parser.parse_args()

    frozen = {
        "iters": 576,
        "batch_size": 4,
        "num_layers": 8,
        "learning_rate": 1e-4,
        "max_seq_length": 256,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise ValueError(f"Protocol freezes {name}={expected}")
    if args.seed not in FROZEN_SEEDS:
        raise ValueError(f"Seed {args.seed} is outside the frozen set")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    lock = json.loads(args.data_lock.read_text())
    if lock.get("status") != "PASS" or lock.get("max_seq_length") != 256:
        raise ValueError("Dataset lock is not a passing 256-token lock")
    if sha256(args.data / "manifest.json") != lock["data_manifest_sha256"]:
        raise ValueError("Dataset manifest hash does not match the lock")
    args.output.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    mx.reset_peak_memory()
    started = time.time()
    model, tokenizer = load(str(args.model))
    model.freeze()
    lora_parameters = {"rank": 8, "dropout": 0.0, "scale": 20.0}
    linear_to_lora_layers(model, args.num_layers, lora_parameters)
    print_trainable_parameters(model)

    train_data = NaturalInstructionDataset(args.data / "train.jsonl", tokenizer)
    valid_data = NaturalInstructionDataset(args.data / "valid.jsonl", tokenizer)
    dataset_validation = {
        "train": validate_tokenized_dataset(train_data, args.max_seq_length),
        "valid": validate_tokenized_dataset(valid_data, args.max_seq_length),
    }
    if dataset_validation["train"]["records"] != 1000:
        raise ValueError("Frozen train count is 1000")
    if dataset_validation["valid"]["records"] != 256:
        raise ValueError("Frozen validation count is 256")

    iterator = DeterministicBatchIterator(order_seed=args.seed + 1000)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.0)
    config = {
        "protocol": "N1-GSM8K-A1",
        "condition": args.condition,
        "seed": args.seed,
        "model": str(args.model),
        "fine_tune_type": "lora",
        "num_layers": args.num_layers,
        "lora_parameters": lora_parameters,
        "batch_size": args.batch_size,
        "iters": args.iters,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.0,
        "max_seq_length": args.max_seq_length,
        "grad_accumulation_steps": 1,
        "dataset_validation": dataset_validation,
        "fingerprints": {
            "canonical_protocol": sha256(PROTOCOL),
            "data_lock": sha256(args.data_lock),
            "data_manifest": sha256(args.data / "manifest.json"),
            "train_split": sha256(args.data / "train.jsonl"),
            "valid_split": sha256(args.data / "valid.jsonl"),
            "train_condition": sha256(Path(__file__)),
            "losses": sha256(PROJECT / "losses.py"),
            "natural_data": sha256(PROJECT / "natural_data.py"),
            "model_safetensors": sha256(args.model / "model.safetensors"),
            "model_config": sha256(args.model / "config.json"),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "mlx": importlib.metadata.version("mlx"),
            "mlx-lm": importlib.metadata.version("mlx-lm"),
            "numpy": importlib.metadata.version("numpy"),
        },
    }
    (args.output / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=-1,
        steps_per_report=48,
        steps_per_eval=576,
        steps_per_save=576,
        adapter_file=args.output / "adapters.safetensors",
        max_seq_length=args.max_seq_length,
        grad_accumulation_steps=1,
        clear_cache_threshold=2_000_000_000,
    )
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_data,
        val_dataset=valid_data,
        args=training_args,
        loss=LOSSES[args.condition],
        iterate_batches=iterator,
    )
    resource = {
        "condition": args.condition,
        "seed": args.seed,
        "wall_seconds": time.time() - started,
        "peak_mlx_memory_gb": mx.get_peak_memory() / 1e9,
        "active_mlx_memory_gb": mx.get_active_memory() / 1e9,
        "cache_mlx_memory_gb": mx.get_cache_memory() / 1e9,
    }
    (args.output / "resource.json").write_text(
        json.dumps(resource, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(resource, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
