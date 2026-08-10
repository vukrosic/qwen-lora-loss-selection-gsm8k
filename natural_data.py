#!/usr/bin/env python3
"""Shared natural-instruction dataset and deterministic batching primitives."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np


REQUIRED_FIELDS = {"id", "prompt", "completion", "stratum"}
ALLOWED_STRATA = {"short", "long"}


def load_records(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"Missing {sorted(missing)} at {path}:{line_number}")
        if record["stratum"] not in ALLOWED_STRATA:
            raise ValueError(f"Invalid stratum at {path}:{line_number}")
        if not all(isinstance(record[key], str) for key in REQUIRED_FIELDS):
            raise TypeError(f"Required fields must be strings at {path}:{line_number}")
        if not record["id"] or not record["prompt"] or not record["completion"]:
            raise ValueError(f"Empty required value at {path}:{line_number}")
        records.append(record)
    if not records:
        raise ValueError(f"No records in {path}")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return records


class NaturalInstructionDataset:
    def __init__(self, path: Path, tokenizer):
        self.path = path
        self.records = load_records(path)
        self.tokenizer = tokenizer
        self.cache: list[tuple | None] = [None] * len(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def processed(self, index: int):
        if self.cache[index] is None:
            record = self.records[index]
            user = [{"role": "user", "content": record["prompt"]}]
            messages = user + [
                {"role": "assistant", "content": record["completion"]}
            ]
            tokens = self.tokenizer.apply_chat_template(
                messages,
                return_dict=False,
                enable_thinking=False,
            )
            prefix = self.tokenizer.apply_chat_template(
                user,
                add_generation_prompt=True,
                return_dict=False,
                enable_thinking=False,
            )
            self.cache[index] = (
                tokens,
                len(prefix),
                record["stratum"],
                record["id"],
            )
        return self.cache[index]


def validate_tokenized_dataset(
    dataset: NaturalInstructionDataset, max_seq_length: int
) -> dict:
    ids: set[str] = set()
    strata = {name: 0 for name in sorted(ALLOWED_STRATA)}
    counts = []
    total_lengths = []
    for index in range(len(dataset)):
        tokens, offset, stratum, record_id = dataset.processed(index)
        if record_id in ids:
            raise ValueError(f"Duplicate processed ID: {record_id}")
        ids.add(record_id)
        if len(tokens) > max_seq_length:
            raise ValueError(
                f"Record {record_id} has {len(tokens)} tokens; max is {max_seq_length}"
            )
        if offset >= len(tokens):
            raise ValueError(f"Record {record_id} has no supervised completion tokens")
        strata[stratum] += 1
        counts.append(len(tokens) - offset)
        total_lengths.append(len(tokens))
    return {
        "records": len(dataset),
        "strata": strata,
        "supervised_tokens_min": min(counts),
        "supervised_tokens_max": max(counts),
        "supervised_tokens_total": sum(counts),
        "sequence_tokens_max": max(total_lengths),
    }


class StratifiedBatchIterator:
    """Yield equal short/long batches in a deterministic order for each seed."""

    def __init__(self, order_seed: int):
        self.order_seed = order_seed

    def __call__(
        self,
        dataset,
        batch_size,
        max_seq_length,
        loop=False,
        seed=None,
        comm_group=None,
    ):
        if comm_group is not None and comm_group.size() != 1:
            raise ValueError("This bounded experiment permits one local worker only")
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("batch_size must be a positive even number")
        half = batch_size // 2
        by_stratum = {
            stratum: [
                index
                for index, record in enumerate(dataset.records)
                if record["stratum"] == stratum
            ]
            for stratum in sorted(ALLOWED_STRATA)
        }
        sizes = {name: len(indices) for name, indices in by_stratum.items()}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"Strata must be equal-sized, got {sizes}")
        if next(iter(sizes.values())) % half:
            raise ValueError(f"Each stratum size must be divisible by {half}, got {sizes}")

        epoch = 0
        while True:
            rng = np.random.default_rng(self.order_seed + epoch)
            short = rng.permutation(by_stratum["short"])
            long = rng.permutation(by_stratum["long"])
            batches = []
            for start in range(0, len(short), half):
                indices = list(short[start : start + half]) + list(
                    long[start : start + half]
                )
                rng.shuffle(indices)
                batches.append(indices)
            rng.shuffle(batches)

            for indices in batches:
                items = [dataset.processed(index) for index in indices]
                lengths = [min(len(item[0]), max_seq_length) for item in items]
                padded = min(
                    1 + 32 * ((max(lengths) + 31) // 32), max_seq_length
                )
                batch = np.zeros((batch_size, padded), dtype=np.int32)
                spans = []
                for row, (tokens, offset, _, _) in enumerate(items):
                    length = min(len(tokens), max_seq_length)
                    batch[row, :length] = tokens[:length]
                    spans.append((min(offset, length), length))
                yield mx.array(batch), mx.array(spans)

            if not loop:
                return
            epoch += 1


class DeterministicBatchIterator:
    """Shuffle every locked record once per epoch, without replacement."""

    def __init__(self, order_seed: int):
        self.order_seed = order_seed

    def ordered_indices(self, dataset, batch_size: int, epoch: int) -> list[int]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(dataset) % batch_size:
            raise ValueError("dataset size must be divisible by batch_size")
        rng = np.random.default_rng(self.order_seed + epoch)
        return rng.permutation(len(dataset)).tolist()

    def __call__(
        self,
        dataset,
        batch_size,
        max_seq_length,
        loop=False,
        seed=None,
        comm_group=None,
    ):
        if comm_group is not None and comm_group.size() != 1:
            raise ValueError("This bounded experiment permits one local worker only")
        epoch = 0
        while True:
            ordered = self.ordered_indices(dataset, batch_size, epoch)
            for start in range(0, len(ordered), batch_size):
                indices = ordered[start : start + batch_size]
                items = [dataset.processed(index) for index in indices]
                lengths = [min(len(item[0]), max_seq_length) for item in items]
                padded = min(
                    1 + 32 * ((max(lengths) + 31) // 32), max_seq_length
                )
                batch = np.zeros((batch_size, padded), dtype=np.int32)
                spans = []
                for row, (tokens, offset, _, _) in enumerate(items):
                    length = min(len(tokens), max_seq_length)
                    batch[row, :length] = tokens[:length]
                    spans.append((min(offset, length), length))
                yield mx.array(batch), mx.array(spans)
            if not loop:
                return
            epoch += 1
