#!/usr/bin/env python3
"""Completion-only token-mean and example-mean loss implementations."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def numpy_completion_mask(width: int, spans: np.ndarray) -> np.ndarray:
    """Mask next-token positions where prefix <= target index < sequence length."""
    steps = np.arange(1, width + 1)[None, :]
    return np.logical_and(steps >= spans[:, 0:1], steps < spans[:, 1:2])


def numpy_reduce(losses: np.ndarray, mask: np.ndarray, mode: str) -> float:
    masked = losses * mask
    counts = mask.sum(axis=1)
    if np.any(counts <= 0):
        raise ValueError("Every example must have at least one supervised token")
    if mode == "token":
        return float(masked.sum() / counts.sum())
    if mode == "example":
        return float((masked.sum(axis=1) / counts).mean())
    raise ValueError(f"Unknown mode: {mode}")


def mlx_masked_losses(model, batch, spans):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= spans[:, 0:1], steps < spans[:, 1:2])
    losses = nn.losses.cross_entropy(logits, targets).astype(mx.float32) * mask
    counts = mask.sum(axis=1)
    return losses, counts, mask.sum()


def token_mean_loss(model, batch, spans):
    losses, _, total = mlx_masked_losses(model, batch, spans)
    return losses.sum() / total, total


def example_mean_loss(model, batch, spans):
    losses, counts, total = mlx_masked_losses(model, batch, spans)
    return (losses.sum(axis=1) / counts).mean(), total
