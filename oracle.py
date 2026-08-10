#!/usr/bin/env python3
"""Small deterministic checks for mask boundaries and loss aggregation."""

from __future__ import annotations

import json

import numpy as np

from losses import numpy_completion_mask, numpy_reduce


def main() -> None:
    spans = np.array([[2, 4], [1, 5]], dtype=np.int64)
    mask = numpy_completion_mask(width=5, spans=spans)
    expected_mask = np.array(
        [
            [False, True, True, False, False],
            [True, True, True, True, False],
        ]
    )
    if not np.array_equal(mask, expected_mask):
        raise AssertionError({"mask": mask.tolist(), "expected": expected_mask.tolist()})

    losses = np.array(
        [
            [99.0, 1.0, 3.0, 99.0, 99.0],
            [2.0, 2.0, 2.0, 2.0, 99.0],
        ]
    )
    token = numpy_reduce(losses, mask, "token")
    example = numpy_reduce(losses, mask, "example")
    if not np.isclose(token, 2.0):
        raise AssertionError(f"token mean {token} != 2.0")
    if not np.isclose(example, 2.0):
        raise AssertionError(f"example mean {example} != 2.0")

    unequal_losses = losses.copy()
    unequal_losses[0, 1:3] = [0.0, 0.0]
    unequal_token = numpy_reduce(unequal_losses, mask, "token")
    unequal_example = numpy_reduce(unequal_losses, mask, "example")
    if not np.isclose(unequal_token, 8.0 / 6.0):
        raise AssertionError(f"unequal token mean {unequal_token}")
    if not np.isclose(unequal_example, 1.0):
        raise AssertionError(f"unequal example mean {unequal_example}")
    if np.isclose(unequal_token, unequal_example):
        raise AssertionError("Oracle failed to distinguish the two objectives")

    print(
        json.dumps(
            {
                "status": "PASS",
                "mask": mask.astype(int).tolist(),
                "equal_case": {"token": token, "example": example},
                "unequal_case": {
                    "token": unequal_token,
                    "example": unequal_example,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
