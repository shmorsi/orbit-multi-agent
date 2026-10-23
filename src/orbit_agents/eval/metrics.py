"""Means and a paired percentile bootstrap. The seed is fixed."""

from __future__ import annotations

import math
import random


# CPython 3.11 and 3.12 can sum the same schema-valid series to values a few
# binary steps apart (about 1e-17 here). This bound stays far below one task
# on the 12-task rates. It does not round a published figure.
FLOAT_NOISE = 1e-12


def measurements_match(left, right) -> bool:
    """True when two generated measurements are the same record.

    Floats may differ by floating-point noise between interpreters. Integers,
    strings, and structure still have to match.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(measurements_match(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(measurements_match(a, b) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if left == right:
            return True
        if isinstance(left, int) or isinstance(right, int):
            return False
        if math.isnan(left) or math.isnan(right):
            return False
        return abs(left - right) <= FLOAT_NOISE
    return left == right


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def bootstrap_mean_difference(
    treatment: list[float],
    control: list[float],
    *,
    n_resamples: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    if len(treatment) != len(control) or not treatment:
        raise ValueError("bootstrap needs two equal non-empty series")
    point = sum(a - b for a, b in zip(treatment, control, strict=True)) / len(treatment)
    rng = random.Random(seed)
    count = len(treatment)
    diffs = []
    for _ in range(n_resamples):
        total = 0.0
        for _index in range(count):
            draw = rng.randrange(count)
            total += treatment[draw] - control[draw]
        diffs.append(total / count)
    diffs.sort()
    low_index = int((alpha / 2) * n_resamples)
    high_index = min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)
    return {
        "point": point,
        "low": diffs[low_index],
        "high": diffs[high_index],
        "n": count,
        "n_resamples": n_resamples,
        "seed": seed,
        "alpha": alpha,
        "method": "paired_percentile_bootstrap",
    }


def summarize(rows: list[dict]) -> dict:
    successes = [1.0 if row["success"] else 0.0 for row in rows]
    steps = [float(row["n_steps"]) for row in rows]
    schema = [row["schema_valid_rate"] for row in rows if row["schema_valid_rate"] is not None]
    refusals = [row for row in rows if row["expect_refusal"]]
    correct = sum(1 for row in refusals if row["correct_refusal"])
    return {
        "n_tasks": len(rows),
        "n_success": sum(1 for row in rows if row["success"]),
        "task_success_rate": mean(successes),
        "mean_steps": mean(steps),
        "schema_valid_rate": mean(schema),
        "n_traps": len(refusals),
        "n_correct_refusals": correct,
    }
