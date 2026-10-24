"""Means and a paired percentile bootstrap. The seed is fixed."""

from __future__ import annotations

import random


def mean(values: list[float]) -> float | None:
    """Arithmetic mean, or None for an empty series."""
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
    """Paired percentile interval for the mean of treatment minus control. Seed is fixed."""
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
    """Task success, schema-valid rate, and trap refusals for one scored split."""
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
