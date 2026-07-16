"""Deterministic hyper-parameter tuning for the ALNS policy.

The evaluator's score is lexicographic (distinct brands, daily coverage, then
servings), so tuning must use the same ordering rather than collapsing the
objectives into an arbitrary weighted sum.  Wall-clock limits are deliberately
not part of this tuner: fixed iteration budgets make a tuning report
reproducible across machines.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .runner import (
    OBJECTIVES,
    find_binary,
    normalized_performance,
    run_core,
    structural_optimum,
)


@dataclass(frozen=True, order=True)
class ALNSParameters:
    """A deterministic ALNS search budget evaluated by :func:`tune_alns`."""

    fixed_iterations: int
    min_iterations: int = 32
    stagnation_iterations: int = 0

    def __post_init__(self) -> None:
        if self.fixed_iterations < 1:
            raise ValueError("fixed_iterations must be positive")
        if self.min_iterations < 1:
            raise ValueError("min_iterations must be positive")
        if self.min_iterations > self.fixed_iterations:
            raise ValueError("min_iterations cannot exceed fixed_iterations")
        if self.stagnation_iterations < 0:
            raise ValueError("stagnation_iterations cannot be negative")

    def as_search(self) -> dict[str, int]:
        return {
            "minIterations": self.min_iterations,
            "maxIterations": self.fixed_iterations,
            "stagnationIterations": self.stagnation_iterations,
        }

    def as_dict(self) -> dict[str, int]:
        return {
            "fixed_iterations": self.fixed_iterations,
            "min_iterations": self.min_iterations,
            "stagnation_iterations": self.stagnation_iterations,
        }


def _values(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(value) for value in values))
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    return result


def parameter_grid(
    fixed_iterations: Iterable[int],
    min_iterations: Iterable[int] = (32,),
    stagnation_iterations: Iterable[int] = (0,),
) -> list[ALNSParameters]:
    """Build and validate the Cartesian product of the requested budgets."""

    fixed = _values(fixed_iterations, "fixed_iterations")
    minimum = _values(min_iterations, "min_iterations")
    stagnation = _values(stagnation_iterations, "stagnation_iterations")
    candidates: list[ALNSParameters] = []
    for budget in fixed:
        for lower in minimum:
            for stagnant in stagnation:
                candidates.append(ALNSParameters(budget, lower, stagnant))
    return candidates


def _aggregate(
    parameters: ALNSParameters,
    evaluated: list[tuple[dict[str, Any], dict[str, int]]],
) -> dict[str, Any]:
    case_count = len(evaluated)
    aggregate: dict[str, Any] = {
        "parameters": parameters.as_dict(),
        "case_count": case_count,
        "valid_cases": 0,
        "invalid_days": 0,
        "runtime_seconds": 0.0,
        "distinct_types": 0,
        "cumulative_daily_types": 0,
        "total_servings": 0,
        "distinct_percent": 0.0,
        "daily_percent": 0.0,
        "servings_percent": 0.0,
    }
    for result, optimum in evaluated:
        aggregate["valid_cases"] += int(result["invalid_days"] == 0)
        aggregate["invalid_days"] += int(result["invalid_days"])
        aggregate["runtime_seconds"] += float(result["runtime_seconds"])
        for objective in OBJECTIVES:
            aggregate[objective] += int(result["score"][objective])
        percentages = result["objective_percentages"]
        aggregate["distinct_percent"] += percentages["distinct_types"]
        aggregate["daily_percent"] += percentages["cumulative_daily_types"]
        aggregate["servings_percent"] += percentages["total_servings"]
    for objective in OBJECTIVES:
        aggregate[f"mean_{objective}"] = (
            aggregate[objective] / case_count if case_count else 0.0
        )
    for key in ("distinct_percent", "daily_percent", "servings_percent"):
        aggregate[f"mean_{key}"] = aggregate[key] / case_count if case_count else 0.0
    aggregate["score_key"] = [
        aggregate["mean_distinct_percent"],
        aggregate["mean_daily_percent"],
        aggregate["mean_servings_percent"],
        -aggregate["runtime_seconds"],
    ]
    return aggregate


def tune_alns(
    manifest_path: Path,
    report_dir: Path,
    *,
    fixed_iterations: Iterable[int] = (128, 256, 512, 1024, 2048, 3072, 4096, 6000),
    min_iterations: Iterable[int] = (32,),
    stagnation_iterations: Iterable[int] = (0,),
    binary_path: str | None = None,
    jobs: int | None = None,
    timeout: float = 60,
) -> dict[str, Any]:
    """Evaluate ALNS parameter candidates and persist a tuning report.

    ``manifest_path`` has the same format as ``hexbench grade``.  Every
    candidate sees exactly the same scenarios and all cases are evaluated with
    one core thread when subprocesses are parallelized, preserving the
    evaluator's deterministic behavior.
    """

    manifest_path = Path(manifest_path)
    report_dir = Path(report_dir)
    manifest = json.loads(manifest_path.read_text())
    cases = manifest.get("cases", [])
    if not cases:
        raise ValueError("manifest contains no cases")
    scenarios = [
        json.loads((manifest_path.parent / case["path"]).read_text()) for case in cases
    ]
    candidates = parameter_grid(fixed_iterations, min_iterations, stagnation_iterations)
    binary = find_binary(binary_path)
    worker_count = max(1, jobs or min(os.cpu_count() or 1, 8))
    tasks = [
        (candidate_index, case_index)
        for candidate_index in range(len(candidates))
        for case_index in range(len(scenarios))
    ]

    def evaluate(task: tuple[int, int]) -> tuple[int, int, dict[str, Any]]:
        candidate_index, case_index = task
        started = time.perf_counter()
        result = run_core(
            "eval",
            "alns",
            {**scenarios[case_index], "search": candidates[candidate_index].as_search()},
            binary=binary,
            timeout=timeout,
            core_threads=1 if worker_count > 1 else None,
        )
        result["runtime_seconds"] = time.perf_counter() - started
        result.update(
            normalized_performance(result, structural_optimum(scenarios[case_index]))
        )
        return candidate_index, case_index, result

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        outputs = list(executor.map(evaluate, tasks))
    grouped: list[list[tuple[dict[str, Any], dict[str, int]]]] = [
        [] for _ in candidates
    ]
    rows: list[list[dict[str, Any] | None]] = [
        [None for _ in scenarios] for _ in candidates
    ]
    for candidate_index, case_index, result in outputs:
        optimum = structural_optimum(scenarios[case_index])
        grouped[candidate_index].append((result, optimum))
        rows[candidate_index][case_index] = result
    aggregates = [
        _aggregate(candidate, grouped[index])
        for index, candidate in enumerate(candidates)
    ]
    ranking = sorted(
        range(len(candidates)), key=lambda index: aggregates[index]["score_key"], reverse=True
    )
    best_index = ranking[0]
    report: dict[str, Any] = {
        "schema_version": 1,
        "suite": manifest.get("suite"),
        "method": "alns",
        "case_count": len(scenarios),
        "candidate_count": len(candidates),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "jobs": worker_count,
        "wall_seconds": time.perf_counter() - started,
        "objective_order": list(OBJECTIVES),
        "best": aggregates[best_index],
        "candidates": [aggregates[index] for index in ranking],
        "cases": [
            {
                "case": cases[case_index],
                "results": {
                    str(candidates[candidate_index].as_dict()): rows[candidate_index][case_index]
                    for candidate_index in range(len(candidates))
                },
            }
            for case_index in range(len(scenarios))
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    best = report["best"]
    lines = [
        "# ALNS hyperparameter optimization",
        "",
        f"Suite: `{manifest.get('suite', 'unknown')}` ({len(scenarios)} cases)",
        f"Candidates: `{len(candidates)}` · workers: `{worker_count}` · wall time: `{report['wall_seconds']:.3f}s`",
        "",
        "Candidates are ranked lexicographically by mean normalized distinct, daily, and serving percentages; runtime only breaks an exact score tie.",
        "",
        "## Best parameters",
        "",
        f"- fixed iterations: `{best['parameters']['fixed_iterations']}`",
        f"- minimum iterations: `{best['parameters']['min_iterations']}`",
        f"- stagnation iterations: `{best['parameters']['stagnation_iterations']}`",
        f"- mean score: `{best['mean_distinct_percent']:.3f}% / {best['mean_daily_percent']:.3f}% / {best['mean_servings_percent']:.3f}%`",
        "",
        "## Candidate ranking",
        "",
        "| Rank | Fixed | Minimum | Stagnation | Distinct | Daily | Servings | Runtime (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, candidate in enumerate(report["candidates"], start=1):
        parameters = candidate["parameters"]
        lines.append(
            f"| {rank} | {parameters['fixed_iterations']} | {parameters['min_iterations']} | {parameters['stagnation_iterations']} | "
            f"{candidate['mean_distinct_percent']:.3f}% | {candidate['mean_daily_percent']:.3f}% | "
            f"{candidate['mean_servings_percent']:.3f}% | {candidate['runtime_seconds']:.3f} |"
        )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n")
    return report


# A descriptive alias for callers that use the shorter spelling.
optimize_alns = tune_alns
