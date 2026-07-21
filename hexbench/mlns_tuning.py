"""Resumable Tree-structured Parzen optimization for MLNS.

The optimizer deliberately keeps the official objective lexicographic.  TPE
splits completed trials into lexicographically good and bad groups, then seeks
parameter combinations that are more likely under the good group.  It never
needs to collapse distinct coverage, daily coverage, and servings into a
weighted scalar.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .runner import (
    OBJECTIVES,
    find_binary,
    normalized_performance,
    run_core,
    structural_optimum,
)


# This is a safety ceiling only. It is deliberately not part of the Bayesian
# search: Web/Competition requests with a deadline should be anytime-driven.
DEFAULT_ITERATION_CEILING = 10_000_000
DEFAULT_MIN_ITERATIONS = (0, 8, 16, 32, 64)
DEFAULT_STAGNATION_ITERATIONS = (0, 16, 32, 64, 96, 128, 256)
DEFAULT_FUTURE_DISCOUNTS = tuple(range(25, 101, 5))


def _positive_unique(values: Iterable[int], name: str, *, allow_zero: bool) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(value) for value in values))
    minimum = 0 if allow_zero else 1
    if not result or any(value < minimum for value in result):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must contain {qualifier} integers")
    return result


@dataclass(frozen=True, order=True)
class MLNSParameters:
    min_iterations: int
    stagnation_iterations: int
    future_discount_percent: int

    def __post_init__(self) -> None:
        if self.min_iterations < 0:
            raise ValueError("min_iterations cannot be negative")
        if self.stagnation_iterations < 0:
            raise ValueError("stagnation_iterations cannot be negative")
        if not 1 <= self.future_discount_percent <= 100:
            raise ValueError("future_discount_percent must be between 1 and 100")

    def as_dict(self) -> dict[str, int]:
        return {
            "min_iterations": self.min_iterations,
            "stagnation_iterations": self.stagnation_iterations,
            "future_discount_percent": self.future_discount_percent,
        }

    def as_search(
        self,
        time_limit_ms: int,
        iteration_ceiling: int = DEFAULT_ITERATION_CEILING,
    ) -> dict[str, int]:
        return {
            "timeLimitMs": time_limit_ms,
            "maxIterations": iteration_ceiling,
            "minIterations": self.min_iterations,
            "stagnationIterations": self.stagnation_iterations,
            "futureDiscountPercent": self.future_discount_percent,
        }


@dataclass(frozen=True)
class MLNSSearchSpace:
    min_iterations: tuple[int, ...]
    stagnation_iterations: tuple[int, ...]
    future_discount_percent: tuple[int, ...]

    @classmethod
    def build(
        cls,
        min_iterations: Iterable[int] = DEFAULT_MIN_ITERATIONS,
        stagnation_iterations: Iterable[int] = DEFAULT_STAGNATION_ITERATIONS,
        future_discount_percent: Iterable[int] = DEFAULT_FUTURE_DISCOUNTS,
    ) -> "MLNSSearchSpace":
        space = cls(
            _positive_unique(min_iterations, "min_iterations", allow_zero=True),
            _positive_unique(
                stagnation_iterations, "stagnation_iterations", allow_zero=True
            ),
            _positive_unique(
                future_discount_percent,
                "future_discount_percent",
                allow_zero=False,
            ),
        )
        if any(value > 100 for value in space.future_discount_percent):
            raise ValueError("future_discount_percent cannot exceed 100")
        if not space.candidates():
            raise ValueError("MLNS search space has no valid candidates")
        return space

    def dimensions(self) -> tuple[tuple[int, ...], ...]:
        return (
            self.min_iterations,
            self.stagnation_iterations,
            self.future_discount_percent,
        )

    def candidates(self) -> list[MLNSParameters]:
        return [
            MLNSParameters(minimum, stagnant, discount)
            for minimum in self.min_iterations
            for stagnant in self.stagnation_iterations
            for discount in self.future_discount_percent
        ]

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "min_iterations": list(self.min_iterations),
            "stagnation_iterations": list(self.stagnation_iterations),
            "future_discount_percent": list(self.future_discount_percent),
        }


def _parameters_tuple(parameters: MLNSParameters) -> tuple[int, int, int]:
    return (
        parameters.min_iterations,
        parameters.stagnation_iterations,
        parameters.future_discount_percent,
    )


def _kernel_density(
    value: int,
    observations: Sequence[int],
    choices: tuple[int, ...],
) -> float:
    """Ordinal Parzen density with a uniform prior.

    Search values are intentionally discrete.  Measuring distance in choice
    indexes prevents a wide numeric range (iterations) from dominating a
    narrow one (discount percentages).
    """

    if not observations:
        return 1.0 / len(choices)
    indexes = {choice: index for index, choice in enumerate(choices)}
    target = indexes[value]
    bandwidth = max(0.75, len(choices) / math.sqrt(12.0 * len(observations)))
    kernel = sum(
        math.exp(-0.5 * ((target - indexes[item]) / bandwidth) ** 2)
        for item in observations
    )
    # One uniform pseudo-observation prevents zero-density traps.
    return (kernel + 1.0 / len(choices)) / (len(observations) + 1.0)


def propose_tpe(
    space: MLNSSearchSpace,
    completed: Sequence[tuple[MLNSParameters, Sequence[float]]],
    *,
    seed: int,
    startup_trials: int = 8,
    candidate_pool: int = 256,
    gamma: float = 0.25,
) -> tuple[MLNSParameters, str]:
    """Choose the next unevaluated configuration using a discrete TPE model."""

    if startup_trials < 1:
        raise ValueError("startup_trials must be positive")
    if candidate_pool < 1:
        raise ValueError("candidate_pool must be positive")
    if not 0 < gamma < 1:
        raise ValueError("gamma must be between zero and one")
    all_candidates = space.candidates()
    used = {parameters for parameters, _ in completed}
    remaining = [candidate for candidate in all_candidates if candidate not in used]
    if not remaining:
        raise ValueError("requested trials exceed the MLNS search space")
    rng = random.Random(seed + len(completed) * 1_000_003)

    default = MLNSParameters(32, 16, 90)
    if not completed and default in remaining:
        return default, "default"
    if len(completed) < startup_trials:
        return rng.choice(remaining), "random"

    ordered = sorted(completed, key=lambda item: tuple(item[1]), reverse=True)
    good_count = max(2, min(len(ordered) - 1, math.ceil(len(ordered) * gamma)))
    good = [item[0] for item in ordered[:good_count]]
    bad = [item[0] for item in ordered[good_count:]]
    dimensions = space.dimensions()
    good_values = list(zip(*(_parameters_tuple(item) for item in good), strict=True))
    bad_values = list(zip(*(_parameters_tuple(item) for item in bad), strict=True))

    # A bounded pool is enough for TPE and avoids enumerating a potentially
    # large user-supplied Cartesian product for every trial.
    pool = remaining if len(remaining) <= candidate_pool else rng.sample(remaining, candidate_pool)

    def likelihood_ratio(parameters: MLNSParameters) -> float:
        score = 0.0
        for value, choices, good_observations, bad_observations in zip(
            _parameters_tuple(parameters), dimensions, good_values, bad_values, strict=True
        ):
            good_density = _kernel_density(value, good_observations, choices)
            bad_density = _kernel_density(value, bad_observations, choices)
            score += math.log(good_density) - math.log(bad_density)
        return score

    best = max(pool, key=lambda candidate: (likelihood_ratio(candidate), rng.random()))
    return best, "tpe"


def _aggregate(parameters: MLNSParameters, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    case_count = len(rows)
    invalid_days = sum(int(row["invalid_days"]) for row in rows)
    runtime = sum(float(row["runtime_seconds"]) for row in rows)
    percentages = {
        objective: sum(float(row["objective_percentages"][objective]) for row in rows)
        / case_count
        for objective in OBJECTIVES
    }
    result: dict[str, Any] = {
        "parameters": parameters.as_dict(),
        "case_count": case_count,
        "valid_cases": sum(int(row["invalid_days"] == 0) for row in rows),
        "invalid_days": invalid_days,
        "runtime_seconds": runtime,
        "mean_distinct_percent": percentages["distinct_types"],
        "mean_daily_percent": percentages["cumulative_daily_types"],
        "mean_servings_percent": percentages["total_servings"],
    }
    result["score_key"] = [
        int(invalid_days == 0),
        -invalid_days,
        *(percentages[objective] for objective in OBJECTIVES),
        -runtime,
    ]
    return result


def _profile_aggregates(
    cases: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case, row in zip(cases, rows, strict=True):
        grouped.setdefault(str(case.get("profile", "unprofiled")), []).append(row)
    return {
        profile: {
            "case_count": len(items),
            "invalid_days": sum(int(item["invalid_days"]) for item in items),
            **{
                f"mean_{name}": sum(
                    float(item["objective_percentages"][objective]) for item in items
                )
                / len(items)
                for objective, name in zip(
                    OBJECTIVES,
                    ("distinct_percent", "daily_percent", "servings_percent"),
                    strict=True,
                )
            },
        }
        for profile, items in sorted(grouped.items())
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _comparison(best_rows: Sequence[dict[str, Any]], baseline_rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {"wins": 0, "ties": 0, "losses": 0}
    for best, baseline in zip(best_rows, baseline_rows, strict=True):
        left = tuple(int(best["score"][objective]) for objective in OBJECTIVES)
        right = tuple(int(baseline["score"][objective]) for objective in OBJECTIVES)
        counts["wins" if left > right else "losses" if left < right else "ties"] += 1
    return counts


def optimize_mlns(
    manifest_path: Path,
    report_dir: Path,
    *,
    trials: int = 32,
    startup_trials: int = 8,
    time_limit_ms: int = 1_000,
    min_iterations: Iterable[int] = DEFAULT_MIN_ITERATIONS,
    stagnation_iterations: Iterable[int] = DEFAULT_STAGNATION_ITERATIONS,
    future_discount_percent: Iterable[int] = DEFAULT_FUTURE_DISCOUNTS,
    seed: int = 20260720,
    binary_path: str | None = None,
    jobs: int | None = None,
    timeout: float = 180,
    expected_cases: int | None = 96,
    resume: bool = True,
    iteration_ceiling: int = DEFAULT_ITERATION_CEILING,
) -> dict[str, Any]:
    """Optimize MLNS over a complete local suite and persist every trial.

    A trial always evaluates every manifest case.  ``state.json`` is updated
    after each case, so an interrupted multi-hour search resumes without
    repeating completed solver calls.
    """

    if trials < 1:
        raise ValueError("trials must be positive")
    if time_limit_ms < 0:
        raise ValueError("time_limit_ms cannot be negative")
    if iteration_ceiling < 1:
        raise ValueError("iteration_ceiling must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    manifest_path = Path(manifest_path)
    report_dir = Path(report_dir)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    cases = manifest.get("cases", [])
    if not cases:
        raise ValueError("manifest contains no cases")
    if expected_cases is not None and len(cases) != expected_cases:
        raise ValueError(
            f"expected {expected_cases} cases for MLNS tuning, found {len(cases)}"
        )
    scenarios = [
        json.loads((manifest_path.parent / case["path"]).read_text()) for case in cases
    ]
    space = MLNSSearchSpace.build(
        min_iterations,
        stagnation_iterations,
        future_discount_percent,
    )
    if trials > len(space.candidates()):
        raise ValueError("trials exceed the number of valid parameter combinations")
    binary = find_binary(binary_path)
    binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    worker_count = max(1, jobs or min(os.cpu_count() or 1, 8))
    signature = {
        "schema_version": 1,
        "method": "mlns",
        "manifest_sha256": manifest_sha256,
        "binary_sha256": binary_sha256,
        "case_count": len(cases),
        "time_limit_ms": time_limit_ms,
        "iteration_ceiling": iteration_ceiling,
        "search_space": space.as_dict(),
        "seed": seed,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    state_path = report_dir / "state.json"
    state: dict[str, Any]
    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        actual_signature = {key: state.get(key) for key in signature}
        if actual_signature != signature:
            raise ValueError(
                "existing MLNS tuning state does not match this manifest, binary, "
                "budget, seed, or search space; use --no-resume or another report directory"
            )
    else:
        state = {**signature, "suite": manifest.get("suite"), "trials": []}
        _atomic_json(state_path, state)

    started = time.perf_counter()

    def completed_for_model() -> list[tuple[MLNSParameters, Sequence[float]]]:
        completed: list[tuple[MLNSParameters, Sequence[float]]] = []
        for trial in state["trials"]:
            if trial.get("aggregate") is None:
                continue
            completed.append(
                (MLNSParameters(**trial["parameters"]), trial["aggregate"]["score_key"])
            )
        return completed

    while len(state["trials"]) < trials or any(
        trial.get("aggregate") is None for trial in state["trials"]
    ):
        incomplete = next(
            (trial for trial in state["trials"] if trial.get("aggregate") is None),
            None,
        )
        if incomplete is None:
            parameters, source = propose_tpe(
                space,
                completed_for_model(),
                seed=seed,
                startup_trials=startup_trials,
            )
            incomplete = {
                "index": len(state["trials"]),
                "source": source,
                "parameters": parameters.as_dict(),
                "results": [None] * len(cases),
                "aggregate": None,
            }
            state["trials"].append(incomplete)
            _atomic_json(state_path, state)
        parameters = MLNSParameters(**incomplete["parameters"])

        def evaluate(case_index: int) -> tuple[int, dict[str, Any]]:
            case_started = time.perf_counter()
            result = run_core(
                "eval",
                "mlns",
                {
                    **scenarios[case_index],
                    "search": parameters.as_search(time_limit_ms, iteration_ceiling),
                },
                binary=binary,
                timeout=timeout,
                core_threads=1 if worker_count > 1 else None,
            )
            result["runtime_seconds"] = time.perf_counter() - case_started
            result.update(
                normalized_performance(result, structural_optimum(scenarios[case_index]))
            )
            return case_index, result

        missing = [
            index for index, result in enumerate(incomplete["results"]) if result is None
        ]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures: dict[Future[tuple[int, dict[str, Any]]], int] = {
                executor.submit(evaluate, case_index): case_index for case_index in missing
            }
            for future in as_completed(futures):
                case_index, result = future.result()
                incomplete["results"][case_index] = result
                _atomic_json(state_path, state)
        rows = incomplete["results"]
        if any(row is None for row in rows):
            raise RuntimeError("MLNS trial ended with missing case results")
        incomplete["aggregate"] = _aggregate(parameters, rows)
        incomplete["profiles"] = _profile_aggregates(cases, rows)
        _atomic_json(state_path, state)

    complete_trials = [trial for trial in state["trials"] if trial["aggregate"]]
    ranking = sorted(
        complete_trials,
        key=lambda trial: tuple(trial["aggregate"]["score_key"]),
        reverse=True,
    )
    best = ranking[0]
    default = next(
        (
            trial
            for trial in complete_trials
            if MLNSParameters(**trial["parameters"]) == MLNSParameters(32, 16, 90)
        ),
        None,
    )
    report: dict[str, Any] = {
        **signature,
        "suite": manifest.get("suite"),
        "optimizer": {
            "name": "tree_structured_parzen_estimator",
            "startup_trials": startup_trials,
            "lexicographic_good_fraction": 0.25,
        },
        "jobs": worker_count,
        "requested_trials": trials,
        "completed_trials": len(complete_trials),
        "wall_seconds_this_run": time.perf_counter() - started,
        "objective_order": list(OBJECTIVES),
        "best": {**best["aggregate"], "profiles": best["profiles"]},
        "trials": [
            {
                "rank": rank,
                "index": trial["index"],
                "source": trial["source"],
                **trial["aggregate"],
                "profiles": trial["profiles"],
            }
            for rank, trial in enumerate(ranking, start=1)
        ],
        "versus_default": (
            _comparison(best["results"], default["results"]) if default else None
        ),
        "case_results": [
            {
                "case": case,
                "best": best["results"][index],
                "default": default["results"][index] if default else None,
            }
            for index, case in enumerate(cases)
        ],
    }
    _atomic_json(report_dir / "report.json", report)
    _atomic_json(
        report_dir / "best-search.json",
        {
            "method": "mlns",
            "hyperparameters": best["parameters"],
            # Deliberately omit the resource controls (time/max iterations):
            # this is a Web/UI-ready anytime configuration. The UI supplies
            # its own deadline and the evaluator supplies its own ceiling.
            "search": {
                "minIterations": best["parameters"]["min_iterations"],
                "stagnationIterations": best["parameters"][
                    "stagnation_iterations"
                ],
                "futureDiscountPercent": best["parameters"][
                    "future_discount_percent"
                ],
            },
        },
    )
    best_aggregate = best["aggregate"]
    lines = [
        "# MLNS Bayesian hyperparameter optimization",
        "",
        f"Suite: `{manifest.get('suite', 'unknown')}` ({len(cases)} cases)",
        f"Trials: `{len(complete_trials)}` · workers: `{worker_count}` · fixed budget: `{time_limit_ms} ms/day`",
        "",
        "Trials are ranked lexicographically by validity, normalized distinct coverage, daily coverage, and servings. Runtime only breaks an exact quality tie.",
        "",
        "## Best parameters",
        "",
        *(
            f"- {key.replace('_', ' ')}: `{value}`"
            for key, value in best["parameters"].items()
        ),
        f"- mean score: `{best_aggregate['mean_distinct_percent']:.3f}% / {best_aggregate['mean_daily_percent']:.3f}% / {best_aggregate['mean_servings_percent']:.3f}%`",
        f"- valid cases: `{best_aggregate['valid_cases']}/{len(cases)}`",
        "",
        "## Trial ranking",
        "",
        "| Rank | Source | Min | Stagnation | Discount | Distinct | Daily | Servings | Runtime (s) |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, trial in enumerate(ranking, start=1):
        parameters = trial["parameters"]
        aggregate = trial["aggregate"]
        lines.append(
            f"| {rank} | {trial['source']} | {parameters['min_iterations']} | "
            f"{parameters['stagnation_iterations']} | "
            f"{parameters['future_discount_percent']}% | "
            f"{aggregate['mean_distinct_percent']:.3f}% | "
            f"{aggregate['mean_daily_percent']:.3f}% | "
            f"{aggregate['mean_servings_percent']:.3f}% | "
            f"{aggregate['runtime_seconds']:.3f} |"
        )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n")
    return report


# Match the naming used by the deterministic ALNS and PALNS tuners.
tune_mlns = optimize_mlns
