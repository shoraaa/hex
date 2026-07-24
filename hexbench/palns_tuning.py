"""Deterministic tuning and validation for Projection ALNS."""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .runner import OBJECTIVES, find_binary, normalized_performance, run_core, structural_optimum


@dataclass(frozen=True, order=True)
class PALNSParameters:
    total_iterations: int
    projection_iterations: int
    restarts: int

    def as_search(self) -> dict[str, int]:
        return {
            "totalIterations": self.total_iterations,
            "palnsProjectionIterations": self.projection_iterations,
            "palnsRestarts": self.restarts,
        }

    def as_dict(self) -> dict[str, int]:
        return {
            "total_iterations": self.total_iterations,
            "projection_iterations": self.projection_iterations,
            "restarts": self.restarts,
        }


def _values(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(value) for value in values))
    if not result or any(value < 1 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


def _aggregate(parameters: PALNSParameters, rows: list[dict]) -> dict:
    percentages = {
        objective: sum(row["objective_percentages"][objective] for row in rows)
        / len(rows)
        for objective in OBJECTIVES
    }
    runtime = sum(float(row["runtime_seconds"]) for row in rows)
    diagnostics = {
        key: sum(int(row.get("palns_diagnostics", {}).get(key, 0)) for row in rows)
        for key in (
            "total_iterations",
            "iterations_used",
            "outer_iterations",
            "projection_iterations",
            "projection_requests",
            "projection_completed",
            "projection_cache_hits",
            "projection_iteration_fallbacks",
            "projection_deadline_fallbacks",
        )
    }
    fallbacks = (
        diagnostics["projection_iteration_fallbacks"]
        + diagnostics["projection_deadline_fallbacks"]
    )
    requests = diagnostics["projection_requests"]
    diagnostics["projection_fallback_percentage"] = (
        0.0 if requests == 0 else 100.0 * fallbacks / requests
    )
    score_key = [percentages[objective] for objective in OBJECTIVES] + [-runtime]
    return {
        "parameters": parameters.as_dict(),
        "objective_percentages": percentages,
        "runtime_seconds": runtime,
        "valid_cases": sum(row["invalid_days"] == 0 for row in rows),
        "diagnostics": diagnostics,
        "score_key": score_key,
    }


def tune_palns(
    manifest_path: Path,
    report_dir: Path,
    *,
    tuning_total_iterations: int = 1536,
    projection_iterations: Iterable[int] = (1, 2, 4, 8, 16, 32),
    restarts: Iterable[int] = (1, 2, 3),
    total_iteration_curve: Iterable[int] = (
        128,
        256,
        288,
        512,
        1024,
        1536,
        2048,
        3072,
        4096,
        6000,
    ),
    binary_path: str | None = None,
    jobs: int | None = None,
    timeout: float = 180,
) -> dict:
    manifest_path = Path(manifest_path)
    report_dir = Path(report_dir)
    manifest = json.loads(manifest_path.read_text())
    cases = manifest.get("cases", [])
    if not cases:
        raise ValueError("manifest contains no cases")
    scenarios = [
        json.loads((manifest_path.parent / case["path"]).read_text()) for case in cases
    ]
    projection_values = _values(projection_iterations, "projection_iterations")
    restart_values = _values(restarts, "restarts")
    curve_values = _values(total_iteration_curve, "total_iteration_curve")
    candidates = [
        PALNSParameters(tuning_total_iterations, depth, restart)
        for depth in projection_values
        for restart in restart_values
    ]
    binary = find_binary(binary_path)
    worker_count = max(1, jobs or min(os.cpu_count() or 1, 8))

    def run(
        method: str,
        scenario: dict,
        search: dict[str, int],
        *,
        per_case_timeout: float | None = None,
    ) -> dict:
        started = time.perf_counter()
        result = run_core(
            "eval",
            method,
            {**scenario, "search": search},
            binary=binary,
            timeout=timeout if per_case_timeout is None else per_case_timeout,
            core_threads=1 if worker_count > 1 else None,
        )
        result["runtime_seconds"] = time.perf_counter() - started
        result.update(normalized_performance(result, structural_optimum(scenario)))
        return result

    def evaluate_candidate(task: tuple[int, int]) -> tuple[int, int, dict]:
        candidate_index, case_index = task
        return (
            candidate_index,
            case_index,
            run("palns", scenarios[case_index], candidates[candidate_index].as_search()),
        )

    started = time.perf_counter()
    tasks = [
        (candidate_index, case_index)
        for candidate_index in range(len(candidates))
        for case_index in range(len(scenarios))
    ]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        outputs = list(executor.map(evaluate_candidate, tasks))
    rows: list[list[dict | None]] = [
        [None for _ in scenarios] for _ in candidates
    ]
    for candidate_index, case_index, result in outputs:
        rows[candidate_index][case_index] = result
    aggregates = [
        _aggregate(candidate, [row for row in candidate_rows if row is not None])
        for candidate, candidate_rows in zip(candidates, rows, strict=True)
    ]
    ranking = sorted(
        range(len(candidates)), key=lambda index: aggregates[index]["score_key"], reverse=True
    )
    best_parameters = candidates[ranking[0]]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "palns-interim.json").write_text(
        json.dumps(
            {
                "suite": manifest.get("suite"),
                "case_count": len(cases),
                "best_fixed_parameters": best_parameters.as_dict(),
                "tuning_candidates": [aggregates[index] for index in ranking],
            },
            indent=2,
        )
        + "\n"
    )

    curve_candidates = [
        PALNSParameters(total, best_parameters.projection_iterations, best_parameters.restarts)
        for total in curve_values
    ]

    def evaluate_curve(task: tuple[int, int]) -> tuple[int, int, dict]:
        candidate_index, case_index = task
        return (
            candidate_index,
            case_index,
            run("palns", scenarios[case_index], curve_candidates[candidate_index].as_search()),
        )

    curve_rows: list[list[dict | None]] = [
        [None for _ in scenarios] for _ in curve_candidates
    ]
    for curve_index, curve_candidate in enumerate(curve_candidates):
        if curve_candidate in candidates:
            source_index = candidates.index(curve_candidate)
            curve_rows[curve_index] = list(rows[source_index])
    curve_tasks = [
        (candidate_index, case_index)
        for candidate_index in range(len(curve_candidates))
        for case_index in range(len(scenarios))
        if curve_rows[candidate_index][case_index] is None
    ]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        curve_outputs = list(executor.map(evaluate_curve, curve_tasks))
        production_baselines = list(
            executor.map(
                lambda scenario: run(
                    "alns",
                    scenario,
                    {
                        "minIterations": 32,
                        "maxIterations": 96,
                        "stagnationIterations": 0,
                        "alnsRestarts": 3,
                    },
                    per_case_timeout=max(timeout, 600),
                ),
                scenarios,
            )
        )
        single_restart_baselines = list(
            executor.map(
                lambda scenario: run(
                    "alns",
                    scenario,
                    {
                        "minIterations": 32,
                        "maxIterations": 96,
                        "stagnationIterations": 0,
                        "alnsRestarts": 1,
                    },
                    per_case_timeout=max(timeout, 600),
                ),
                scenarios,
            )
        )
    for candidate_index, case_index, result in curve_outputs:
        curve_rows[candidate_index][case_index] = result
    curve = [
        _aggregate(candidate, [row for row in candidate_rows if row is not None])
        for candidate, candidate_rows in zip(curve_candidates, curve_rows, strict=True)
    ]

    def compare(candidate_rows: list[dict | None], baseline_rows: list[dict]) -> dict[str, int]:
        comparisons = {"wins": 0, "ties": 0, "losses": 0}
        for candidate, baseline in zip(candidate_rows, baseline_rows, strict=True):
            assert candidate is not None
            candidate_score = tuple(candidate["score"][objective] for objective in OBJECTIVES)
            baseline_score = tuple(baseline["score"][objective] for objective in OBJECTIVES)
            key = "wins" if candidate_score > baseline_score else "losses" if candidate_score < baseline_score else "ties"
            comparisons[key] += 1
        return comparisons

    for aggregate, candidate_rows in zip(curve, curve_rows, strict=True):
        aggregate["versus_alns"] = compare(candidate_rows, production_baselines)
        aggregate["versus_alns_single_restart"] = compare(
            candidate_rows, single_restart_baselines
        )
        tier_indexes: dict[str, list[int]] = {}
        for case_index, case in enumerate(cases):
            tier = str(
                case.get("source_tier")
                or (case.get("validation") or {}).get("source_tier")
                or case.get("profile")
                or "unknown"
            )
            tier_indexes.setdefault(tier, []).append(case_index)
        aggregate["tiers"] = {
            tier: _aggregate(
                PALNSParameters(**aggregate["parameters"]),
                [candidate_rows[index] for index in indexes if candidate_rows[index] is not None],
            )["objective_percentages"]
            for tier, indexes in tier_indexes.items()
        }

    production_baseline = _aggregate(
        PALNSParameters(288, 1, 3), production_baselines
    )
    production_baseline["parameters"] = {
        "max_iterations_per_restart": 96,
        "restarts": 3,
        "nominal_outer_iterations_per_day": 288,
    }
    single_restart_baseline = _aggregate(
        PALNSParameters(96, 1, 1), single_restart_baselines
    )
    single_restart_baseline["parameters"] = {
        "max_iterations_per_restart": 96,
        "restarts": 1,
        "nominal_outer_iterations_per_day": 96,
    }
    report = {
        "schema_version": 1,
        "suite": manifest.get("suite"),
        "case_count": len(cases),
        "method": "palns",
        "objective_order": list(OBJECTIVES),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "wall_seconds": time.perf_counter() - started,
        "best_fixed_parameters": best_parameters.as_dict(),
        "tuning_candidates": [aggregates[index] for index in ranking],
        "total_iteration_curve": curve,
        "baseline": production_baseline,
        "single_restart_baseline": single_restart_baseline,
    }
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# PALNS validation",
        "",
        f"Cases: `{len(cases)}`",
        f"Best fixed parameters: `{best_parameters.as_dict()}`",
        "",
        "| Total iterations | Distinct % | Daily % | Servings % | Fallback % | W/T/L vs ALNS-3 | W/T/L vs ALNS-1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in curve:
        score = row["objective_percentages"]
        comparison = row["versus_alns"]
        single_comparison = row["versus_alns_single_restart"]
        lines.append(
            f"| {row['parameters']['total_iterations']} | {score['distinct_types']:.5f} | "
            f"{score['cumulative_daily_types']:.5f} | {score['total_servings']:.5f} | "
            f"{row['diagnostics']['projection_fallback_percentage']:.2f} | "
            f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']} | "
            f"{single_comparison['wins']}/{single_comparison['ties']}/{single_comparison['losses']} |"
        )
    if curve:
        lines.extend(("", "## Per-tier scores at each total iteration budget", ""))
        for row in curve:
            lines.append(f"### {row['parameters']['total_iterations']} iterations")
            lines.extend(("", "| Tier | Distinct % | Daily % | Servings % |", "|---|---:|---:|---:|"))
            for tier, score in sorted(row["tiers"].items()):
                lines.append(
                    f"| {tier} | {score['distinct_types']:.5f} | "
                    f"{score['cumulative_daily_types']:.5f} | {score['total_servings']:.5f} |"
                )
            lines.append("")
    (report_dir / "report.md").write_text("\n".join(lines) + "\n")
    return report
