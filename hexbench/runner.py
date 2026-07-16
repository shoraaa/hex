from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def find_binary(explicit: str | None = None) -> Path:
    candidates = [Path(explicit)] if explicit else []
    candidates.extend((ROOT / "build" / "hexudon", ROOT / "hexudon"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("hexudon binary not found; run cmake --build build")


def run_core(
    command: str,
    policy: str,
    payload: dict[str, Any] | list[Any],
    *,
    binary: Path,
    timeout: float = 60,
    core_threads: int | None = None,
) -> Any:
    environment = None
    if core_threads is not None:
        environment = dict(os.environ)
        environment["HEXUDON_THREADS"] = str(max(1, core_threads))
    completed = subprocess.run(
        [str(binary), command, policy],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=environment,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "hexudon core failed")
    return json.loads(completed.stdout)


def _grade_key(result: dict[str, Any]) -> tuple[int, int, int, int]:
    score = result["score"]
    return (
        int(result["invalid_days"] == 0),
        score["distinct_types"],
        score["cumulative_daily_types"],
        score["total_servings"],
    )


def grade_suite(
    manifest_path: Path,
    method: str,
    baselines: list[str],
    report_dir: Path,
    binary_path: str | None = None,
    jobs: int | None = None,
    timeout: float = 60,
) -> dict[str, Any]:
    binary = find_binary(binary_path)
    manifest = json.loads(manifest_path.read_text())
    methods = list(dict.fromkeys([method, *baselines]))
    worker_count = max(1, jobs or min(os.cpu_count() or 1, 8))
    scenarios = [
        json.loads((manifest_path.parent / case["path"]).read_text())
        for case in manifest["cases"]
    ]
    rows: list[dict[str, Any]] = [
        {"case": case, "results": {}} for case in manifest["cases"]
    ]
    aggregates = {
        name: {
            "valid_cases": 0,
            "invalid_days": 0,
            "distinct_types": 0,
            "cumulative_daily_types": 0,
            "total_servings": 0,
            "runtime_seconds": 0.0,
        }
        for name in methods
    }
    tasks = [
        (case_index, name)
        for case_index in range(len(scenarios))
        for name in methods
    ]

    def evaluate(task: tuple[int, str]) -> tuple[int, str, dict[str, Any]]:
        case_index, name = task
        started = time.perf_counter()
        result = run_core(
            "eval",
            name,
            scenarios[case_index],
            binary=binary,
            timeout=timeout,
            core_threads=1 if worker_count > 1 else None,
        )
        result["runtime_seconds"] = time.perf_counter() - started
        return case_index, name, result

    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        evaluated = list(executor.map(evaluate, tasks))
    wall_seconds = time.perf_counter() - wall_started
    for case_index, name, result in evaluated:
        rows[case_index]["results"][name] = result
        aggregate = aggregates[name]
        aggregate["valid_cases"] += int(result["invalid_days"] == 0)
        aggregate["invalid_days"] += result["invalid_days"]
        aggregate["runtime_seconds"] += result["runtime_seconds"]
        for objective in (
            "distinct_types",
            "cumulative_daily_types",
            "total_servings",
        ):
            aggregate[objective] += result["score"][objective]

    comparisons: dict[str, dict[str, int]] = {}
    for baseline in baselines:
        if baseline == method:
            continue
        count = {"wins": 0, "ties": 0, "losses": 0}
        for row in rows:
            own = _grade_key(row["results"][method])
            other = _grade_key(row["results"][baseline])
            key = "wins" if own > other else ("losses" if own < other else "ties")
            count[key] += 1
        comparisons[baseline] = count

    case_count = len(rows)
    for aggregate in aggregates.values():
        for key in (
            "distinct_types",
            "cumulative_daily_types",
            "total_servings",
            "runtime_seconds",
        ):
            aggregate[f"mean_{key}"] = aggregate[key] / case_count if case_count else 0

    report = {
        "schema_version": 1,
        "suite": manifest["suite"],
        "method": method,
        "case_count": case_count,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "jobs": worker_count,
        "wall_seconds": wall_seconds,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "cases": rows,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        f"# HEXUDON benchmark: {method}",
        "",
        f"Suite: `{manifest['suite']}` ({case_count} cases)",
        f"Workers: `{worker_count}` · wall time: `{wall_seconds:.3f}s`",
        "",
        "| Method | Valid cases | Invalid days | Mean brands | Mean daily brands | Mean servings | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, aggregate in aggregates.items():
        lines.append(
            f"| {name} | {aggregate['valid_cases']}/{case_count} | "
            f"{aggregate['invalid_days']} | {aggregate['mean_distinct_types']:.2f} | "
            f"{aggregate['mean_cumulative_daily_types']:.2f} | "
            f"{aggregate['mean_total_servings']:.2f} | "
            f"{aggregate['runtime_seconds']:.3f} |"
        )
    if comparisons:
        lines.extend(("", "## Pairwise lexicographic result", ""))
        for baseline, counts in comparisons.items():
            lines.append(
                f"- vs `{baseline}`: {counts['wins']} wins, {counts['ties']} ties, "
                f"{counts['losses']} losses"
            )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n")
    return report
