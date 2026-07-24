"""Component profiler for the production MLNS + LNS-DP pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Mapping

from .runner import find_binary, run_core

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITES = {
    "brutal": ROOT / "cases/mlns-profile/brutal/manifest.json",
    "steady": ROOT / "cases/mlns-profile/steady/manifest.json",
    "easy": ROOT / "cases/mlns-profile/easy/manifest.json",
    "online": ROOT / "cases/mlns-profile/online/manifest.json",
}
VECTOR_FIELDS = ("current_score_gain", "projected_score_gain")


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _log_progress(message: str, *, enabled: bool) -> None:
    if enabled:
        print(f"[mlns-profile] {message}", file=sys.stderr, flush=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _empty_component(name: str) -> dict[str, Any]:
    return {
        "component": name,
        "calls": 0,
        "elapsed_microseconds": 0,
        "incumbent_updates": 0,
        "final_selections": 0,
        "current_score_gain": [0, 0, 0],
        "projected_score_gain": [0, 0, 0],
        "ending_patrol_fuel_gain": 0,
    }


def _merge_component(destination: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key in (
        "calls",
        "elapsed_microseconds",
        "incumbent_updates",
        "final_selections",
        "ending_patrol_fuel_gain",
    ):
        destination[key] += int(source.get(key, 0))
    for key in VECTOR_FIELDS:
        values = source.get(key, (0, 0, 0))
        for index in range(3):
            destination[key][index] += int(values[index])


def _finalize_aggregate(aggregate: dict[str, Any]) -> None:
    planner_us = int(aggregate["elapsed_microseconds"])
    planner_calls = int(aggregate["planner_calls"])
    accounted_us = sum(
        int(component["elapsed_microseconds"])
        for component in aggregate["components"].values()
    )
    residual_us = max(0, planner_us - accounted_us)
    if residual_us:
        residual = _empty_component("other_overhead")
        residual["calls"] = planner_calls
        residual["elapsed_microseconds"] = residual_us
        aggregate["components"]["other_overhead"] = residual

    finalized: list[dict[str, Any]] = []
    for component in aggregate["components"].values():
        elapsed_us = int(component["elapsed_microseconds"])
        calls = int(component["calls"])
        updates = int(component["incumbent_updates"])
        selections = int(component["final_selections"])
        time_percentage = 100.0 * elapsed_us / planner_us if planner_us else 0.0
        row = {
            **component,
            "elapsed_seconds": elapsed_us / 1_000_000,
            "time_percentage": time_percentage,
            "productive_call_percentage": 100.0 * updates / calls if calls else 0.0,
            "final_selection_percentage": (
                100.0 * selections / planner_calls if planner_calls else 0.0
            ),
        }
        if time_percentage >= 5.0 and updates == 0 and selections == 0:
            row["signal"] = "review: >=5% time with no attributed gain"
        elif time_percentage <= 5.0 and (updates > 0 or selections > 0):
            row["signal"] = "high leverage: gain at <=5% time"
        else:
            row["signal"] = ""
        finalized.append(row)
    finalized.sort(key=lambda row: (-row["elapsed_microseconds"], row["component"]))
    aggregate["accounted_microseconds"] = accounted_us + residual_us
    aggregate["components"] = finalized


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "case_count": len(rows),
        "valid_cases": 0,
        "external_runtime_seconds": 0.0,
        "planned_budget_seconds": 0.0,
        "budget_overrun_seconds": 0.0,
        "planner_calls": 0,
        "elapsed_microseconds": 0,
        "score": {
            "distinct_types": 0,
            "cumulative_daily_types": 0,
            "total_servings": 0,
        },
        "components": {},
    }
    for row in rows:
        result = row["result"]
        aggregate["valid_cases"] += int(result["invalid_days"] == 0)
        aggregate["external_runtime_seconds"] += row["runtime_seconds"]
        aggregate["planned_budget_seconds"] += row["planned_budget_seconds"]
        aggregate["budget_overrun_seconds"] += max(
            0.0, row["runtime_seconds"] - row["planned_budget_seconds"]
        )
        for objective in aggregate["score"]:
            aggregate["score"][objective] += int(result["score"][objective])
        profile = result.get("mlns_profile")
        if not profile:
            raise RuntimeError(
                "core result has no mlns_profile; rebuild the instrumented binary"
            )
        aggregate["planner_calls"] += int(profile["planner_calls"])
        aggregate["elapsed_microseconds"] += int(profile["elapsed_microseconds"])
        for component in profile["components"]:
            name = str(component["component"])
            destination = aggregate["components"].setdefault(
                name, _empty_component(name)
            )
            _merge_component(destination, component)
    _finalize_aggregate(aggregate)
    planned_seconds = aggregate["planned_budget_seconds"]
    aggregate["runtime_to_budget_percentage"] = (
        100.0 * aggregate["external_runtime_seconds"] / planned_seconds
        if planned_seconds
        else 0.0
    )
    return aggregate


def _vector(values: list[int]) -> str:
    return "/".join(f"{value:+d}" for value in values)


def _component_table(aggregate: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Component | Time | Time % | Calls | Productive | Final source | Current gain D/Daily/S | Projected gain D/Daily/S | Fuel gain | Signal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in aggregate["components"]:
        lines.append(
            f"| `{row['component']}` | {row['elapsed_seconds']:.3f}s | "
            f"{row['time_percentage']:.2f}% | {row['calls']} | "
            f"{row['incumbent_updates']} ({row['productive_call_percentage']:.1f}%) | "
            f"{row['final_selections']} ({row['final_selection_percentage']:.1f}%) | "
            f"{_vector(row['current_score_gain'])} | "
            f"{_vector(row['projected_score_gain'])} | "
            f"{row['ending_patrol_fuel_gain']:+d} | {row['signal']} |"
        )
    return lines


def _write_markdown(report: Mapping[str, Any], destination: Path) -> None:
    lines = [
        "# MLNS + DP component profile",
        "",
        f"Per-day wall-clock limit: `{report['time_limit_ms']} ms`; "
        f"workers: `{report['jobs']}`; core threads per case: `{report['core_threads']}`.",
        f"Neighborhood mode: `{report['neighborhood_mode']}`.",
        f"The per-case watchdog uses max(`{report['minimum_watchdog_seconds']:.0f}s`, "
        "`1.25 x planned case budget + 60s`) and is separate from the solver budget.",
        "",
        "Gain is observational phase-boundary attribution in the production run. "
        "Current gain is the authoritative score change for the submitted day; "
        "projected gain is the predicted final-match score change. It is not a "
        "leave-one-component-out causal ablation.",
        "",
        "## Suite summary",
        "",
        "| Suite | Valid | Planner calls | Planner time | External / planned | Score D/Daily/S |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, suite in report["suites"].items():
        aggregate = suite["aggregate"]
        score = aggregate["score"]
        lines.append(
            f"| {name} | {aggregate['valid_cases']}/{aggregate['case_count']} | "
            f"{aggregate['planner_calls']} | "
            f"{aggregate['elapsed_microseconds'] / 1_000_000:.3f}s | "
            f"{aggregate['external_runtime_seconds']:.3f}s / "
            f"{aggregate['planned_budget_seconds']:.3f}s "
            f"({aggregate['runtime_to_budget_percentage']:.1f}%) | "
            f"{score['distinct_types']}/{score['cumulative_daily_types']}/"
            f"{score['total_servings']} |"
        )
    lines.extend(("", "## Overall components", ""))
    lines.extend(_component_table(report["aggregate"]))
    for name, suite in report["suites"].items():
        lines.extend(("", f"## {name}", ""))
        lines.extend(_component_table(suite["aggregate"]))
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            "- `Productive` counts phase calls that changed the incumbent; the gain vectors sum the before/after deltas for those calls.",
            "- `Final source` counts days whose submitted plan last came from that component.",
            "- `other_overhead` is total MLNS wall time not enclosed by a named phase (setup, ranking, callbacks, and bookkeeping).",
            "- A zero gain can still support downstream search. Confirm removal candidates with a follow-up ablation before deleting them.",
            "- The `online` suite uses all seven saved online-practice fixtures (Q01-Q06 plus New Question), so this run is reproducible and does not require a live token.",
        )
    )
    destination.write_text("\n".join(lines) + "\n")


def profile_mlns(
    suites: Mapping[str, Path],
    report_dir: Path,
    *,
    time_limit_ms: int = 5_000,
    binary_path: str | None = None,
    jobs: int | None = None,
    timeout: float = 120.0,
    core_threads: int = 1,
    max_cases: int | None = None,
    log_interval_seconds: float = 10.0,
    progress: bool = True,
    resume: bool = True,
    neighborhood_mode: str = "current",
) -> dict[str, Any]:
    if time_limit_ms < 1_000:
        raise ValueError("MLNS + DP profiling requires --time-limit-ms >= 1000")
    if log_interval_seconds <= 0:
        raise ValueError("log_interval_seconds must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if neighborhood_mode not in {"current", "refined", "disabled"}:
        raise ValueError(
            "neighborhood_mode must be current, refined, or disabled"
        )
    binary = find_binary(binary_path)
    binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    worker_count = max(1, jobs or min(os.cpu_count() or 1, 4))
    suite_inputs: dict[str, dict[str, Any]] = {}
    tasks: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = []
    for name, manifest_path in suites.items():
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text())
        entries = list(manifest.get("cases", []))
        if max_cases is not None:
            entries = entries[:max_cases]
        if not entries:
            raise ValueError(f"suite '{name}' contains no cases")
        suite_inputs[name] = {
            "manifest": str(path),
            "suite": manifest.get("suite", name),
            "rows": [None] * len(entries),
        }
        for index, entry in enumerate(entries):
            scenario = json.loads((path.parent / entry["path"]).read_text())
            search = dict(scenario.get("search", {}))
            search.update(
                {
                    "timeLimitMs": time_limit_ms,
                    "maxIterations": 10_000_000,
                    "stagnationIterations": 0,
                    "useLnsDpProposals": True,
                }
            )
            scenario["search"] = search
            tasks.append((name, index, entry, scenario))

    def task_key(
        task: tuple[str, int, dict[str, Any], dict[str, Any]]
    ) -> str:
        name, index, entry, _ = task
        return f"{name}:{index}:{entry.get('path', index)}"

    signature = {
        "schema_version": 1,
        "binary_sha256": binary_sha256,
        "time_limit_ms": time_limit_ms,
        "core_threads": core_threads,
        "neighborhood_mode": neighborhood_mode,
        "cases": {
            task_key(task): hashlib.sha256(
                json.dumps(task[3], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for task in tasks
        },
    }
    task_keys_by_position = {(task[0], task[1]): task_key(task) for task in tasks}
    report_dir.mkdir(parents=True, exist_ok=True)
    progress_path = report_dir / "progress.json"
    checkpoint_rows: dict[str, dict[str, Any]] = {}
    if resume and progress_path.is_file():
        try:
            saved = json.loads(progress_path.read_text())
            if saved.get("signature") == signature and isinstance(
                saved.get("rows"), dict
            ):
                checkpoint_rows = saved["rows"]
            else:
                _log_progress(
                    f"ignoring incompatible checkpoint {progress_path}",
                    enabled=progress,
                )
        except (OSError, ValueError):
            _log_progress(
                f"ignoring unreadable checkpoint {progress_path}",
                enabled=progress,
            )
    for task in tasks:
        key = task_key(task)
        if key in checkpoint_rows:
            suite_inputs[task[0]]["rows"][task[1]] = checkpoint_rows[key]

    def evaluate(
        task: tuple[str, int, dict[str, Any], dict[str, Any]]
    ) -> tuple[str, int, dict[str, Any]]:
        name, index, entry, scenario = task
        day_count = len(scenario["config"]["daySteps"])
        planned_budget_seconds = day_count * time_limit_ms / 1_000
        watchdog_seconds = max(timeout, planned_budget_seconds * 1.25 + 60.0)
        started = time.perf_counter()
        result = run_core(
            "eval",
            "mlns",
            scenario,
            binary=binary,
            timeout=watchdog_seconds,
            core_threads=core_threads,
            environment_overrides={
                "HEXUDON_MLNS_NEIGHBORHOOD_MODE": (
                    None if neighborhood_mode == "current" else neighborhood_mode
                )
            },
        )
        return name, index, {
            "case": entry,
            "runtime_seconds": time.perf_counter() - started,
            "planned_budget_seconds": planned_budget_seconds,
            "watchdog_seconds": watchdog_seconds,
            "result": result,
        }

    total_tasks = len(tasks)
    tasks_to_run = [task for task in tasks if task_key(task) not in checkpoint_rows]
    suite_completed = {
        name: sum(row is not None for row in suite["rows"])
        for name, suite in suite_inputs.items()
    }
    suite_totals = {
        name: len(suite["rows"]) for name, suite in suite_inputs.items()
    }
    scope = ", ".join(
        f"{name}={suite_totals[name]}" for name in suite_inputs
    )
    wall_started = time.perf_counter()
    initial_completed = total_tasks - len(tasks_to_run)
    _log_progress(
        f"starting {total_tasks} cases ({scope}); resumed={initial_completed}, "
        f"workers={worker_count}, core_threads={core_threads}, "
        f"limit={time_limit_ms}ms/day, minimum_watchdog={timeout:.0f}s",
        enabled=progress,
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures: dict[Future[tuple[str, int, dict[str, Any]]], tuple[str, int]] = {
            executor.submit(evaluate, task): (task[0], task[1])
            for task in tasks_to_run
        }
        pending = set(futures)
        completed = initial_completed
        run_completed = 0
        while pending:
            done, pending = wait(
                pending,
                timeout=log_interval_seconds,
                return_when=FIRST_COMPLETED,
            )
            elapsed = time.perf_counter() - wall_started
            if not done:
                eta = (
                    elapsed / run_completed * (total_tasks - completed)
                    if run_completed
                    else 0
                )
                eta_text = _duration(eta) if run_completed else "calculating"
                _log_progress(
                    f"heartbeat {completed}/{total_tasks} completed; "
                    f"remaining={len(pending)}, elapsed={_duration(elapsed)}, "
                    f"eta={eta_text}",
                    enabled=progress,
                )
                continue
            for future in done:
                task_name, task_index = futures[future]
                try:
                    name, index, row = future.result()
                except Exception:
                    _log_progress(
                        f"FAILED suite={task_name} case_index={task_index}; "
                        "see the exception below",
                        enabled=progress,
                    )
                    for remaining in pending:
                        remaining.cancel()
                    raise
                suite_inputs[name]["rows"][index] = row
                completed += 1
                run_completed += 1
                suite_completed[name] += 1
                checkpoint_rows[task_keys_by_position[(name, index)]] = row
                _atomic_json(
                    progress_path,
                    {
                        "signature": signature,
                        "completed": len(checkpoint_rows),
                        "total": total_tasks,
                        "updated_at_unix": time.time(),
                        "rows": checkpoint_rows,
                    },
                )
                elapsed = time.perf_counter() - wall_started
                eta = elapsed / run_completed * (total_tasks - completed)
                result = row["result"]
                score = result["score"]
                case = row["case"]
                case_name = case.get("name") or case.get("path") or str(index)
                _log_progress(
                    f"{completed}/{total_tasks} ({100.0 * completed / total_tasks:.1f}%) "
                    f"suite={name} {suite_completed[name]}/{suite_totals[name]} "
                    f"case={case_name} runtime={row['runtime_seconds']:.1f}s "
                    f"budget={row['planned_budget_seconds']:.0f}s "
                    f"watchdog={row['watchdog_seconds']:.0f}s "
                    f"valid={result['invalid_days'] == 0} "
                    f"score={score['distinct_types']}/"
                    f"{score['cumulative_daily_types']}/{score['total_servings']} "
                    f"elapsed={_duration(elapsed)} eta={_duration(eta)}",
                    enabled=progress,
                )
    wall_seconds = time.perf_counter() - wall_started
    _log_progress(
        f"completed {total_tasks}/{total_tasks} cases in {_duration(wall_seconds)}; "
        "aggregating component profiles",
        enabled=progress,
    )

    all_rows: list[dict[str, Any]] = []
    suite_reports: dict[str, Any] = {}
    for name, suite in suite_inputs.items():
        rows = suite.pop("rows")
        if any(row is None for row in rows):
            raise RuntimeError(f"suite '{name}' did not produce every case result")
        all_rows.extend(rows)
        suite_reports[name] = {
            **suite,
            "aggregate": _aggregate_rows(rows),
            "cases": rows,
        }
    report = {
        "schema_version": 1,
        "method": "mlns",
        "lns_dp_proposals": True,
        "time_limit_ms": time_limit_ms,
        "jobs": worker_count,
        "core_threads": core_threads,
        "neighborhood_mode": neighborhood_mode,
        "minimum_watchdog_seconds": timeout,
        "resumed_cases": initial_completed,
        "checkpoint": str(progress_path),
        "wall_seconds": wall_seconds,
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "suites": suite_reports,
        "aggregate": _aggregate_rows(all_rows),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, report_dir / "report.md")
    _log_progress(
        f"reports written: {report_dir / 'report.json'} and "
        f"{report_dir / 'report.md'}",
        enabled=progress,
    )
    return report
