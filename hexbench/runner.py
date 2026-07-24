from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
OBJECTIVES = (
    "distinct_types",
    "cumulative_daily_types",
    "total_servings",
)


def _traffic_gnn_enabled(payload: dict[str, Any]) -> bool:
    for source in (
        payload,
        payload.get("search") if isinstance(payload.get("search"), dict) else {},
        payload.get("hyperparameters")
        if isinstance(payload.get("hyperparameters"), dict)
        else {},
    ):
        if source.get("use_traffic_gnn") or source.get("useTrafficGnn"):
            return True
    return os.environ.get("HEXUDON_USE_TRAFFIC_GNN", "") not in {"", "0"}


def _traffic_gnn_checkpoint(payload: dict[str, Any]) -> Path:
    explicit = payload.get("traffic_model_path") or os.environ.get(
        "HEXUDON_TRAFFIC_GNN_CHECKPOINT"
    )
    candidates = [Path(explicit)] if explicit else []
    candidates.append(ROOT / "reports" / "traffic-1k-test" / "model.pt")
    candidates.extend(
        sorted(
            (ROOT / "reports").glob("traffic-gnn*/model.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "GNN prediction requested but no traffic checkpoint was found; "
        "expected reports/traffic-1k-test/model.pt; set "
        "HEXUDON_TRAFFIC_GNN_CHECKPOINT or train traffic-train first"
    )


def _prepare_traffic_gnn_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not _traffic_gnn_enabled(payload) or "predictedTraffic" in payload:
        return payload
    from .traffic_gnn import predict_future_traffic

    scenario = dict(payload)
    if isinstance(payload.get("day_info"), dict):
        scenario["day_info"] = payload["day_info"]
        known_day = int(payload["day_info"].get("day", 0))
        known_traffic = {
            int(item["pos"]): int(item["status"])
            for item in payload["day_info"].get("traffics", [])
        }
    else:
        known_day = None
        known_traffic = None
    predictions = predict_future_traffic(
        scenario,
        _traffic_gnn_checkpoint(payload),
        known_day=known_day,
        known_traffic=known_traffic,
    )
    prepared = dict(payload)
    prepared["predictedTraffic"] = predictions
    return prepared
OPTIMUM_DEFINITION = (
    "Per-case structural upper bound: every brand is collected, every brand "
    "is collected on every day, and every spot serves up to one bowl per "
    "available agent per day; travel, fuel, and congestion feasibility are "
    "ignored."
)


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
    environment_overrides: dict[str, str | None] | None = None,
) -> Any:
    if isinstance(payload, dict) and policy == "mlns":
        payload = _prepare_traffic_gnn_payload(payload)
    environment = None
    if core_threads is not None or environment_overrides:
        environment = dict(os.environ)
    if core_threads is not None:
        assert environment is not None
        environment["HEXUDON_THREADS"] = str(max(1, core_threads))
    if environment_overrides:
        assert environment is not None
        for key, value in environment_overrides.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
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


def stream_core(
    policy: str,
    payload: dict[str, Any],
    *,
    binary: Path,
    on_improve: Callable[[dict[str, Any]], None],
    timeout: float = 60,
    should_stop: Callable[[], bool] | None = None,
    core_threads: int | None = None,
) -> dict[str, Any] | None:
    """Run the anytime `solve` command, calling `on_improve` per NDJSON line.

    The core streams one record with ``score``, ``internal_rank``, and ``actions``
    for every improving incumbent and terminates itself at ``search.timeLimitMs``.
    The ``timeout`` is a backstop watchdog: it should exceed the solver deadline
    plus a margin. Returns the final (best) record, or ``None`` if nothing streamed.
    """
    payload = _prepare_traffic_gnn_payload(payload) if policy == "mlns" else payload
    environment = None
    if core_threads is not None:
        environment = dict(os.environ)
        environment["HEXUDON_THREADS"] = str(max(1, core_threads))
    process = subprocess.Popen(
        [str(binary), "solve", policy],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    timed_out = threading.Event()

    def _kill() -> None:
        timed_out.set()
        process.kill()

    watchdog = threading.Timer(max(0.1, timeout), _kill)
    watchdog.start()
    last: dict[str, Any] | None = None
    stopped = False
    try:
        assert process.stdin is not None and process.stdout is not None
        # The core reads all of stdin before producing output, so writing the
        # whole payload and closing the pipe cannot deadlock against stdout.
        process.stdin.write(json.dumps(payload))
        process.stdin.close()
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            last = record
            if record.get("kind") != "final":
                on_improve(record)
            if should_stop is not None and should_stop():
                stopped = True
                process.kill()
                break
        process.wait()
    finally:
        watchdog.cancel()
    if timed_out.is_set() or stopped:
        # A backstop kill (deadline overrun) or a caller-requested stop leaves the
        # last streamed plan as the best answer we have.
        return last
    if process.returncode:
        stderr = (process.stderr.read() if process.stderr else "").strip()
        raise RuntimeError(stderr or "hexudon core solve failed")
    return last


def structural_optimum(scenario: dict[str, Any]) -> dict[str, int]:
    config = scenario["config"]
    days = len(config["daySteps"])
    agents = len(config["agents"])
    distinct = len({int(spot["brand"]) for spot in config["spots"]})
    daily_servings = sum(
        min(int(spot["stocks"]), agents) for spot in config["spots"]
    )
    return {
        "distinct_types": distinct,
        "cumulative_daily_types": distinct * days,
        "total_servings": daily_servings * days,
    }


def normalized_performance(
    result: dict[str, Any], optimum: dict[str, int]
) -> dict[str, Any]:
    percentages: dict[str, float] = {}
    valid = result["invalid_days"] == 0
    for objective in OBJECTIVES:
        denominator = optimum[objective]
        percentage = (
            100.0
            if denominator == 0
            else 100.0 * result["score"][objective] / denominator
        )
        percentages[objective] = min(100.0, percentage) if valid else 0.0
    return {
        "optimum_score": optimum,
        "objective_percentages": percentages,
    }


def grade_suite(
    manifest_path: Path,
    method: str,
    baselines: list[str],
    report_dir: Path,
    binary_path: str | None = None,
    jobs: int | None = None,
    timeout: float = 60,
    time_limit_ms: int | None = None,
    core_environment: dict[str, str | None] | None = None,
    core_threads: int | None = None,
    policy_hyperparameters: dict[str, dict[str, int | float | bool]] | None = None,
) -> dict[str, Any]:
    binary = find_binary(binary_path)
    manifest = json.loads(manifest_path.read_text())
    methods = list(dict.fromkeys([method, *baselines]))
    worker_count = max(1, jobs or min(os.cpu_count() or 1, 8))
    threads_per_core = (
        max(1, core_threads)
        if core_threads is not None
        else (1 if worker_count > 1 else None)
    )
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
            "distinct_percent": 0.0,
            "daily_percent": 0.0,
            "servings_percent": 0.0,
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
        scenario = scenarios[case_index]
        if time_limit_ms is not None:
            scenario = dict(scenario)
            scenario["search"] = {
                "timeLimitMs": time_limit_ms,
                "maxIterations": 10_000_000,
                "stagnationIterations": 0,
            }
        if policy_hyperparameters and name in policy_hyperparameters:
            scenario = dict(scenario)
            scenario["hyperparameters"] = dict(policy_hyperparameters[name])
        try:
            result = run_core(
                "eval",
                name,
                scenario,
                binary=binary,
                timeout=timeout,
                core_threads=threads_per_core,
                environment_overrides=core_environment,
            )
        except subprocess.TimeoutExpired as error:
            case_path = manifest["cases"][case_index]["path"]
            raise RuntimeError(
                f"policy '{name}' timed out on case '{case_path}' after "
                f"{timeout:.1f} seconds; raise --timeout or reduce --jobs"
            ) from error
        result["runtime_seconds"] = time.perf_counter() - started
        return case_index, name, result

    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        evaluated = list(executor.map(evaluate, tasks))
    wall_seconds = time.perf_counter() - wall_started
    for case_index, name, result in evaluated:
        optimum = structural_optimum(scenarios[case_index])
        performance = normalized_performance(result, optimum)
        result.update(performance)
        rows[case_index]["results"][name] = result
        rows[case_index]["optimum_score"] = optimum
        aggregate = aggregates[name]
        aggregate["valid_cases"] += int(result["invalid_days"] == 0)
        aggregate["invalid_days"] += result["invalid_days"]
        aggregate["runtime_seconds"] += result["runtime_seconds"]
        for objective in OBJECTIVES:
            aggregate[objective] += result["score"][objective]
        percentages = performance["objective_percentages"]
        aggregate["distinct_percent"] += percentages["distinct_types"]
        aggregate["daily_percent"] += percentages["cumulative_daily_types"]
        aggregate["servings_percent"] += percentages["total_servings"]

    case_count = len(rows)
    for aggregate in aggregates.values():
        for key in (
            "distinct_types",
            "cumulative_daily_types",
            "total_servings",
            "runtime_seconds",
        ):
            aggregate[f"mean_{key}"] = aggregate[key] / case_count if case_count else 0
        for key in (
            "distinct_percent",
            "daily_percent",
            "servings_percent",
        ):
            aggregate[f"mean_{key}"] = (
                aggregate[key] / case_count if case_count else 0.0
            )

    ranking = sorted(
        methods,
        key=lambda name: (
            aggregates[name]["mean_distinct_percent"],
            aggregates[name]["mean_daily_percent"],
            aggregates[name]["mean_servings_percent"],
            -aggregates[name]["runtime_seconds"],
        ),
        reverse=True,
    )
    comparisons: dict[str, dict[str, Any]] = {}
    own_average = {
        "distinct_types": aggregates[method]["mean_distinct_percent"],
        "cumulative_daily_types": aggregates[method]["mean_daily_percent"],
        "total_servings": aggregates[method]["mean_servings_percent"],
    }
    own_key = tuple(own_average[objective] for objective in OBJECTIVES)
    for baseline in baselines:
        if baseline == method:
            continue
        baseline_average = {
            "distinct_types": aggregates[baseline]["mean_distinct_percent"],
            "cumulative_daily_types": aggregates[baseline]["mean_daily_percent"],
            "total_servings": aggregates[baseline]["mean_servings_percent"],
        }
        baseline_key = tuple(
            baseline_average[objective] for objective in OBJECTIVES
        )
        if own_key > baseline_key:
            lexicographic_result = "win"
        elif own_key < baseline_key:
            lexicographic_result = "loss"
        else:
            lexicographic_result = "tie"
        comparisons[baseline] = {
            "method_average_percentages": own_average,
            "baseline_average_percentages": baseline_average,
            "delta_percentage_points": {
                objective: own_average[objective] - baseline_average[objective]
                for objective in OBJECTIVES
            },
            "lexicographic_result": lexicographic_result,
        }

    report = {
        "schema_version": 2,
        "suite": manifest["suite"],
        "method": method,
        "case_count": case_count,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "jobs": worker_count,
        "core_threads": threads_per_core,
        "time_limit_ms": time_limit_ms,
        "wall_seconds": wall_seconds,
        "optimum_definition": OPTIMUM_DEFINITION,
        "aggregates": aggregates,
        "ranking": ranking,
        "comparisons": comparisons,
        "cases": rows,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        f"# HEXUDON benchmark: {method}",
        "",
        f"Suite: `{manifest['suite']}` ({case_count} cases)",
        f"Workers: `{worker_count}` · core threads: `{threads_per_core or 'default'}` · wall time: `{wall_seconds:.3f}s`",
        f"Per-day solver limit: `{time_limit_ms} ms`" if time_limit_ms is not None else "Per-day solver limit: policy default",
        "",
        f"Optimum: {OPTIMUM_DEFINITION}",
        "",
        "The grade is the lexicographic vector of macro-averaged objective percentages: distinct types first, then cumulative daily types, then servings. Every case has equal weight and an invalid case contributes 0% to every component.",
        "",
        "| Rank | Method | Valid | Distinct optimum | Daily optimum | Servings optimum | Runtime (s) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, name in enumerate(ranking, start=1):
        aggregate = aggregates[name]
        lines.append(
            f"| {rank} | {name} | {aggregate['valid_cases']}/{case_count} | "
            f"{aggregate['mean_distinct_percent']:.2f}% | "
            f"{aggregate['mean_daily_percent']:.2f}% | "
            f"{aggregate['mean_servings_percent']:.2f}% | "
            f"{aggregate['runtime_seconds']:.3f} |"
        )
    if comparisons:
        lines.extend(("", "## Average-performance comparison", ""))
        for baseline, comparison in comparisons.items():
            own = comparison["method_average_percentages"]
            other = comparison["baseline_average_percentages"]
            lines.append(
                f"- `{method}` ({own['distinct_types']:.2f}%, "
                f"{own['cumulative_daily_types']:.2f}%, "
                f"{own['total_servings']:.2f}%) vs `{baseline}` "
                f"({other['distinct_types']:.2f}%, "
                f"{other['cumulative_daily_types']:.2f}%, "
                f"{other['total_servings']:.2f}%): "
                f"**{comparison['lexicographic_result']}**"
            )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n")
    return report
