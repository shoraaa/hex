#!/usr/bin/env python3
"""Benchmark hybrid LNS policies with LNS-DP proposals and standalone LNS-DP."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hexbench.generator import generate_validation_suite  # noqa: E402
from hexbench.runner import find_binary, grade_suite  # noqa: E402


# The proposal toggle only applies to these hybrid policies. ``lns_dp`` is an
# independent policy, so it is benchmarked once rather than being shown twice
# under proposal-enabled and proposal-disabled labels.
HYBRID_METHODS = ("alns", "mlns", "palns")
STANDALONE_METHODS = ("lns_dp",)
METHODS = (*HYBRID_METHODS, *STANDALONE_METHODS)
PROFILES = ("hard", "medium", "easy")
PROFILE_LABELS = {"hard": "brutal", "medium": "steady", "easy": "easy"}
DP_ENABLE = "HEXUDON_ENABLE_LNS_DP_PROPOSALS"
DP_DISABLE = "HEXUDON_DISABLE_LNS_DP_PROPOSALS"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Generate brutal/steady/easy validation cases, benchmark "
            "ALNS/MLNS/PALNS with and without LNS-DP proposals plus "
            "standalone LNS-DP, and write one final comparison table."
        )
    )
    result.add_argument("--per-profile", type=int, default=100)
    result.add_argument("--cases-dir", type=Path, default=Path("cases/lns-300"))
    result.add_argument(
        "--report-dir", type=Path, default=Path("reports/lns-seven")
    )
    result.add_argument(
        "--time-limit-ms",
        type=int,
        help=(
            "per-day solver budget; default derives it from the generated "
            "daySeconds minus --deadline-margin-ms"
        ),
    )
    result.add_argument("--deadline-margin-ms", type=int, default=2000)
    result.add_argument(
        "--timeout",
        type=float,
        help="complete-match watchdog; default is derived from days and budget",
    )
    result.add_argument(
        "--jobs",
        type=int,
        help="policy-case workers per mode; default avoids CPU oversubscription",
    )
    result.add_argument(
        "--threads-per-case",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="C++ threads available to each solver case (default: up to 4)",
    )
    result.add_argument("--binary", type=Path)
    result.add_argument(
        "--skip-generate",
        action="store_true",
        help="reuse the existing combined manifest under --cases-dir",
    )
    result.add_argument(
        "--skip-build",
        action="store_true",
        help="do not run cmake --build build --parallel first",
    )
    result.add_argument(
        "--rerun",
        action="store_true",
        help="ignore compatible raw reports and run both benchmark modes again",
    )
    result.add_argument(
        "--sequential-modes",
        action="store_true",
        help="run no-DP and with-DP one after another to reduce peak CPU/RAM",
    )
    return result


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def suite_timing(manifest: Path, deadline_margin_ms: int) -> tuple[int, int]:
    data = json.loads(manifest.read_text())
    day_seconds: list[float] = []
    maximum_days = 0
    for case in data["cases"]:
        scenario = json.loads((manifest.parent / case["path"]).read_text())
        schedule = scenario["config"]["daySeconds"]
        day_seconds.extend(float(value) for value in schedule)
        maximum_days = max(maximum_days, len(schedule))
    if not day_seconds:
        raise SystemExit(f"suite has no daySeconds schedule: {manifest}")
    budget_ms = max(50, int(min(day_seconds) * 1000) - deadline_margin_ms)
    return budget_ms, maximum_days


def run_grade(
    manifest: Path,
    report_dir: Path,
    *,
    enabled: bool,
    methods: tuple[str, ...],
    binary: str | None,
    jobs: int,
    timeout: float,
    time_limit_ms: int,
    reuse: bool,
    core_threads: int,
) -> dict[str, Any]:
    mode = "with-dp" if enabled else "no-dp"
    raw_report = report_dir / mode / "report.json"
    if reuse:
        cached = compatible_report(
            raw_report,
            manifest=manifest,
            methods=methods,
            binary=binary,
            time_limit_ms=time_limit_ms,
            core_threads=core_threads,
        )
        if cached is not None:
            print(f"Reusing completed {mode} report: {raw_report}", flush=True)
            return cached
    print(f"Benchmarking {mode}: {', '.join(methods)}", flush=True)
    environment = (
        {DP_ENABLE: "1", DP_DISABLE: None}
        if enabled
        else {DP_DISABLE: "1", DP_ENABLE: None}
    )
    return grade_suite(
        manifest,
        methods[0],
        list(methods[1:]),
        report_dir / mode,
        binary_path=binary,
        jobs=jobs,
        timeout=timeout,
        time_limit_ms=time_limit_ms,
        core_environment=environment,
        core_threads=core_threads,
    )


def compatible_report(
    report_path: Path,
    *,
    manifest: Path,
    methods: tuple[str, ...],
    binary: str | None,
    time_limit_ms: int,
    core_threads: int,
) -> dict[str, Any] | None:
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text())
        manifest_data = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    expected_cases = [case["path"] for case in manifest_data["cases"]]
    reported_cases = [row["case"]["path"] for row in report.get("cases", [])]
    actual_binary = find_binary(binary)
    binary_hash = hashlib.sha256(actual_binary.read_bytes()).hexdigest()
    if (
        report.get("suite") != manifest_data.get("suite")
        or report.get("time_limit_ms") != time_limit_ms
        or reported_cases != expected_cases
        or set(report.get("aggregates", {})) != set(methods)
        or report.get("binary_sha256") != binary_hash
        or report.get("core_threads") != core_threads
    ):
        return None
    return report


def empty_scope() -> dict[str, Any]:
    return {
        "cases": 0,
        "valid_cases": 0,
        "runtime_seconds": 0.0,
        "distinct_percent": 0.0,
        "daily_percent": 0.0,
        "servings_percent": 0.0,
    }


def scope_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    summary = empty_scope()
    for row in rows:
        result = row["results"][method]
        percentages = result["objective_percentages"]
        summary["cases"] += 1
        summary["valid_cases"] += int(result["invalid_days"] == 0)
        summary["runtime_seconds"] += result["runtime_seconds"]
        summary["distinct_percent"] += percentages["distinct_types"]
        summary["daily_percent"] += percentages["cumulative_daily_types"]
        summary["servings_percent"] += percentages["total_servings"]
    count = summary["cases"]
    for key in ("distinct_percent", "daily_percent", "servings_percent"):
        summary[key] = summary[key] / count if count else 0.0
    return summary


def variant_summaries(
    reports: list[tuple[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    variants: dict[str, dict[str, dict[str, Any]]] = {}
    for suffix, report in reports:
        for method in report["aggregates"]:
            name = f"{method}{suffix}"
            scopes: dict[str, dict[str, Any]] = {}
            for profile in PROFILES:
                rows = [
                    row
                    for row in report["cases"]
                    if row["case"].get("profile") == profile
                ]
                scopes[PROFILE_LABELS[profile]] = scope_summary(rows, method)
            scopes["overall"] = scope_summary(report["cases"], method)
            variants[name] = scopes
    return variants


def ranking(variants: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    return sorted(
        variants,
        key=lambda name: (
            -variants[name]["overall"]["distinct_percent"],
            -variants[name]["overall"]["daily_percent"],
            -variants[name]["overall"]["servings_percent"],
            variants[name]["overall"]["runtime_seconds"],
        ),
    )


def score_cell(scope: dict[str, Any]) -> str:
    return (
        f"{scope['distinct_percent']:.3f} / "
        f"{scope['daily_percent']:.3f} / "
        f"{scope['servings_percent']:.3f}"
    )


def markdown_table(
    variants: dict[str, dict[str, dict[str, Any]]],
    ordered: list[str],
    *,
    profile_counts: dict[str, int],
    time_limit_ms: int,
) -> str:
    total_cases = sum(profile_counts.values())
    lines = [
        "# Seven-variant LNS benchmark",
        "",
        "Cases: "
        + ", ".join(
            f"`{PROFILE_LABELS[profile]}={profile_counts[profile]}`"
            for profile in PROFILES
        )
        + f"; `{total_cases}` total.",
        f"Per-day wall-clock budget: `{time_limit_ms} ms`.",
        "Scores are `distinct / daily / servings` as percentages of the structural optimum.",
        "",
        "| Rank | Variant | Valid | Brutal | Steady | Easy | Overall | Runtime (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for position, name in enumerate(ordered, start=1):
        scopes = variants[name]
        overall = scopes["overall"]
        lines.append(
            f"| {position} | `{name}` | "
            f"{overall['valid_cases']}/{overall['cases']} | "
            f"{score_cell(scopes['brutal'])} | "
            f"{score_cell(scopes['steady'])} | "
            f"{score_cell(scopes['easy'])} | "
            f"{score_cell(overall)} | "
            f"{overall['runtime_seconds']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parser().parse_args()
    if args.per_profile < 1:
        raise SystemExit("--per-profile must be positive")
    if args.jobs is not None and args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    if args.threads_per_case < 1:
        raise SystemExit("--threads-per-case must be positive")
    if args.time_limit_ms is not None and args.time_limit_ms < 1:
        raise SystemExit("--time-limit-ms must be positive")
    if args.deadline_margin_ms < 0:
        raise SystemExit("--deadline-margin-ms must be nonnegative")

    cases_dir = repo_path(args.cases_dir)
    report_dir = repo_path(args.report_dir)
    binary = str(repo_path(args.binary)) if args.binary else None

    if not args.skip_build:
        subprocess.run(
            ["cmake", "--build", str(ROOT / "build"), "--parallel"],
            cwd=ROOT,
            check=True,
        )

    manifest = cases_dir / "manifest.json"
    if args.skip_generate:
        if not manifest.is_file():
            raise SystemExit(f"manifest not found: {manifest}")
    else:
        manifest = generate_validation_suite(
            cases_dir,
            per_profile=args.per_profile,
            profiles=PROFILES,
        )
        print(f"Generated suite: {manifest}", flush=True)

    official_budget_ms, maximum_days = suite_timing(
        manifest, args.deadline_margin_ms
    )
    time_limit_ms = args.time_limit_ms or official_budget_ms
    timeout = args.timeout or max(
        120.0, maximum_days * time_limit_ms / 1000.0 + 60.0
    )
    if args.time_limit_ms is None:
        print(
            f"Using official daySeconds budget: {time_limit_ms} ms/day "
            f"({args.deadline_margin_ms} ms submission margin)",
            flush=True,
        )
    print(f"Complete-match watchdog: {timeout:.1f} seconds", flush=True)
    concurrent_modes = 1 if args.sequential_modes else 2
    jobs = args.jobs or max(
        1,
        (os.cpu_count() or 1) //
        (args.threads_per_case * concurrent_modes),
    )
    mode_arguments = {
        "manifest": manifest,
        "report_dir": report_dir,
        "binary": binary,
        "jobs": jobs,
        "timeout": timeout,
        "time_limit_ms": time_limit_ms,
        "reuse": not args.rerun,
        "core_threads": args.threads_per_case,
    }
    if args.sequential_modes:
        without_dp = run_grade(
            enabled=False, methods=METHODS, **mode_arguments
        )
        with_dp = run_grade(
            enabled=True, methods=HYBRID_METHODS, **mode_arguments
        )
    else:
        print(
            f"Parallel modes: {jobs} cases/mode x "
            f"{args.threads_per_case} threads/case, up to "
            f"{jobs * args.threads_per_case * 2} threads total",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            no_dp_future = executor.submit(
                run_grade, enabled=False, methods=METHODS, **mode_arguments
            )
            with_dp_future = executor.submit(
                run_grade,
                enabled=True,
                methods=HYBRID_METHODS,
                **mode_arguments,
            )
            without_dp = no_dp_future.result()
            with_dp = with_dp_future.result()

    variants = variant_summaries([("", without_dp), ("+dp", with_dp)])
    ordered = ranking(variants)
    profile_counts = {
        profile: sum(
            row["case"].get("profile") == profile for row in without_dp["cases"]
        )
        for profile in PROFILES
    }
    table = markdown_table(
        variants,
        ordered,
        profile_counts=profile_counts,
        time_limit_ms=time_limit_ms,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    table_path = report_dir / "final-table.md"
    json_path = report_dir / "final-table.json"
    table_path.write_text(table)
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_case_counts": {
                    PROFILE_LABELS[profile]: count
                    for profile, count in profile_counts.items()
                },
                "time_limit_ms": time_limit_ms,
                "deadline_margin_ms": args.deadline_margin_ms,
                "ranking": ordered,
                "variants": variants,
            },
            indent=2,
        )
        + "\n"
    )
    print()
    print(table, end="")
    print(f"Markdown: {table_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
