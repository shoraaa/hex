#!/usr/bin/env python3
"""Compare MLNS's symmetric suffix forecast with the optional traffic GNN."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexbench.runner import grade_suite


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL = ROOT / "cases/alns-validation/manifest.json"
DEFAULT_ONLINE = ROOT / "reports/mlns-quality/online-5-manifest.json"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--online", type=Path, default=DEFAULT_ONLINE)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/mlns-gnn")
    parser.add_argument("--time-limit-ms", type=int, default=1_000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads-per-case", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--binary")
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def _run(
    manifest: Path,
    output: Path,
    *,
    use_gnn: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    environment = None
    if use_gnn and args.checkpoint:
        environment = {"HEXUDON_TRAFFIC_GNN_CHECKPOINT": str(args.checkpoint)}
    return grade_suite(
        manifest,
        "mlns",
        [],
        output,
        binary_path=args.binary,
        jobs=args.jobs,
        timeout=args.timeout,
        time_limit_ms=args.time_limit_ms,
        core_threads=args.threads_per_case,
        policy_hyperparameters={
            "mlns": {"use_traffic_gnn": use_gnn},
        },
        core_environment=environment,
    )


def _compare(baseline: dict[str, Any], gnn: dict[str, Any]) -> dict[str, Any]:
    rows = []
    wins = ties = losses = 0
    for left, right in zip(baseline["cases"], gnn["cases"], strict=True):
        base = left["results"]["mlns"]
        model = right["results"]["mlns"]
        base_key = tuple(base["score"][key] for key in ("distinct_types", "cumulative_daily_types", "total_servings"))
        model_key = tuple(model["score"][key] for key in ("distinct_types", "cumulative_daily_types", "total_servings"))
        if model_key > base_key:
            wins += 1
        elif model_key == base_key:
            ties += 1
        else:
            losses += 1
        rows.append(
            {
                "case": left["case"],
                "baseline": base,
                "gnn": model,
                "delta": [model_key[index] - base_key[index] for index in range(3)],
            }
        )
    return {"wins": wins, "ties": ties, "losses": losses, "cases": rows}


def _main() -> None:
    args = _args()
    if args.checkpoint:
        os.environ["HEXUDON_TRAFFIC_GNN_CHECKPOINT"] = str(
            args.checkpoint.resolve()
        )
    summary: dict[str, Any] = {"time_limit_ms": args.time_limit_ms, "suites": {}}
    for label, raw_manifest in (("local-96", args.local), ("online", args.online)):
        manifest = raw_manifest if raw_manifest.is_absolute() else ROOT / raw_manifest
        if not manifest.is_file():
            summary["suites"][label] = {"status": "missing", "manifest": str(manifest)}
            continue
        baseline = _run(manifest, args.report / label / "baseline", use_gnn=False, args=args)
        gnn = _run(manifest, args.report / label / "gnn", use_gnn=True, args=args)
        summary["suites"][label] = {
            "manifest": str(manifest),
            "baseline": baseline["aggregates"]["mlns"],
            "gnn": gnn["aggregates"]["mlns"],
            "comparison": _compare(baseline, gnn),
        }
    args.report.mkdir(parents=True, exist_ok=True)
    (args.report / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    printable = {
        "time_limit_ms": summary["time_limit_ms"],
        "suites": {
            label: {
                key: value
                for key, value in suite.items()
                if key != "comparison"
            }
            | (
                {
                    "comparison": {
                        key: suite["comparison"][key]
                        for key in ("wins", "ties", "losses")
                    }
                }
                if "comparison" in suite
                else {}
            )
            for label, suite in summary["suites"].items()
        },
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    _main()
