#!/usr/bin/env python3
"""Run MLNS+DP on the three resettable online practice questions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hexbench.api import GameClient, load_token, practice_benchmark  # noqa: E402


QUESTIONS = {
    "q01": {
        "game_id": "d2d87157-9158-484f-be37-814a0cf44524:13",
        "future_discount_percent": 50,
        "max_iterations": 96,
        "time_limit_ms": None,
        "verified_score": (4, 28, 126),
    },
    "q04": {
        "game_id": "b73d272a-7fca-4106-b8d3-0f650fd2ddde:13",
        "future_discount_percent": 90,
        "max_iterations": 96,
        "time_limit_ms": None,
        "verified_score": (3, 24, 88),
    },
    "new": {
        "game_id": "52962f8f-4ac3-4587-9493-9c45ae947243:13",
        "future_discount_percent": 90,
        "max_iterations": 10_000_000,
        "time_limit_ms": 5_000,
        "verified_score": (5, 35, 218),
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reset, optimize, and submit MLNS+DP under each question's "
            "official daySeconds budget."
        )
    )
    parser.add_argument(
        "questions",
        nargs="*",
        choices=tuple(QUESTIONS),
        help="questions to run (default: q01 q04 new)",
    )
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/hexudon")
    parser.add_argument(
        "--state-dir", type=Path, default=ROOT / ".hexbench-state/mlns-dp-online"
    )
    parser.add_argument(
        "--report-dir", type=Path, default=ROOT / "reports/mlns-dp-online"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--force",
        action="store_true",
        help="reset even when the current online score meets the verified target",
    )
    return parser


def emit(question: str, event: dict[str, Any]) -> None:
    print(json.dumps({"question": question, **event}), flush=True)


def main() -> None:
    args = build_parser().parse_args()
    os.environ["HEXUDON_THREADS"] = str(max(1, args.threads))
    client = GameClient(load_token(args.env))
    summary: list[dict[str, Any]] = []
    for question in args.questions or list(QUESTIONS):
        settings = QUESTIONS[question]
        discount = settings["future_discount_percent"]
        max_iterations = settings["max_iterations"]
        time_limit_ms = settings["time_limit_ms"]
        team_id = settings["game_id"].rsplit(":", 1)[1]
        current = client.get("/game/practice/score", settings["game_id"])
        detail = current.get("detail", {}).get(team_id, {})
        current_score = (
            int(detail.get("distinct_types", 0)),
            int(detail.get("cumulative_daily_types", 0)),
            int(detail.get("total_servings", 0)),
        )
        if not args.force and current_score >= settings["verified_score"]:
            row = {
                "question": question,
                "game_id": settings["game_id"],
                "status": "preserved",
                "distinct_types": current_score[0],
                "cumulative_daily_types": current_score[1],
                "total_servings": current_score[2],
            }
            summary.append(row)
            emit(question, row)
            continue
        emit(
            question,
            {
                "status": "starting",
                "game_id": settings["game_id"],
                "future_discount_percent": discount,
                "max_iterations": max_iterations,
                "time_limit_ms": time_limit_ms,
                "threads": max(1, args.threads),
            },
        )
        report = practice_benchmark(
            settings["game_id"],
            ["mlns"],
            args.env,
            args.state_dir / question,
            args.report_dir / question,
            leave_best=True,
            peer_team_ids=[],
            poll_interval=0.05,
            binary_path=str(args.binary),
            quiet=True,
            progress=lambda event, name=question: emit(name, event),
            hyperparameters={
                "mlns": {
                    "min_iterations": 32,
                    "max_iterations": max_iterations,
                    "stagnation_iterations": 0,
                    "future_discount_percent": discount,
                    "use_lns_dp_proposals": True,
                    **(
                        {"time_limit_ms": time_limit_ms}
                        if time_limit_ms is not None
                        else {}
                    ),
                }
            },
        )
        row = report["results"][0]
        summary.append(
            {
                "question": question,
                "game_id": report["game_id"],
                "future_discount_percent": discount,
                "distinct_types": row["distinct_types"],
                "cumulative_daily_types": row["cumulative_daily_types"],
                "total_servings": row["total_servings"],
                "wall_seconds": row["wall_seconds"],
            }
        )
        emit(question, {"status": "submitted", "score": summary[-1]})
    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "complete", "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
