from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import (
    BASE_URL,
    deploy,
    fetch_fixture,
    fuel_stress_benchmark,
    lns_time_benchmark,
    practice_benchmark,
    practice_suite,
)
from .generator import (
    HARD_TIERS,
    VALIDATION_PROFILES,
    generate_hard_suite,
    generate_suite,
    generate_validation_suite,
)
from .runner import grade_suite
from .mlns_tuning import optimize_mlns
from .palns_tuning import tune_palns
from .tuning import SEED_PROFILES, tune_alns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hexbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a deterministic virtual suite")
    generate.add_argument("--suite", choices=("quick", "full"), default="quick")
    generate.add_argument("--out", type=Path, default=Path("cases/quick"))

    hard = subparsers.add_parser(
        "generate-hard",
        help="construct the solver-verified graded hard suite (brutal/steady/easy)",
    )
    hard.add_argument("--out", type=Path, default=Path("cases/hard"))
    hard.add_argument("--per-tier", type=int, default=6)
    hard.add_argument(
        "--tiers",
        default=",".join(HARD_TIERS),
        help="comma-separated subset of " + ",".join(HARD_TIERS),
    )
    hard.add_argument(
        "--no-verify",
        action="store_true",
        help="skip ALNS band verification (constructive only; not recommended)",
    )
    hard.add_argument("--verify-policy", default="alns")
    hard.add_argument("--max-attempts", type=int, default=80)
    hard.add_argument("--binary")

    validation = subparsers.add_parser(
        "generate-validation",
        help="generate deterministic ALNS validation cases per profile",
    )
    validation.add_argument(
        "--out", type=Path, default=Path("cases/alns-validation")
    )
    validation.add_argument("--per-profile", type=int, default=32)
    validation.add_argument(
        "--profiles",
        default=",".join(VALIDATION_PROFILES),
        help="comma-separated subset of " + ",".join(VALIDATION_PROFILES),
    )

    grade = subparsers.add_parser("grade", help="grade a C++ policy locally")
    grade.add_argument("--cases", type=Path, required=True, help="suite manifest.json")
    grade.add_argument("--method", default="greedy")
    grade.add_argument("--baselines", default="wait,greedy")
    grade.add_argument("--report", type=Path, default=Path("reports/latest"))
    grade.add_argument("--binary")
    grade.add_argument("--jobs", type=int, default=0, help="parallel case/method workers; 0=auto")
    grade.add_argument("--timeout", type=float, default=60, help="per policy-case timeout in seconds")
    grade.add_argument(
        "--time-limit-ms",
        type=int,
        help="inject the same per-day wall-clock solver limit into every case",
    )

    fetch = subparsers.add_parser("fetch", help="fetch a read-only live fixture")
    fetch.add_argument("--game-id", required=True)
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--env", type=Path, default=Path(".env"))
    fetch.add_argument("--base-url", default=BASE_URL)

    live = subparsers.add_parser(
        "deploy", aliases=["play"], help="autonomously play a practice or real game"
    )
    live.add_argument("--game-id", required=True)
    live.add_argument("--method", default="greedy")
    live.add_argument("--env", type=Path, default=Path(".env"))
    live.add_argument("--state-dir", type=Path, default=Path(".hexbench-state"))
    live.add_argument("--dry-run", action="store_true")
    live.add_argument("--once", action="store_true")
    live.add_argument("--deadline-margin", type=float, default=2.0)
    live.add_argument("--poll-interval", type=float, default=0.5)
    live.add_argument("--binary")
    live.add_argument("--base-url", default=BASE_URL)

    practice = subparsers.add_parser(
        "practice-benchmark",
        help="reset and benchmark policies on the same resettable practice game",
    )
    practice.add_argument("--game-id", required=True)
    practice.add_argument(
        "--methods",
        default=(
            "greedy,utility_greedy,fuel_aware,stock_maximiser,"
            "coordinated,local_search,lns,alns,aco,aco_ls"
        ),
    )
    practice.add_argument("--env", type=Path, default=Path(".env"))
    practice.add_argument("--state-dir", type=Path, default=Path(".hexbench-state"))
    practice.add_argument("--report", type=Path, default=Path("reports/practice"))
    practice.add_argument(
        "--peer-team-ids",
        default="auto",
        help="comma-separated peer ids, 'auto' for manager discovery, or 'none'",
    )
    practice.add_argument("--leave-last", action="store_true")
    practice.add_argument("--poll-interval", type=float, default=0.05)
    practice.add_argument("--binary")
    practice.add_argument("--base-url", default=BASE_URL)

    suite = subparsers.add_parser(
        "practice-suite",
        help="benchmark resettable practice maps and rank against configured teams",
    )
    suite.add_argument(
        "--game-ids",
        default="auto",
        help="comma-separated question ids or 'auto' for every safe practice map",
    )
    suite.add_argument("--methods", default="local_search")
    suite.add_argument("--env", type=Path, default=Path(".env"))
    suite.add_argument("--state-dir", type=Path, default=Path(".hexbench-state"))
    suite.add_argument("--report", type=Path, default=Path("reports/practice-suite"))
    suite.add_argument(
        "--peer-team-ids",
        default="auto",
        help="comma-separated peer ids, 'auto' per map, or 'none'",
    )
    suite.add_argument("--poll-interval", type=float, default=0.05)
    suite.add_argument("--binary")
    suite.add_argument("--base-url", default=BASE_URL)

    fuel = subparsers.add_parser(
        "fuel-benchmark",
        help="locally benchmark lower-fuel variants of authoritative practice maps",
    )
    fuel.add_argument(
        "--methods", default="lns,alns,local_search,aco,coordinated,fuel_aware"
    )
    fuel.add_argument(
        "--game-ids",
        default="auto",
        help="comma-separated question ids or 'auto' for every safe practice map",
    )
    fuel.add_argument(
        "--fuel-multipliers",
        default="1.0,0.5,0.25",
        help="multiples of Day-1 steps; server fuel is always included",
    )
    fuel.add_argument("--report", type=Path, default=Path("reports/fuel-stress"))
    fuel.add_argument("--jobs", type=int, default=0)
    fuel.add_argument("--env", type=Path, default=Path(".env"))
    fuel.add_argument("--binary")
    fuel.add_argument("--base-url", default=BASE_URL)

    time_curve = subparsers.add_parser(
        "lns-time-benchmark",
        help="measure LNS/ALNS score as the per-day time budget increases",
    )
    time_curve.add_argument(
        "--method",
        choices=("lns", "alns", "palns", "mlns", "simple_lns", "lns_dp"),
        default="lns",
    )
    time_curve.add_argument(
        "--game-ids",
        default="auto",
        help="comma-separated question ids or 'auto' for every safe practice map",
    )
    time_curve.add_argument("--fuel-multiplier", type=float, default=0.5)
    time_curve.add_argument(
        "--time-limits-ms", default="25,100,500,2000,10000"
    )
    time_curve.add_argument(
        "--report", type=Path, default=Path("reports/lns-time-curve")
    )
    time_curve.add_argument("--jobs", type=int, default=0)
    time_curve.add_argument("--env", type=Path, default=Path(".env"))
    time_curve.add_argument("--binary")
    time_curve.add_argument("--base-url", default=BASE_URL)
    time_curve.add_argument(
        "--seed-profile", choices=tuple(SEED_PROFILES), default="production"
    )

    tune = subparsers.add_parser(
        "alns-tune",
        help="optimize deterministic ALNS iteration and stopping controls on a local suite",
    )
    tune.add_argument("--cases", type=Path, required=True, help="suite manifest.json")
    tune.add_argument(
        "--alns-iterations",
        default="128,256,512,1024,2048,3072,4096,6000",
        help="comma-separated ALNS iteration counts",
    )

    palns_tune = subparsers.add_parser(
        "palns-tune",
        help="tune PALNS projection depth/restarts and report its total-iteration curve",
    )
    palns_tune.add_argument("--cases", type=Path, required=True)
    palns_tune.add_argument("--tuning-total-iterations", type=int, default=1536)
    palns_tune.add_argument("--projection-iterations", default="1,2,4,8,16,32")
    palns_tune.add_argument("--restarts", default="1,2,3")
    palns_tune.add_argument(
        "--total-iteration-curve",
        default="128,256,512,1024,1536,2048,3072,4096,6000",
    )
    palns_tune.add_argument(
        "--report", type=Path, default=Path("reports/palns-validation")
    )
    palns_tune.add_argument("--binary")
    palns_tune.add_argument("--jobs", type=int, default=0)
    palns_tune.add_argument("--timeout", type=float, default=180)

    mlns_tune = subparsers.add_parser(
        "mlns-tune",
        help="optimize MLNS controls with resumable Bayesian TPE on the 96-case suite",
    )
    mlns_tune.add_argument(
        "--cases",
        type=Path,
        default=Path("cases/alns-validation/manifest.json"),
    )
    mlns_tune.add_argument("--trials", type=int, default=32)
    mlns_tune.add_argument("--startup-trials", type=int, default=8)
    mlns_tune.add_argument("--time-limit-ms", type=int, default=1_000)
    mlns_tune.add_argument("--min-iterations", default="0,8,16,32,64")
    mlns_tune.add_argument("--stagnation-iterations", default="0,16,32,64,96,128,256")
    mlns_tune.add_argument(
        "--future-discount-percent",
        default="25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100",
    )
    mlns_tune.add_argument("--seed", type=int, default=20260720)
    mlns_tune.add_argument("--expected-cases", type=int, default=96)
    mlns_tune.add_argument(
        "--report", type=Path, default=Path("reports/mlns-validation")
    )
    mlns_tune.add_argument("--binary")
    mlns_tune.add_argument("--jobs", type=int, default=0)
    mlns_tune.add_argument("--timeout", type=float, default=180)
    mlns_tune.add_argument(
        "--no-resume", action="store_true", help="discard state.json and start a new search"
    )
    tune.add_argument("--min-iterations", default="32", help="comma-separated minimum iteration values")
    tune.add_argument(
        "--stagnation-iterations",
        default="0",
        help="comma-separated stopping-after-stagnation values; 0 disables it",
    )
    tune.add_argument(
        "--seed-profiles",
        default="production",
        help="comma-separated seed profiles: " + ",".join(SEED_PROFILES),
    )
    tune.add_argument("--report", type=Path, default=Path("reports/alns-tuning"))
    tune.add_argument("--binary")
    tune.add_argument("--jobs", type=int, default=0, help="parallel workers; 0=auto")
    tune.add_argument("--timeout", type=float, default=60, help="per case timeout in seconds")
    tune.add_argument(
        "--case-stride",
        type=int,
        default=1,
        help="evaluate every Nth manifest case; 1 evaluates the complete suite",
    )

    traffic_data = subparsers.add_parser(
        "traffic-generate",
        help="generate a reusable traffic dataset with 16 diversely seeded ALNS players",
    )
    traffic_data.add_argument("--train-cases", type=int, default=100)
    traffic_data.add_argument("--validation-cases", type=int, default=20)
    traffic_data.add_argument("--seed", type=int, default=20260718)
    traffic_data.add_argument("--alns-iterations", type=int, default=32)
    traffic_data.add_argument(
        "--out", type=Path, default=Path("datasets/traffic-gnn")
    )
    traffic_data.add_argument("--binary")
    traffic_data.add_argument("--simulation-timeout", type=float, default=180.0)
    traffic_data.add_argument("--core-threads", type=int, default=1)
    traffic_data.add_argument("--jobs", type=int, default=8)
    traffic_data.add_argument("--overwrite", action="store_true")

    traffic = subparsers.add_parser(
        "traffic-train",
        help="train a minimal GNN from a previously generated traffic dataset",
    )
    traffic.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/traffic-gnn/dataset.pt"),
    )
    traffic.add_argument("--epochs", type=int, default=200)
    traffic.add_argument("--seed", type=int, default=20260718)
    traffic.add_argument("--hidden-size", type=int, default=128)
    traffic.add_argument("--layers", type=int, default=4)
    traffic.add_argument("--learning-rate", type=float, default=1e-3)
    traffic.add_argument("--batch-size", type=int, default=64)
    traffic.add_argument("--patience", type=int, default=30)
    traffic.add_argument("--minimum-epochs", type=int, default=50)
    traffic.add_argument("--device", default="auto")
    traffic.add_argument("--report", type=Path, default=Path("reports/traffic-gnn"))

    web = subparsers.add_parser(
        "web", help="serve the Practice and Competition operations console"
    )
    web.add_argument("--host", default="0.0.0.0")
    web.add_argument("--port", type=int, default=5678)
    web.add_argument("--env", type=Path, default=Path(".env"))
    web.add_argument("--state-dir", type=Path, default=Path(".hexbench-state"))
    web.add_argument("--report", type=Path, default=Path("reports/web"))
    web.add_argument("--poll-interval", type=float, default=0.05)
    web.add_argument("--binary")
    web.add_argument("--base-url", default=BASE_URL)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        path = generate_suite(args.suite, args.out)
        print(path)
    elif args.command == "generate-hard":
        tiers = tuple(item.strip() for item in args.tiers.split(",") if item.strip())
        path = generate_hard_suite(
            args.out,
            per_tier=args.per_tier,
            tiers=tiers,
            binary_path=args.binary,
            verify=not args.no_verify,
            verify_policy=args.verify_policy,
            max_attempts=args.max_attempts,
        )
        print(path)
    elif args.command == "generate-validation":
        profiles = tuple(
            item.strip() for item in args.profiles.split(",") if item.strip()
        )
        path = generate_validation_suite(
            args.out, per_profile=args.per_profile, profiles=profiles
        )
        print(path)
    elif args.command == "grade":
        baselines = [item for item in args.baselines.split(",") if item]
        report = grade_suite(
            args.cases,
            args.method,
            baselines,
            args.report,
            args.binary,
            None if args.jobs == 0 else args.jobs,
            args.timeout,
            args.time_limit_ms,
        )
        print(json.dumps({"report": str(args.report / "report.json"), "cases": report["case_count"]}))
    elif args.command == "fetch":
        fetch_fixture(args.game_id, args.out, args.env, args.base_url)
        print(args.out)
    elif args.command in {"deploy", "play"}:
        deploy(
            args.game_id,
            args.method,
            args.env,
            args.state_dir,
            dry_run=args.dry_run,
            once=args.once,
            deadline_margin=args.deadline_margin,
            poll_interval=args.poll_interval,
            binary_path=args.binary,
            base_url=args.base_url,
        )
    elif args.command == "practice-benchmark":
        methods = [item for item in args.methods.split(",") if item]
        peer_team_ids = (
            None
            if args.peer_team_ids == "auto"
            else ([] if args.peer_team_ids == "none" else args.peer_team_ids.split(","))
        )
        report = practice_benchmark(
            args.game_id,
            methods,
            args.env,
            args.state_dir,
            args.report,
            leave_best=not args.leave_last,
            peer_team_ids=peer_team_ids,
            poll_interval=args.poll_interval,
            binary_path=args.binary,
            base_url=args.base_url,
        )
        print(
            json.dumps(
                {
                    "report": str(args.report / "report.json"),
                    "best_policy": report["best_policy"],
                    "final_policy": report["final_policy"],
                }
            )
        )
    elif args.command == "practice-suite":
        methods = [item for item in args.methods.split(",") if item]
        game_ids = None if args.game_ids == "auto" else [
            item for item in args.game_ids.split(",") if item
        ]
        peer_team_ids = (
            None
            if args.peer_team_ids == "auto"
            else ([] if args.peer_team_ids == "none" else args.peer_team_ids.split(","))
        )
        report = practice_suite(
            methods,
            args.env,
            args.state_dir,
            args.report,
            game_ids=game_ids,
            peer_team_ids=peer_team_ids,
            poll_interval=args.poll_interval,
            binary_path=args.binary,
            base_url=args.base_url,
        )
        print(
            json.dumps(
                {
                    "report": str(args.report / "summary.json"),
                    "maps": report["map_count"],
                    "completed_peer_comparison": report["completed_peer_comparison"],
                    "errors": len(report["errors"]),
                }
            )
        )
    elif args.command == "fuel-benchmark":
        methods = [item for item in args.methods.split(",") if item]
        game_ids = (
            None
            if args.game_ids == "auto"
            else [item for item in args.game_ids.split(",") if item]
        )
        multipliers = tuple(
            float(item) for item in args.fuel_multipliers.split(",") if item
        )
        report = fuel_stress_benchmark(
            methods,
            args.env,
            args.report,
            game_ids=game_ids,
            fuel_multipliers=multipliers,
            jobs=args.jobs,
            binary_path=args.binary,
            base_url=args.base_url,
        )
        print(
            json.dumps(
                {
                    "report": str(args.report / "report.json"),
                    "maps": report["map_count"],
                    "cases": report["case_count"],
                }
            )
        )
    elif args.command == "lns-time-benchmark":
        game_ids = (
            None
            if args.game_ids == "auto"
            else [item for item in args.game_ids.split(",") if item]
        )
        time_limits = tuple(
            int(item) for item in args.time_limits_ms.split(",") if item
        )
        report = lns_time_benchmark(
            args.env,
            args.report,
            method=args.method,
            game_ids=game_ids,
            fuel_multiplier=args.fuel_multiplier,
            time_limits_ms=time_limits,
            jobs=args.jobs,
            binary_path=args.binary,
            base_url=args.base_url,
            seed_profile=args.seed_profile,
        )
        print(
            json.dumps(
                {
                    "report": str(args.report / "report.json"),
                    "maps": report["map_count"],
                    "budgets": report["time_limits_ms"],
                }
            )
        )
    elif args.command == "alns-tune":
        def parse_values(raw: str) -> list[int]:
            return [int(item.strip()) for item in raw.split(",") if item.strip()]

        report = tune_alns(
            args.cases,
            args.report,
            alns_iterations=parse_values(args.alns_iterations),
            min_iterations=parse_values(args.min_iterations),
            stagnation_iterations=parse_values(args.stagnation_iterations),
            seed_profiles=[
                item.strip() for item in args.seed_profiles.split(",") if item.strip()
            ],
            binary_path=args.binary,
            jobs=None if args.jobs == 0 else args.jobs,
            timeout=args.timeout,
            case_stride=args.case_stride,
        )
        print(json.dumps({"report": str(args.report / "report.json"), "best": report["best"]["parameters"]}))
    elif args.command == "palns-tune":
        def parse_palns_values(raw: str) -> list[int]:
            return [int(item.strip()) for item in raw.split(",") if item.strip()]

        report = tune_palns(
            args.cases,
            args.report,
            tuning_total_iterations=args.tuning_total_iterations,
            projection_iterations=parse_palns_values(args.projection_iterations),
            restarts=parse_palns_values(args.restarts),
            total_iteration_curve=parse_palns_values(args.total_iteration_curve),
            binary_path=args.binary,
            jobs=None if args.jobs == 0 else args.jobs,
            timeout=args.timeout,
        )
        print(
            json.dumps(
                {
                    "report": str(args.report / "report.json"),
                    "best": report["best_fixed_parameters"],
                }
            )
        )
    elif args.command == "mlns-tune":
        def parse_mlns_values(raw: str) -> list[int]:
            return [int(item.strip()) for item in raw.split(",") if item.strip()]

        report = optimize_mlns(
            args.cases,
            args.report,
            trials=args.trials,
            startup_trials=args.startup_trials,
            time_limit_ms=args.time_limit_ms,
            min_iterations=parse_mlns_values(args.min_iterations),
            stagnation_iterations=parse_mlns_values(args.stagnation_iterations),
            future_discount_percent=parse_mlns_values(
                args.future_discount_percent
            ),
            seed=args.seed,
            expected_cases=args.expected_cases,
            binary_path=args.binary,
            jobs=None if args.jobs == 0 else args.jobs,
            timeout=args.timeout,
            resume=not args.no_resume,
        )
        print(
            json.dumps(
                {
                    "report": str(args.report / "report.json"),
                    "best": report["best"]["parameters"],
                }
            )
        )
    elif args.command == "traffic-generate":
        from .traffic_gnn import generate_traffic_dataset

        manifest = generate_traffic_dataset(
            output_dir=args.out,
            train_cases=args.train_cases,
            validation_cases=args.validation_cases,
            seed=args.seed,
            alns_iterations=args.alns_iterations,
            binary_path=args.binary,
            simulation_timeout=args.simulation_timeout,
            core_threads=args.core_threads,
            jobs=args.jobs,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "dataset": manifest["dataset"],
                    "manifest": str(args.out / "manifest.json"),
                    "train_samples": manifest["train_samples"],
                    "validation_samples": manifest["validation_samples"],
                }
            )
        )
    elif args.command == "traffic-train":
        from .traffic_gnn import train_traffic_gnn

        report = train_traffic_gnn(
            dataset_path=args.dataset,
            epochs=args.epochs,
            seed=args.seed,
            hidden_size=args.hidden_size,
            layers=args.layers,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            patience=args.patience,
            minimum_epochs=args.minimum_epochs,
            device_name=args.device,
            report_dir=args.report,
        )
        print(
            json.dumps(
                {
                    "report": str(args.report / "report.json"),
                    "checkpoint": report["checkpoint"],
                    "best_epoch": report["best_epoch"],
                    "train_loss": report["best_train"]["loss"],
                    "validation_loss": report["best_validation"]["loss"],
                    "validation_accuracy": report["best_validation"]["accuracy"],
                }
            )
        )
    elif args.command == "web":
        from .web import serve_dashboard

        serve_dashboard(
            args.host,
            args.port,
            args.env,
            args.state_dir,
            args.report,
            binary_path=args.binary,
            base_url=args.base_url,
            poll_interval=args.poll_interval,
        )


if __name__ == "__main__":
    main()
