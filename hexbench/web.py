from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .api import (
    BASE_URL,
    _official_score_key,
    _score_detail,
    _suite_map_summary,
    discover_assigned_games,
    discover_practice_questions,
    fetch_game_snapshot,
    fuel_stress_benchmark,
    load_token,
    lns_time_benchmark,
    normalize_hyperparameters,
    POLICY_HYPERPARAMETERS,
    practice_benchmark,
    practice_suite,
)
from .competition import CompetitionSessionManager
from .runner import (
    ROOT,
    find_binary,
    normalized_performance,
    run_core,
    structural_optimum,
)

LOCAL_CASE_ROOT = ROOT / "cases"

POLICIES = (
    "local_search",
    "lns",
    "alns",
    "aco",
    "aco_ls",
)
# Explicit deterministic profile that preserves the saved New Question 219
# serving trajectory. ALNS and exact work are separate literal counts.
PRACTICE_BENCHMARK_DEFAULT_HYPERPARAMETERS = {
    "alns_iterations": 1_536,
    "final_alns_iterations": 1_024,
    "min_iterations": 1_536,
    "stagnation_iterations": 2_048,
    "seed_iterations": 2_048,
    "exact_nodes": 512,
    "final_exact_nodes": 1_024,
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DashboardApp:
    """Own dashboard state and serialize benchmarks that reset practice games."""

    def __init__(
        self,
        env_path: Path,
        state_dir: Path,
        report_dir: Path,
        *,
        binary_path: str | None = None,
        base_url: str = BASE_URL,
        poll_interval: float = 0.05,
    ) -> None:
        self.env_path = env_path
        self.state_dir = state_dir
        self.report_dir = report_dir
        self.binary_path = binary_path
        self.base_url = base_url
        self.poll_interval = poll_interval
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load_jobs()
        self._competition = CompetitionSessionManager(
            env_path,
            state_dir,
            report_dir,
            binary_path=binary_path,
            base_url=base_url,
            poll_interval=max(0.2, poll_interval),
        )
        # Practice resets are stateful. One worker prevents two jobs from
        # resetting the same team game underneath each other.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hexbench-web")

    def close(self) -> None:
        self._competition.close()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _job_path(self, job_id: str) -> Path:
        return self.report_dir / job_id / "job.json"

    def _persist_job(self, job: dict[str, Any]) -> None:
        path = self._job_path(str(job["id"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=2) + "\n")
        temporary.replace(path)

    def _load_jobs(self) -> None:
        if not self.report_dir.exists():
            return
        for path in self.report_dir.glob("*/job.json"):
            try:
                job = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(job, dict) or not job.get("id"):
                continue
            if job.get("status") in {"queued", "running"}:
                job["status"] = "interrupted"
                job["error"] = "dashboard restarted before the run completed"
                job["updated_at"] = _now()
                self._persist_job(job)
            self._jobs[str(job["id"])] = job
        for path in self.report_dir.glob("*/report.json"):
            job_id = path.parent.name
            if job_id in self._jobs:
                continue
            try:
                report = json.loads(path.read_text())
                timestamp = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
            except (OSError, ValueError):
                continue
            if not isinstance(report, dict):
                continue
            mode = (
                "practice"
                if "best_policy" in report
                else ("lns_time" if "time_limits_ms" in report else "fuel_stress")
            )
            self._jobs[job_id] = {
                "id": job_id,
                "status": "completed",
                "mode": mode,
                "created_at": timestamp,
                "updated_at": timestamp,
                "game": {
                    "name": report.get("game_id", "Legacy run"),
                    "question_id": str(report.get("game_id", "")).split(":", 1)[0],
                },
                "methods": [row.get("policy") for row in report.get("results", []) if isinstance(row, dict)],
                "legacy": True,
                "report": str(path),
            }

    def list_games(self, mode: str = "practice") -> list[dict[str, Any]]:
        token = load_token(self.env_path)
        if mode == "competition":
            return [
                game
                for game in discover_assigned_games(token, self.base_url)
                if game.get("mode") == "competition"
            ]
        return [
            {
                **game,
                "mode": "practice",
                "competition_kind": "practice",
                "is_practice": True,
                "no_reset": False,
                "capabilities": {
                    "reset": True,
                    "benchmark": True,
                    "local_evaluation": True,
                    "submit": True,
                    "peer_rank": True,
                    "replay": True,
                },
            }
            for game in discover_practice_questions(token, self.base_url)
        ]

    def list_all_games(self) -> list[dict[str, Any]]:
        token = load_token(self.env_path)
        return discover_assigned_games(token, self.base_url)

    def local_cases(self) -> dict[str, Any]:
        """Return only manifest-declared local cases, grouped for the UI."""
        groups: dict[str, dict[str, Any]] = {}
        manifests = sorted(LOCAL_CASE_ROOT.glob("*/manifest.json"))
        for manifest_path in manifests:
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, ValueError):
                continue
            suite = str(manifest.get("suite") or manifest_path.parent.name)
            for entry in manifest.get("cases", []):
                if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                    continue
                relative = (manifest_path.parent.relative_to(LOCAL_CASE_ROOT) / entry["path"])
                case_id = relative.as_posix()
                tier = str(entry.get("tier") or suite)
                group_id = f"{suite}/{tier}" if tier != suite else suite
                group = groups.setdefault(
                    group_id,
                    {
                        "id": group_id,
                        "suite": suite,
                        "tier": tier,
                        "target": entry.get("target") or manifest.get("target"),
                        "cases": [],
                    },
                )
                group["cases"].append(
                    {
                        "id": case_id,
                        "name": Path(entry["path"]).stem,
                        "seed": entry.get("seed"),
                        "target": entry.get("target"),
                        "design": entry.get("design"),
                        "verification": entry.get("verification"),
                    }
                )
        return {"groups": list(groups.values()), "case_count": sum(len(row["cases"]) for row in groups.values())}

    def _local_case_path(self, case_id: str) -> Path:
        allowed = {
            case["id"]
            for group in self.local_cases()["groups"]
            for case in group["cases"]
        }
        if case_id not in allowed:
            raise ValueError("unknown local case")
        path = (LOCAL_CASE_ROOT / case_id).resolve()
        if not path.is_relative_to(LOCAL_CASE_ROOT.resolve()) or not path.is_file():
            raise ValueError("local case is unavailable")
        return path

    def local_case(self, case_id: str) -> dict[str, Any]:
        scenario = json.loads(self._local_case_path(case_id).read_text())
        return {
            "id": case_id,
            "scenario": scenario,
            "optimum_score": structural_optimum(scenario),
        }

    def run_local_case(
        self,
        case_id: str,
        method: str,
        hyperparameters: dict[str, int | float] | None = None,
    ) -> dict[str, Any]:
        if method not in POLICIES:
            raise ValueError(f"unknown policy: {method}")
        normalized = normalize_hyperparameters(
            [method], {method: hyperparameters or {}}
        ).get(method, {})
        scenario = json.loads(self._local_case_path(case_id).read_text())
        if normalized:
            scenario["hyperparameters"] = normalized
        binary = find_binary(self.binary_path)
        started = time.perf_counter()
        result = run_core(
            "visualize", method, scenario, binary=binary, timeout=180
        )
        result["runtime_seconds"] = time.perf_counter() - started
        optimum = structural_optimum(scenario)
        result.update(normalized_performance(result, optimum))
        return {
            "case_id": case_id,
            "method": method,
            "hyperparameters": normalized,
            "scenario": scenario,
            "result": result,
        }

    def snapshot(self, game_id: str) -> dict[str, Any]:
        return fetch_game_snapshot(load_token(self.env_path), game_id, self.base_url)

    def replay(self, game_id: str, team_id: str | None = None) -> dict[str, Any]:
        from .api import GameClient

        client = GameClient(load_token(self.env_path), self.base_url)
        try:
            board = client.get("/game/board", game_id)
            resolved = str(board.get("game_id", game_id))
            if team_id:
                question_id = resolved.split(":", 1)[0]
                resolved = f"{question_id}:{team_id}"
                endpoint = "/game/practice/peer"
            else:
                endpoint = "/game/replay"
            return {"game_id": resolved, "replay": client.get(endpoint, resolved)}
        finally:
            client.close()

    def standings(self, game_id: str) -> dict[str, Any]:
        """Build the official practice ranking for every configured match team."""
        from .api import GameClient

        token = load_token(self.env_path)
        games = discover_assigned_games(token, self.base_url)
        descriptor = next(
            (game for game in games if game.get("question_id") == game_id),
            None,
        )
        if descriptor is None:
            raise ValueError("game not found")
        if not descriptor.get("is_practice"):
            raise ValueError("peer standings are only available for practice games")
        client = GameClient(token, self.base_url)
        try:
            board = client.get("/game/board", game_id)
            resolved = str(board.get("game_id", game_id))
            question_id, own_team_id = resolved.rsplit(":", 1)
            detail: dict[str, dict[str, Any]] = {}
            errors: dict[str, str] = {}
            zero = {
                "distinct_types": 0,
                "cumulative_daily_types": 0,
                "total_servings": 0,
                "cumulative_response_time": 0.0,
            }
            for team_id in descriptor.get("team_ids", []):
                team_id = str(team_id)
                composite = f"{question_id}:{team_id}"
                try:
                    detail[team_id] = {
                        **zero,
                        **_score_detail(
                            client.get("/game/practice/score", composite),
                            composite,
                        ),
                    }
                except RuntimeError as error:
                    detail[team_id] = dict(zero)
                    errors[team_id] = str(error)
            ranking = sorted(
                detail,
                key=lambda team_id: _official_score_key(detail[team_id]),
                reverse=True,
            )
            return {
                "ranking": ranking,
                "detail": detail,
                "teams": descriptor.get("teams", []),
                "own_team_id": own_team_id,
                "errors": errors,
            }
        finally:
            client.close()

    def answers(self, game_id: str) -> dict[str, Any]:
        from .api import GameClient

        client = GameClient(load_token(self.env_path), self.base_url)
        try:
            board = client.get("/game/board", game_id)
            resolved = str(board.get("game_id", game_id))
            return {"game_id": resolved, **client.get("/game/actions", resolved)}
        finally:
            client.close()

    def reset_game(self, game_id: str) -> dict[str, Any]:
        from .api import GameClient

        client = GameClient(load_token(self.env_path), self.base_url)
        try:
            board = client.get("/game/board", game_id)
            if not board.get("is_practice") or board.get("no_reset"):
                raise ValueError("only resettable practice games can be reset")
            resolved = str(board.get("game_id", game_id))
            self._competition.cancel_game_sessions(resolved)
            response = client.post("/game/practice/reset", {"game_id": resolved})
            self._competition.clear_game_journal(resolved)
            return response
        finally:
            client.close()

    def start_competition(
        self,
        game_id: str,
        method: str,
        hyperparameters: dict[str, int | float] | None = None,
        *,
        execution_mode: str = "manual",
        target_day: int | None = None,
        time_limit_seconds: float | None = None,
    ) -> dict[str, Any]:
        return self._competition.start_session(
            game_id,
            method,
            hyperparameters,
            execution_mode=execution_mode,
            target_day=target_day,
            time_limit_seconds=time_limit_seconds,
        )

    def get_competition(self, session_id: str) -> dict[str, Any] | None:
        return self._competition.get_session(session_id)

    def competition_sessions(self) -> list[dict[str, Any]]:
        return self._competition.recent_sessions()

    def approve_competition(
        self, session_id: str, fingerprint: str, allow_fallback: bool = False
    ) -> dict[str, Any]:
        return self._competition.approve(
            session_id, fingerprint=fingerprint, allow_fallback=allow_fallback
        )

    def submit_competition(
        self,
        session_id: str,
        fingerprint: str,
        *,
        types: list[int] | None = None,
        actions: list[list[int]] | None = None,
        allow_fallback: bool = False,
    ) -> dict[str, Any]:
        return self._competition.submit_proposal(
            session_id,
            fingerprint=fingerprint,
            types=types,
            actions=actions,
            allow_fallback=allow_fallback,
        )

    def competition_curl(
        self,
        session_id: str,
        fingerprint: str,
        *,
        types: list[int] | None = None,
        actions: list[list[int]] | None = None,
    ) -> dict[str, Any]:
        return self._competition.curl_command(
            session_id,
            fingerprint=fingerprint,
            types=types,
            actions=actions,
        )

    def control_competition(self, session_id: str, action: str) -> dict[str, Any]:
        return self._competition.control(session_id, action)

    def start_job(
        self,
        game_id: str,
        methods: list[str],
        hyperparameters: dict[str, dict[str, int | float]] | None = None,
        mode: str = "practice",
        fuel_multipliers: list[int | float] | None = None,
        time_limits_ms: list[int | float] | None = None,
        time_fuel_multiplier: int | float = 0.5,
    ) -> dict[str, Any]:
        methods = list(dict.fromkeys(methods))
        if not methods:
            raise ValueError("select at least one policy")
        unknown = [method for method in methods if method not in POLICIES]
        if unknown:
            raise ValueError(f"unknown policies: {', '.join(unknown)}")
        if mode not in {"practice", "practice_suite", "fuel_stress", "lns_time"}:
            raise ValueError("unknown benchmark mode")
        if mode == "lns_time" and (
            len(methods) != 1 or methods[0] not in {"lns", "alns"}
        ):
            raise ValueError("time curves require exactly one of lns or alns")
        multipliers = tuple(float(value) for value in (fuel_multipliers or [1, 0.5, 0.25]))
        if not multipliers or any(
            not 0 < value <= 3 for value in multipliers
        ):
            raise ValueError("fuel multipliers must be values in (0, 3]")
        time_limits = tuple(int(value) for value in (time_limits_ms or [25, 100, 500, 2000, 10000]))
        if not time_limits or any(value < 1 or value > 60_000 for value in time_limits):
            raise ValueError("time limits must be values from 1 to 60000 ms")
        time_fuel = float(time_fuel_multiplier)
        if not 0 < time_fuel <= 3:
            raise ValueError("time-curve fuel multiplier must be in (0, 3]")
        normalized_hyperparameters = normalize_hyperparameters(methods, hyperparameters)

        games = self.list_games("practice")
        if mode == "practice_suite":
            question = {
                "question_id": "all",
                "name": "All resettable practice maps",
                "width": None,
                "height": None,
                "total_days": sum(int(game.get("total_days") or 0) for game in games),
                "map_count": len(games),
                "team_ids": [],
            }
            if not games:
                raise ValueError("no assigned resettable practice games")
        else:
            question = next(
                (game for game in games if game["question_id"] == game_id), None
            )
            if question is None:
                raise ValueError("game is not an assigned resettable practice game")

        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "status": "queued",
            "created_at": _now(),
            "updated_at": _now(),
            "game": question,
            "methods": methods,
            "mode": mode,
            "hyperparameters": normalized_hyperparameters,
            "fuel_multipliers": list(multipliers),
            "time_limits_ms": list(time_limits),
            "time_fuel_multiplier": time_fuel,
            "progress": {"status": "queued"},
            "events": [{"at": _now(), "status": "queued"}],
        }
        with self._lock:
            self._jobs[job_id] = job
            self._persist_job(job)
        if mode == "practice":
            self._executor.submit(
                self._run_job, job_id, question, methods, normalized_hyperparameters
            )
        elif mode == "practice_suite":
            self._executor.submit(
                self._run_suite_job,
                job_id,
                methods,
                normalized_hyperparameters,
            )
        elif mode == "fuel_stress":
            self._executor.submit(
                self._run_fuel_job,
                job_id,
                question,
                methods,
                normalized_hyperparameters,
                multipliers,
            )
        else:
            self._executor.submit(
                self._run_time_job,
                job_id,
                question,
                methods[0],
                time_limits,
                time_fuel,
            )
        return copy.deepcopy(job)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job is not None else None

    def recent_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(), key=lambda row: row["created_at"], reverse=True
            )
            return copy.deepcopy(jobs[:20])

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            updated_at = _now()
            self._jobs[job_id].update(values, updated_at=updated_at)
            if "progress" in values or "status" in values:
                progress = values.get("progress") or {}
                self._jobs[job_id].setdefault("events", []).append(
                    {
                        "at": updated_at,
                        "status": values.get("status", progress.get("status")),
                        **({"policy": progress["policy"]} if "policy" in progress else {}),
                        **({"day": progress["day"]} if "day" in progress else {}),
                    }
                )
                self._jobs[job_id]["events"] = self._jobs[job_id]["events"][-200:]
            self._persist_job(self._jobs[job_id])

    def _run_job(
        self,
        job_id: str,
        question: dict[str, Any],
        methods: list[str],
        hyperparameters: dict[str, dict[str, int | float]],
    ) -> None:
        self._update(job_id, status="running", progress={"status": "starting"})

        def progress(event: dict[str, Any]) -> None:
            self._update(job_id, progress=event)

        destination = self.report_dir / job_id
        try:
            effective_hyperparameters = copy.deepcopy(hyperparameters)
            for method in methods:
                if method in {"lns", "alns"}:
                    parameters = effective_hyperparameters.setdefault(method, {})
                    if (
                        "time_limit_ms" not in parameters
                        and "alns_iterations" not in parameters
                    ):
                        parameters.update(PRACTICE_BENCHMARK_DEFAULT_HYPERPARAMETERS)
            self._update(
                job_id,
                effective_hyperparameters=effective_hyperparameters,
            )
            report = practice_benchmark(
                question["question_id"],
                methods,
                self.env_path,
                self.state_dir,
                destination,
                leave_best=True,
                peer_team_ids=question.get("team_ids") or None,
                poll_interval=self.poll_interval,
                binary_path=self.binary_path,
                base_url=self.base_url,
                quiet=True,
                progress=progress,
                hyperparameters=effective_hyperparameters,
            )
            summary = _suite_map_summary(
                question, report, destination / "report.json"
            )
            result = {
                "summary": summary,
                "policy_results": report["results"],
                "peer_baselines": report["peer_baselines"],
                "completed_peer_comparison": report["completed_peer_comparison"],
                "combined_ranking": report["combined_ranking"],
            }
            self._update(
                job_id,
                status="completed",
                progress={"status": "completed", "policy": report["final_policy"]},
                result=result,
            )
        except Exception as error:  # Keep worker failures visible to the browser.
            self._update(
                job_id,
                status="failed",
                progress={"status": "failed"},
                error=str(error),
            )

    def _run_fuel_job(
        self,
        job_id: str,
        question: dict[str, Any],
        methods: list[str],
        hyperparameters: dict[str, dict[str, int | float]],
        fuel_multipliers: tuple[float, ...],
    ) -> None:
        self._update(
            job_id,
            status="running",
            progress={"status": "building fuel variants"},
        )
        destination = self.report_dir / job_id
        try:
            report = fuel_stress_benchmark(
                methods,
                self.env_path,
                destination,
                game_ids=[question["question_id"]],
                fuel_multipliers=fuel_multipliers,
                hyperparameters=hyperparameters,
                binary_path=self.binary_path,
                base_url=self.base_url,
            )
            self._update(
                job_id,
                status="completed",
                progress={"status": "completed"},
                result={
                    "question": question,
                    "maximum_score": report["cases"][0]["maximum_score"],
                    "aggregates": report["aggregates"],
                    "cases": report["cases"],
                    "wall_seconds": report["wall_seconds"],
                    "report": str(destination / "report.json"),
                },
            )
        except Exception as error:
            self._update(
                job_id,
                status="failed",
                progress={"status": "failed"},
                error=str(error),
            )

    def _run_suite_job(
        self,
        job_id: str,
        methods: list[str],
        hyperparameters: dict[str, dict[str, int | float]],
    ) -> None:
        self._update(
            job_id,
            status="running",
            progress={"status": "discovering_maps"},
        )

        def progress(event: dict[str, Any]) -> None:
            self._update(job_id, progress=event)

        destination = self.report_dir / job_id
        try:
            report = practice_suite(
                methods,
                self.env_path,
                self.state_dir,
                destination,
                peer_team_ids=None,
                poll_interval=self.poll_interval,
                binary_path=self.binary_path,
                base_url=self.base_url,
                quiet=True,
                hyperparameters=hyperparameters,
                progress=progress,
            )
            self._update(
                job_id,
                status="completed",
                progress={"status": "completed"},
                result=report,
            )
        except Exception as error:
            self._update(
                job_id,
                status="failed",
                progress={"status": "failed"},
                error=str(error),
            )

    def _run_time_job(
        self,
        job_id: str,
        question: dict[str, Any],
        method: str,
        time_limits_ms: tuple[int, ...],
        fuel_multiplier: float,
    ) -> None:
        self._update(
            job_id,
            status="running",
            progress={"status": "running LNS time curve"},
        )
        destination = self.report_dir / job_id
        try:
            report = lns_time_benchmark(
                self.env_path,
                destination,
                method=method,
                game_ids=[question["question_id"]],
                fuel_multiplier=fuel_multiplier,
                time_limits_ms=time_limits_ms,
                binary_path=self.binary_path,
                base_url=self.base_url,
            )
            self._update(
                job_id,
                status="completed",
                progress={"status": "completed"},
                result={
                    "question": question,
                    "maximum_score": report["cases"][0]["maximum_score"],
                    "fuel_multiplier": fuel_multiplier,
                    "time_limits_ms": list(time_limits_ms),
                    "aggregates": report["aggregates"],
                    "cases": report["cases"],
                    "wall_seconds": report["wall_seconds"],
                    "report": str(destination / "report.json"),
                },
            )
        except Exception as error:
            self._update(
                job_id,
                status="failed",
                progress={"status": "failed"},
                error=str(error),
            )


class DashboardHandler(BaseHTTPRequestHandler):
    app: DashboardApp

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _html(self) -> None:
        body = DASHBOARD_HTML.encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _asset(self, name: str) -> None:
        safe_name = Path(unquote(name)).name
        if safe_name != name or safe_name not in {"app.js", "styles.css"}:
            self._json(HTTPStatus.NOT_FOUND, {"error": "asset not found"})
            return
        asset = STATIC_ROOT / safe_name
        if not asset.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"error": "asset not found"})
            return
        body = asset.read_bytes()
        content_type = "text/javascript; charset=utf-8" if safe_name.endswith(".js") else "text/css; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/local"} or path.startswith("/competition/game/"):
            self._html()
            return
        if path.startswith("/assets/"):
            self._asset(path.removeprefix("/assets/"))
            return
        if path == "/api/health":
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        if path == "/api/bootstrap":
            online_error = None
            try:
                games = self.app.list_all_games()
            except RuntimeError as error:
                # The LOCAL workspace is deliberately usable without a token or
                # network connection; surface the online failure without making
                # the whole static application fail to boot.
                games = []
                online_error = str(error)
            self._json(
                HTTPStatus.OK,
                {
                    "schema_version": 3,
                    "games": games,
                    "online_error": online_error,
                    "policies": POLICIES,
                    "hyperparameters": POLICY_HYPERPARAMETERS,
                    "modes": {"practice": True, "competition": True, "local": True},
                },
            )
            return
        if path == "/api/local/cases":
            self._json(HTTPStatus.OK, self.app.local_cases())
            return
        if path == "/api/local/case":
            try:
                case_id = parse_qs(parsed.query).get("case_id", [""])[0]
                self._json(HTTPStatus.OK, self.app.local_case(case_id))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        if path == "/api/games":
            try:
                mode = parse_qs(parsed.query).get("mode", ["practice"])[0]
                if mode not in {"practice", "competition", "all"}:
                    raise ValueError("mode must be practice, competition, or all")
                self._json(
                    HTTPStatus.OK,
                    {
                        "games": self.app.list_all_games() if mode == "all" else self.app.list_games(mode),
                        "policies": POLICIES,
                        "hyperparameters": POLICY_HYPERPARAMETERS,
                    },
                )
            except (RuntimeError, ValueError) as error:
                self._json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})
            return
        if path == "/api/jobs":
            self._json(HTTPStatus.OK, {"jobs": self.app.recent_jobs()})
            return
        if path.startswith("/api/jobs/"):
            job = self.app.get_job(path.removeprefix("/api/jobs/"))
            if job is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "job not found"})
            else:
                self._json(HTTPStatus.OK, job)
            return
        if path == "/api/runs":
            self._json(HTTPStatus.OK, {"runs": self.app.recent_jobs()})
            return
        if path.startswith("/api/runs/"):
            job = self.app.get_job(path.removeprefix("/api/runs/"))
            if job is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "run not found"})
            else:
                self._json(HTTPStatus.OK, job)
            return
        if path.startswith("/api/games/") and path.endswith("/snapshot"):
            game_id = unquote(path.removeprefix("/api/games/").removesuffix("/snapshot")).strip("/")
            try:
                self._json(HTTPStatus.OK, self.app.snapshot(game_id))
            except (RuntimeError, ValueError) as error:
                self._json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})
            return
        if path.startswith("/api/games/") and path.endswith("/standings"):
            game_id = unquote(path.removeprefix("/api/games/").removesuffix("/standings")).strip("/")
            try:
                self._json(HTTPStatus.OK, self.app.standings(game_id))
            except (RuntimeError, ValueError) as error:
                self._json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})
            return
        if path.startswith("/api/games/") and path.endswith("/replay"):
            game_id = unquote(path.removeprefix("/api/games/").removesuffix("/replay")).strip("/")
            team_id = parse_qs(parsed.query).get("team_id", [None])[0]
            try:
                self._json(HTTPStatus.OK, self.app.replay(game_id, team_id))
            except (RuntimeError, ValueError) as error:
                self._json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})
            return
        if path.startswith("/api/games/") and path.endswith("/answers"):
            game_id = unquote(path.removeprefix("/api/games/").removesuffix("/answers")).strip("/")
            try:
                self._json(HTTPStatus.OK, self.app.answers(game_id))
            except (RuntimeError, ValueError) as error:
                self._json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})
            return
        if path in {"/api/competition/sessions", "/api/play/sessions"}:
            self._json(HTTPStatus.OK, {"sessions": self.app.competition_sessions()})
            return
        if path.startswith("/api/competition/sessions/") or path.startswith("/api/play/sessions/"):
            prefix = "/api/play/sessions/" if path.startswith("/api/play/") else "/api/competition/sessions/"
            session = self.app.get_competition(path.removeprefix(prefix))
            if session is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "session not found"})
            else:
                self._json(HTTPStatus.OK, session)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path not in {
            "/api/jobs",
            "/api/practice/runs",
            "/api/competition/sessions",
            "/api/play/sessions",
            "/api/local/run",
        } and not path.startswith("/api/competition/sessions/") and not path.startswith("/api/play/sessions/") and not (
            path.startswith("/api/games/") and path.endswith("/reset")
        ):
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65_536:
                raise ValueError("invalid request size")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Content-Type must be application/json")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            if path == "/api/local/run":
                case_id = payload.get("case_id")
                method = payload.get("method", "alns")
                hyperparameters = payload.get("hyperparameters")
                if not isinstance(case_id, str) or not isinstance(method, str):
                    raise ValueError("case_id and method are required")
                if hyperparameters is not None and not isinstance(hyperparameters, dict):
                    raise ValueError("hyperparameters must be an object")
                self._json(
                    HTTPStatus.OK,
                    self.app.run_local_case(case_id, method, hyperparameters),
                )
                return
            if path.startswith("/api/games/") and path.endswith("/reset"):
                game_id = unquote(path.removeprefix("/api/games/").removesuffix("/reset")).strip("/")
                self._json(HTTPStatus.OK, self.app.reset_game(game_id))
                return
            if path in {"/api/competition/sessions", "/api/play/sessions"}:
                game_id = payload.get("game_id")
                method = payload.get("method", "alns")
                hyperparameters = payload.get("hyperparameters")
                execution_mode = payload.get("execution_mode", "manual")
                target_day = payload.get("target_day")
                time_limit_seconds = payload.get("time_limit_seconds")
                if not isinstance(game_id, str) or not isinstance(method, str):
                    raise ValueError("game_id and method are required")
                if hyperparameters is not None and not isinstance(hyperparameters, dict):
                    raise ValueError("hyperparameters must be an object")
                if not isinstance(execution_mode, str):
                    raise ValueError("execution_mode must be a string")
                if time_limit_seconds is not None and (
                    isinstance(time_limit_seconds, bool)
                    or not isinstance(time_limit_seconds, (int, float))
                ):
                    raise ValueError("time_limit_seconds must be numeric")
                session = self.app.start_competition(
                    game_id,
                    method,
                    hyperparameters,
                    execution_mode=execution_mode,
                    target_day=target_day,
                    time_limit_seconds=time_limit_seconds,
                )
                self._json(HTTPStatus.ACCEPTED, session)
                return
            if path.startswith("/api/competition/sessions/") or path.startswith("/api/play/sessions/"):
                prefix = "/api/play/sessions/" if path.startswith("/api/play/") else "/api/competition/sessions/"
                session_id = path.removeprefix(prefix).removesuffix("/approve").removesuffix("/submit").removesuffix("/curl").removesuffix("/control").strip("/")
                if path.endswith("/approve"):
                    fingerprint = payload.get("fingerprint")
                    if not isinstance(fingerprint, str):
                        raise ValueError("fingerprint is required")
                    session = self.app.approve_competition(
                        session_id,
                        fingerprint,
                        bool(payload.get("allow_fallback", False)),
                    )
                elif path.endswith("/submit"):
                    fingerprint = payload.get("fingerprint")
                    if not isinstance(fingerprint, str):
                        raise ValueError("fingerprint is required")
                    session = self.app.submit_competition(
                        session_id,
                        fingerprint,
                        types=payload.get("types"),
                        actions=payload.get("actions"),
                        allow_fallback=bool(payload.get("allow_fallback", False)),
                    )
                elif path.endswith("/curl"):
                    fingerprint = payload.get("fingerprint")
                    if not isinstance(fingerprint, str):
                        raise ValueError("fingerprint is required")
                    session = self.app.competition_curl(
                        session_id,
                        fingerprint,
                        types=payload.get("types"),
                        actions=payload.get("actions"),
                    )
                elif path.endswith("/control"):
                    action = payload.get("action")
                    if not isinstance(action, str):
                        raise ValueError("control action is required")
                    session = self.app.control_competition(session_id, action)
                else:
                    raise ValueError("unsupported competition session operation")
                self._json(HTTPStatus.ACCEPTED, session)
                return
            game_id = payload.get("game_id")
            methods = payload.get("methods")
            hyperparameters = payload.get("hyperparameters")
            mode = payload.get("mode", "practice")
            fuel_multipliers = payload.get("fuel_multipliers")
            time_limits_ms = payload.get("time_limits_ms")
            time_fuel_multiplier = payload.get("time_fuel_multiplier", 0.5)
            if not isinstance(game_id, str) or not isinstance(methods, list) or not all(
                isinstance(method, str) for method in methods
            ):
                raise ValueError("game_id and a methods array are required")
            if not isinstance(mode, str):
                raise ValueError("mode must be a string")
            if fuel_multipliers is not None and (
                not isinstance(fuel_multipliers, list)
                or not all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in fuel_multipliers
                )
            ):
                raise ValueError("fuel_multipliers must be a numeric array")
            if time_limits_ms is not None and (
                not isinstance(time_limits_ms, list)
                or not all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in time_limits_ms
                )
            ):
                raise ValueError("time_limits_ms must be a numeric array")
            if isinstance(time_fuel_multiplier, bool) or not isinstance(
                time_fuel_multiplier, (int, float)
            ):
                raise ValueError("time_fuel_multiplier must be numeric")
            job = self.app.start_job(
                game_id,
                methods,
                hyperparameters,
                mode,
                fuel_multipliers,
                time_limits_ms,
                time_fuel_multiplier,
            )
            self._json(HTTPStatus.ACCEPTED, job)
        except (ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except RuntimeError as error:
            self._json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})


def serve_dashboard(
    host: str,
    port: int,
    env_path: Path,
    state_dir: Path,
    report_dir: Path,
    *,
    binary_path: str | None = None,
    base_url: str = BASE_URL,
    poll_interval: float = 0.05,
) -> None:
    app = DashboardApp(
        env_path,
        state_dir,
        report_dir,
        binary_path=binary_path,
        base_url=base_url,
        poll_interval=poll_interval,
    )
    handler = type("ConfiguredDashboardHandler", (DashboardHandler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"HEXUDON dashboard listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HEXUDON Practice Console</title>
  <style>
    :root { --ink:#182018; --muted:#627066; --paper:#f4f0e7; --card:#fffdf8; --line:#d8d3c7; --green:#1f6b45; --green2:#dcecdf; --orange:#bc5f2c; --red:#a43d38; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:radial-gradient(circle at 85% 0,#dbe9d9 0,transparent 32rem),var(--paper); font:15px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif; }
    main { width:min(1120px,calc(100% - 32px)); margin:0 auto; padding:42px 0 64px; }
    header { display:flex; justify-content:space-between; align-items:end; gap:24px; margin-bottom:28px; }
    h1 { font:700 clamp(30px,5vw,52px)/1.02 ui-serif,Georgia,serif; letter-spacing:-.04em; margin:6px 0; }
    h2 { font:700 22px/1.2 ui-serif,Georgia,serif; margin:0 0 18px; }
    .eyebrow { color:var(--green); font-weight:800; letter-spacing:.14em; text-transform:uppercase; font-size:12px; }
    .lede,.muted { color:var(--muted); }
    .live { display:flex; align-items:center; gap:8px; white-space:nowrap; font-weight:700; }
    .dot { width:9px; height:9px; border-radius:50%; background:#32a166; box-shadow:0 0 0 5px #32a16622; }
    .grid { display:grid; grid-template-columns:minmax(0,1.1fr) minmax(320px,.9fr); gap:20px; align-items:start; }
    .card { background:color-mix(in srgb,var(--card) 94%,transparent); border:1px solid var(--line); border-radius:18px; padding:24px; box-shadow:0 16px 38px #2734260b; }
    label.title { display:block; font-weight:800; margin-bottom:8px; }
    select { width:100%; border:1px solid #b9beb6; background:white; color:var(--ink); border-radius:10px; padding:12px 14px; font:inherit; }
    .policies { display:grid; grid-template-columns:1fr 1fr; gap:9px; margin:16px 0 20px; }
    .policy { display:flex; align-items:center; gap:9px; border:1px solid var(--line); border-radius:10px; padding:10px; cursor:pointer; background:#fff; }
    .policy:has(input:checked) { border-color:var(--green); background:var(--green2); }
    .hyperparameters { display:grid; gap:12px; margin:4px 0 20px; }
    .parameter-card { border:1px solid var(--line); border-radius:12px; padding:13px; background:#fff; }
    .parameter-card h3 { margin:0 0 10px; font-size:14px; text-transform:capitalize; }
    .parameter-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
    .parameter-grid label { display:grid; gap:4px; color:var(--muted); font-size:12px; }
    .parameter-grid input { width:100%; border:1px solid #b9beb6; border-radius:8px; padding:8px 9px; font:inherit; color:var(--ink); }
    .parameter-help { color:var(--muted); font-size:12px; }
    .fuel-input { width:100%; border:1px solid #b9beb6; border-radius:9px; padding:9px 10px; font:inherit; margin:0 0 14px; }
    input { accent-color:var(--green); }
    button { width:100%; border:0; border-radius:11px; padding:13px 18px; background:var(--green); color:white; font:800 15px/1 inherit; cursor:pointer; }
    button:hover { background:#175838; }
    button:disabled { opacity:.55; cursor:wait; }
    .button-row { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
    button.secondary { background:#fff; color:var(--green); border:1px solid var(--green); }
    button.secondary:hover { background:var(--green2); }
    .warning { margin-top:12px; color:#78684d; font-size:13px; }
    .status { min-height:150px; display:grid; place-items:center; text-align:center; }
    .spinner { width:30px; height:30px; border:3px solid var(--green2); border-top-color:var(--green); border-radius:50%; animation:spin .8s linear infinite; margin:0 auto 12px; }
    @keyframes spin { to { transform:rotate(360deg); } }
    .error { color:var(--red); background:#f7e7e4; border-radius:10px; padding:12px; }
    #results { margin-top:20px; display:none; }
    .metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin:14px 0 24px; }
    .metric { border:1px solid var(--line); border-radius:12px; padding:14px; background:#fff; }
    .metric strong { display:block; font:700 25px/1.1 ui-serif,Georgia,serif; }
    .metric span { color:var(--muted); font-size:12px; }
    .table-wrap { overflow:auto; margin-top:12px; }
    table { border-collapse:collapse; width:100%; white-space:nowrap; }
    th,td { padding:10px 12px; text-align:left; border-bottom:1px solid var(--line); }
    th { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }
    .tag { display:inline-block; padding:3px 8px; border-radius:99px; background:#ecece6; font-size:12px; }
    .tag.completed { color:var(--green); background:var(--green2); }
    footer { margin-top:18px; color:var(--muted); font-size:12px; }
    @media (max-width:800px) { .grid { grid-template-columns:1fr; } header { align-items:start; flex-direction:column; } .metrics { grid-template-columns:1fr 1fr; } }
    @media (max-width:480px) { main { width:min(100% - 20px,1120px); padding-top:24px; } .policies,.button-row,.parameter-grid { grid-template-columns:1fr; } .card { padding:18px; } }
  </style>
</head>
<body>
<main>
  <header>
    <div><div class="eyebrow">Practice operations</div><h1>HEXUDON Console</h1><div class="lede">Benchmark policies on the authoritative server and see where they rank.</div></div>
    <div class="live"><span class="dot"></span> API connected</div>
  </header>
  <div class="grid">
    <section class="card">
      <h2>Configure benchmark</h2>
      <label class="title" for="game">Practice game</label>
      <select id="game" disabled><option>Loading safe practice games…</option></select>
      <label class="title" style="margin-top:20px">Policies</label>
      <div id="policies" class="policies"></div>
      <label class="title" style="margin-top:20px">Method hyper-parameters</label>
      <div id="hyperparameters" class="hyperparameters"><div class="parameter-help">Select a policy to configure its optional controls.</div></div>
      <div class="parameter-help">Untimed ALNS exposes literal warm-start, normal-day/final-day ALNS, and normal-day/final-day exact-search counts. Enter a wall-clock limit for timed search or set every fixed-search phase below.</div>
      <label class="title" for="fuel-multipliers">Fuel stress multipliers</label>
      <input class="fuel-input" id="fuel-multipliers" value="1.0, 0.5, 0.25" aria-describedby="fuel-help">
      <div class="parameter-help" id="fuel-help" style="margin:-8px 0 14px">Multiples of Day-1 steps; the server fuel value is always included.</div>
      <label class="title" for="time-limits">LNS time curve</label>
      <div class="parameter-grid" style="margin-bottom:14px"><label>Budgets per day (ms)<input id="time-limits" value="25, 100, 500, 2000, 10000"></label><label>Fuel multiplier<input id="time-fuel" type="number" min="0.01" max="3" step="0.05" value="0.5"></label></div>
      <div class="button-row"><button id="run" disabled>Server benchmark</button><button id="fuel-run" class="secondary" disabled>Fuel stress</button><button id="time-run" class="secondary" disabled>LNS time curve</button></div>
      <div class="warning">Server benchmark resets the selected practice game and leaves the best run submitted. Fuel stress reads the same map but evaluates lower-fuel variants entirely locally. Jobs execute one at a time.</div>
    </section>
    <section class="card status" id="status"><div><div class="eyebrow">Ready</div><h2 style="margin-top:8px">Choose a game and policies</h2><div class="muted">Results and peer rankings will appear here.</div></div></section>
  </div>
  <section class="card" id="results">
    <div class="eyebrow" id="result-game"></div><h2>Server result</h2>
    <div class="metrics" id="metrics"></div>
    <h2>Policy comparison</h2><div class="table-wrap"><table><thead><tr><th>Rank</th><th>Policy</th><th>Distinct</th><th>Daily</th><th>Servings</th><th>Wall time</th></tr></thead><tbody id="policy-rows"></tbody></table></div>
    <h2 style="margin-top:28px">Other teams</h2><div class="table-wrap"><table><thead><tr><th>Team</th><th>Status</th><th>Days</th><th>Distinct</th><th>Daily</th><th>Servings</th></tr></thead><tbody id="peer-rows"></tbody></table></div>
  </section>
  <section class="card" id="fuel-results" style="display:none;margin-top:20px">
    <div class="eyebrow" id="fuel-result-game"></div><h2>Local fuel-stress result</h2>
    <div class="metrics" id="fuel-metrics"></div>
    <div class="table-wrap"><table><thead><tr><th>Fuel</th><th>Method</th><th>Distinct</th><th>Daily</th><th>Servings</th><th>Refuels</th><th>Refuel cars</th></tr></thead><tbody id="fuel-rows"></tbody></table></div>
  </section>
  <section class="card" id="time-results" style="display:none;margin-top:20px">
    <div class="eyebrow" id="time-result-game"></div><h2>LNS score versus time</h2>
    <div class="metrics" id="time-metrics"></div>
    <div class="table-wrap"><table><thead><tr><th>Budget/day</th><th>Distinct</th><th>Daily</th><th>Servings</th><th>Refuels</th><th>Actual runtime</th></tr></thead><tbody id="time-rows"></tbody></table></div>
  </section>
  <footer>The token remains server-side. Only resettable practice games assigned to this team are offered.</footer>
</main>
<script>
const game = document.querySelector('#game'), policies = document.querySelector('#policies'), hyperparameters = document.querySelector('#hyperparameters'), fuelMultipliers = document.querySelector('#fuel-multipliers'), timeLimits = document.querySelector('#time-limits'), timeFuel = document.querySelector('#time-fuel'), run = document.querySelector('#run'), fuelRun = document.querySelector('#fuel-run'), timeRun = document.querySelector('#time-run'), status = document.querySelector('#status'), results = document.querySelector('#results'), fuelResults = document.querySelector('#fuel-results'), timeResults = document.querySelector('#time-results');
let parameterDefinitions = {};
const parameterValues = {};
const esc = value => String(value ?? '—').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
async function jsonFetch(url, options) { const response = await fetch(url, options); const data = await response.json(); if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`); return data; }
function selectedPolicies() { return [...document.querySelectorAll('input[name=policy]:checked')].map(input => input.value); }
function parameterBounds(field) {
  const minimum = field.min ?? field.exclusive_min;
  const maximum = field.max ?? field.exclusive_max;
  return `${minimum === undefined ? '' : ` min="${minimum}"`}${maximum === undefined ? '' : ` max="${maximum}"`} step="${field.step}"`;
}
function renderHyperparameters() {
  const selected = selectedPolicies();
  hyperparameters.innerHTML = selected.map(method => {
    const fields = parameterDefinitions[method] || [];
    if (!fields.length) return `<div class="parameter-card"><h3>${esc(method)}</h3><div class="parameter-help">This method has no exposed hyper-parameters.</div></div>`;
    const values = parameterValues[method] || (parameterValues[method] = {});
    return `<div class="parameter-card"><h3>${esc(method)}</h3><div class="parameter-grid">${fields.map(field => `<label>${esc(field.label)}<input data-method="${esc(method)}" data-key="${esc(field.key)}" type="number"${parameterBounds(field)} value="${values[field.key] ?? ''}" placeholder="compiled default"></label>`).join('')}</div></div>`;
  }).join('') || '<div class="parameter-help">Select a policy to configure its optional controls.</div>';
  hyperparameters.querySelectorAll('input[data-method]').forEach(input => input.addEventListener('input', () => {
    parameterValues[input.dataset.method] ||= {};
    if (input.value === '') delete parameterValues[input.dataset.method][input.dataset.key];
    else parameterValues[input.dataset.method][input.dataset.key] = input.value;
  }));
}
function selectedHyperparameters() {
  const result = {};
  hyperparameters.querySelectorAll('input[data-method]').forEach(input => {
    if (input.value === '') return;
    result[input.dataset.method] ||= {};
    result[input.dataset.method][input.dataset.key] = input.step === '1' ? Number.parseInt(input.value, 10) : Number.parseFloat(input.value);
  });
  return result;
}
function setStatus(kind, title, detail='') { status.innerHTML = kind === 'running' ? `<div><div class="spinner"></div><div class="eyebrow">Running</div><h2 style="margin-top:8px">${esc(title)}</h2><div class="muted">${esc(detail)}</div></div>` : `<div class="${kind === 'error' ? 'error' : ''}"><div class="eyebrow">${esc(kind)}</div><h2 style="margin-top:8px">${esc(title)}</h2><div class="muted">${esc(detail)}</div></div>`; }
async function load() {
  try {
    const data = await jsonFetch('/api/games');
    game.innerHTML = data.games.map(g => `<option value="${esc(g.question_id)}">${esc(g.name)} · ${g.width}×${g.height} · ${g.total_days} days</option>`).join('');
    parameterDefinitions = data.hyperparameters || {};
    policies.innerHTML = data.policies.map(p => `<label class="policy"><input type="checkbox" name="policy" value="${esc(p)}" ${p === 'local_search' ? 'checked' : ''}><span>${esc(p)}</span></label>`).join('');
    policies.querySelectorAll('input[name=policy]').forEach(input => input.addEventListener('change', renderHyperparameters));
    renderHyperparameters();
    game.disabled = !data.games.length; run.disabled = !data.games.length; fuelRun.disabled = !data.games.length; timeRun.disabled = !data.games.length;
    if (!data.games.length) setStatus('error','No safe practice games','No assigned resettable games were discovered.');
  } catch (error) { setStatus('error','Could not load games',error.message); }
}
function render(job) {
  const r = job.result, s = r.summary, score = s.score, maximum = s.maximum_score || score, structural = s.structural_maximum_score || maximum, compare = r.completed_peer_comparison;
  document.querySelector('#result-game').textContent = `${s.name} · ${s.width}×${s.height} · best: ${s.best_policy}`;
  document.querySelector('#metrics').innerHTML = [
    [`${s.completed_rank}/${s.completed_players}`,'Completed rank'],
    [`${s.provisional_rank}/${s.configured_players}`,'Provisional rank'],
    [`${score.distinct_types}/${maximum.distinct_types}`,'Distinct types'],
    [`${score.cumulative_daily_types}/${maximum.cumulative_daily_types}`,'Cumulative daily types'],
    [`${score.total_servings}/${structural.total_servings} · raw ${maximum.total_servings}`,'Total servings / structural ceiling'],
    [`${compare.wins}–${compare.ties}–${compare.losses}`,'Wins / ties / losses']
  ].map(([v,l]) => `<div class="metric"><strong>${esc(v)}</strong><span>${esc(l)}</span></div>`).join('');
  document.querySelector('#policy-rows').innerHTML = r.policy_results.map(x => `<tr><td>${x.rank}</td><td><strong>${esc(x.policy)}</strong></td><td>${x.distinct_types}/${maximum.distinct_types}</td><td>${x.cumulative_daily_types}/${maximum.cumulative_daily_types}</td><td>${x.total_servings}/${maximum.total_servings}</td><td>${Number(x.wall_seconds).toFixed(2)}s</td></tr>`).join('');
  document.querySelector('#peer-rows').innerHTML = r.peer_baselines.map(x => `<tr><td>${esc(x.team_id)}</td><td><span class="tag ${esc(x.status)}">${esc(x.status)}</span></td><td>${x.submitted_days}/${x.total_days}</td><td>${x.distinct_types === undefined ? '—' : `${x.distinct_types}/${maximum.distinct_types}`}</td><td>${x.cumulative_daily_types === undefined ? '—' : `${x.cumulative_daily_types}/${maximum.cumulative_daily_types}`}</td><td>${x.total_servings === undefined ? '—' : `${x.total_servings}/${maximum.total_servings}`}</td></tr>`).join('');
  fuelResults.style.display = 'none'; timeResults.style.display = 'none'; results.style.display = 'block'; results.scrollIntoView({behavior:'smooth',block:'start'});
}
function renderFuel(job) {
  const r = job.result, labels = Object.keys(r.aggregates), maximum = r.maximum_score;
  document.querySelector('#fuel-result-game').textContent = `${r.question.name} · server map with local fuel variants`;
  document.querySelector('#fuel-metrics').innerHTML = [
    [labels.length,'Fuel levels'],
    [job.methods.length,'Methods'],
    [`${Number(r.wall_seconds).toFixed(2)}s`,'Wall time'],
    ['Read only','Server impact']
  ].map(([v,l]) => `<div class="metric"><strong>${esc(v)}</strong><span>${esc(l)}</span></div>`).join('');
  document.querySelector('#fuel-rows').innerHTML = labels.flatMap(label => job.methods.map(method => {
    const x = r.aggregates[label][method];
    return `<tr><td><strong>${esc(label)}</strong></td><td>${esc(method)}</td><td>${x.distinct_types}/${x.maximum_distinct_types ?? maximum.distinct_types}</td><td>${x.cumulative_daily_types}/${x.maximum_cumulative_daily_types ?? maximum.cumulative_daily_types}</td><td>${x.total_servings}/${x.maximum_total_servings ?? maximum.total_servings}</td><td>${x.refuel_events}</td><td>${x.refuel_agents}</td></tr>`;
  })).join('');
  results.style.display = 'none'; timeResults.style.display = 'none'; fuelResults.style.display = 'block'; fuelResults.scrollIntoView({behavior:'smooth',block:'start'});
}
function renderTime(job) {
  const r = job.result, budgets = r.time_limits_ms;
  document.querySelector('#time-result-game').textContent = `${r.question.name} · ${r.fuel_multiplier}× Day-1 fuel`;
  document.querySelector('#time-metrics').innerHTML = [
    [budgets.length,'Budgets'],
    [`${r.fuel_multiplier}×`,'Fuel level'],
    [`${Number(r.wall_seconds).toFixed(2)}s`,'Wall time'],
    ['Deadline','Stopping rule']
  ].map(([v,l]) => `<div class="metric"><strong>${esc(v)}</strong><span>${esc(l)}</span></div>`).join('');
  document.querySelector('#time-rows').innerHTML = budgets.map(budget => {
    const x = r.aggregates[String(budget)];
    return `<tr><td><strong>${budget} ms</strong></td><td>${x.distinct_types}/${x.maximum_distinct_types}</td><td>${x.cumulative_daily_types}/${x.maximum_cumulative_daily_types}</td><td>${x.total_servings}/${x.maximum_total_servings}</td><td>${x.refuel_events}</td><td>${Number(x.runtime_seconds).toFixed(2)}s</td></tr>`;
  }).join('');
  results.style.display = 'none'; fuelResults.style.display = 'none'; timeResults.style.display = 'block'; timeResults.scrollIntoView({behavior:'smooth',block:'start'});
}
function setRunDisabled(disabled) { run.disabled = disabled; fuelRun.disabled = disabled; timeRun.disabled = disabled; }
async function poll(id) {
  try {
    const job = await jsonFetch(`/api/jobs/${id}`), p = job.progress || {};
    if (job.status === 'completed') { if (job.mode === 'fuel_stress') { setStatus('completed','Fuel stress complete','Authoritative map variants were evaluated locally; no server state changed.'); renderFuel(job); } else if (job.mode === 'lns_time') { setStatus('completed','LNS time curve complete','Each budget was applied independently per match day.'); renderTime(job); } else { setStatus('completed','Benchmark complete',`Best policy ${job.result.summary.best_policy} is left on the server.`); render(job); } setRunDisabled(false); return; }
    if (job.status === 'failed') { setStatus('error','Benchmark failed',job.error); setRunDisabled(false); return; }
    const phase = String(p.status || 'running').replaceAll('_',' ');
    const title = p.policy ? `${p.policy}: ${phase}` : (job.status === 'queued' ? 'Waiting for earlier job' : 'Benchmarking on server');
    const day = Number.isInteger(p.day) ? `Day ${p.day + 1} of ${job.game.total_days}` : '';
    const budget = Number.isFinite(p.budget_seconds) ? `planning budget ${Number(p.budget_seconds).toFixed(1)}s` : '';
    const iterations = Number.isInteger(p.alns_iterations) ? `${p.alns_iterations.toLocaleString()} ALNS iterations` : '';
    const detail = [day,budget,iterations,`${job.methods.length} selected ${job.methods.length === 1 ? 'policy' : 'policies'}`].filter(Boolean).join(' · ');
    setStatus('running',title,detail); setTimeout(() => poll(id),1000);
  } catch (error) { setStatus('error','Lost benchmark status',error.message); setRunDisabled(false); }
}
run.addEventListener('click', async () => {
  const methods = selectedPolicies(); if (!methods.length) { setStatus('error','Select a policy','Choose at least one policy to benchmark.'); return; }
  setRunDisabled(true); results.style.display = 'none'; fuelResults.style.display = 'none'; timeResults.style.display = 'none'; setStatus('running','Creating benchmark job');
  try { const job = await jsonFetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'practice',game_id:game.value,methods,hyperparameters:selectedHyperparameters()})}); poll(job.id); }
  catch (error) { setStatus('error','Could not start benchmark',error.message); setRunDisabled(false); }
});
fuelRun.addEventListener('click', async () => {
  const methods = selectedPolicies(); if (!methods.length) { setStatus('error','Select a policy','Choose at least one policy to benchmark.'); return; }
  const multipliers = fuelMultipliers.value.split(',').map(value => Number.parseFloat(value.trim()));
  if (!multipliers.length || multipliers.some(value => Number.isNaN(value) || value <= 0 || value > 3)) { setStatus('error','Invalid fuel multipliers','Enter comma-separated values greater than 0 and at most 3.'); return; }
  setRunDisabled(true); results.style.display = 'none'; fuelResults.style.display = 'none'; timeResults.style.display = 'none'; setStatus('running','Building local fuel variants','No server reset or submission will occur.');
  try { const job = await jsonFetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'fuel_stress',game_id:game.value,methods,hyperparameters:selectedHyperparameters(),fuel_multipliers:multipliers})}); poll(job.id); }
  catch (error) { setStatus('error','Could not start fuel benchmark',error.message); setRunDisabled(false); }
});
timeRun.addEventListener('click', async () => {
  const budgets = timeLimits.value.split(',').map(value => Number.parseInt(value.trim(),10));
  const fuel = Number.parseFloat(timeFuel.value);
  if (!budgets.length || budgets.some(value => Number.isNaN(value) || value < 1 || value > 60000)) { setStatus('error','Invalid time budgets','Enter comma-separated milliseconds from 1 to 60000.'); return; }
  if (Number.isNaN(fuel) || fuel <= 0 || fuel > 3) { setStatus('error','Invalid fuel multiplier','Enter a value greater than 0 and at most 3.'); return; }
  setRunDisabled(true); results.style.display = 'none'; fuelResults.style.display = 'none'; timeResults.style.display = 'none'; setStatus('running','Starting LNS time curve','Long budgets are applied once per match day.');
  const method = selectedPolicies().find(value => value === 'alns' || value === 'lns') || 'alns';
  try { const job = await jsonFetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'lns_time',game_id:game.value,methods:[method],time_limits_ms:budgets,time_fuel_multiplier:fuel})}); poll(job.id); }
  catch (error) { setStatus('error','Could not start time curve',error.message); setRunDisabled(false); }
});
load();
</script>
</body>
</html>
"""

# The previous embedded page is kept above as a compatibility fallback for
# downstream imports.  The served dashboard is now an offline static bundle;
# keeping the constant name preserves the existing test and plugin surface.
STATIC_ROOT = Path(__file__).with_name("static")
if (STATIC_ROOT / "index.html").is_file():
    DASHBOARD_HTML = (STATIC_ROOT / "index.html").read_text()
