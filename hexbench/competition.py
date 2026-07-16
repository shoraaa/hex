"""Safe, approval-gated live game controller for the web console.

The original ``deploy`` command intentionally remains autonomous for CLI
compatibility.  The web controller uses the same planner and simulator but
splits planning from submission so an operator must approve each plan.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .api import (
    BASE_URL,
    GameClient,
    POLICY_HYPERPARAMETERS,
    discover_assigned_games,
    fetch_game_snapshot,
    load_token,
    normalize_competitive_state,
    normalize_hyperparameters,
    planning_budget,
    trace_action_plan,
)
from .models import validate_action_shape, validate_agent_types
from .runner import find_binary, run_core


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _history_for_planner(journal: dict[str, Any]) -> dict[str, Any]:
    submitted = journal.get("submitted_days", {})
    return {
        "distinct_brands": list(journal.get("distinct_brands", [])),
        "submitted_actions": [
            submitted[str(day)]
            for day in range(len(submitted))
            if str(day) in submitted
        ],
    }


class CompetitionSessionManager:
    """Manage approval-gated competition sessions and durable proposals."""

    def __init__(
        self,
        env_path: Path,
        state_dir: Path,
        report_dir: Path,
        *,
        binary_path: str | None = None,
        base_url: str = BASE_URL,
        poll_interval: float = 1.0,
    ) -> None:
        self.env_path = env_path
        self.state_dir = state_dir
        self.report_dir = report_dir
        self.binary_path = binary_path
        self.base_url = base_url
        self.poll_interval = max(0.2, poll_interval)
        self._sessions: dict[str, dict[str, Any]] = {}
        self._events: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._load_sessions()
        self._resume_result_sessions()

    def _session_path(self, session_id: str) -> Path:
        return self.report_dir / "sessions" / session_id / "session.json"

    def _load_sessions(self) -> None:
        root = self.report_dir / "sessions"
        if not root.exists():
            return
        for path in root.glob("*/session.json"):
            try:
                session = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(session, dict) or not session.get("id"):
                continue
            if session.get("state") in {
                "planning",
                "awaiting_role_approval",
                "awaiting_plan_approval",
                "submitting",
                "waiting_for_day",
            }:
                session["state"] = "interrupted"
                session["error"] = "dashboard restarted; resume explicitly"
                session["updated_at"] = _now()
                _write_json(path, session)
            self._sessions[str(session["id"])] = session

    def _resume_result_sessions(self) -> None:
        """Safely resume read-only result polling after a dashboard restart."""
        for session_id, session in self._sessions.items():
            if session.get("state") != "waiting_for_result":
                continue
            self._events[session_id] = threading.Event()
            thread = threading.Thread(
                target=self._run,
                args=(session_id,),
                daemon=True,
                name=f"hexbench-competition-{session_id[:8]}",
            )
            self._threads[session_id] = thread
            thread.start()

    def close(self) -> None:
        self._closed = True
        with self._lock:
            for event in self._events.values():
                event.set()

    def list_games(self) -> list[dict[str, Any]]:
        return discover_assigned_games(load_token(self.env_path), self.base_url)

    def recent_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = sorted(
                self._sessions.values(),
                key=lambda row: row.get("created_at", ""),
                reverse=True,
            )
            return copy.deepcopy(rows[:20])

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._sessions.get(session_id)
            return copy.deepcopy(value) if value is not None else None

    def start_session(
        self,
        game_id: str,
        method: str,
        hyperparameters: dict[str, int | float] | None = None,
    ) -> dict[str, Any]:
        if method not in POLICY_HYPERPARAMETERS:
            raise ValueError(f"unknown policy: {method}")
        normalized = normalize_hyperparameters([method], {method: hyperparameters or {}}).get(method, {})
        token = load_token(self.env_path)
        snapshot = fetch_game_snapshot(token, game_id, self.base_url)
        board = snapshot.get("board", {})
        if not board.get("game_id"):
            raise ValueError("game snapshot did not contain a canonical id")
        if snapshot.get("state", {}).get("status") == "reset_incomplete":
            raise ValueError(str(snapshot["state"].get("error")))
        descriptor = next(
            (row for row in discover_assigned_games(token, self.base_url)
             if row["question_id"] == game_id),
            None,
        )
        if descriptor is None:
            # Direct IDs are allowed when a manager is temporarily stale, but
            # preserve the safety property that the board must be readable.
            descriptor = {
                "question_id": game_id,
                "name": game_id,
                "mode": (
                    "practice"
                    if board.get("is_practice") and not board.get("no_reset")
                    else "competition"
                ),
                "competition_kind": "practice_competition" if board.get("is_practice") else "competition",
                "is_practice": bool(board.get("is_practice")),
                "no_reset": bool(board.get("no_reset")),
                "capabilities": {"reset": False, "submit": True},
            }
        if descriptor["mode"] != "competition":
            raise ValueError("resettable practice games belong in Practice mode")
        canonical = str(snapshot.get("game_id", game_id))
        with self._lock:
            for row in self._sessions.values():
                if row.get("game_id") == canonical and row.get("state") not in {
                    "finished", "cancelled", "failed", "interrupted",
                }:
                    raise ValueError("an active session already exists for this game")
            session_id = uuid.uuid4().hex
            session = {
                "id": session_id,
                "game": descriptor,
                "requested_game_id": game_id,
                "game_id": canonical,
                "method": method,
                "hyperparameters": normalized,
                "state": "starting",
                "snapshot": snapshot,
                "proposal": None,
                "last_submission": None,
                "created_at": _now(),
                "updated_at": _now(),
                "progress": {"status": "starting"},
                "events": [{"at": _now(), "status": "starting"}],
            }
            self._sessions[session_id] = session
            self._events[session_id] = threading.Event()
            _write_json(self._session_path(session_id), session)
            thread = threading.Thread(
                target=self._run,
                args=(session_id,),
                daemon=True,
                name=f"hexbench-competition-{session_id[:8]}",
            )
            self._threads[session_id] = thread
            thread.start()
            return copy.deepcopy(session)

    def approve(
        self,
        session_id: str,
        *,
        fingerprint: str,
        allow_fallback: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError("session not found")
            proposal = session.get("proposal")
            if session.get("state") not in {"awaiting_role_approval", "awaiting_plan_approval"} or not proposal:
                raise ValueError("session is not waiting for approval")
            if fingerprint != proposal.get("fingerprint"):
                raise ValueError("proposal is stale; refresh before approving")
            if proposal.get("fallback") and not allow_fallback:
                raise ValueError("fallback wait requires explicit allow_fallback")
            session["approval"] = {
                "fingerprint": fingerprint,
                "allow_fallback": bool(allow_fallback),
                "approved_at": _now(),
            }
            session["progress"] = {"status": "approval_received"}
            session["updated_at"] = _now()
            session.setdefault("events", []).append(
                {"at": session["updated_at"], "status": "approval_received"}
            )
            _write_json(self._session_path(session_id), session)
            self._events[session_id].set()
            return copy.deepcopy(session)

    def control(self, session_id: str, action: str) -> dict[str, Any]:
        if action not in {"pause", "resume", "cancel", "replan"}:
            raise ValueError("control action must be pause, resume, cancel, or replan")
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError("session not found")
            if action == "cancel":
                session["state"] = "cancelled"
            elif action == "pause":
                session["paused"] = True
                session["progress"] = {"status": "paused"}
            elif action == "resume":
                session["paused"] = False
                session["progress"] = {"status": "resuming"}
                if session.get("state") in {"interrupted", "paused"}:
                    session["state"] = "starting"
                    event = self._events.setdefault(session_id, threading.Event())
                    thread = self._threads.get(session_id)
                    if thread is None or not thread.is_alive():
                        thread = threading.Thread(
                            target=self._run,
                            args=(session_id,),
                            daemon=True,
                            name=f"hexbench-competition-{session_id[:8]}",
                        )
                        self._threads[session_id] = thread
                        thread.start()
                    event.set()
            elif action == "replan":
                if session.get("state") != "awaiting_plan_approval":
                    raise ValueError("replan is only available for a proposed day plan")
                session["proposal"] = None
                session["state"] = "waiting_for_day"
                session["progress"] = {"status": "replanning"}
            session["updated_at"] = _now()
            session.setdefault("events", []).append(
                {"at": session["updated_at"], "status": action}
            )
            _write_json(self._session_path(session_id), session)
            self._events[session_id].set()
            return copy.deepcopy(session)

    def _update(self, session_id: str, **values: Any) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            updated_at = _now()
            previous_state = session.get("state")
            session.update(values, updated_at=updated_at)
            if "progress" in values or (
                "state" in values and values.get("state") != previous_state
            ):
                progress = values.get("progress") or {}
                event = {
                    "at": updated_at,
                    "status": progress.get("status", values.get("state")),
                    **({"day": progress["day"]} if "day" in progress else {}),
                }
                events = session.setdefault("events", [])
                compacted: list[dict[str, Any]] = []
                for existing in events:
                    if compacted and (
                        existing.get("status"), existing.get("day")
                    ) == (
                        compacted[-1].get("status"),
                        compacted[-1].get("day"),
                    ):
                        continue
                    compacted.append(existing)
                if not compacted or (
                    event.get("status"), event.get("day")
                ) != (
                    compacted[-1].get("status"), compacted[-1].get("day")
                ):
                    compacted.append(event)
                session["events"] = compacted[-300:]
            _write_json(self._session_path(session_id), session)

    def _wait(self, session_id: str, timeout: float | None = None) -> bool:
        event = self._events[session_id]
        signalled = event.wait(timeout)
        event.clear()
        return signalled

    def _journal(self, resolved_id: str) -> tuple[Path, dict[str, Any]]:
        digest = hashlib.sha256(resolved_id.encode()).hexdigest()[:24]
        path = self.state_dir / f"{digest}.json"
        if path.exists():
            try:
                journal = json.loads(path.read_text())
            except (OSError, ValueError):
                journal = {}
        else:
            journal = {}
        journal.setdefault("distinct_brands", [])
        journal.setdefault("submitted_days", {})
        journal.setdefault("day_snapshots", {})
        journal.setdefault("types", None)
        return path, journal

    def _plan_search(self, session: dict[str, Any], config: dict[str, Any], day: dict[str, Any]) -> dict[str, int]:
        params = session.get("hyperparameters", {})
        budget = planning_budget(
            config,
            day,
            is_practice=bool(session.get("game", {}).get("is_practice")),
            deadline_margin=2.0,
        )
        alns_iterations = params.get("alns_iterations")
        if alns_iterations is not None:
            alns_iterations = int(alns_iterations)
            return {
                "minIterations": int(params.get("min_iterations", min(2048, alns_iterations))),
                "maxIterations": alns_iterations,
                "stagnationIterations": int(params.get("stagnation_iterations", alns_iterations)),
            }
        return {
            "timeLimitMs": min(int(params.get("time_limit_ms", budget * 850)), max(50, int(budget * 850))),
            "minIterations": int(params.get("min_iterations", 32)),
            "maxIterations": int(params.get("max_iterations", 10_000_000 if session["method"] in {"lns", "alns"} else 2048)),
            "stagnationIterations": int(params.get("stagnation_iterations", 0 if session["method"] in {"lns", "alns"} else 96)),
        }

    def _build_proposal(
        self,
        session: dict[str, Any],
        config: dict[str, Any],
        day: dict[str, Any],
        journal: dict[str, Any],
    ) -> dict[str, Any]:
        binary = find_binary(self.binary_path)
        types = [int(agent["kind"]) for agent in day["agents"]]
        history = _history_for_planner(journal)
        payload = {
            "config": config,
            "day_info": day,
            "history": history,
            "types": types,
            "search": self._plan_search(session, config, day),
            "hyperparameters": {
                key: value
                for key, value in session.get("hyperparameters", {}).items()
                if key != "alns_iterations"
            },
        }
        fallback = False
        planner_error: str | None = None
        try:
            actions = run_core(
                "plan",
                session["method"],
                payload,
                binary=binary,
                timeout=max(
                    0.2,
                    planning_budget(
                        config,
                        day,
                        is_practice=bool(session.get("game", {}).get("is_practice")),
                        deadline_margin=2.0,
                    ),
                ),
            )
            validate_action_shape(actions, len(types))
            check = run_core(
                "check",
                session["method"],
                {"config": config, "day_info": day, "history": history, "actions": actions},
                binary=binary,
            )
            if not check.get("valid"):
                planner_error = str(check.get("error") or "planner returned an invalid plan")
                raise RuntimeError(planner_error)
        except Exception as error:  # The safe fallback is proposed, never hidden.
            planner_error = str(error)
            actions = [[-int(config["daySteps"][int(day["day"])])] for _ in types]
            check = run_core(
                "check",
                session["method"],
                {"config": config, "day_info": day, "history": history, "actions": actions},
                binary=binary,
            )
            if not check.get("valid"):
                raise RuntimeError(
                    f"safe wait fallback failed validation: {check.get('error')}"
                )
            fallback = True
        trace = trace_action_plan(config, day, history, actions, binary_path=self.binary_path)
        proposal = {
            "kind": "day_plan",
            "day": int(day["day"]),
            "actions": actions,
            "validation": check,
            "trace": trace,
            "fallback": fallback,
            "planner_error": planner_error,
            "deadline": day.get("endsAt"),
            "created_at": _now(),
        }
        proposal["fingerprint"] = _fingerprint(proposal)
        return proposal

    def _run(self, session_id: str) -> None:
        client: GameClient | None = None
        try:
            session = self.get_session(session_id)
            if session is None:
                return
            token = load_token(self.env_path)
            client = GameClient(token, self.base_url)
            resolved_id = session["game_id"]
            competitive_practice = bool(
                session.get("game", {}).get("is_practice")
                and session.get("game", {}).get("no_reset")
            )
            api_game_id = (
                str(session.get("requested_game_id") or resolved_id)
                if competitive_practice
                else resolved_id
            )
            config = client.get("/game/config", resolved_id)
            state_path, journal = self._journal(resolved_id)
            competitive_selection_initialized = False
            while not self._closed:
                current = self.get_session(session_id)
                if current is None or current.get("state") in {"cancelled", "finished", "failed"}:
                    return
                if current.get("paused"):
                    self._update(session_id, state="paused")
                    self._wait(session_id, 2.0)
                    continue
                competitive_state: dict[str, Any] | None = None
                day: dict[str, Any] | None = None
                if competitive_practice:
                    competitive_state = client.get(
                        "/game/competitive/state", api_game_id
                    )
                    status, day = normalize_competitive_state(
                        competitive_state, config
                    )
                else:
                    status = client.get("/game/state", resolved_id)
                current_snapshot = {
                    "requested_game_id": current.get("requested_game_id"),
                    "game_id": resolved_id,
                    "board": current.get("snapshot", {}).get("board", {}),
                    "config": config,
                    "state": status,
                    "day": day,
                    "fetched_at": _now(),
                }
                if competitive_state is not None:
                    current_snapshot["competitive_state"] = competitive_state
                self._update(session_id, snapshot=current_snapshot)
                if status.get("status") == "reset_incomplete":
                    raise RuntimeError(str(status.get("error")))
                day_count = len(config.get("daySteps", []))
                day_index = int(status.get("day", 0))
                terminal_status = status.get("status") == "finished"
                terminal_day = not competitive_practice and (
                    status.get("status") == "in_progress"
                    and day_count > 0
                    and day_index >= day_count
                )
                if terminal_status or terminal_day:
                    if competitive_practice:
                        result = competitive_state
                        self._update(
                            session_id,
                            state="finished",
                            result=result,
                            progress={"status": "finished"},
                        )
                        return
                    endpoint = (
                        "/game/practice/score"
                        if current["game"].get("is_practice")
                        else "/game/result"
                    )
                    try:
                        result = client.get(endpoint, resolved_id)
                    except RuntimeError:
                        if terminal_status:
                            raise
                        self._update(
                            session_id,
                            state="waiting_for_result",
                            progress={"status": "waiting_for_result"},
                        )
                        self._wait(session_id, self.poll_interval)
                        continue
                    self._update(
                        session_id,
                        state="finished",
                        result=result,
                        progress={"status": "finished"},
                    )
                    return
                if status.get("status") == "selecting_agents":
                    if competitive_practice and not competitive_selection_initialized:
                        competitive_selection_initialized = True
                        if any(
                            (
                                journal.get("types"),
                                journal.get("submitted_days"),
                                journal.get("day_snapshots"),
                                journal.get("distinct_brands"),
                            )
                        ):
                            journal = {
                                "types": None,
                                "submitted_days": {},
                                "day_snapshots": {},
                                "distinct_brands": [],
                            }
                            _write_json(state_path, journal)
                    if current.get("state") == "awaiting_role_approval" and current.get("approval"):
                        proposal = current.get("proposal") or {}
                        approval = current.get("approval")
                        if approval.get("fingerprint") != proposal.get("fingerprint"):
                            raise RuntimeError("role approval fingerprint mismatch")
                        client.post(
                            "/game/agent-types",
                            {"game_id": api_game_id, "types": proposal.get("types", [])},
                        )
                        journal["types"] = proposal.get("types", [])
                        _write_json(state_path, journal)
                        self._update(
                            session_id,
                            state="waiting_for_day",
                            proposal=None,
                            approval=None,
                            progress={"status": "roles_submitted"},
                        )
                        self._wait(session_id, self.poll_interval)
                        continue
                    if current.get("state") == "awaiting_role_approval":
                        self._wait(session_id, 1.0)
                        continue
                    types = journal.get("types") or run_core(
                        "types",
                        current["method"],
                        config,
                        binary=find_binary(self.binary_path),
                    )
                    validate_agent_types(types, len(config["agents"]))
                    proposal = {
                        "kind": "agent_types",
                        "types": types,
                        "created_at": _now(),
                    }
                    proposal["fingerprint"] = _fingerprint(proposal)
                    self._update(
                        session_id,
                        state="awaiting_role_approval",
                        proposal=proposal,
                        progress={"status": "awaiting_role_approval"},
                    )
                    self._wait(session_id, 1.0)
                    continue
                if status.get("status") != "in_progress":
                    self._update(session_id, state="waiting_for_day", progress={"status": status.get("status", "waiting")})
                    self._wait(session_id, self.poll_interval)
                    continue
                if day_index >= len(config.get("daySteps", [])):
                    self._update(
                        session_id,
                        state="waiting_for_result",
                        progress={"status": "waiting_for_result"},
                    )
                    self._wait(session_id, self.poll_interval)
                    continue
                key = str(day_index)
                if key in journal.get("submitted_days", {}):
                    self._update(session_id, state="waiting_for_day", progress={"status": "submitted", "day": day_index})
                    self._wait(session_id, self.poll_interval)
                    continue
                if day is None:
                    day = client.get("/game/day", resolved_id)
                    current_snapshot["day"] = day
                    self._update(session_id, snapshot=current_snapshot)
                if current.get("state") != "awaiting_plan_approval" or not current.get("proposal"):
                    proposal = self._build_proposal(current, config, day, journal)
                    self._update(
                        session_id,
                        state="awaiting_plan_approval",
                        proposal=proposal,
                        progress={"status": "awaiting_plan_approval", "day": day_index},
                    )
                    self._wait(session_id, 1.0)
                    continue
                approval = current.get("approval")
                if not approval:
                    self._wait(session_id, 1.0)
                    continue
                if approval.get("fingerprint") != current["proposal"].get("fingerprint"):
                    self._update(session_id, approval=None, state="waiting_for_day", proposal=None, error="approval fingerprint mismatch")
                    continue
                if (
                    not current.get("game", {}).get("is_practice")
                    and day.get("endsAt") is not None
                    and float(day["endsAt"]) <= time.time()
                ):
                    self._update(session_id, approval=None, state="waiting_for_day", proposal=None, error="deadline passed before approval")
                    continue
                proposal = current["proposal"]
                self._update(session_id, state="submitting", progress={"status": "submitting", "day": day_index})
                endpoint = (
                    "/game/competitive/actions"
                    if competitive_practice
                    else (
                        "/game/practice/actions"
                        if current["game"].get("is_practice")
                        else "/game/actions"
                    )
                )
                response = client.post(endpoint, {"game_id": api_game_id, "day": day_index, "actions": proposal["actions"]})
                journal["submitted_days"][key] = proposal["actions"]
                journal["day_snapshots"][key] = {
                    "day_info": day,
                    "actions": proposal["actions"],
                    "validation": proposal["validation"],
                    "trace": proposal.get("trace"),
                    "submitted_at": _now(),
                }
                if proposal.get("validation", {}).get("score"):
                    collected_positions = {
                        int(position)
                        for frame in proposal.get("trace", {}).get("frames", [])
                        for position in frame.get("collected", [])
                    }
                    journal["distinct_brands"] = sorted(
                        set(journal.get("distinct_brands", []))
                        | {
                            int(item["brand"])
                            for item in config.get("spots", [])
                            if int(item.get("pos", -1)) in collected_positions
                        }
                    )
                _write_json(state_path, journal)
                self._update(
                    session_id,
                    state="submitted",
                    proposal=None,
                    approval=None,
                    last_submission={"day": day_index, "endpoint": endpoint, "response": response, "submitted_at": _now()},
                    progress={"status": "submitted", "day": day_index},
                )
                self._wait(session_id, self.poll_interval)
        except Exception as error:
            self._update(session_id, state="failed", error=str(error), progress={"status": "failed"})
        finally:
            if client is not None:
                client.close()
