"""Safe, approval-gated live game controller for the web console.

The original ``deploy`` command intentionally remains autonomous for CLI
compatibility.  The web controller uses the same planner and simulator but
splits planning from submission so an operator must approve each plan.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shlex
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .api import (
    BASE_URL,
    GameClient,
    MLNS_ANYTIME_ITERATION_CEILING,
    POLICY_HYPERPARAMETERS,
    STATEFUL_DEFAULTS,
    STATEFUL_POLICIES,
    _token_team_id,
    discover_assigned_games,
    fetch_game_snapshot,
    load_token,
    normalize_competitive_day,
    normalize_competitive_state,
    normalize_hyperparameters,
    planning_budget,
    solver_time_limit_ms,
    trace_action_plan,
)
from .models import validate_action_shape, validate_agent_types
from .runner import find_binary, run_core, stream_core


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


def _history_for_planner(
    journal: dict[str, Any], before_day: int | None = None
) -> dict[str, Any]:
    submitted = journal.get("submitted_days", {})
    days = sorted(
        int(day)
        for day in submitted
        if before_day is None or int(day) < before_day
    )
    return {
        "distinct_brands": list(journal.get("distinct_brands", [])),
        "cumulative_daily_types": int(
            journal.get("cumulative_daily_types", 0)
        ),
        "total_servings": int(journal.get("total_servings", 0)),
        "submitted_actions": [submitted[str(day)] for day in days],
    }


def _score_triplet(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError("competitive holder score is missing or invalid")
    return tuple(int(value[index]) for index in range(3))


def _challenge_score(
    holder_score: Any,
    local_score: dict[str, Any] | None,
    baseline_score: Any = None,
) -> tuple[list[int], list[int] | None, bool | None]:
    """Project a replacement from the score before the challenged day.

    ``holder_score`` already includes the holder's contribution for this day.
    Adding a new day contribution to it double-counts that day.  When the
    pre-day baseline is not observable, leave the projection unknown and let
    the competitive endpoint make the authoritative comparison.
    """
    holder = _score_triplet(holder_score)
    if baseline_score is None:
        return list(holder), None, None
    baseline = _score_triplet(baseline_score)
    local = local_score or {}
    candidate = (
        max(baseline[0], int(local.get("distinct_types", 0))),
        baseline[1] + int(local.get("daily_types", 0)),
        baseline[2] + int(local.get("servings", 0)),
    )
    return list(holder), list(candidate), candidate > holder


def _challenge_baseline(candidate_score: Any, day_score: Any) -> list[int]:
    """Recover the fixed pre-day score from one authoritative candidate."""
    candidate = _score_triplet(candidate_score)
    local = day_score if isinstance(day_score, dict) else {}
    return [
        candidate[0],
        candidate[1] - int(local.get("daily_types", 0)),
        candidate[2] - int(local.get("servings", 0)),
    ]


def _competitive_rejection_scores(
    message: str,
) -> tuple[list[int], list[int]] | None:
    match = re.search(
        r"yours\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)\s*"
        r"<=\s*current\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)",
        message,
    )
    if match is None:
        return None
    values = [int(value) for value in match.groups()]
    return values[:3], values[3:]


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

    def cancel_game_sessions(self, game_id: str, *, timeout: float = 5.0) -> list[str]:
        """Stop every local controller before a reset mutates the remote game."""
        cancelled: list[str] = []
        threads: list[threading.Thread] = []
        with self._lock:
            for session_id, session in self._sessions.items():
                if session.get("game_id") != game_id or session.get("state") in {
                    "finished",
                    "cancelled",
                    "failed",
                }:
                    continue
                session.update(
                    state="cancelled",
                    proposal=None,
                    approval=None,
                    error=None,
                    progress={"status": "cancelled"},
                    updated_at=_now(),
                )
                session.setdefault("events", []).append(
                    {"at": session["updated_at"], "status": "cancelled"}
                )
                _write_json(self._session_path(session_id), session)
                event = self._events.get(session_id)
                if event is not None:
                    event.set()
                thread = self._threads.get(session_id)
                if thread is not None and thread.is_alive():
                    threads.append(thread)
                cancelled.append(session_id)
        deadline = time.monotonic() + timeout
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            raise RuntimeError("planner session is still stopping; retry reset")
        return cancelled

    def clear_game_journal(self, game_id: str) -> None:
        digest = hashlib.sha256(game_id.encode()).hexdigest()[:24]
        _write_json(
            self.state_dir / f"{digest}.json",
            {
                "types": None,
                "submitted_days": {},
                "day_snapshots": {},
                "distinct_brands": [],
                "cumulative_daily_types": 0,
                "total_servings": 0,
                "planner_state": None,
            },
        )

    def start_session(
        self,
        game_id: str,
        method: str,
        hyperparameters: dict[str, int | float | bool] | None = None,
        *,
        execution_mode: str = "manual",
        target_day: int | None = None,
        time_limit_seconds: float | None = None,
    ) -> dict[str, Any]:
        if execution_mode not in {"manual", "auto", "curl"}:
            raise ValueError("execution_mode must be manual, auto, or curl")
        if time_limit_seconds is not None:
            if isinstance(time_limit_seconds, bool) or not isinstance(
                time_limit_seconds, (int, float)
            ):
                raise ValueError("time_limit_seconds must be numeric")
            if not 0.1 <= float(time_limit_seconds) <= 600:
                raise ValueError("time_limit_seconds must be between 0.1 and 600")
            time_limit_seconds = float(time_limit_seconds)
        if target_day is not None and (
            isinstance(target_day, bool) or not isinstance(target_day, int) or target_day < 0
        ):
            raise ValueError("target_day must be a non-negative integer")
        if execution_mode == "auto" and target_day is not None:
            raise ValueError("auto mode cannot target an earlier practice day")
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
                "capabilities": {
                    "reset": bool(board.get("is_practice") and not board.get("no_reset")),
                    "submit": True,
                },
            }
        competitive_previous = snapshot.get("competitive_state", {}).get("prev")
        competitive_challenge = bool(
            descriptor.get("is_practice")
            and descriptor.get("no_reset")
            and isinstance(competitive_previous, dict)
            and target_day == int(competitive_previous.get("day", -1))
            and isinstance(competitive_previous.get("holder_score"), list)
        )
        if target_day is not None and not (
            (descriptor.get("is_practice") and not descriptor.get("no_reset"))
            or competitive_challenge
        ):
            raise ValueError(
                "target_day must be a resettable practice day or the current "
                "competitive previous-day challenge"
            )
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
                "execution_mode": execution_mode,
                "target_day": target_day,
                "time_limit_seconds": time_limit_seconds,
                "state": "starting",
                "snapshot": snapshot,
                "proposal": None,
                "last_submission": None,
                "incumbents": [],
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
            if (
                proposal.get("challenge")
                and proposal["challenge"].get("beats_holder") is False
            ):
                raise ValueError(
                    "candidate score does not strictly beat the current holder"
                )
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

    def submit_proposal(
        self,
        session_id: str,
        *,
        fingerprint: str,
        types: list[int] | None = None,
        actions: list[list[int]] | None = None,
        allow_fallback: bool = False,
    ) -> dict[str, Any]:
        """Validate the exact editor value and queue one authoritative submission."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError("session not found")
            proposal = session.get("proposal")
            if session.get("state") not in {
                "awaiting_role_approval",
                "awaiting_plan_approval",
            } or not proposal:
                raise ValueError("session is not waiting for a proposal")
            if fingerprint != proposal.get("fingerprint"):
                raise ValueError("proposal is stale; refresh before submitting")
            if proposal.get("fallback") and not allow_fallback:
                raise ValueError("fallback wait requires explicit allow_fallback")
            if proposal["kind"] == "agent_types":
                candidate = proposal.get("types") if types is None else types
                if not isinstance(candidate, list):
                    raise ValueError("types must be an array")
                try:
                    validate_agent_types(
                        candidate, len(session["snapshot"]["config"]["agents"])
                    )
                except Exception as error:
                    raise ValueError(str(error)) from None
                proposal["types"] = [int(value) for value in candidate]
            else:
                candidate = proposal.get("actions") if actions is None else actions
                if not isinstance(candidate, list):
                    raise ValueError("actions must be an array")
                try:
                    validate_action_shape(
                        candidate,
                        len((proposal.get("day_snapshot") or {}).get("agents", [])),
                    )
                except Exception as error:
                    raise ValueError(str(error)) from None
                if proposal.get("challenge"):
                    check, trace, comparison = self._validate_challenge_actions(
                        session, proposal, candidate
                    )
                    if comparison["beats_holder"] is False:
                        raise ValueError(
                            "candidate score does not strictly beat the current holder"
                        )
                else:
                    day = session.get("snapshot", {}).get("day")
                    if not isinstance(day, dict):
                        day = proposal.get("day_snapshot")
                    config = session.get("snapshot", {}).get("config")
                    if not isinstance(day, dict) or not isinstance(config, dict):
                        raise ValueError("proposal no longer has a day snapshot")
                    _, journal = self._journal(session["game_id"])
                    history = _history_for_planner(journal, int(day["day"]))
                    check = run_core(
                        "check",
                        session["method"],
                        {
                            "config": config,
                            "day_info": day,
                            "history": history,
                            "actions": candidate,
                        },
                        binary=find_binary(self.binary_path),
                    )
                    if not check.get("valid"):
                        raise ValueError(
                            str(check.get("error") or "invalid action plan")
                        )
                    trace = trace_action_plan(
                        config,
                        day,
                        history,
                        candidate,
                        binary_path=self.binary_path,
                    )
                    comparison = None
                edited_actions = candidate != proposal.get("actions")
                proposal.update(
                    actions=candidate,
                    validation=check,
                    trace=trace,
                    fallback=False,
                    planner_error=None,
                )
                if comparison is not None:
                    proposal["challenge"] = comparison
                if edited_actions:
                    proposal["planner_state"] = None
            proposal["fingerprint"] = _fingerprint(
                {key: value for key, value in proposal.items() if key != "fingerprint"}
            )
            session["proposal"] = proposal
            session["approval"] = {
                "fingerprint": proposal["fingerprint"],
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

    def curl_command(
        self,
        session_id: str,
        *,
        fingerprint: str,
        types: list[int] | None = None,
        actions: list[list[int]] | None = None,
    ) -> dict[str, Any]:
        """Return an explicitly requested, executable command. Never persist the token."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not session.get("proposal"):
                raise ValueError("session has no proposal")
            proposal = copy.deepcopy(session["proposal"])
            if fingerprint != proposal.get("fingerprint"):
                raise ValueError("proposal is stale; refresh before generating curl")
            if proposal.get("challenge") and not proposal["challenge"].get(
                "beats_holder"
            ):
                raise ValueError(
                    "candidate score does not strictly beat the current holder"
                )
            if proposal["kind"] == "agent_types":
                candidate = proposal.get("types") if types is None else types
                try:
                    validate_agent_types(
                        candidate, len(session["snapshot"]["config"]["agents"])
                    )
                except Exception as error:
                    raise ValueError(str(error)) from None
                body = {
                    "game_id": proposal["submission_game_id"],
                    "types": candidate,
                }
            else:
                candidate = proposal.get("actions") if actions is None else actions
                if actions is not None:
                    # Reuse the exact manual validation path without approving it.
                    try:
                        validate_action_shape(
                            candidate,
                            len(
                                (proposal.get("day_snapshot") or {}).get(
                                    "agents", []
                                )
                            ),
                        )
                    except Exception as error:
                        raise ValueError(str(error)) from None
                    if proposal.get("challenge"):
                        _, _, comparison = self._validate_challenge_actions(
                            session, proposal, candidate
                        )
                        if not comparison["beats_holder"]:
                            raise ValueError(
                                "candidate score does not strictly beat the current holder"
                            )
                    else:
                        day = session.get("snapshot", {}).get("day")
                        config = session.get("snapshot", {}).get("config")
                        _, journal = self._journal(session["game_id"])
                        check = run_core(
                            "check",
                            session["method"],
                            {
                                "config": config,
                                "day_info": day,
                                "history": _history_for_planner(
                                    journal, int(day["day"])
                                ),
                                "actions": candidate,
                            },
                            binary=find_binary(self.binary_path),
                        )
                        if not check.get("valid"):
                            raise ValueError(
                                str(check.get("error") or "invalid action plan")
                            )
                body = {
                    "game_id": proposal["submission_game_id"],
                    "day": int(proposal["day"]),
                    "actions": candidate,
                }
            token = load_token(self.env_path)
            url = f"{self.base_url.rstrip('/')}{proposal['submission_endpoint']}"
            command = " ".join(
                (
                    "curl --fail-with-body -X POST",
                    shlex.quote(url),
                    "-H",
                    shlex.quote(f"Authorization: Bearer {token}"),
                    "-H",
                    shlex.quote("Content-Type: application/json"),
                    "--data-raw",
                    shlex.quote(json.dumps(body, separators=(",", ":"))),
                )
            )
            return {"command": command, "endpoint": proposal["submission_endpoint"]}

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
            if session.get("state") == "cancelled" and values.get("state") != "cancelled":
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

    def _record_incumbent(
        self,
        session_id: str,
        *,
        day: int,
        score: Any,
        internal_rank: Any,
        elapsed_seconds: float,
    ) -> int | None:
        """Persist a solver improvement as soon as it is found."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.get("state") in {"cancelled", "failed"}:
                return None
            incumbents = session.setdefault("incumbents", [])
            sequence = len(incumbents) + 1
            day_sequence = sum(
                1 for row in incumbents if int(row.get("day", -1)) == day
            ) + 1
            found_at = _now()
            for previous in reversed(incumbents):
                if int(previous.get("day", -1)) != day:
                    continue
                if previous.get("submission_status") == "pending":
                    previous.update(
                        submission_status="superseded",
                        superseded_at=found_at,
                    )
                break
            incumbents.append(
                {
                    "sequence": sequence,
                    "day_sequence": day_sequence,
                    "day": day,
                    "score": copy.deepcopy(score),
                    "internal_rank": copy.deepcopy(internal_rank),
                    "found_at": found_at,
                    "elapsed_seconds": round(max(0.0, elapsed_seconds), 6),
                    "submitted": False,
                    "submission_status": "pending",
                }
            )
            progress = dict(session.get("progress") or {})
            progress.update(
                best_score=copy.deepcopy(score),
                incumbent_count=day_sequence,
            )
            session.update(progress=progress, updated_at=found_at)
            _write_json(self._session_path(session_id), session)
            return sequence

    def _mark_incumbent_submitted(
        self,
        session_id: str,
        sequence: int | None,
        *,
        submitted_at: str,
        submission_count: int,
    ) -> None:
        if sequence is None:
            return
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            row = next(
                (
                    item
                    for item in session.get("incumbents", [])
                    if int(item.get("sequence", -1)) == sequence
                ),
                None,
            )
            if row is None:
                return
            row.update(
                submitted=True,
                submission_status="submitted",
                submitted_at=submitted_at,
                submission_count=submission_count,
            )
            session["updated_at"] = submitted_at
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
        journal.setdefault("cumulative_daily_types", 0)
        journal.setdefault("total_servings", 0)
        journal.setdefault("submitted_days", {})
        journal.setdefault("day_snapshots", {})
        journal.setdefault("competitive_day_baselines", {})
        journal.setdefault("types", None)
        journal.setdefault("planner_state", None)
        return path, journal

    def _sync_actions(
        self,
        client: GameClient,
        resolved_id: str,
        journal: dict[str, Any],
        team_id: str | None = None,
    ) -> None:
        """Import accepted actions, including submissions made by copied curl."""
        try:
            response = client.get("/game/actions", resolved_id)
        except Exception:
            # This read-only synchronization is optional on older hosts.
            return
        team_id = team_id or resolved_id.rsplit(":", 1)[-1]
        for row in response.get("actions", []):
            if str(row.get("team_id")) != team_id or not isinstance(row.get("plan"), list):
                continue
            journal["submitted_days"][str(int(row["day"]))] = row["plan"]

    @staticmethod
    def _accepted_action(
        client: GameClient,
        resolved_id: str,
        day: int,
        team_id: str | None = None,
    ) -> dict[str, Any] | None:
        try:
            response = client.get("/game/actions", resolved_id)
        except Exception:
            return None
        team_id = team_id or resolved_id.rsplit(":", 1)[-1]
        return next(
            (
                row
                for row in response.get("actions", [])
                if str(row.get("team_id")) == team_id
                and int(row.get("day", -1)) == day
            ),
            None,
        )

    @staticmethod
    def _action_marker(row: dict[str, Any] | None) -> tuple[Any, ...] | None:
        if row is None:
            return None
        return (
            row.get("submit_count"),
            row.get("submitted_at"),
            json.dumps(row.get("plan"), sort_keys=True),
        )

    def _historical_day(
        self,
        client: GameClient,
        resolved_id: str,
        config: dict[str, Any],
        target_day: int,
        journal: dict[str, Any],
    ) -> dict[str, Any]:
        replay = client.get("/game/replay", resolved_id)
        replay_day = next(
            (row for row in replay.get("days", []) if int(row.get("day", -1)) == target_day),
            None,
        )
        if replay_day is None:
            raise ValueError(f"replay does not contain day {target_day + 1}")
        team_id = resolved_id.rsplit(":", 1)[-1]
        team = next(
            (row for row in replay_day.get("teams", []) if str(row.get("team_id")) == team_id),
            None,
        )
        if team is None or not team.get("frames"):
            raise ValueError("replay does not contain this team's starting frame")
        frame = team["frames"][0]
        journal["submitted_days"] = {
            key: value
            for key, value in journal.get("submitted_days", {}).items()
            if int(key) < target_day
        }
        journal["day_snapshots"] = {
            key: value
            for key, value in journal.get("day_snapshots", {}).items()
            if int(key) < target_day
        }
        journal["planner_state"] = None
        brands: set[int] = set()
        spot_brand = {int(spot["pos"]): int(spot["brand"]) for spot in config.get("spots", [])}
        for prior in replay.get("days", []):
            if int(prior.get("day", -1)) >= target_day:
                continue
            own = next(
                (row for row in prior.get("teams", []) if str(row.get("team_id")) == team_id),
                None,
            )
            for prior_frame in (own or {}).get("frames", []):
                for position in prior_frame.get("collected", []):
                    if int(position) in spot_brand:
                        brands.add(spot_brand[int(position)])
        journal["distinct_brands"] = sorted(brands)
        agents = [
            {
                "kind": 1 if agent.get("type") == "refuel" else 0,
                "pos": int(agent["cell"]),
                "fuel": agent.get("fuel"),
            }
            for agent in frame.get("agents", [])
        ]
        timed_max = int(
            params.get(
                "max_iterations",
                MLNS_ANYTIME_ITERATION_CEILING
                if session["method"] in {"lns", "alns", "mlns", "simple_lns", "lns_dp"}
                else 2048,
            )
        )
        timed_min = int(
            params.get(
                "min_iterations",
                STATEFUL_DEFAULTS[session["method"]]["min_iterations"]
                if session["method"] in STATEFUL_POLICIES
                else 32,
            )
        )
        if session["method"] == "lns_dp" and timed_max == 16:
            timed_max = MLNS_ANYTIME_ITERATION_CEILING
            timed_min = 0
        return {
            "day": target_day,
            "steps": int(replay_day.get("steps", config["daySteps"][target_day])),
            "agents": agents,
            "others": [],
            "traffics": [
                {"pos": int(pos), "status": int(value)}
                for pos, value in replay_day.get("road_condition", {}).items()
            ],
            "endsAt": None,
        }

    @staticmethod
    def _decorate_proposal(
        proposal: dict[str, Any], endpoint: str, game_id: str
    ) -> dict[str, Any]:
        proposal["submission_endpoint"] = endpoint
        proposal["submission_game_id"] = game_id
        proposal["fingerprint"] = _fingerprint(
            {key: value for key, value in proposal.items() if key != "fingerprint"}
        )
        return proposal

    @staticmethod
    def _challenge_planner_inputs(
        session: dict[str, Any], day: dict[str, Any], holder_score: Any
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Build a single-day planner view for an unbounded competitive round."""
        config = copy.deepcopy(session["snapshot"]["config"])
        budget = float(session.get("time_limit_seconds") or 60.0)
        config["daySteps"] = [int(day["steps"])]
        config["daySeconds"] = [budget]
        planner_day = {**copy.deepcopy(day), "day": 0}
        holder = _score_triplet(holder_score)
        all_brands = sorted({int(spot["brand"]) for spot in config.get("spots", [])})
        history = {
            "distinct_brands": all_brands if holder[0] >= len(all_brands) else [],
            "cumulative_daily_types": holder[1],
            "total_servings": holder[2],
            "submitted_actions": [],
        }
        return config, planner_day, history

    def _build_challenge_proposal(
        self,
        session: dict[str, Any],
        previous: dict[str, Any],
        api_game_id: str,
        baseline_score: list[int] | None,
    ) -> dict[str, Any]:
        day = normalize_competitive_day(previous)
        holder_score = previous.get("holder_score")
        config, planner_day, history = self._challenge_planner_inputs(
            session, day, baseline_score or holder_score
        )
        journal = {
            "distinct_brands": history["distinct_brands"],
            "cumulative_daily_types": history["cumulative_daily_types"],
            "total_servings": history["total_servings"],
            "submitted_days": {},
            "planner_state": None,
        }
        proposal = self._build_proposal(session, config, planner_day, journal)
        holder, candidate, beats_holder = _challenge_score(
            holder_score,
            (proposal.get("trace") or {}).get("score"),
            baseline_score,
        )
        proposal.update(
            day=int(day["day"]),
            day_snapshot=day,
            challenge={
                "owner": str(previous.get("owner", "")),
                "holder_score": holder,
                "baseline_score": baseline_score,
                "candidate_score": candidate,
                "day_score": (proposal.get("trace") or {}).get("score", {}),
                "beats_holder": beats_holder,
            },
        )
        return self._decorate_proposal(
            proposal, "/game/competitive/actions", api_game_id
        )

    def _resolve_challenge_baseline(
        self,
        client: GameClient,
        session: dict[str, Any],
        previous: dict[str, Any],
        resolved_id: str,
        journal: dict[str, Any],
        own_team_id: str | None,
    ) -> list[int] | None:
        """Find the fixed score immediately before this competitive day.

        The server does not expose that score directly.  A locally cached
        authoritative result is preferred.  If we currently hold the day, our
        accepted action is observable and its day gain can be subtracted from
        the holder score.
        """
        day_index = int(previous["day"])
        cached = journal.get("competitive_day_baselines", {}).get(str(day_index))
        try:
            return list(_score_triplet(cached))
        except (TypeError, ValueError):
            pass
        if own_team_id is None or str(previous.get("owner", "")) != own_team_id:
            return None
        accepted = self._accepted_action(
            client, resolved_id, day_index, own_team_id
        )
        actions = accepted.get("plan") if isinstance(accepted, dict) else None
        if not isinstance(actions, list):
            return None
        try:
            day = normalize_competitive_day(previous)
            config, planner_day, history = self._challenge_planner_inputs(
                session, day, previous.get("holder_score")
            )
            trace = trace_action_plan(
                config,
                planner_day,
                history,
                actions,
                binary_path=self.binary_path,
            )
            baseline = _challenge_baseline(
                previous.get("holder_score"), trace.get("score")
            )
        except Exception:
            return None
        journal.setdefault("competitive_day_baselines", {})[
            str(day_index)
        ] = baseline
        return baseline

    def _validate_challenge_actions(
        self,
        session: dict[str, Any],
        proposal: dict[str, Any],
        actions: list[list[int]],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        challenge = proposal.get("challenge")
        day = proposal.get("day_snapshot")
        if not isinstance(challenge, dict) or not isinstance(day, dict):
            raise ValueError("competitive challenge context is missing")
        config, planner_day, history = self._challenge_planner_inputs(
            session,
            day,
            challenge.get("baseline_score") or challenge.get("holder_score"),
        )
        check = run_core(
            "check",
            session["method"],
            {
                "config": config,
                "day_info": planner_day,
                "history": history,
                "actions": actions,
            },
            binary=find_binary(self.binary_path),
        )
        if not check.get("valid"):
            raise ValueError(str(check.get("error") or "invalid action plan"))
        trace = trace_action_plan(
            config,
            planner_day,
            history,
            actions,
            binary_path=self.binary_path,
        )
        holder, candidate, beats_holder = _challenge_score(
            challenge.get("holder_score"),
            trace.get("score"),
            challenge.get("baseline_score"),
        )
        comparison = {
            **challenge,
            "holder_score": holder,
            "candidate_score": candidate,
            "day_score": trace.get("score", {}),
            "beats_holder": beats_holder,
        }
        return check, trace, comparison

    def _plan_search(self, session: dict[str, Any], config: dict[str, Any], day: dict[str, Any]) -> dict[str, int]:
        params = session.get("hyperparameters", {})
        budget = planning_budget(
            config,
            day,
            is_practice=bool(session.get("game", {}).get("is_practice")),
            deadline_margin=2.0,
        )
        alns_iterations = params.get("alns_iterations")
        safe_time_limit_ms = solver_time_limit_ms(session["method"], budget)
        timed_max = int(
            params.get(
                "max_iterations",
                MLNS_ANYTIME_ITERATION_CEILING
                if session["method"]
                in {"lns", "alns", "mlns", "simple_lns", "lns_dp"}
                else 2048,
            )
        )
        timed_min = int(
            params.get(
                "min_iterations",
                STATEFUL_DEFAULTS[session["method"]]["min_iterations"]
                if session["method"] in STATEFUL_POLICIES
                else 32,
            )
        )
        if session["method"] == "lns_dp" and timed_max == 16:
            timed_max = MLNS_ANYTIME_ITERATION_CEILING
            timed_min = 0
        if session["method"] == "palns":
            total_iterations = int(params.get("total_iterations", 1_536))
            search = {"totalIterations": total_iterations}
            if "time_limit_ms" in params:
                search["timeLimitMs"] = min(
                    int(params["time_limit_ms"]), safe_time_limit_ms
                )
            if "exact_nodes" in params:
                search["exactNodes"] = int(params["exact_nodes"])
            if "final_exact_nodes" in params:
                search["finalExactNodes"] = int(params["final_exact_nodes"])
            if "exact_time_percent" in params:
                search["exactTimePercent"] = int(params["exact_time_percent"])
            return search
        if alns_iterations is not None:
            alns_iterations = int(alns_iterations)
            return {
                "minIterations": int(params.get("min_iterations", min(2048, alns_iterations))),
                "maxIterations": alns_iterations,
                "stagnationIterations": int(params.get("stagnation_iterations", alns_iterations)),
            }
        return {
            "timeLimitMs": min(
                int(params.get("time_limit_ms", safe_time_limit_ms)),
                safe_time_limit_ms,
            ),
            "minIterations": timed_min,
            "maxIterations": timed_max,
            "stagnationIterations": int(
                params.get(
                    "stagnation_iterations",
                    STATEFUL_DEFAULTS[session["method"]]["stagnation_iterations"]
                    if session["method"] in STATEFUL_POLICIES
                    else 0
                    if session["method"] in {"lns", "alns"}
                    else 96,
                )
            ),
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
        history = _history_for_planner(journal, int(day["day"]))
        payload = {
            "config": config,
            "day_info": day,
            "history": history,
            "types": types,
            "search": self._plan_search(session, config, day),
            "hyperparameters": {
                key: value
                for key, value in (
                    {
                        **(
                            STATEFUL_DEFAULTS[session["method"]]
                            if session["method"] in STATEFUL_POLICIES
                            else {}
                        ),
                        **session.get("hyperparameters", {}),
                    }
                ).items()
                if key not in {"alns_iterations", "total_iterations"}
            },
        }
        saved_state = journal.get("planner_state")
        if (
            session["method"] in STATEFUL_POLICIES
            and isinstance(saved_state, dict)
            and saved_state.get("method") == session["method"]
        ):
            payload["planner_state"] = saved_state.get("state")
        if session["method"] in STATEFUL_POLICIES:
            payload["include_planner_state"] = True
        fallback = False
        planner_error: str | None = None
        planner_state: dict[str, Any] | None = None
        try:
            planned = run_core(
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
            if session["method"] in STATEFUL_POLICIES:
                if not isinstance(planned, dict) or "actions" not in planned:
                    raise RuntimeError(
                        f"{session['method']} planner did not return a state envelope"
                    )
                actions = planned["actions"]
                planner_state = planned.get("planner_state")
            else:
                actions = planned
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
            "day_snapshot": copy.deepcopy(day),
            "actions": actions,
            "validation": check,
            "trace": trace,
            "fallback": fallback,
            "planner_error": planner_error,
            "planner_state": planner_state if not fallback else None,
            "deadline": day.get("endsAt"),
            "created_at": _now(),
        }
        proposal["fingerprint"] = _fingerprint(proposal)
        return proposal

    def _stream_day(
        self,
        session_id: str,
        client: GameClient,
        session: dict[str, Any],
        config: dict[str, Any],
        day: dict[str, Any],
        journal: dict[str, Any],
        state_path: Path,
        api_game_id: str,
        competitive_practice: bool,
    ) -> None:
        """Search this day to its budget, resubmitting every improving plan.

        The competition scores the last valid submission, so the ALNS core
        streams each new best and we resubmit it. A short ``time_limit_seconds``
        overrides the day budget for fast local iteration; otherwise the real
        per-day ``daySeconds`` (clamped to any live deadline) is used.
        """
        day_index = int(day["day"])
        key = str(day_index)
        types = [int(agent["kind"]) for agent in day["agents"]]
        history = _history_for_planner(journal, day_index)
        is_practice = bool(session.get("game", {}).get("is_practice"))
        override = session.get("time_limit_seconds")
        base_budget = planning_budget(
            config, day, is_practice=is_practice, deadline_margin=2.0
        )
        if override is None:
            budget = base_budget
        elif is_practice:
            budget = float(override)
        else:
            # Never let a fast-iteration override outrun a real match deadline.
            budget = min(float(override), base_budget)
        budget = max(0.2, budget)
        endpoint = (
            "/game/competitive/actions"
            if competitive_practice
            else (
                "/game/practice/actions" if is_practice else "/game/actions"
            )
        )
        binary = find_binary(self.binary_path)

        stream_state: dict[str, Any] = {
            "count": 0,
            "incumbent_count": 0,
            "best": None,
            "best_score": None,
            "pending": None,
            "last_submit": 0.0,
        }
        debounce = 0.25
        previous_incumbents = (self.get_session(session_id) or {}).get(
            "incumbents", []
        )
        elapsed_offset = max(
            (
                float(row.get("elapsed_seconds", 0.0))
                for row in previous_incumbents
                if int(row.get("day", -1)) == day_index
            ),
            default=0.0,
        )
        search_started = time.monotonic()

        def submit(
            actions: list[list[int]], score: Any, incumbent_sequence: int | None = None
        ) -> None:
            response = client.post(
                endpoint,
                {"game_id": api_game_id, "day": day_index, "actions": actions},
            )
            stream_state["count"] += 1
            stream_state["last_submit"] = time.monotonic()
            stream_state["best"] = actions
            stream_state["best_score"] = score
            stream_state["pending"] = None
            journal["submitted_days"][key] = actions
            _write_json(state_path, journal)
            submitted_at = _now()
            self._mark_incumbent_submitted(
                session_id,
                incumbent_sequence,
                submitted_at=submitted_at,
                submission_count=stream_state["count"],
            )
            self._update(
                session_id,
                state="streaming",
                last_submission={
                    "day": day_index,
                    "endpoint": endpoint,
                    "response": response,
                    "submitted_at": submitted_at,
                },
                progress={
                    "status": "streaming",
                    "day": day_index,
                    "best_score": score,
                    "submission_count": stream_state["count"],
                    "incumbent_count": stream_state["incumbent_count"],
                    "budget_seconds": budget,
                },
            )

        def on_improve(record: dict[str, Any]) -> None:
            actions = record.get("actions")
            score = record.get("score")
            internal_rank = record.get("internal_rank")
            if not isinstance(actions, list):
                return
            try:
                validate_action_shape(actions, len(types))
            except Exception:
                return
            stream_state["incumbent_count"] += 1
            incumbent_sequence = self._record_incumbent(
                session_id,
                day=day_index,
                score=score,
                internal_rank=internal_rank,
                elapsed_seconds=(
                    elapsed_offset + time.monotonic() - search_started
                ),
            )
            if time.monotonic() - stream_state["last_submit"] >= debounce:
                submit(actions, score, incumbent_sequence)
            else:
                # Coalesce a burst of rapid improvements; the final flush below
                # guarantees the best is submitted before the day advances.
                stream_state["pending"] = (actions, score, incumbent_sequence)
                stream_state["best"] = actions
                stream_state["best_score"] = score

        def should_stop() -> bool:
            current = self.get_session(session_id)
            return (
                current is None
                or current.get("state") in {"cancelled", "failed"}
                or bool(current.get("paused"))
            )

        self._update(
            session_id,
            state="streaming",
            proposal=None,
            approval=None,
            progress={
                "status": "streaming",
                "day": day_index,
                "budget_seconds": budget,
                "submission_count": 0,
                "incumbent_count": 0,
            },
        )
        payload = {
            "config": config,
            "day_info": day,
            "history": history,
            "types": types,
            "search": {
                "timeLimitMs": max(50, int(budget * 1000)),
                "minIterations": 1,
                "maxIterations": 10_000_000,
                "stagnationIterations": 0,
            },
            # Production defaults are percentage-driven. Supplying values here
            # is the retained manual mode for controlled tuning; the separate
            # Web time field remains authoritative for the total day budget.
            "hyperparameters": {
                key: value
                for key, value in session.get("hyperparameters", {}).items()
                if key != "time_limit_ms"
            },
        }
        saved_state = journal.get("planner_state")
        if (
            session["method"] in STATEFUL_POLICIES
            and isinstance(saved_state, dict)
            and saved_state.get("method") == session["method"]
        ):
            payload["planner_state"] = saved_state.get("state")
        final_record: dict[str, Any] | None = None
        try:
            final_record = stream_core(
                session["method"],
                payload,
                binary=binary,
                on_improve=on_improve,
                timeout=budget + 15,
                should_stop=should_stop,
            )
        except Exception as error:  # A safe wait plan keeps the day valid.
            if stream_state["best"] is None:
                fallback = [[-int(config["daySteps"][day_index])] for _ in types]
                try:
                    submit(fallback, [0, 0, 0])
                except Exception:
                    self._update(
                        session_id,
                        state="failed",
                        error=str(error),
                        progress={"status": "failed"},
                    )
                    return
        # Ensure the best plan is the last valid submission for this day.
        if stream_state["pending"] is not None:
            submit(*stream_state["pending"])
        if isinstance(final_record, dict) and final_record.get("kind") == "final":
            final_actions = final_record.get("actions")
            if isinstance(final_actions, list):
                validate_action_shape(final_actions, len(types))
                if journal["submitted_days"].get(key) != final_actions:
                    submit(final_actions, final_record.get("score"))
            if (
                session["method"] in STATEFUL_POLICIES
                and journal["submitted_days"].get(key) == final_actions
                and isinstance(final_record.get("planner_state"), dict)
            ):
                journal["planner_state"] = {
                    "method": session["method"],
                    "state": final_record["planner_state"],
                }
                _write_json(state_path, journal)

        current = self.get_session(session_id)
        if current is None or current.get("state") in {"cancelled", "failed"}:
            return
        if current.get("paused"):
            # Resume re-streams this day; the partial best already stands.
            self._update(
                session_id,
                progress={
                    "status": "stream_paused",
                    "day": day_index,
                    "submission_count": stream_state["count"],
                    "incumbent_count": stream_state["incumbent_count"],
                    "budget_seconds": budget,
                },
            )
            return
        if stream_state["best"] is None:
            fallback = [[-int(config["daySteps"][day_index])] for _ in types]
            submit(fallback, [0, 0, 0])

        final_actions = journal["submitted_days"].get(key)
        if final_actions is not None:
            trace = trace_action_plan(
                config, day, history, final_actions, binary_path=self.binary_path
            )
            journal["day_snapshots"][key] = {
                "day_info": day,
                "actions": final_actions,
                "trace": trace,
                "submitted_at": _now(),
            }
            collected_positions = {
                int(position)
                for frame in trace.get("frames", [])
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
            scores = [
                (snapshot.get("trace") or {}).get("score") or {}
                for snapshot in journal.get("day_snapshots", {}).values()
            ]
            journal["cumulative_daily_types"] = sum(
                int(score.get("daily_types", 0)) for score in scores
            )
            journal["total_servings"] = sum(
                int(score.get("servings", 0)) for score in scores
            )
            _write_json(state_path, journal)

        self._update(
            session_id,
            state="waiting_for_day",
            last_streamed_day=day_index,
            proposal=None,
            approval=None,
            progress={
                "status": "submitted",
                "day": day_index,
                "best_score": stream_state["best_score"],
                "submission_count": stream_state["count"],
                "incumbent_count": stream_state["incumbent_count"],
                "budget_seconds": budget,
            },
        )

    def _run(self, session_id: str) -> None:
        client: GameClient | None = None
        try:
            session = self.get_session(session_id)
            if session is None:
                return
            token = load_token(self.env_path)
            own_team_id = _token_team_id(token)
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
            self._sync_actions(client, resolved_id, journal, own_team_id)
            _write_json(state_path, journal)
            if session.get("target_day") is not None and competitive_practice:
                target_day = int(session["target_day"])
                competitive_state = client.get(
                    "/game/competitive/state", api_game_id
                )
                previous = competitive_state.get("prev")
                if not isinstance(previous, dict) or int(
                    previous.get("day", -1)
                ) != target_day:
                    raise ValueError(
                        "the selected previous-day challenge is no longer available"
                    )
                baseline_score = self._resolve_challenge_baseline(
                    client,
                    session,
                    previous,
                    resolved_id,
                    journal,
                    own_team_id,
                )
                _write_json(state_path, journal)
                proposal = self._build_challenge_proposal(
                    session, previous, api_game_id, baseline_score
                )
                challenge_snapshot = {
                    **session.get("snapshot", {}),
                    "state": {"status": "in_progress", "day": target_day},
                    "day": proposal["day_snapshot"],
                    "competitive_state": competitive_state,
                    "previous_day": proposal["day_snapshot"],
                    "fetched_at": _now(),
                }
                self._update(
                    session_id,
                    state="awaiting_plan_approval",
                    snapshot=challenge_snapshot,
                    proposal=proposal,
                    error=None,
                    progress={
                        "status": "challenge_ready",
                        "day": target_day,
                        "best_score": proposal["challenge"]["candidate_score"],
                    },
                )
                while not self._closed:
                    current = self.get_session(session_id)
                    if current is None or current.get("state") in {
                        "cancelled",
                        "failed",
                    }:
                        return
                    if current.get("approval"):
                        active = current["proposal"]
                        challenge = active.get("challenge", {})
                        if challenge.get("beats_holder") is False:
                            raise RuntimeError(
                                "candidate score does not strictly beat the current holder"
                            )
                        fresh = client.get(
                            "/game/competitive/state", api_game_id
                        ).get("prev")
                        if not isinstance(fresh, dict) or (
                            int(fresh.get("day", -1)) != target_day
                            or str(fresh.get("owner", ""))
                            != str(challenge.get("owner", ""))
                            or list(fresh.get("holder_score", []))
                            != list(challenge.get("holder_score", []))
                        ):
                            raise RuntimeError(
                                "previous-day holder changed; run the comparison again"
                            )
                        try:
                            response = client.post(
                                "/game/competitive/actions",
                                {
                                    "game_id": api_game_id,
                                    "day": target_day,
                                    "actions": active["actions"],
                                },
                            )
                        except RuntimeError as error:
                            scores = _competitive_rejection_scores(str(error))
                            if scores is None:
                                raise
                            candidate_score, holder_score = scores
                            day_score = (active.get("trace") or {}).get(
                                "score", {}
                            )
                            baseline_score = _challenge_baseline(
                                candidate_score, day_score
                            )
                            journal.setdefault("competitive_day_baselines", {})[
                                str(target_day)
                            ] = baseline_score
                            _write_json(state_path, journal)
                            challenge.update(
                                holder_score=holder_score,
                                baseline_score=baseline_score,
                                candidate_score=candidate_score,
                                day_score=day_score,
                                beats_holder=False,
                            )
                            active["challenge"] = challenge
                            active["fingerprint"] = _fingerprint(
                                {
                                    key: value
                                    for key, value in active.items()
                                    if key != "fingerprint"
                                }
                            )
                            self._update(
                                session_id,
                                state="awaiting_plan_approval",
                                proposal=active,
                                approval=None,
                                error=str(error),
                                progress={
                                    "status": "challenge_not_better",
                                    "day": target_day,
                                    "best_score": candidate_score,
                                },
                            )
                            continue
                        submitted_at = _now()
                        day_score = (active.get("trace") or {}).get("score", {})
                        authoritative_score = response.get("score")
                        try:
                            baseline_score = _challenge_baseline(
                                authoritative_score, day_score
                            )
                        except (TypeError, ValueError):
                            baseline_score = challenge.get("baseline_score")
                        if baseline_score is not None:
                            journal.setdefault("competitive_day_baselines", {})[
                                str(target_day)
                            ] = baseline_score
                            _write_json(state_path, journal)
                        self._update(
                            session_id,
                            state="finished",
                            target_day=None,
                            proposal=None,
                            approval=None,
                            error=None,
                            result=response,
                            last_submission={
                                "day": target_day,
                                "endpoint": "/game/competitive/actions",
                                "response": response,
                                "day_score": day_score,
                                "baseline_score": baseline_score,
                                "submitted_at": submitted_at,
                            },
                            progress={
                                "status": "challenge_submitted",
                                "day": target_day,
                                "best_score": authoritative_score
                                or challenge.get("candidate_score"),
                            },
                        )
                        return
                    self._wait(session_id, 1.0)
            if session.get("target_day") is not None:
                target_day = int(session["target_day"])
                if target_day >= len(config.get("daySteps", [])):
                    raise ValueError("target_day is outside the configured match")
                historical_day = self._historical_day(
                    client, resolved_id, config, target_day, journal
                )
                _write_json(state_path, journal)
                historical_snapshot = {
                    **session.get("snapshot", {}),
                    "config": config,
                    "day": historical_day,
                    "state": {"status": "in_progress", "day": target_day},
                    "fetched_at": _now(),
                }
                proposal = self._decorate_proposal(
                    self._build_proposal(session, config, historical_day, journal),
                    "/game/practice/actions",
                    resolved_id,
                )
                self._update(
                    session_id,
                    state="awaiting_plan_approval",
                    snapshot=historical_snapshot,
                    proposal=proposal,
                    progress={"status": "awaiting_plan_approval", "day": target_day},
                )
                baseline_action = self._action_marker(
                    self._accepted_action(client, resolved_id, target_day)
                )
                while not self._closed:
                    current = self.get_session(session_id)
                    if current is None or current.get("state") in {"cancelled", "failed"}:
                        return
                    if current.get("approval"):
                        active = current["proposal"]
                        response = client.post(
                            "/game/practice/actions",
                            {
                                "game_id": resolved_id,
                                "day": target_day,
                                "actions": active["actions"],
                            },
                        )
                        journal["submitted_days"][str(target_day)] = active["actions"]
                        if (
                            current["method"] in STATEFUL_POLICIES
                            and isinstance(active.get("planner_state"), dict)
                        ):
                            journal["planner_state"] = {
                                "method": current["method"],
                                "state": active["planner_state"],
                            }
                        else:
                            journal["planner_state"] = None
                        journal["day_snapshots"][str(target_day)] = {
                            "day_info": historical_day,
                            "actions": active["actions"],
                            "validation": active["validation"],
                            "trace": active.get("trace"),
                            "submitted_at": _now(),
                        }
                        _write_json(state_path, journal)
                        self._update(
                            session_id,
                            state="submitted",
                            target_day=None,
                            proposal=None,
                            approval=None,
                            last_submission={
                                "day": target_day,
                                "endpoint": "/game/practice/actions",
                                "response": response,
                                "submitted_at": _now(),
                            },
                            progress={"status": "submitted", "day": target_day},
                        )
                        break
                    if current.get("execution_mode") == "curl":
                        accepted = self._accepted_action(
                            client, resolved_id, target_day
                        )
                        marker = self._action_marker(accepted)
                        if marker is not None and marker != baseline_action:
                            journal["submitted_days"][str(target_day)] = accepted["plan"]
                            _write_json(state_path, journal)
                            self._update(
                                session_id,
                                state="submitted",
                                target_day=None,
                                proposal=None,
                                approval=None,
                                last_submission={
                                    "day": target_day,
                                    "endpoint": "/game/practice/actions",
                                    "external": True,
                                    "submitted_at": _now(),
                                },
                                progress={
                                    "status": "external_submission_detected",
                                    "day": target_day,
                                },
                            )
                            break
                    self._wait(session_id, 1.0)
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
                    previous_snapshot = current.get("snapshot", {})
                    previous_day = previous_snapshot.get("day")
                    previous_state = previous_snapshot.get("state", {})
                    if (
                        isinstance(previous_day, dict)
                        and isinstance(previous_state, dict)
                        and int(previous_day.get("day", -1))
                        == int(status.get("day", -2))
                        == int(previous_state.get("day", -3))
                    ):
                        day = previous_day
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
                    previous = competitive_state.get("prev")
                    if isinstance(previous, dict):
                        current_snapshot["previous_day"] = (
                            normalize_competitive_day(previous)
                        )
                self._update(session_id, snapshot=current_snapshot)
                self._sync_actions(client, resolved_id, journal, own_team_id)
                _write_json(state_path, journal)
                if status.get("status") == "reset_incomplete":
                    raise RuntimeError(str(status.get("error")))
                previous_challenge = (
                    competitive_state.get("prev")
                    if isinstance(competitive_state, dict)
                    else None
                )
                if (
                    competitive_practice
                    and isinstance(previous_challenge, dict)
                    and isinstance(previous_challenge.get("holder_score"), list)
                ):
                    self._update(
                        session_id,
                        state="interrupted",
                        error=(
                            "this competitive-practice match requires a "
                            "previewed previous-day challenge"
                        ),
                        progress={"status": "challenge_required"},
                    )
                    return
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
                                "cumulative_daily_types": 0,
                                "total_servings": 0,
                                "planner_state": None,
                            }
                            _write_json(state_path, journal)
                    if journal.get("types"):
                        # Roles are chosen once per match and already submitted;
                        # wait for the first day to open.
                        self._update(
                            session_id,
                            state="waiting_for_day",
                            progress={"status": "roles_submitted"},
                        )
                        self._wait(session_id, self.poll_interval)
                        continue
                    type_payload = (
                        {
                            "config": config,
                            "hyperparameters": current.get("hyperparameters", {}),
                        }
                        if current["method"] in {"simple_lns", "lns_dp"}
                        else config
                    )
                    types = run_core(
                        "types",
                        current["method"],
                        type_payload,
                        binary=find_binary(self.binary_path),
                    )
                    validate_agent_types(types, len(config["agents"]))
                    client.post(
                        "/game/agent-types",
                        {"game_id": api_game_id, "types": types},
                    )
                    journal["types"] = types
                    _write_json(state_path, journal)
                    self._update(
                        session_id,
                        state="waiting_for_day",
                        progress={"status": "roles_submitted"},
                    )
                    self._wait(session_id, self.poll_interval)
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
                if current.get("last_streamed_day") == day_index:
                    # This day's search budget is already spent. The last valid
                    # submission stands; wait for the server to open the next day.
                    self._update(session_id, state="waiting_for_day", progress={"status": "submitted", "day": day_index})
                    self._wait(session_id, self.poll_interval)
                    continue
                if day is None:
                    day = client.get("/game/day", resolved_id)
                    current_snapshot["day"] = day
                    self._update(session_id, snapshot=current_snapshot)
                assert day is not None
                self._stream_day(
                    session_id,
                    client,
                    current,
                    config,
                    day,
                    journal,
                    state_path,
                    api_game_id,
                    competitive_practice,
                )
                self._wait(session_id, self.poll_interval)
        except Exception as error:
            self._update(session_id, state="failed", error=str(error), progress={"status": "failed"})
        finally:
            if client is not None:
                client.close()
