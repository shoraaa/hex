from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hexbench import competition


def test_no_reset_practice_competition_requires_day_approval(
    monkeypatch, tmp_path: Path
) -> None:
    config = {
        "startsAt": time.time() - 10,
        "daySeconds": [60],
        "daySteps": [4],
        "map": {"height": 2, "width": 2, "cells": [[0, 0], [0, 0]]},
        "spots": [{"brand": 0, "pos": 1, "stocks": 1}],
        "agents": [0],
        "fuelLimits": 10,
        "players": 1,
        "busyThreshold": 2,
        "jammedThreshold": 4,
    }
    day = {
        # Practice endpoints remain usable after their display deadline. The
        # supervised controller must not treat this as a live-match cutoff.
        "endsAt": time.time() - 60,
        "day": 0,
        "agents": [{"kind": 0, "pos": 0, "fuel": 10}],
        "others": [],
        "traffics": [],
    }
    descriptor = {
        "question_id": "practice-comp",
        "name": "Practice competition",
        "mode": "competition",
        "competition_kind": "practice_competition",
        "is_practice": True,
        "no_reset": True,
        "total_days": 1,
        "capabilities": {"reset": False, "submit": True},
    }

    class FakeClient:
        posts: list[tuple[str, dict]] = []
        finished = False

        def __init__(self, token: str, base_url: str):
            pass

        def get(self, path: str, game_id: str):
            if path == "/game/config":
                return config
            if path == "/game/competitive/state":
                if self.finished:
                    return {
                        "selecting": False,
                        "open": None,
                        "standings": {
                            "ranking": ["13"],
                            "timeline": {"distinct_types": 1},
                        },
                    }
                return {
                    "selecting": False,
                    "open": {
                        "day": 0,
                        "steps": 4,
                        "agents": [{"kind": 0, "pos": 0, "fuel": 10}],
                        "road_condition": {},
                    },
                    "standings": {
                        "ranking": ["13"],
                        "timeline": {"distinct_types": 0},
                    },
                }
            raise AssertionError(path)

        def post(self, path: str, payload: dict):
            self.posts.append((path, payload))
            assert path != "/game/practice/reset"
            self.finished = True
            return {"accepted": True}

        def close(self) -> None:
            pass

    monkeypatch.setattr(competition, "load_token", lambda path: "token")
    monkeypatch.setattr(
        competition, "discover_assigned_games", lambda *args: [descriptor]
    )
    monkeypatch.setattr(
        competition,
        "fetch_game_snapshot",
        lambda *args: {
            "requested_game_id": "practice-comp",
            "game_id": "practice-comp:13",
            "board": {
                "game_id": "practice-comp:13",
                "is_practice": True,
                "no_reset": True,
            },
            "config": config,
            "state": {"status": "in_progress", "day": 0},
            "day": day,
        },
    )
    monkeypatch.setattr(competition, "GameClient", FakeClient)
    monkeypatch.setattr(competition, "find_binary", lambda path: tmp_path / "hexudon")

    def fake_core(command, method, payload, **kwargs):
        if command == "plan":
            assert payload["search"]["timeLimitMs"] == 49_300
            assert kwargs["timeout"] >= 58
            return [[-4]]
        if command == "check":
            return {
                "valid": True,
                "error": None,
                "score": {"distinct_types": 0, "daily_types": 0, "servings": 0},
            }
        raise AssertionError(command)

    monkeypatch.setattr(competition, "run_core", fake_core)
    monkeypatch.setattr(
        competition,
        "trace_action_plan",
        lambda *args, **kwargs: {
            "valid": True,
            "score": {"distinct_types": 0, "daily_types": 0, "servings": 0},
            "frames": [
                {"step": 0, "agents": [{"cell": 0, "fuel": 10, "type": 0}], "collected": []},
                {"step": 4, "agents": [{"cell": 0, "fuel": 10, "type": 0}], "collected": []},
            ],
        },
    )

    manager = competition.CompetitionSessionManager(
        tmp_path / ".env", tmp_path / "state", tmp_path / "reports", poll_interval=0.01
    )
    try:
        session = manager.start_session("practice-comp", "greedy")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            current = manager.get_session(session["id"])
            assert current is not None
            if current["state"] == "awaiting_plan_approval":
                break
            time.sleep(0.01)
        assert current["state"] == "awaiting_plan_approval"
        assert FakeClient.posts == []

        manager.approve(
            session["id"], fingerprint=current["proposal"]["fingerprint"]
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not FakeClient.posts:
            time.sleep(0.01)
        assert FakeClient.posts[0][0] == "/game/competitive/actions"
        assert all(path != "/game/practice/reset" for path, _ in FakeClient.posts)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            current = manager.get_session(session["id"])
            assert current is not None
            if current["state"] == "finished":
                break
            time.sleep(0.01)
        assert current["state"] == "finished"
        assert current["result"]["standings"]["timeline"]["distinct_types"] == 1
        waiting_events = [
            event
            for event in current["events"]
            if event.get("status") == "waiting_for_result"
        ]
        assert len(waiting_events) <= 1
    finally:
        manager.close()


def test_waiting_for_result_resumes_safely_after_restart(
    monkeypatch, tmp_path: Path
) -> None:
    config = {
        "daySteps": [4],
        "daySeconds": [60],
        "agents": [0],
    }
    session = {
        "id": "recover-result",
        "game_id": "practice-comp:13",
        "requested_game_id": "practice-comp",
        "game": {"is_practice": True},
        "method": "greedy",
        "state": "waiting_for_result",
        "snapshot": {"board": {}},
        "events": [
            {"at": "1", "status": "waiting_for_result"},
            {"at": "2", "status": "waiting_for_result"},
        ],
    }
    report_dir = tmp_path / "reports"
    session_path = report_dir / "sessions" / session["id"] / "session.json"
    session_path.parent.mkdir(parents=True)
    session_path.write_text(json.dumps(session))

    class FakeClient:
        def __init__(self, token: str, base_url: str):
            pass

        def get(self, path: str, game_id: str):
            if path == "/game/config":
                return config
            if path == "/game/state":
                return {"status": "in_progress", "day": 1}
            if path == "/game/practice/score":
                return {"detail": {"13": {"total_servings": 7}}}
            raise AssertionError(path)

        def close(self) -> None:
            pass

    monkeypatch.setattr(competition, "load_token", lambda path: "token")
    monkeypatch.setattr(competition, "GameClient", FakeClient)
    manager = competition.CompetitionSessionManager(
        tmp_path / ".env", tmp_path / "state", report_dir, poll_interval=0.01
    )
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            current = manager.get_session(session["id"])
            assert current is not None
            if current["state"] == "finished":
                break
            time.sleep(0.01)
        assert current["state"] == "finished"
        assert current["result"]["detail"]["13"]["total_servings"] == 7
        assert [event["status"] for event in current["events"]] == [
            "waiting_for_result",
            "finished",
        ]
    finally:
        manager.close()


def test_curl_is_explicit_and_token_is_not_persisted(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(competition, "load_token", lambda path: "secret.jwt")
    manager = competition.CompetitionSessionManager(
        tmp_path / ".env", tmp_path / "state", tmp_path / "reports"
    )
    proposal = {
        "kind": "agent_types",
        "types": [0, 0],
        "submission_endpoint": "/game/agent-types",
        "submission_game_id": "question:13",
        "created_at": "now",
    }
    proposal["fingerprint"] = competition._fingerprint(proposal)
    manager._sessions["curl"] = {
        "id": "curl",
        "game_id": "question:13",
        "snapshot": {"config": {"agents": [1, 2]}},
        "proposal": proposal,
        "state": "awaiting_role_approval",
    }
    try:
        result = manager.curl_command(
            "curl", fingerprint=proposal["fingerprint"], types=[0, 1]
        )
        assert "Authorization: Bearer secret.jwt" in result["command"]
        assert '"types":[0,1]' in result["command"]
        assert "secret.jwt" not in json.dumps(manager.get_session("curl"))
        assert not manager._session_path("curl").exists()
    finally:
        manager.close()


def test_manual_editor_value_is_revalidated_before_approval(
    monkeypatch, tmp_path: Path
) -> None:
    manager = competition.CompetitionSessionManager(
        tmp_path / ".env", tmp_path / "state", tmp_path / "reports"
    )
    day = {
        "day": 0,
        "steps": 4,
        "agents": [{"kind": 0, "pos": 0, "fuel": 9}],
        "others": [],
        "traffics": [],
        "endsAt": None,
    }
    config = {"agents": [0], "daySteps": [4]}
    proposal = {
        "kind": "day_plan",
        "day": 0,
        "day_snapshot": day,
        "actions": [[-4]],
        "validation": {"valid": True},
        "trace": {},
        "fallback": False,
        "planner_error": None,
        "submission_endpoint": "/game/practice/actions",
        "submission_game_id": "question:13",
        "created_at": "now",
    }
    proposal["fingerprint"] = competition._fingerprint(proposal)
    original_fingerprint = proposal["fingerprint"]
    session = {
        "id": "manual",
        "game_id": "question:13",
        "method": "greedy",
        # The polling loop used to transiently replace this with None while
        # preserving the proposal. Submission must retain the immutable copy.
        "snapshot": {"config": config, "day": None},
        "proposal": proposal,
        "state": "awaiting_plan_approval",
        "events": [],
    }
    manager._sessions["manual"] = session
    manager._events["manual"] = competition.threading.Event()
    monkeypatch.setattr(competition, "find_binary", lambda path: tmp_path / "hexudon")
    monkeypatch.setattr(
        competition,
        "run_core",
        lambda command, method, payload, **kwargs: {
            "valid": payload["actions"] == [[0, -2]],
            "error": None,
        },
    )
    monkeypatch.setattr(
        competition,
        "trace_action_plan",
        lambda *args, **kwargs: {"frames": [], "valid": True},
    )
    try:
        approved = manager.submit_proposal(
            "manual", fingerprint=original_fingerprint, actions=[[0, -2]]
        )
        assert approved["proposal"]["actions"] == [[0, -2]]
        assert approved["approval"]["fingerprint"] == approved["proposal"]["fingerprint"]
        with pytest.raises(ValueError, match="stale"):
            manager.submit_proposal(
                "manual", fingerprint=original_fingerprint, actions=[[-4]]
            )
    finally:
        manager.close()


def test_auto_mode_submits_normal_role_proposal_without_manual_approval(
    monkeypatch, tmp_path: Path
) -> None:
    config = {"agents": [0, 1], "daySteps": [4], "daySeconds": [60]}
    descriptor = {
        "question_id": "practice",
        "name": "Practice",
        "mode": "practice",
        "is_practice": True,
        "no_reset": False,
        "total_days": 1,
    }

    class FakeClient:
        posts: list[tuple[str, dict]] = []
        selected = False

        def __init__(self, token: str, base_url: str):
            pass

        def get(self, path: str, game_id: str):
            if path == "/game/config":
                return config
            if path == "/game/actions":
                raise RuntimeError("not visible yet")
            if path == "/game/state":
                return (
                    {"status": "finished", "day": 1}
                    if self.selected
                    else {"status": "selecting_agents", "day": -1}
                )
            if path == "/game/practice/score":
                return {"detail": {"13": {"total_servings": 0}}}
            raise AssertionError(path)

        def post(self, path: str, payload: dict):
            self.posts.append((path, payload))
            self.selected = True
            return {"accepted": True}

        def close(self) -> None:
            pass

    monkeypatch.setattr(competition, "load_token", lambda path: "token")
    monkeypatch.setattr(competition, "discover_assigned_games", lambda *args: [descriptor])
    monkeypatch.setattr(
        competition,
        "fetch_game_snapshot",
        lambda *args: {
            "game_id": "practice:13",
            "board": {"game_id": "practice:13", "is_practice": True, "no_reset": False},
            "config": config,
            "state": {"status": "selecting_agents", "day": -1},
            "day": None,
        },
    )
    monkeypatch.setattr(competition, "GameClient", FakeClient)
    monkeypatch.setattr(competition, "find_binary", lambda path: tmp_path / "hexudon")
    monkeypatch.setattr(
        competition,
        "run_core",
        lambda command, method, payload, **kwargs: [0, 1]
        if command == "types"
        else (_ for _ in ()).throw(AssertionError(command)),
    )
    manager = competition.CompetitionSessionManager(
        tmp_path / ".env", tmp_path / "state", tmp_path / "reports", poll_interval=0.01
    )
    try:
        session = manager.start_session(
            "practice", "greedy", execution_mode="auto"
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not FakeClient.posts:
            time.sleep(0.01)
        assert FakeClient.posts == [
            ("/game/agent-types", {"game_id": "practice:13", "types": [0, 1]})
        ]
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            current = manager.get_session(session["id"])
            if current and current["state"] == "finished":
                break
            time.sleep(0.01)
        assert current["state"] == "finished"
    finally:
        manager.close()
