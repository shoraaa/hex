from __future__ import annotations

import json
import time
from pathlib import Path

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
            if path == "/game/state":
                # The real practice server can advance beyond the configured
                # days without changing its textual status to ``finished``.
                return {"status": "in_progress", "day": 1} if self.finished else {
                    "status": "in_progress",
                    "day": 0,
                }
            if path == "/game/day":
                return day
            if path == "/game/practice/score":
                return {"detail": {"13": {"distinct_types": 1}}}
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
        assert FakeClient.posts[0][0] == "/game/practice/actions"
        assert all(path != "/game/practice/reset" for path, _ in FakeClient.posts)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            current = manager.get_session(session["id"])
            assert current is not None
            if current["state"] == "finished":
                break
            time.sleep(0.01)
        assert current["state"] == "finished"
        assert current["result"] == {
            "detail": {"13": {"distinct_types": 1}}
        }
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
