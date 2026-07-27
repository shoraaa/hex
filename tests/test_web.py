from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from hexbench import api, competition, web
from hexbench.competition import (
    CompetitionSessionManager,
    _advanced_day_from_submission_error,
    _agent_selection_open_wait_seconds,
    _agent_selection_opening,
    _agent_selection_wait_seconds,
    _is_rate_limit_error,
    _adaptive_submission_margin,
    _simulation_traffic_prediction,
    _traffic_prediction_accuracy,
)


def test_dashboard_job_benchmarks_selected_policies(monkeypatch, tmp_path: Path) -> None:
    question = {
        "question_id": "question",
        "name": "Q01",
        "width": 12,
        "height": 12,
        "total_days": 7,
        "team_ids": ["13", "18"],
    }
    monkeypatch.setattr(web, "load_token", lambda _: "redacted-test-token")
    monkeypatch.setattr(
        web, "discover_practice_questions", lambda token, base_url: [question]
    )

    def fake_benchmark(game_id, methods, *args, progress=None, **kwargs):
        assert game_id == "question"
        assert methods == ["local_search", "alns"]
        assert kwargs["peer_team_ids"] == ["13", "18"]
        assert kwargs["hyperparameters"] == {
            "local_search": {"passes": 3},
            "alns": web.PRACTICE_BENCHMARK_DEFAULT_HYPERPARAMETERS,
        }
        progress({"policy": "local_search", "status": "finished"})
        return {
            "game_id": "question:13",
            "best_policy": "alns",
            "final_policy": "alns",
            "results": [
                {
                    "policy": "alns",
                    "rank": 1,
                    "distinct_types": 4,
                    "cumulative_daily_types": 28,
                    "total_servings": 100,
                    "cumulative_response_time": 0.0,
                    "wall_seconds": 0.1,
                },
                {
                    "policy": "local_search",
                    "rank": 2,
                    "distinct_types": 3,
                    "cumulative_daily_types": 20,
                    "total_servings": 80,
                    "cumulative_response_time": 0.0,
                    "wall_seconds": 0.1,
                },
            ],
            "peer_baselines": [
                {
                    "team_id": "18",
                    "status": "completed",
                    "submitted_days": 7,
                    "total_days": 7,
                    "distinct_types": 4,
                    "cumulative_daily_types": 23,
                    "total_servings": 42,
                    "cumulative_response_time": 0.0,
                }
            ],
            "completed_peer_comparison": {"wins": 1, "ties": 0, "losses": 0},
            "combined_ranking": [],
        }

    monkeypatch.setattr(web, "practice_benchmark", fake_benchmark)
    app = web.DashboardApp(
        tmp_path / ".env", tmp_path / "state", tmp_path / "reports"
    )
    try:
        created = app.start_job(
            "question",
            ["local_search", "alns"],
            {"local_search": {"passes": 3}},
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            job = app.get_job(created["id"])
            assert job is not None
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert job["status"] == "completed"
        assert job["result"]["summary"]["completed_rank"] == 1
        assert job["result"]["summary"]["configured_players"] == 2
    finally:
        app.close()


def test_dashboard_rejects_unknown_policy(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web, "load_token", lambda _: "redacted-test-token")
    app = web.DashboardApp(
        tmp_path / ".env", tmp_path / "state", tmp_path / "reports"
    )
    try:
        with pytest.raises(ValueError, match="unknown policies"):
            app.start_job("question", ["not_a_policy"])
    finally:
        app.close()


def test_dashboard_fuel_job_uses_read_only_local_benchmark(
    monkeypatch, tmp_path: Path
) -> None:
    question = {
        "question_id": "question",
        "name": "Q01",
        "width": 12,
        "height": 12,
        "total_days": 7,
        "team_ids": [],
    }
    monkeypatch.setattr(web, "load_token", lambda _: "token")
    monkeypatch.setattr(
        web, "discover_practice_questions", lambda token, base_url: [question]
    )

    def fake_fuel(methods, env_path, report_dir, **kwargs):
        assert methods == ["lns"]
        assert kwargs["game_ids"] == ["question"]
        assert kwargs["fuel_multipliers"] == (1.0, 0.5, 0.25)
        return {
            "aggregates": {
                "server": {
                    "lns": {
                        "distinct_types": 4,
                        "cumulative_daily_types": 28,
                        "total_servings": 100,
                        "refuel_events": 3,
                        "refuel_agents": 1,
                    }
                }
            },
            "cases": [
                {
                    "maximum_score": {
                        "distinct_types": 4,
                        "cumulative_daily_types": 28,
                        "total_servings": 120,
                    }
                }
            ],
            "wall_seconds": 0.1,
        }

    monkeypatch.setattr(web, "fuel_stress_benchmark", fake_fuel)
    app = web.DashboardApp(
        tmp_path / ".env", tmp_path / "state", tmp_path / "reports"
    )
    try:
        created = app.start_job("question", ["lns"], mode="fuel_stress")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            job = app.get_job(created["id"])
            assert job is not None
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert job["status"] == "completed"
        assert job["mode"] == "fuel_stress"
        assert job["result"]["aggregates"]["server"]["lns"]["refuel_events"] == 3
    finally:
        app.close()


def test_dashboard_lns_time_job_passes_budgets(monkeypatch, tmp_path: Path) -> None:
    question = {
        "question_id": "question",
        "name": "Q01",
        "width": 12,
        "height": 12,
        "total_days": 7,
        "team_ids": [],
    }
    monkeypatch.setattr(web, "load_token", lambda _: "token")
    monkeypatch.setattr(
        web, "discover_practice_questions", lambda token, base_url: [question]
    )

    def fake_curve(env_path, report_dir, **kwargs):
        assert kwargs["time_limits_ms"] == (10, 50)
        assert kwargs["fuel_multiplier"] == 0.5
        assert kwargs["method"] == "lns"
        return {
            "aggregates": {
                "10": {"distinct_types": 4, "cumulative_daily_types": 20, "total_servings": 50, "refuel_events": 3, "runtime_seconds": 0.1},
                "50": {"distinct_types": 4, "cumulative_daily_types": 22, "total_servings": 60, "refuel_events": 4, "runtime_seconds": 0.3},
            },
            "cases": [
                {
                    "maximum_score": {
                        "distinct_types": 4,
                        "cumulative_daily_types": 28,
                        "total_servings": 120,
                    }
                }
            ],
            "wall_seconds": 0.3,
        }

    monkeypatch.setattr(web, "lns_time_benchmark", fake_curve)
    app = web.DashboardApp(
        tmp_path / ".env", tmp_path / "state", tmp_path / "reports"
    )
    try:
        created = app.start_job(
            "question",
            ["lns"],
            mode="lns_time",
            time_limits_ms=[10, 50],
            time_fuel_multiplier=0.5,
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            job = app.get_job(created["id"])
            assert job is not None
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert job["status"] == "completed"
        assert job["result"]["aggregates"]["50"]["total_servings"] == 60
    finally:
        app.close()


def test_dashboard_practice_suite_runs_all_resettable_maps(
    monkeypatch, tmp_path: Path
) -> None:
    question = {
        "question_id": "question",
        "name": "Q01",
        "width": 12,
        "height": 12,
        "total_days": 7,
        "team_ids": [],
    }
    monkeypatch.setattr(web, "load_token", lambda _: "token")
    monkeypatch.setattr(
        web, "discover_practice_questions", lambda token, base_url: [question]
    )

    def fake_suite(methods, env_path, state_dir, report_dir, **kwargs):
        assert methods == ["local_search"]
        assert kwargs["hyperparameters"] == {"local_search": {"passes": 3}}
        kwargs["progress"]({"status": "finished_map", "map": 1, "maps": 1})
        return {
            "schema_version": 1,
            "methods": methods,
            "map_count": 1,
            "maps": [],
            "completed_peer_comparison": {"wins": 0, "ties": 0, "losses": 0},
            "errors": [],
        }

    monkeypatch.setattr(web, "practice_suite", fake_suite)
    app = web.DashboardApp(
        tmp_path / ".env", tmp_path / "state", tmp_path / "reports"
    )
    try:
        created = app.start_job(
            "all",
            ["local_search"],
            {"local_search": {"passes": 3}},
            mode="practice_suite",
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            job = app.get_job(created["id"])
            assert job is not None
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert job["status"] == "completed"
        assert job["mode"] == "practice_suite"
    finally:
        app.close()


def test_dashboard_page_has_selection_and_result_regions() -> None:
    assert 'id="app"' in web.DASHBOARD_HTML
    assert 'id="locale-button"' in web.DASHBOARD_HTML
    assert 'id="modal-root"' in web.DASHBOARD_HTML
    assert "PROCON 2026" in web.DASHBOARD_HTML
    assert '<script type="module" src="/assets/app.js"></script>' in web.DASHBOARD_HTML


def test_day_one_waits_until_agent_selection_deadline() -> None:
    config = {"startsAt": 1_000.0}

    assert _agent_selection_wait_seconds(config, 0, now=990.5) == 9.5
    assert _agent_selection_wait_seconds(config, 0, now=1_000.0) == 0.0
    assert _agent_selection_wait_seconds(config, 1, now=990.5) == 0.0
    assert _agent_selection_wait_seconds({}, 0, now=990.5) == 0.0
    assert _agent_selection_wait_seconds({"startsAt": "invalid"}, 0, now=990.5) == 0.0


def test_agent_selection_budget_uses_remaining_window_and_margin() -> None:
    board = {"agent_selection_time_limit": 30.0}
    config = {"startsAt": 1_000.0}

    assert api.agent_selection_budget(
        board, config, deadline_margin=2.0, now=990.0
    ) == 8.0
    assert api.agent_selection_budget(
        board, config, deadline_margin=2.0, now=950.0
    ) == 28.0


def test_agent_selection_budget_uses_explicit_practice_limit() -> None:
    assert api.agent_selection_budget(
        {},
        {},
        deadline_margin=2.0,
        selection_time_limit_seconds=30.0,
    ) == 28.0

    payload, budget = api.agent_type_payload(
        "mlns",
        {},
        {},
        {"use_lns_dp_proposals": True},
        deadline_margin=2.0,
        selection_time_limit_seconds=30.0,
    )
    assert budget == 28.0
    assert payload["search"]["timeLimitMs"] == 25_200

    assert api.agent_selection_budget(
        {"agent_selection_time_limit": 10.0},
        {},
        deadline_margin=2.0,
        selection_time_limit_seconds=30.0,
    ) == 8.0


def test_mlns_agent_type_payload_spends_selection_budget(monkeypatch) -> None:
    monkeypatch.setattr(api.time, "time", lambda: 990.0)
    payload, budget = api.agent_type_payload(
        "mlns",
        {"startsAt": 1_000.0},
        {"agent_selection_time_limit": 30.0},
        {"time_limit_ms": 123, "use_lns_dp_proposals": True},
        deadline_margin=2.0,
    )

    assert budget == 8.0
    assert payload["search"] == {
        "timeLimitMs": 7_200,
        "minIterations": 0,
        "maxIterations": api.MLNS_ANYTIME_ITERATION_CEILING,
        "stagnationIterations": 0,
    }
    assert payload["hyperparameters"] == {"use_lns_dp_proposals": True}

    monkeypatch.setattr(api.time, "time", lambda: 950.0)
    payload, budget = api.agent_type_payload(
        "mlns",
        {"startsAt": 1_000.0},
        {"agent_selection_time_limit": 30.0},
        {"use_lns_dp_proposals": True},
        deadline_margin=2.0,
    )
    assert budget == 28.0
    assert payload["search"]["timeLimitMs"] == 25_200


def test_agent_selection_not_open_error_exposes_server_timing() -> None:
    error = RuntimeError(
        "POST /game/agent-types failed (400): agent type selection has not "
        "opened yet (opens at 1785053340.0, now 1785052619.049738)"
    )

    assert _agent_selection_opening(error) == (
        1785053340.0,
        1785052619.049738,
    )
    assert _agent_selection_opening("agent type selection is closed") is None


def test_agent_selection_open_wait_uses_live_state_countdown() -> None:
    assert _agent_selection_open_wait_seconds({"selection_opens_in": 313.9}) == 313.9
    assert _agent_selection_open_wait_seconds({"selection_opens_in": -0.1}) == 0.0
    assert _agent_selection_open_wait_seconds({"selection_opens_in": None}) == 0.0


def test_rate_limit_error_is_retryable() -> None:
    assert _is_rate_limit_error(RuntimeError("GET /game/state failed (429)"))
    assert not _is_rate_limit_error(RuntimeError("GET /game/state failed (401)"))


def test_online_submission_margin_adapts_to_recent_latency() -> None:
    config = {"daySeconds": [45]}

    assert _adaptive_submission_margin({}, config, 0) == 5.0
    assert _adaptive_submission_margin(
        {"submission_latency_samples": [{"seconds": 0.8, "late": False}]},
        config,
        0,
    ) == 3.2
    assert _adaptive_submission_margin(
        {"submission_latency_samples": [{"seconds": 6.0, "late": False}]},
        config,
        0,
    ) == 11.0
    assert _adaptive_submission_margin(
        {"submission_latency_samples": [{"seconds": 0.1, "late": True}]},
        config,
        0,
    ) == 11.25
    assert _adaptive_submission_margin({}, {"daySeconds": [2]}, 0) == 0.5


def test_controller_retries_rate_limited_state_poll(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            self.state_requests = 0

        def get(self, path: str, _game_id: str) -> dict:
            if path == "/game/config":
                return {"startsAt": 1_010.0, "daySteps": [10], "agents": [0]}
            if path == "/game/state":
                self.state_requests += 1
                if self.state_requests == 1:
                    raise RuntimeError("GET /game/state failed (429)")
                return {"status": "selecting_agents", "day": 0}
            raise AssertionError(path)

        def close(self) -> None:
            return None

    fake_client = FakeClient()
    monkeypatch.setattr(
        competition, "GameClient", lambda *_args, **_kwargs: fake_client
    )
    monkeypatch.setattr(competition, "load_token", lambda _path: "token")
    monkeypatch.setattr(competition, "_token_team_id", lambda _token: None)
    monkeypatch.setattr(competition.time, "time", lambda: 1_000.0)

    manager = CompetitionSessionManager.__new__(CompetitionSessionManager)
    manager.env_path = tmp_path / ".env"
    manager.state_dir = tmp_path / "state"
    manager.report_dir = tmp_path / "reports"
    manager.binary_path = None
    manager.base_url = "https://example.invalid"
    manager.poll_interval = 0.2
    manager._lock = threading.RLock()
    manager._closed = False
    manager._events = {}
    manager._threads = {}
    manager._sessions = {
        "session": {
            "id": "session",
            "game_id": "game",
            "requested_game_id": "game",
            "game": {"is_practice": False, "no_reset": False},
            "method": "alns",
            "hyperparameters": {},
            "state": "starting",
            "snapshot": {"board": {}},
            "events": [],
        }
    }
    journal = {
        "match_starts_at": 1_010.0,
        "types": [0],
        "submitted_days": {},
        "day_snapshots": {},
        "distinct_brands": [],
        "cumulative_daily_types": 0,
        "total_servings": 0,
        "planner_state": None,
    }
    manager._journal = lambda _game_id: (  # type: ignore[method-assign]
        tmp_path / "state.json",
        journal,
    )
    manager._sync_actions = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    observed: list[tuple[str, float]] = []

    def retry_then_stop(_session_id: str, seconds: float) -> None:
        status = manager._sessions["session"]["progress"]["status"]
        observed.append((status, seconds))
        if status == "roles_submitted":
            manager._closed = True

    manager._wait = retry_then_stop  # type: ignore[method-assign]

    manager._run("session")

    assert fake_client.state_requests == 2
    assert observed == [
        ("waiting_for_server_rate_limit", 2.0),
        ("roles_submitted", 10.0),
    ]
    assert manager._sessions["session"]["state"] == "waiting_for_day"
    assert manager._sessions["session"].get("error") is None


def test_reused_game_id_does_not_skip_agent_type_post(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            self.posts: list[tuple[str, dict]] = []

        def get(self, path: str, _game_id: str) -> dict:
            if path == "/game/config":
                return {"startsAt": 1_100.0, "daySteps": [10], "agents": [0]}
            if path == "/game/state":
                return {
                    "status": "selecting_agents",
                    "day": 0,
                    "teams": {"13": {"types_selected": False}},
                }
            raise AssertionError(path)

        def post(self, path: str, payload: dict) -> dict:
            self.posts.append((path, payload))
            return {"ok": True}

        def close(self) -> None:
            return None

    fake_client = FakeClient()
    monkeypatch.setattr(
        competition, "GameClient", lambda *_args, **_kwargs: fake_client
    )
    monkeypatch.setattr(competition, "load_token", lambda _path: "token")
    monkeypatch.setattr(competition, "_token_team_id", lambda _token: "13")
    monkeypatch.setattr(competition.time, "time", lambda: 1_000.0)
    monkeypatch.setattr(competition, "find_binary", lambda _path: "hexudon")
    monkeypatch.setattr(competition, "run_core", lambda *_args, **_kwargs: [1])

    manager = CompetitionSessionManager.__new__(CompetitionSessionManager)
    manager.env_path = tmp_path / ".env"
    manager.state_dir = tmp_path / "state"
    manager.report_dir = tmp_path / "reports"
    manager.binary_path = None
    manager.base_url = "https://example.invalid"
    manager.poll_interval = 0.05
    manager._lock = threading.RLock()
    manager._closed = False
    manager._events = {}
    manager._threads = {}
    manager._sessions = {
        "session": {
            "id": "session",
            "game_id": "reused-game",
            "requested_game_id": "reused-game",
            "game": {"is_practice": False, "no_reset": False},
            "method": "alns",
            "hyperparameters": {},
            "state": "starting",
            "snapshot": {"board": {}},
            "events": [],
        }
    }
    journal = {
        "match_starts_at": 900.0,
        "types": [0],
        "submitted_days": {"0": [[-10]]},
        "day_snapshots": {"0": {"day": 0}},
        "distinct_brands": [1],
        "cumulative_daily_types": 1,
        "total_servings": 1,
        "planner_state": None,
    }
    manager._journal = lambda _game_id: (  # type: ignore[method-assign]
        tmp_path / "state.json",
        journal,
    )
    manager._sync_actions = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    def stop_after_submission(_session_id: str, _seconds: float) -> None:
        if manager._sessions["session"]["progress"]["status"] == "roles_submitted":
            manager._closed = True

    manager._wait = stop_after_submission  # type: ignore[method-assign]

    manager._run("session")

    assert fake_client.posts == [
        ("/game/agent-types", {"game_id": "reused-game", "types": [1]})
    ]
    assert journal["match_starts_at"] == 1_100.0
    assert journal["types"] == [1]
    assert journal["submitted_days"] == {}
    assert journal["day_snapshots"] == {}


def test_start_queues_until_agent_selection_opens(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            self.posts = 0

        def get(self, path: str, _game_id: str) -> dict:
            if path == "/game/config":
                return {"startsAt": 1_100.0, "daySteps": [10], "agents": [0]}
            if path == "/game/state":
                return {"status": "selecting_agents", "day": 0}
            raise AssertionError(path)

        def post(self, path: str, payload: dict) -> dict:
            assert path == "/game/agent-types"
            assert payload == {"game_id": "game", "types": [0]}
            self.posts += 1
            if self.posts == 1:
                raise RuntimeError(
                    "POST /game/agent-types failed (400): agent type selection "
                    "has not opened yet (opens at 1000.0, now 900.0)"
                )
            return {"ok": True}

        def close(self) -> None:
            return None

    fake_client = FakeClient()
    clock = [900.0]
    monkeypatch.setattr(
        competition, "GameClient", lambda *_args, **_kwargs: fake_client
    )
    monkeypatch.setattr(competition, "load_token", lambda _path: "token")
    monkeypatch.setattr(competition, "_token_team_id", lambda _token: None)
    monkeypatch.setattr(competition.time, "time", lambda: clock[0])
    monkeypatch.setattr(competition, "find_binary", lambda _path: "hexudon")
    monkeypatch.setattr(competition, "run_core", lambda *_args, **_kwargs: [0])

    manager = CompetitionSessionManager.__new__(CompetitionSessionManager)
    manager.env_path = tmp_path / ".env"
    manager.state_dir = tmp_path / "state"
    manager.report_dir = tmp_path / "reports"
    manager.binary_path = None
    manager.base_url = "https://example.invalid"
    manager.poll_interval = 0.05
    manager._lock = threading.RLock()
    manager._closed = False
    manager._events = {}
    manager._threads = {}
    manager._sessions = {
        "session": {
            "id": "session",
            "game_id": "game",
            "requested_game_id": "game",
            "game": {"is_practice": False, "no_reset": False},
            "method": "alns",
            "hyperparameters": {},
            "state": "starting",
            "snapshot": {"board": {"agent_selection_time_limit": 60.0}},
            "events": [],
        }
    }
    journal = {
        "types": None,
        "submitted_days": {},
        "day_snapshots": {},
        "distinct_brands": [],
        "cumulative_daily_types": 0,
        "total_servings": 0,
        "planner_state": None,
    }
    manager._journal = lambda _game_id: (  # type: ignore[method-assign]
        tmp_path / "state.json",
        journal,
    )
    manager._sync_actions = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    observed_statuses: list[str] = []
    observed_waits: list[float] = []

    def advance_or_stop(_session_id: str, seconds: float) -> None:
        observed_statuses.append(manager._sessions["session"]["progress"]["status"])
        observed_waits.append(seconds)
        if observed_statuses[-1] == "waiting_for_agent_selection_open":
            clock[0] = 1_000.1
        else:
            manager._closed = True

    manager._wait = advance_or_stop  # type: ignore[method-assign]

    manager._run("session")

    assert fake_client.posts == 2
    assert journal["types"] == [0]
    assert observed_statuses == [
        "waiting_for_agent_selection_open",
        "roles_submitted",
    ]
    assert observed_waits == pytest.approx([100.0, 99.9])
    assert manager._sessions["session"]["state"] == "waiting_for_day"
    assert manager._sessions["session"]["agent_selection_retry_at"] is None


def test_agent_selection_streams_improving_incumbents_and_metrics(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.posts: list[dict] = []

        def post(self, endpoint: str, payload: dict) -> dict:
            assert endpoint == "/game/agent-types"
            self.posts.append(payload)
            return {"ok": True}

    manager = CompetitionSessionManager.__new__(CompetitionSessionManager)
    manager.binary_path = None
    manager.report_dir = tmp_path / "reports"
    manager._lock = threading.RLock()
    manager._sessions = {
        "session": {
            "id": "session",
            "state": "starting",
            "game": {"is_practice": True},
            "method": "mlns",
            "events": [],
            "agent_selection_incumbents": [],
            "agent_selection_metric": None,
        }
    }
    monkeypatch.setattr(competition, "find_binary", lambda _path: "hexudon")

    def fake_stream_types_core(_method, _payload, *, on_improve, **_kwargs):
        on_improve(
            {
                "types": [0, 0],
                "score": [3, 8, 12],
                "phase": "robust_screen",
            }
        )
        on_improve(
            {
                "types": [0, 1],
                "score": [4, 9, 14],
                "phase": "timed_screen",
            }
        )
        on_improve(
            {
                "types": [0, 1],
                "score": [4, 10, 15],
                "phase": "confirmed",
            }
        )
        return {
            "kind": "final",
            "types": [0, 1],
            "score": [4, 10, 15],
            "phase": "confirmed",
        }

    monkeypatch.setattr(
        competition, "stream_types_core", fake_stream_types_core
    )
    client = FakeClient()
    journal = {"types": None}

    result = manager._stream_agent_selection(
        "session",
        client,  # type: ignore[arg-type]
        manager._sessions["session"],
        {"agents": [0, 1]},
        {"config": {"agents": [0, 1]}, "search": {"timeLimitMs": 900}},
        1.0,
        "game:13",
        journal,
        tmp_path / "state.json",
    )

    assert [post["types"] for post in client.posts] == [[0, 0], [0, 1]]
    assert result["types"] == [0, 1]
    assert journal["types"] == [0, 1]
    rows = manager._sessions["session"]["agent_selection_incumbents"]
    assert [row["phase"] for row in rows] == [
        "robust_screen",
        "timed_screen",
        "confirmed",
    ]
    assert [row["submission_status"] for row in rows] == [
        "submitted",
        "superseded",
        "submitted",
    ]
    metric = manager._sessions["session"]["agent_selection_metric"]
    assert metric["status"] == "accepted"
    assert metric["submission_count"] == 2
    assert metric["incumbent_count"] == 3
    assert metric["best_score"] == [4, 10, 15]
    assert metric["result"] == {
        "patrol_agents": 1,
        "refuel_agents": 1,
        "types": [0, 1],
    }


def test_controller_does_not_open_or_submit_day_one_before_start(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            self.requested_paths: list[str] = []

        def get(self, path: str, _game_id: str) -> dict:
            self.requested_paths.append(path)
            if path == "/game/config":
                return {"startsAt": 1_000.0, "daySteps": [10]}
            if path == "/game/state":
                return {"status": "in_progress", "day": 0}
            raise AssertionError(f"Day 1 was opened early through {path}")

        def close(self) -> None:
            return None

    fake_client = FakeClient()
    monkeypatch.setattr(
        competition, "GameClient", lambda *_args, **_kwargs: fake_client
    )
    monkeypatch.setattr(competition, "load_token", lambda _path: "token")
    monkeypatch.setattr(competition, "_token_team_id", lambda _token: None)
    monkeypatch.setattr(competition.time, "time", lambda: 900.0)

    manager = CompetitionSessionManager.__new__(CompetitionSessionManager)
    manager.env_path = tmp_path / ".env"
    manager.state_dir = tmp_path / "state"
    manager.report_dir = tmp_path / "reports"
    manager.binary_path = None
    manager.base_url = "https://example.invalid"
    manager.poll_interval = 1.0
    manager._lock = threading.RLock()
    manager._closed = False
    manager._events = {}
    manager._threads = {}
    manager._sessions = {
        "session": {
            "id": "session",
            "game_id": "game",
            "requested_game_id": "game",
            "game": {"is_practice": False, "no_reset": False},
            "method": "alns",
            "hyperparameters": {},
            "state": "starting",
            "snapshot": {"board": {"agent_selection_time_limit": 30.0}},
            "events": [],
        }
    }
    journal = {
        "types": [0],
        "submitted_days": {},
        "day_snapshots": {},
        "distinct_brands": [],
        "cumulative_daily_types": 0,
        "total_servings": 0,
        "planner_state": None,
    }
    manager._journal = lambda _game_id: (  # type: ignore[method-assign]
        tmp_path / "state.json",
        journal,
    )
    manager._sync_actions = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    waited: list[float] = []

    def stop_after_wait(_session_id: str, seconds: float) -> None:
        waited.append(seconds)
        manager._closed = True

    manager._wait = stop_after_wait  # type: ignore[method-assign]
    manager._stream_day = lambda *_args, **_kwargs: pytest.fail(  # type: ignore[method-assign]
        "Day 1 solver started before agent selection ended"
    )

    manager._run("session")

    assert fake_client.requested_paths == ["/game/config", "/game/state"]
    assert waited == [100.0]
    assert manager._sessions["session"]["state"] == "waiting_for_day"
    assert manager._sessions["session"]["progress"]["status"] == (
        "waiting_for_agent_selection"
    )


def test_session_ui_explains_agent_selection_wait() -> None:
    script = (web.STATIC_ROOT / "app.js").read_text()

    assert 'waiting_for_agent_selection:"Waiting for agent selection to close"' in script
    assert 'waiting_for_agent_selection:"Đang chờ hết thời gian chọn loại xe"' in script
    assert (
        'waiting_for_agent_selection_open:"Waiting for agent selection to open"'
        in script
    )
    assert "b.agent_selection_time_limit??c.agent_selection_time_limit??0" in script
    assert 'waiting_for_server_rate_limit:"Server rate limit reached; retrying shortly"' in script


def test_saved_match_card_and_snapshot_survive_online_removal(
    monkeypatch, tmp_path: Path
) -> None:
    session_path = (
        tmp_path / "reports" / "sessions" / "saved-session" / "session.json"
    )
    session_path.parent.mkdir(parents=True)
    session_path.write_text(
        json.dumps(
            {
                "id": "saved-session",
                "requested_game_id": "saved-game",
                "game_id": "saved-game",
                "game": {
                    "question_id": "saved-game",
                    "name": "Mock Q03",
                    "is_practice": False,
                    "no_reset": False,
                    "width": 32,
                    "height": 30,
                    "total_days": 10,
                    "teams": [{"id": "13", "name": "Us"}],
                },
                "state": "finished",
                "snapshot": {
                    "game_id": "saved-game",
                    "board": {"game_id": "saved-game"},
                    "config": {
                        "map": {"width": 32, "height": 30, "cells": [[0]]},
                        "daySteps": [10] * 10,
                        "daySeconds": [30] * 10,
                    },
                    "state": {"status": "finished", "day": 10},
                    "day": None,
                },
                "result": {
                    "ranking": ["13", "8"],
                    "detail": {
                        "13": {
                            "distinct_types": 20,
                            "cumulative_daily_types": 132,
                            "total_servings": 534,
                        }
                    },
                },
                "created_at": "2026-07-25T14:51:50+00:00",
                "updated_at": "2026-07-25T14:59:01+00:00",
                "events": [],
            }
        )
    )
    monkeypatch.setattr(web, "load_token", lambda _path: "token")
    monkeypatch.setattr(web, "_token_team_id", lambda _token: "13")
    monkeypatch.setattr(web, "discover_assigned_games", lambda *_args: [])
    monkeypatch.setattr(
        web,
        "fetch_game_snapshot",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("match expired")),
    )

    app = web.DashboardApp(
        tmp_path / ".env",
        tmp_path / "state",
        tmp_path / "reports",
    )
    try:
        journal_path, journal = app._competition._journal("saved-game")
        journal["submitted_days"] = {"0": [[-10]]}
        journal["day_snapshots"] = {
            "0": {
                "day_info": {
                    "day": 0,
                    "traffics": [{"pos": 4, "status": 2}],
                },
                "trace": {
                    "frames": [
                        {
                            "step": 0,
                            "agents": [{"cell": 0, "type": 0, "fuel": 10}],
                            "servings": 0,
                            "types": 0,
                        },
                        {
                            "step": 10,
                            "agents": [{"cell": 0, "type": 0, "fuel": 10}],
                            "servings": 3,
                            "types": 2,
                        },
                    ]
                },
                "submitted_at": "2026-07-25T14:52:00+00:00",
            }
        }
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path.write_text(json.dumps(journal))

        games = app.list_all_games()
        assert games == [
            {
                "question_id": "saved-game",
                "name": "Mock Q03",
                "is_practice": False,
                "no_reset": False,
                "width": 32,
                "height": 30,
                "total_days": 10,
                "teams": [{"id": "13", "name": "Us"}],
                "saved": True,
                "archived": True,
                "saved_session_id": "saved-session",
                "saved_at": "2026-07-25T14:59:01+00:00",
                "session_state": "finished",
                "score": {
                    "distinct_types": 20,
                    "cumulative_daily_types": 132,
                    "total_servings": 534,
                },
                "rank": 1,
                "rank_count": 2,
            }
        ]
        snapshot = app.snapshot("saved-game")
        assert snapshot["archived"] is True
        assert snapshot["archived_session_id"] == "saved-session"
        assert snapshot["archived_result"]["ranking"] == ["13", "8"]
        replay = app._saved_replay("saved-game")
        assert replay is not None
        assert replay["archived"] is True
        assert replay["replay"]["days"][0]["road_condition"] == {"4": 2}
        assert replay["replay"]["days"][0]["teams"][0]["servings"] == 3
    finally:
        app.close()


def test_saved_match_ui_exposes_analysis_card_and_exact_session() -> None:
    script = (web.STATIC_ROOT / "app.js").read_text()

    assert "saved-card-meta" in script
    assert 'retained?T("analyze"):T("enter")' in script
    assert "state.game.archived&&state.game.saved_session_id" in script
    assert "state.snapshot.archived_result" in script
    assert (
        "if(state.game?.archived&&session.result)"
        "state.snapshot.archived_result=session.result"
    ) in script
    assert "function archivedResult(){return state.game?.archived?" in script
    assert "!state.game.archived" in script


def test_live_practice_ranking_does_not_use_saved_session_result() -> None:
    script = (web.STATIC_ROOT / "app.js").read_text()

    assert (
        "if(state.game.archived&&state.game.saved_session_id)"
        "attachSession(await api("
    ) in script
    assert "else await restoreSession();renderGame();if(resettable)loadStandings()" in script
    assert "function standingsMarkup(){const archived=archivedResult();" in script


def test_stale_action_post_identifies_a_forward_day_transition() -> None:
    error = RuntimeError(
        "POST /game/actions failed (409): day 3 is not the current day (4)"
    )

    assert _advanced_day_from_submission_error(error, 3) == 4
    assert _advanced_day_from_submission_error(error, 2) is None
    assert _advanced_day_from_submission_error(
        RuntimeError("POST /game/actions failed (500): request rejected"), 3
    ) is None
    assert _advanced_day_from_submission_error(
        RuntimeError(
            "POST /game/actions failed (409): game has ended "
            "(stopped at 1785056850.0, now 1785056853.6900246)"
        ),
        9,
    ) == 10


def test_streaming_stops_cleanly_when_server_advances_day(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.posts = 0

        def post(
            self, _endpoint: str, _payload: dict, *, timeout: float | None = None
        ) -> dict:
            assert timeout is not None and 0 < timeout <= 2.0
            self.posts += 1
            raise RuntimeError(
                "POST /game/actions failed (409): day 3 is not the current day (4)"
            )

    manager = CompetitionSessionManager.__new__(CompetitionSessionManager)
    manager.binary_path = None
    manager._lock = threading.RLock()
    manager._sessions = {
        "session": {
            "id": "session",
            "state": "streaming",
            "game": {"is_practice": False},
            "method": "alns",
            "hyperparameters": {},
            "incumbents": [],
            "day_metrics": [],
            "events": [],
        }
    }
    manager.report_dir = tmp_path / "reports"
    manager._record_prediction_accuracy = lambda *_args: None  # type: ignore[method-assign]
    recorded_metrics: list[dict] = []
    manager._update_day_metric = (  # type: ignore[method-assign]
        lambda *_args, **kwargs: recorded_metrics.append(kwargs)
    )
    manager._record_incumbent = lambda *_args, **_kwargs: 1  # type: ignore[method-assign]
    manager._mark_incumbent_submitted = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    stream_called = False

    def fake_stream_core(_method, _payload, *, on_improve, should_stop, **_kwargs):
        nonlocal stream_called
        stream_called = True
        on_improve({"actions": [[-1]], "score": [1, 1, 1]})
        assert should_stop() is True
        return {"kind": "final", "actions": [[-1]], "score": [1, 1, 1]}

    monkeypatch.setattr(competition, "find_binary", lambda _path: "hexudon")
    monkeypatch.setattr(competition, "stream_core", fake_stream_core)
    monkeypatch.setattr(competition, "planning_budget", lambda *_args, **_kwargs: 1.0)
    client = FakeClient()
    journal = {
        "submitted_days": {},
        "day_snapshots": {},
        "distinct_brands": [],
        "cumulative_daily_types": 0,
        "total_servings": 0,
        "planner_state": None,
    }

    manager._stream_day(
        "session",
        client,  # type: ignore[arg-type]
        manager._sessions["session"],
        {"daySteps": [1, 1, 1, 1], "daySeconds": [1, 1, 1, 1]},
        {"day": 3, "agents": [{"kind": 0}], "traffics": []},
        journal,
        tmp_path / "state.json",
        "game",
        False,
    )

    assert client.posts == 1
    assert stream_called is True
    assert manager._sessions["session"]["state"] == "waiting_for_day"
    assert manager._sessions["session"]["last_streamed_day"] == 3
    assert manager._sessions["session"]["error"] is None
    assert manager._sessions["session"]["progress"]["status"] == "day_advanced"
    assert recorded_metrics[-1]["submission_status"] == "missed"


def test_streaming_submits_first_real_incumbent_without_emergency_seed(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.posts: list[dict] = []
            self.failed_improvement_once = False

        def post(
            self, _endpoint: str, payload: dict, *, timeout: float | None = None
        ) -> dict:
            assert timeout is not None and 0 < timeout <= 2.0
            self.posts.append(payload)
            if (
                payload["actions"] == [[-2]]
                and not self.failed_improvement_once
            ):
                self.failed_improvement_once = True
                raise RuntimeError(
                    "POST /game/actions failed (ambiguous network error)"
                )
            return {"ok": True}

    manager = CompetitionSessionManager.__new__(CompetitionSessionManager)
    manager.binary_path = None
    manager._lock = threading.RLock()
    manager._sessions = {
        "session": {
            "id": "session",
            "state": "streaming",
            "game": {"is_practice": False},
            "method": "mlns",
            "hyperparameters": {},
            "incumbents": [],
            "day_metrics": [],
            "events": [],
        }
    }
    manager.report_dir = tmp_path / "reports"
    manager._record_prediction_accuracy = lambda *_args: None  # type: ignore[method-assign]
    monkeypatch.setattr(competition, "find_binary", lambda _path: "hexudon")
    monkeypatch.setattr(competition, "planning_budget", lambda *_args, **_kwargs: 5.0)
    monkeypatch.setattr(
        competition,
        "trace_action_plan",
        lambda *_args, **_kwargs: {
            "score": {"distinct_types": 1, "daily_types": 1, "servings": 2},
            "frames": [],
        },
    )
    client = FakeClient()

    def fake_stream_core(_method, payload, *, on_improve, **_kwargs):
        assert client.posts == []
        assert 50 <= payload["search"]["timeLimitMs"] <= 5_000
        on_improve({"actions": [[-1]], "score": [1, 1, 1]})
        on_improve({"actions": [[-2]], "score": [1, 1, 2]})
        # The timing guard submits the first incumbent immediately and
        # coalesces a faster follow-up until the final flush.
        assert client.posts == [
            {"game_id": "game", "day": 0, "actions": [[-1]]}
        ]
        return None

    monkeypatch.setattr(competition, "stream_core", fake_stream_core)
    journal = {
        "submitted_days": {},
        "day_snapshots": {},
        "distinct_brands": [],
        "cumulative_daily_types": 0,
        "total_servings": 0,
        "planner_state": None,
    }

    manager._stream_day(
        "session",
        client,  # type: ignore[arg-type]
        manager._sessions["session"],
        {"daySteps": [1], "daySeconds": [45], "spots": []},
        {"day": 0, "agents": [{"kind": 0}], "traffics": []},
        journal,
        tmp_path / "state.json",
        "game",
        False,
    )

    assert journal["submitted_days"] == {"0": [[-2]]}
    assert [post["actions"] for post in client.posts] == [
        [[-1]],
        [[-2]],
        [[-2]],
    ]
    assert journal["submitted_metadata"]["0"]["source"] == "incumbent"
    assert manager._sessions["session"]["progress"]["status"] == "submitted"
    assert manager._sessions["session"]["day_metrics"][0]["submission_status"] == (
        "accepted"
    )
    assert manager._sessions["session"]["day_metrics"][0]["result"] == {
        "distinct_types": 1,
        "daily_types": 1,
        "servings": 2,
    }
    assert manager._sessions["session"]["day_metrics"][0]["accepted_source"] == (
        "incumbent"
    )
    assert manager._sessions["session"]["day_metrics"][0][
        "failed_submission_count"
    ] == 1
    assert manager._sessions["session"]["submission_latency_samples"][-2][
        "accepted"
    ] is False


def test_streaming_reconciles_lost_practice_submission_response(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.posts: list[dict] = []

        def post(
            self, endpoint: str, payload: dict, *, timeout: float | None = None
        ) -> dict:
            assert endpoint == "/game/practice/actions"
            assert timeout is not None and 0 < timeout <= 2.0
            self.posts.append(payload)
            raise RuntimeError(
                "POST /game/practice/actions failed (ambiguous network error)"
            )

        def get(self, endpoint: str, game_id: str) -> dict:
            assert endpoint == "/game/actions"
            assert game_id == "game:13"
            return {
                "actions": [
                    {
                        "team_id": "13",
                        "day": 0,
                        "plan": [[-1]],
                        "submit_count": 1,
                        "submitted_at": 123.0,
                    }
                ]
            }

    manager = CompetitionSessionManager.__new__(CompetitionSessionManager)
    manager.binary_path = None
    manager._lock = threading.RLock()
    manager._sessions = {
        "session": {
            "id": "session",
            "state": "streaming",
            "game": {"is_practice": True},
            "method": "mlns",
            "hyperparameters": {},
            "incumbents": [],
            "day_metrics": [],
            "events": [],
        }
    }
    manager.report_dir = tmp_path / "reports"
    manager._record_prediction_accuracy = lambda *_args: None  # type: ignore[method-assign]
    monkeypatch.setattr(competition, "find_binary", lambda _path: "hexudon")
    monkeypatch.setattr(competition, "planning_budget", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(
        competition,
        "trace_action_plan",
        lambda *_args, **_kwargs: {
            "score": {"distinct_types": 1, "daily_types": 1, "servings": 2},
            "frames": [],
        },
    )

    def fake_stream_core(_method, _payload, *, on_improve, **_kwargs):
        on_improve({"actions": [[-1]], "score": [1, 1, 2]})
        return {"kind": "final", "actions": [[-1]], "score": [1, 1, 2]}

    monkeypatch.setattr(competition, "stream_core", fake_stream_core)
    journal = {
        "submitted_days": {},
        "day_snapshots": {},
        "distinct_brands": [],
        "cumulative_daily_types": 0,
        "total_servings": 0,
        "planner_state": None,
    }

    manager._stream_day(
        "session",
        FakeClient(),  # type: ignore[arg-type]
        manager._sessions["session"],
        {"daySteps": [1], "daySeconds": [1], "spots": []},
        {"day": 0, "agents": [{"kind": 0}], "traffics": []},
        journal,
        tmp_path / "state.json",
        "game:13",
        False,
    )

    assert journal["submitted_days"] == {"0": [[-1]]}
    assert manager._sessions["session"]["day_metrics"][0][
        "submission_status"
    ] == "accepted"
    assert manager._sessions["session"]["day_metrics"][0]["result"] == {
        "distinct_types": 1,
        "daily_types": 1,
        "servings": 2,
    }
    assert manager._sessions["session"]["day_metrics"][0][
        "failed_submission_count"
    ] == 0
    assert manager._sessions["session"]["last_submission"]["response"] == {
        "reconciled": True,
        "submit_count": 1,
        "submitted_at": 123.0,
    }


def test_dashboard_parameter_ui_has_prefills_sliders_and_preview() -> None:
    script = (web.STATIC_ROOT / "app.js").read_text()

    assert "plannerParameters" in script
    assert 'parameterControl(field,values,"data-param",disabled)' in script
    assert 'state.bootstrap.policies.includes("alns")' in script
    assert "collectParams" in script
    assert "agentSelectionMarkup" in script
    assert "agent_selection_metric" in script
    assert "agent_selection_incumbents" in script


def test_mlns_gnn_checkbox_is_exposed_only_for_mlns() -> None:
    mlns_fields = {field["key"] for field in api.POLICY_HYPERPARAMETERS["mlns"]}
    alns_fields = {field["key"] for field in api.POLICY_HYPERPARAMETERS["alns"]}
    script = (web.STATIC_ROOT / "app.js").read_text()

    assert "use_traffic_gnn" in mlns_fields
    assert "use_traffic_gnn" not in alns_fields
    assert '["use_lns_dp_proposals","use_traffic_gnn"]' in script


def test_competition_refresh_clears_historical_session_ui_and_stale_polls() -> None:
    script = (web.STATIC_ROOT / "app.js").read_text()

    assert "restoreSession" in script
    assert "scheduleSessionPoll" in script
    assert "syncProposal" in script
    assert "reset_incomplete" in script
    assert 'terminal.has(state.session.state)' in script


def test_play_ui_restores_and_resumes_interrupted_or_failed_sessions() -> None:
    script = (web.STATIC_ROOT / "app.js").read_text()

    assert "continueFromServerDay" in script
    assert 'id="resume-search"' in script
    assert 'controlSession("resume")' in script
    assert "canResumeSession" in script
    assert "session.recoverable!==true" in script
    assert "recoveryDay<=serverDay" in script
    assert "matches.find(s=>canResumeSession(s))" in script


def test_failed_controller_can_resume_from_authoritative_server_state(
    tmp_path: Path,
) -> None:
    manager = CompetitionSessionManager.__new__(CompetitionSessionManager)
    manager.report_dir = tmp_path
    manager._lock = threading.RLock()
    manager._closed = False
    manager._events = {}
    predecessor_release = threading.Event()
    predecessor = threading.Thread(target=predecessor_release.wait, daemon=True)
    predecessor.start()
    manager._threads = {"session": predecessor}
    manager._sessions = {
        "session": {
            "id": "session",
            "state": "failed",
            "recoverable": True,
            "proposal": {"fingerprint": "stale"},
            "approval": {"fingerprint": "stale"},
            "error": "POST /game/practice/actions failed (409)",
            "progress": {"status": "failed", "day": 2},
            "events": [],
        }
    }
    restarted = threading.Event()
    manager._run = lambda session_id: restarted.set()  # type: ignore[method-assign]

    resumed = manager.control("session", "resume")

    assert not restarted.wait(0.05)
    predecessor_release.set()
    assert restarted.wait(1)
    manager._threads["session"].join(1)
    assert resumed["state"] == "starting"
    assert resumed["recoverable"] is False
    assert resumed["proposal"] is None
    assert resumed["approval"] is None
    assert resumed["error"] is None
    assert resumed["progress"] == {"status": "resuming_from_server"}
    persisted = json.loads(
        (tmp_path / "sessions" / "session" / "session.json").read_text()
    )
    assert persisted["events"][-1]["status"] == "resume"


def test_dashboard_restart_marks_active_controller_recoverable(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "sessions" / "session" / "session.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "id": "session",
                "state": "streaming",
                "error": None,
                "events": [],
            }
        )
    )
    legacy_failed_path = (
        tmp_path / "reports" / "sessions" / "legacy" / "session.json"
    )
    legacy_failed_path.parent.mkdir(parents=True)
    legacy_failed_path.write_text(
        json.dumps(
            {
                "id": "legacy",
                "state": "failed",
                "error": "POST /game/practice/actions failed (409)",
                "events": [],
            }
        )
    )

    manager = CompetitionSessionManager(
        tmp_path / ".env",
        tmp_path / "state",
        tmp_path / "reports",
    )
    try:
        restored = manager.get_session("session")
        assert restored is not None
        assert restored["state"] == "interrupted"
        assert restored["recoverable"] is True
        assert restored["error"] == "dashboard restarted; resume explicitly"
        legacy = manager.get_session("legacy")
        assert legacy is not None
        assert legacy["recoverable"] is True
    finally:
        manager.close()


def test_practice_reset_invalidates_failed_recovery_session(tmp_path: Path) -> None:
    manager = CompetitionSessionManager.__new__(CompetitionSessionManager)
    manager.report_dir = tmp_path
    manager._lock = threading.RLock()
    manager._events = {}
    manager._threads = {}
    manager._sessions = {
        "session": {
            "id": "session",
            "game_id": "game:team",
            "state": "failed",
            "recoverable": True,
            "events": [],
        }
    }

    assert manager.cancel_game_sessions("game:team") == ["session"]
    assert manager._sessions["session"]["state"] == "cancelled"
    assert manager._sessions["session"]["recoverable"] is False


def test_play_ui_exposes_streaming_console_and_original_game_features() -> None:
    script = (web.STATIC_ROOT / "app.js").read_text()

    # Autonomous streaming console: policy + practice time-limit + start/stop.
    assert "start-search" in script
    assert "time-limit" in script
    assert "time_limit_seconds" in script
    assert 'agentSelectionTimeLimit:"30"' in script
    assert "agent-selection-time-limit" in script
    assert "agent_selection_time_limit_seconds" in script
    assert "autoSubmitInfo" in script
    assert "convergenceMarkup" in script
    assert "incumbents" in script
    assert "elapsed_seconds" in script
    assert "day_metrics" in script
    assert "prediction_accuracy" in script
    assert "dayResult" in script
    assert "dayDistinct" in script
    assert "metric.result" in script
    assert '${esc(T("daily"))}: ${score.daily}' in script
    assert "competitive?.prev?.holder_score?.[1]" in script
    assert "peer.cumulative_daily_types" in script
    assert 'id="score-daily"' in script
    assert "updateDayMetricTimers" in script
    assert "day-metrics-grid" in script
    assert "convergence-chart" not in script
    assert "internalRankMarkup" in script
    assert "internal_rank" in script
    assert "trafficRank" in script
    assert "workloadRank" in script
    assert "patrolFuel" in script
    assert "sincePrevious" in script
    assert 'role="img"' in script
    assert 'post("/api/play/sessions"' in script
    assert "showReplay" in script
    assert "showAnswers" in script
    assert "showConfig" in script
    assert "resetGame" in script
    assert "renderMap" in script
    assert "hex-locale" in script


def test_prediction_metrics_compare_simulation_with_authoritative_roads() -> None:
    config = {
        "busyThreshold": 2,
        "jammedThreshold": 4,
        "map": {"cells": [[1, 0, 1]]},
    }
    predicted = _simulation_traffic_prediction(
        config,
        [
            [{"pos": 0, "volume": 2}, {"pos": 2, "volume": 1}],
            [{"pos": 0, "volume": 2}, {"pos": 2, "volume": 3}],
        ],
    )

    assert predicted == [{"pos": 0, "status": 2}, {"pos": 2, "status": 2}]
    assert _traffic_prediction_accuracy(
        predicted,
        [{"pos": 0, "status": 2}, {"pos": 2, "status": 1}],
    ) == {
        "matched_roads": 1,
        "road_count": 2,
        "prediction_accuracy": 0.5,
    }


def test_session_persists_revealed_gnn_accuracy(tmp_path: Path) -> None:
    manager = CompetitionSessionManager.__new__(CompetitionSessionManager)
    manager.report_dir = tmp_path
    manager._lock = threading.RLock()
    manager._sessions = {"session": {"id": "session", "day_metrics": []}}
    journal = {
        "traffic_predictions": {
            "1": {
                "mode": "gnn",
                "traffics": [
                    {"pos": 0, "status": 2},
                    {"pos": 2, "status": 0},
                ],
            }
        }
    }

    manager._record_prediction_accuracy(
        "session",
        journal,
        {
            "day": 1,
            "traffics": [
                {"pos": 0, "status": 2},
                {"pos": 2, "status": 1},
            ],
        },
        "simulation",
    )

    assert manager._sessions["session"]["day_metrics"][0] == {
        "day": 1,
        "prediction_mode": "gnn",
        "prediction_available": True,
        "matched_roads": 1,
        "road_count": 2,
        "prediction_accuracy": 0.5,
    }


def test_day_transition_reuses_the_active_play_session() -> None:
    script = (web.STATIC_ROOT / "app.js").read_text()

    assert "state.selectedDay=Number(proposal.day)" in script
    assert "if(expected===day)" in script
    assert 'await api(`/api/play/sessions/${state.session.id}`)' in script
    assert 'await controlSession("cancel",false)' in script


def test_reset_does_not_restore_or_race_an_interrupted_planner_session() -> None:
    script = (web.STATIC_ROOT / "app.js").read_text()

    assert '"cancelled","interrupted"' in script
    cancel = 'await post(`/api/play/sessions/${state.session.id}/control`,{action:"cancel"})'
    reset = 'await post(`/api/games/${encodeURIComponent(state.game.question_id)}/reset`,{})'
    assert cancel in script
    assert script.index(cancel) < script.index(reset)
    assert 'state.proposalFingerprint=null' in script
    assert "state.session&&!terminal.has(state.session.state)?state.session.proposal" in script


def test_game_map_and_replay_follow_the_official_visual_geometry() -> None:
    script = (web.STATIC_ROOT / "app.js").read_text()
    styles = (web.STATIC_ROOT / "styles.css").read_text()

    assert "viewWidth:(width+1.5)*w" in script
    assert "viewHeight:(height+1)*1.5*radius+radius" in script
    assert '<svg class="hex-map" width="${g.viewWidth}" height="${g.viewHeight}"' in script
    assert 'stroke-dasharray="1 4" opacity=".55"' in script
    assert "replayTrail" in script
    assert "replay?team_id=" in script
    assert '<div class="map-canvas">${renderMap(' in script
    assert ".map-canvas{display:flex;width:fit-content;margin:0 auto}" in styles
    assert ".hex-map{display:block;flex:none;min-width:0;margin:0}" in styles


def test_replay_slider_updates_in_place_and_collection_is_a_count() -> None:
    script = (web.STATIC_ROOT / "app.js").read_text()

    assert '$("#replay-range").oninput' in script
    assert "queueReplayFrame()" in script
    assert "requestAnimationFrame(updateReplayFrame)" in script
    assert "collectedNow=(frame.collected||[]).length" in script
    assert 'collectedNow>0?`+${collectedNow}`:"—"' in script
    assert '(frame.collected||[]).join' not in script


def test_scoreboard_loads_named_peer_results() -> None:
    script = (web.STATIC_ROOT / "app.js").read_text()

    assert "/standings`" in script
    assert "peer.ranking.map" in script
    assert "cumulative_daily_types" in script
    assert "cumulative_response_time" in script
    assert "team.name" in script


def test_dashboard_policy_list_includes_aco() -> None:
    assert "aco" in web.POLICIES
    assert "aco_ls" in web.POLICIES


def test_local_catalog_exposes_manifest_cases_without_online_token(
    monkeypatch, tmp_path: Path
) -> None:
    cases = tmp_path / "cases"
    scenario = {
        "schema_version": 1,
        "tier": "brutal",
        "config": {
            "daySteps": [4],
            "agents": [0],
            "spots": [{"brand": 7, "pos": 1, "stocks": 1}],
        },
        "opponents": [],
    }
    (cases / "hard" / "brutal").mkdir(parents=True)
    (cases / "hard" / "brutal" / "case-0000.json").write_text(
        web.json.dumps(scenario)
    )
    (cases / "hard" / "manifest.json").write_text(
        web.json.dumps(
            {
                "suite": "hard",
                "cases": [
                    {
                        "path": "brutal/case-0000.json",
                        "tier": "brutal",
                        "seed": 3_000_000,
                        "target": "distinct_types",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(web, "LOCAL_CASE_ROOT", cases)
    app = web.DashboardApp(
        tmp_path / "missing.env", tmp_path / "state", tmp_path / "reports"
    )
    try:
        catalog = app.local_cases()
        assert catalog["case_count"] == 1
        assert catalog["groups"][0]["id"] == "hard/brutal"
        loaded = app.local_case("hard/brutal/case-0000.json")
        assert loaded["scenario"]["tier"] == "brutal"
        assert loaded["optimum_score"] == {
            "distinct_types": 1,
            "cumulative_daily_types": 1,
            "total_servings": 1,
        }
    finally:
        app.close()


def test_local_run_uses_trace_capable_core_and_normalizes_score(
    monkeypatch, tmp_path: Path
) -> None:
    cases = tmp_path / "cases"
    scenario = {
        "schema_version": 1,
        "config": {
            "daySteps": [4],
            "agents": [0],
            "spots": [{"brand": 7, "pos": 1, "stocks": 1}],
        },
        "opponents": [],
    }
    (cases / "quick").mkdir(parents=True)
    (cases / "quick" / "case-0000.json").write_text(web.json.dumps(scenario))
    (cases / "quick" / "manifest.json").write_text(
        web.json.dumps(
            {"suite": "quick", "cases": [{"path": "case-0000.json", "seed": 1}]}
        )
    )
    monkeypatch.setattr(web, "LOCAL_CASE_ROOT", cases)
    monkeypatch.setattr(web, "find_binary", lambda _: tmp_path / "hexudon")

    def fake_core(command, method, payload, **kwargs):
        assert command == "visualize"
        assert method == "local_search"
        assert payload["hyperparameters"] == {"passes": 3}
        return {
            "score": {
                "distinct_types": 1,
                "cumulative_daily_types": 1,
                "total_servings": 1,
            },
            "valid_days": 1,
            "invalid_days": 0,
            "replay": {"days": [{"day": 0, "teams": []}]},
        }

    monkeypatch.setattr(web, "run_core", fake_core)
    app = web.DashboardApp(
        tmp_path / ".env", tmp_path / "state", tmp_path / "reports"
    )
    try:
        result = app.run_local_case(
            "quick/case-0000.json", "local_search", {"passes": 3}
        )
        assert result["result"]["objective_percentages"] == {
            "distinct_types": 100.0,
            "cumulative_daily_types": 100.0,
            "total_servings": 100.0,
        }
        assert result["result"]["replay"]["days"][0]["day"] == 0
    finally:
        app.close()


def test_local_tab_exposes_case_controls_scores_and_frame_playback() -> None:
    page = web.DASHBOARD_HTML
    script = (web.STATIC_ROOT / "app.js").read_text()

    assert 'id="local-nav"' in page
    assert 'href="/local"' in page
    assert 'api("/api/local/cases")' in script
    assert 'post("/api/local/run"' in script
    assert "local-score-grid" in script
    assert "local-range" in script
    assert "local-map-canvas" in script
    assert "team.actions" in script
    assert script.index("const params=collectLocalParams()") < script.index(
        "state.local.running=true"
    )
    assert "lns" in web.POLICIES
    assert "alns" in web.POLICIES


def test_local_tab_exposes_traffic_model_prediction_overlay() -> None:
    script = (web.STATIC_ROOT / "app.js").read_text()

    assert "trafficModel" in script
    assert 'id="local-model"' in script
    assert "traffic_model_id" in script
    assert 'api("/api/traffic/models")' in script
    assert "data-local-view" in script
    assert "actualTraffics" in script
    assert "localTrafficBarMarkup" in script
    assert "state.local.view=button.dataset.localView" in script
    assert "state.local.prediction=state.local.result?.result?.traffic||null" in script


def test_traffic_models_discovers_trained_checkpoints(
    monkeypatch, tmp_path: Path
) -> None:
    import torch

    from hexbench.traffic_gnn import FEATURE_NAMES, TRAFFIC_CLASSES, TrafficGNN

    monkeypatch.setattr(web, "ROOT", tmp_path)
    model_dir = tmp_path / "reports" / "traffic-fixture"
    model_dir.mkdir(parents=True)
    checkpoint = model_dir / "model.pt"
    torch.save(
        {
            "state_dict": TrafficGNN(
                len(FEATURE_NAMES), hidden_size=8, layers=1
            ).state_dict(),
            "feature_names": FEATURE_NAMES,
            "hidden_size": 8,
            "layers": 1,
            "classes": TRAFFIC_CLASSES,
        },
        checkpoint,
    )
    (model_dir / "report.json").write_text(
        web.json.dumps(
            {
                "kind": "offline-lns16-traffic-gnn",
                "checkpoint": str(checkpoint),
                "best_epoch": 5,
                "best_validation": {"accuracy": 0.89, "macro_f1": 0.7},
                "dataset": {"policy": "lns", "dataset": "fixture.pt"},
                "train_samples": 10,
                "validation_samples": 2,
            }
        )
    )
    app = web.DashboardApp(
        tmp_path / ".env", tmp_path / "state", tmp_path / "reports"
    )
    try:
        models = app.traffic_models()
        assert len(models) == 1
        assert models[0]["id"] == "traffic-fixture"
        assert models[0]["validation_accuracy"] == 0.89
        assert models[0]["policy"] == "lns"
    finally:
        app.close()


def test_local_run_attaches_traffic_prediction_when_model_selected(
    monkeypatch, tmp_path: Path
) -> None:
    cases = tmp_path / "cases"
    scenario = {
        "schema_version": 1,
        "config": {
            "daySteps": [4, 4],
            "agents": [0],
            "spots": [{"brand": 7, "pos": 1, "stocks": 1}],
        },
        "opponents": [],
    }
    (cases / "quick").mkdir(parents=True)
    (cases / "quick" / "case-0000.json").write_text(web.json.dumps(scenario))
    (cases / "quick" / "manifest.json").write_text(
        web.json.dumps({"suite": "quick", "cases": [{"path": "case-0000.json", "seed": 1}]})
    )
    monkeypatch.setattr(web, "LOCAL_CASE_ROOT", cases)
    monkeypatch.setattr(web, "find_binary", lambda _: tmp_path / "hexudon")

    def fake_core(command, method, payload, **kwargs):
        assert command == "visualize"
        return {
            "score": {
                "distinct_types": 1,
                "cumulative_daily_types": 2,
                "total_servings": 2,
            },
            "valid_days": 2,
            "invalid_days": 0,
            "replay": {
                "days": [
                    {"day": 0, "road_condition": {}, "teams": []},
                    {"day": 1, "road_condition": {"0": 1}, "teams": []},
                ]
            },
        }

    monkeypatch.setattr(web, "run_core", fake_core)
    app = web.DashboardApp(
        tmp_path / ".env", tmp_path / "state", tmp_path / "reports"
    )
    captured: dict[str, object] = {}

    def fake_predict(scenario_arg, result_arg, model_id):
        captured["model_id"] = model_id
        return {
            "classes": ["smooth", "busy", "jammed"],
            "days": [
                {
                    "day": 1,
                    "road_count": 1,
                    "matched": 0,
                    "accuracy": 0.0,
                    "cells": [
                        {
                            "pos": 0,
                            "predicted": 0,
                            "actual": 1,
                            "probability": 0.5,
                            "correct": False,
                        }
                    ],
                }
            ],
            "road_count": 1,
            "matched": 0,
            "accuracy": 0.0,
            "confusion": [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        }

    monkeypatch.setattr(app, "_traffic_prediction", fake_predict)
    try:
        result = app.run_local_case(
            "quick/case-0000.json", "local_search", {}, "traffic-fixture"
        )
        assert captured["model_id"] == "traffic-fixture"
        traffic = result["result"]["traffic"]
        assert traffic["road_count"] == 1
        assert traffic["days"][0]["cells"][0]["actual"] == 1
    finally:
        app.close()


def test_game_ui_proxies_replay_answers_and_safe_practice_reset(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str, object]] = []

    class FakeClient:
        def __init__(self, token: str, base_url: str):
            assert token == "token"

        def get(self, path: str, game_id: str):
            calls.append(("GET", path, game_id))
            if path == "/game/board":
                return {
                    "game_id": f"{game_id}:13",
                    "is_practice": game_id == "practice",
                    "no_reset": False,
                }
            if path == "/game/replay":
                return {"days": [{"day": 0}]}
            if path == "/game/actions":
                return {"actions": [{"day": 0, "team_id": "13", "plan": [[-4]]}]}
            raise AssertionError(path)

        def post(self, path: str, payload: dict):
            calls.append(("POST", path, payload))
            return {"accepted": True}

        def close(self):
            pass

    monkeypatch.setattr(web, "load_token", lambda path: "token")
    monkeypatch.setattr(api, "GameClient", FakeClient)
    app = web.DashboardApp(
        tmp_path / ".env", tmp_path / "state", tmp_path / "reports"
    )
    try:
        assert app.replay("live")["replay"]["days"][0]["day"] == 0
        assert app.answers("live")["actions"][0]["team_id"] == "13"
        app._competition._sessions["stale"] = {
            "id": "stale",
            "game_id": "practice:13",
            "state": "interrupted",
            "proposal": {"kind": "day_plan"},
            "events": [],
        }
        assert app.reset_game("practice") == {"accepted": True}
        stale = app._competition.get_session("stale")
        assert stale is not None
        assert stale["state"] == "cancelled"
        assert stale["proposal"] is None
        journal_path, journal = app._competition._journal("practice:13")
        assert journal_path.exists()
        assert journal["submitted_days"] == {}
        assert journal["day_snapshots"] == {}
        with pytest.raises(ValueError, match="resettable practice"):
            app.reset_game("live")
        assert (
            "POST",
            "/game/practice/reset",
            {"game_id": "practice:13"},
        ) in calls
    finally:
        app.close()


def test_practice_standings_include_named_peers_in_official_order(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeClient:
        def __init__(self, token: str, base_url: str):
            pass

        def get(self, path: str, game_id: str):
            if path == "/game/board":
                return {"game_id": "practice:13", "is_practice": True}
            if path == "/game/practice/score":
                team_id = game_id.rsplit(":", 1)[-1]
                scores = {
                    "13": {
                        "distinct_types": 4,
                        "cumulative_daily_types": 5,
                        "total_servings": 10,
                        "cumulative_response_time": 2.0,
                    },
                    "8": {
                        "distinct_types": 4,
                        "cumulative_daily_types": 6,
                        "total_servings": 1,
                        "cumulative_response_time": 100.0,
                    },
                }
                return {"detail": {team_id: scores[team_id]}}
            raise AssertionError((path, game_id))

        def close(self):
            pass

    monkeypatch.setattr(web, "load_token", lambda _: "token")
    monkeypatch.setattr(api, "GameClient", FakeClient)
    monkeypatch.setattr(
        web,
        "discover_assigned_games",
        lambda *_: [
            {
                "question_id": "practice",
                "is_practice": True,
                "team_ids": ["13", "8"],
                "teams": [
                    {"id": "13", "name": "banned214"},
                    {"id": "8", "name": "BGNA"},
                ],
            }
        ],
    )
    app = web.DashboardApp(
        tmp_path / ".env", tmp_path / "state", tmp_path / "reports"
    )
    try:
        result = app.standings("practice")
        assert result["ranking"] == ["8", "13"]
        assert result["teams"][1] == {"id": "8", "name": "BGNA"}
        assert result["own_team_id"] == "13"
    finally:
        app.close()
