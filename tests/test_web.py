from __future__ import annotations

import time
from pathlib import Path

import pytest

from hexbench import api, web


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


def test_dashboard_parameter_ui_has_prefills_sliders_and_preview() -> None:
    script = (web.STATIC_ROOT / "app.js").read_text()

    assert "plannerParameters" in script
    assert 'data-param=' in script
    assert 'state.bootstrap.policies.includes("alns")' in script
    assert "collectParams" in script


def test_competition_refresh_clears_historical_session_ui_and_stale_polls() -> None:
    script = (web.STATIC_ROOT / "app.js").read_text()

    assert "restoreSession" in script
    assert "scheduleSessionPoll" in script
    assert "syncProposal" in script
    assert "reset_incomplete" in script
    assert 'terminal.has(state.session.state)' in script


def test_play_ui_exposes_streaming_console_and_original_game_features() -> None:
    script = (web.STATIC_ROOT / "app.js").read_text()

    # Autonomous streaming console: policy + practice time-limit + start/stop.
    assert "start-search" in script
    assert "time-limit" in script
    assert "time_limit_seconds" in script
    assert "autoSubmitInfo" in script
    assert 'post("/api/play/sessions"' in script
    assert "showReplay" in script
    assert "showAnswers" in script
    assert "showConfig" in script
    assert "resetGame" in script
    assert "renderMap" in script
    assert "hex-locale" in script


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
