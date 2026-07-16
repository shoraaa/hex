from __future__ import annotations

import time
from pathlib import Path

import pytest

from hexbench import web


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
        assert methods == ["greedy", "alns"]
        assert kwargs["peer_team_ids"] == ["13", "18"]
        assert kwargs["hyperparameters"] == {
            "greedy": {"max_targets": 3},
            "alns": {"fixed_iterations": 3_072},
        }
        progress({"policy": "greedy", "status": "finished"})
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
                    "policy": "greedy",
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
            ["greedy", "alns"],
            {"greedy": {"max_targets": 3}},
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
        assert methods == ["greedy"]
        assert kwargs["hyperparameters"] == {"greedy": {"max_targets": 3}}
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
            ["greedy"],
            {"greedy": {"max_targets": 3}},
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
    assert 'id="game"' in web.DASHBOARD_HTML
    assert 'id="policies"' in web.DASHBOARD_HTML
    assert 'id="results"' in web.DASHBOARD_HTML
    assert 'id="hyperparameters"' in web.DASHBOARD_HTML
    assert 'id="fuel-run"' in web.DASHBOARD_HTML
    assert 'id="fuel-results"' in web.DASHBOARD_HTML
    assert 'id="fuel-multipliers"' in web.DASHBOARD_HTML
    assert 'id="time-run"' in web.DASHBOARD_HTML
    assert 'id="time-results"' in web.DASHBOARD_HTML
    assert 'id="suite-run"' in web.DASHBOARD_HTML
    assert 'id="suite-results"' in web.DASHBOARD_HTML
    assert 'data-mode="practice"' in web.DASHBOARD_HTML
    assert 'data-mode="competition"' in web.DASHBOARD_HTML
    assert 'id="competition-map"' in web.DASHBOARD_HTML
    assert 'id="approve-proposal"' in web.DASHBOARD_HTML
    assert '<script type="module" src="/assets/app.js"></script>' in web.DASHBOARD_HTML


def test_dashboard_policy_list_includes_aco() -> None:
    assert "aco" in web.POLICIES
    assert "aco_ls" in web.POLICIES
    assert "lns" in web.POLICIES
    assert "alns" in web.POLICIES
