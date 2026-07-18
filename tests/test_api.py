from __future__ import annotations

from pathlib import Path
import json
import time
import pytest

from hexbench import api
from hexbench.generator import generate_scenario


def test_game_descriptor_classifies_no_reset_practice_as_competition() -> None:
    question = {
        "id": "6860f3ae-8715-4a5a-97f4-a0d6849b4d6a",
        "name": "New Question X",
    }
    descriptor = api._game_descriptor(
        question,
        {
            "is_practice": True,
            "no_reset": True,
            "map": {"width": 30, "height": 24},
            "daySteps": [171, 121, 177, 178, 61],
            "daySeconds": [60] * 5,
            "teams": [{"team_id": "13"}],
        },
        "13",
    )
    assert descriptor is not None
    assert descriptor["mode"] == "competition"
    assert descriptor["competition_kind"] == "practice_competition"
    assert descriptor["capabilities"]["reset"] is False
    assert descriptor["capabilities"]["submit"] is True


def test_game_descriptor_keeps_resettable_practice_in_lab() -> None:
    descriptor = api._game_descriptor(
        {"id": "practice", "name": "Practice"},
        {
            "is_practice": True,
            "no_reset": False,
            "map": {"width": 12, "height": 12},
            "daySteps": [40],
            "daySeconds": [60],
            "teams": [{"team_id": "13"}],
        },
        "13",
    )
    assert descriptor is not None
    assert descriptor["mode"] == "practice"
    assert descriptor["capabilities"]["reset"] is True


def test_game_descriptor_uses_authoritative_match_team_names() -> None:
    descriptor = api._game_descriptor(
        {"id": "practice", "name": "Practice", "match_id": 11},
        {
            "is_practice": True,
            "teams": [{"team_id": "13"}, {"team_id": "8"}],
            "daySteps": [40],
        },
        "13",
        [{"id": 13, "name": "banned214"}, {"id": 8, "name": "BGNA"}],
    )

    assert descriptor is not None
    assert descriptor["teams"] == [
        {"id": "13", "name": "banned214"},
        {"id": "8", "name": "BGNA"},
    ]


def test_hyperparameters_are_validated_per_selected_method() -> None:
    assert api.normalize_hyperparameters(
        ["lns"], {"lns": {"min_iterations": 4, "max_iterations": 8}}
    ) == {"lns": {"min_iterations": 4, "max_iterations": 8}}
    assert api.normalize_hyperparameters(
        ["alns"], {"alns": {"min_iterations": 4, "max_iterations": 8}}
    ) == {"alns": {"min_iterations": 4, "max_iterations": 8}}
    assert api.normalize_hyperparameters(
        ["alns"], {"alns": {"alns_iterations": 10_000}}
    ) == {"alns": {"alns_iterations": 10_000}}
    with pytest.raises(ValueError, match="unselected"):
        api.normalize_hyperparameters(["greedy"], {"lns": {"max_iterations": 8}})
    assert api.normalize_hyperparameters(
        ["aco"], {"aco": {"ants": 1000, "iterations": 1000, "evaporation": 0.1}}
    ) == {"aco": {"ants": 1000, "iterations": 1000, "evaporation": 0.1}}
    with pytest.raises(ValueError, match="less than"):
        api.normalize_hyperparameters(["aco"], {"aco": {"evaporation": 1.0}})
    with pytest.raises(ValueError, match="cannot be combined"):
        api.normalize_hyperparameters(
            ["alns"],
            {"alns": {"alns_iterations": 10_000, "time_limit_ms": 2_000}},
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        api.normalize_hyperparameters(
            ["alns"],
            {"alns": {"alns_iterations": 10_000, "max_iterations": 2_000}},
        )


def test_no_reset_competition_snapshot_uses_competitive_state(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    config = {
        "daySteps": [4],
        "daySeconds": [60],
        "agents": [0],
    }

    class FakeClient:
        def __init__(self, token: str, base_url: str):
            pass

        def get(self, path: str, game_id: str):
            calls.append((path, game_id))
            if path == "/game/board":
                return {
                    "game_id": "practice-comp:13",
                    "is_practice": True,
                    "no_reset": True,
                }
            if path == "/game/config":
                return config
            if path == "/game/competitive/state":
                return {
                    "selecting": True,
                    "canonical_types": None,
                    "standings": {"timeline": {"total_servings": 0}},
                }
            raise AssertionError(path)

        def close(self) -> None:
            pass

    monkeypatch.setattr(api, "GameClient", FakeClient)
    snapshot = api.fetch_game_snapshot("token", "practice-comp")

    assert snapshot["state"] == {"status": "selecting_agents", "day": -1}
    assert snapshot["day"] is None
    assert snapshot["competitive_state"]["selecting"] is True
    assert ("/game/competitive/state", "practice-comp") in calls
    assert not any(path in {"/game/state", "/game/day"} for path, _ in calls)


def test_competitive_open_day_is_normalized_for_the_planner() -> None:
    config = {"daySteps": [4], "daySeconds": [60], "agents": [0]}
    state, day = api.normalize_competitive_state(
        {
            "selecting": False,
            "open": {
                "day": 0,
                "steps": 4,
                "agents": [{"kind": 0, "pos": 2, "fuel": 9}],
                "road_condition": {"1": 2},
            },
        },
        config,
    )

    assert state == {"status": "in_progress", "day": 0}
    assert day == {
        "day": 0,
        "steps": 4,
        "agents": [{"kind": 0, "pos": 2, "fuel": 9}],
        "others": [],
        "traffics": [{"pos": 1, "status": 2}],
        "endsAt": None,
    }


def test_incomplete_competitive_reset_is_not_treated_as_agent_selection() -> None:
    state, day = api.normalize_competitive_state(
        {
            "selecting": True,
            "canonical_types": None,
            "open_day": 5,
            "standings": {
                "owned_days": {"13": 5},
                "owner_by_day": {str(day): "13" for day in range(5)},
                "timeline": {"distinct_types": 0, "total_servings": 0},
            },
        },
        {"daySteps": [1, 1, 1, 1, 1]},
    )

    assert state["status"] == "reset_incomplete"
    assert state["day"] == 5
    assert "did not clear day ownership" in state["error"]
    assert day is None
def test_search_hyperparameter_metadata_guides_the_dashboard() -> None:
    alns_iterations = next(
        field
        for field in api.POLICY_HYPERPARAMETERS["alns"]
        if field["key"] == "alns_iterations"
    )

    assert alns_iterations["recommended"] == 1_536
    assert alns_iterations["ui_max"] == 12_000
    fields = {field["key"]: field for field in api.POLICY_HYPERPARAMETERS["alns"]}
    assert {
        "final_alns_iterations",
        "seed_iterations",
        "exact_nodes",
        "final_exact_nodes",
        "aco_ants",
        "aco_iterations",
        "aco_evaporation",
    } <= fields.keys()
    assert "alns_restarts" not in fields
    assert fields["final_alns_iterations"]["recommended"] == 1_024
    assert fields["seed_iterations"]["recommended"] == 2_048
    assert fields["exact_nodes"]["recommended"] == 512
    assert fields["final_exact_nodes"]["recommended"] == 1_024


def test_fuel_stress_variants_preserve_authoritative_config_except_fuel() -> None:
    config = generate_scenario(17, "medium", "small", "single")["config"]
    config["fuelLimits"] = 2 * config["daySteps"][0]
    question = {
        "question_id": "practice",
        "name": "Practice",
        "width": config["map"]["width"],
        "height": config["map"]["height"],
        "total_days": len(config["daySteps"]),
    }
    variants = api.build_fuel_stress_variants(question, config)
    assert [row["fuel_label"] for row in variants] == [
        "server",
        "1x",
        "0.5x",
        "0.25x",
    ]
    for variant in variants:
        changed = variant["scenario"]["config"]
        assert changed["fuelLimits"] == variant["fuel_limit"]
        assert {**changed, "fuelLimits": config["fuelLimits"]} == config
        assert variant["scenario"]["opponents"] == []


def test_fuel_stress_benchmark_is_read_only(monkeypatch, tmp_path: Path) -> None:
    config = generate_scenario(18, "medium", "small", "single")["config"]
    config["fuelLimits"] = 2 * config["daySteps"][0]
    question = {
        "question_id": "practice",
        "name": "Practice",
        "width": config["map"]["width"],
        "height": config["map"]["height"],
        "total_days": len(config["daySteps"]),
    }

    class FakeClient:
        def __init__(self, token: str, base_url: str):
            pass

        def get(self, path: str, game_id: str):
            assert path == "/game/config"
            return config

        def post(self, path: str, payload: object):
            raise AssertionError("fuel benchmark must never write to the server")

        def close(self) -> None:
            pass

    def fake_run_core(command, policy, payload, **kwargs):
        assert command == "eval"
        fuel = payload["config"]["fuelLimits"]
        return {
            "score": {
                "distinct_types": 1,
                "cumulative_daily_types": fuel,
                "total_servings": fuel,
            },
            "valid_days": len(payload["config"]["daySteps"]),
            "invalid_days": 0,
            "patrol_agents": 2,
            "refuel_agents": 1,
            "refuel_events": 3,
            "ending_patrol_fuel": 1,
            "errors": [],
        }

    monkeypatch.setattr(api, "load_token", lambda _: "token")
    monkeypatch.setattr(api, "discover_practice_questions", lambda *args: [question])
    monkeypatch.setattr(api, "GameClient", FakeClient)
    monkeypatch.setattr(api, "run_core", fake_run_core)
    report = api.fuel_stress_benchmark(
        ["lns"], tmp_path / ".env", tmp_path / "report", jobs=1
    )
    assert report["map_count"] == 1
    assert report["case_count"] == 4
    assert report["aggregates"]["0.5x"]["lns"]["refuel_events"] == 3
    assert (tmp_path / "report" / "report.md").exists()


def test_lns_time_benchmark_records_score_curve(monkeypatch, tmp_path: Path) -> None:
    config = generate_scenario(19, "medium", "small", "single")["config"]
    question = {
        "question_id": "practice",
        "name": "Practice",
        "width": config["map"]["width"],
        "height": config["map"]["height"],
        "total_days": len(config["daySteps"]),
    }

    class FakeClient:
        def __init__(self, token: str, base_url: str):
            pass

        def get(self, path: str, game_id: str):
            return config

        def close(self) -> None:
            pass

    def fake_run_core(command, policy, payload, **kwargs):
        assert policy == "alns"
        budget = payload["hyperparameters"]["time_limit_ms"]
        return {
            "score": {
                "distinct_types": 1,
                "cumulative_daily_types": budget,
                "total_servings": budget,
            },
            "valid_days": len(payload["config"]["daySteps"]),
            "invalid_days": 0,
            "patrol_agents": 2,
            "refuel_agents": 1,
            "refuel_events": budget,
            "ending_patrol_fuel": 1,
            "errors": [],
        }

    monkeypatch.setattr(api, "load_token", lambda _: "token")
    monkeypatch.setattr(api, "discover_practice_questions", lambda *args: [question])
    monkeypatch.setattr(api, "GameClient", FakeClient)
    monkeypatch.setattr(api, "run_core", fake_run_core)
    report = api.lns_time_benchmark(
        tmp_path / ".env",
        tmp_path / "curve",
        method="alns",
        time_limits_ms=(10, 50),
        jobs=1,
    )
    assert report["method"] == "alns"
    assert report["aggregates"]["10"]["total_servings"] == 10
    assert report["aggregates"]["50"]["total_servings"] == 50
    assert (tmp_path / "curve" / "report.md").exists()


def test_starting_spot_is_acquired_during_pending_move() -> None:
    config = generate_scenario(7, "easy", "small", "single")["config"]
    spot = config["spots"][0]
    width, height = config["map"]["width"], config["map"]["height"]
    direction = next(
        direction
        for direction, destination in api.neighbors(height, width, spot["pos"])
        if config["map"]["cells"][destination // width][destination % width] != 3
    )
    config["daySteps"][0] = 2
    day = {
        "day": 0,
        "endsAt": None,
        "agents": [{"kind": 0, "pos": spot["pos"], "fuel": config["fuelLimits"]}],
        "others": [],
        "traffics": [
            {"pos": pos, "status": 0}
            for pos in range(width * height)
            if config["map"]["cells"][pos // width][pos % width] == 1
        ],
    }
    assert spot["brand"] in api.predict_acquired_brands(config, day, [[direction]])


def test_practice_ignores_stale_deadline() -> None:
    config = generate_scenario(8, "easy", "small", "single")["config"]
    day = {"day": 0, "endsAt": time.time() - 10_000}
    assert api.planning_budget(
        config, day, is_practice=True, deadline_margin=2
    ) == config["daySeconds"][0] - 2
    assert api.planning_budget(
        config, day, is_practice=False, deadline_margin=2
    ) == 0.1


def test_deploy_dry_run_makes_no_posts(monkeypatch, tmp_path: Path) -> None:
    config = generate_scenario(9, "easy", "small", "single")["config"]

    class FakeClient:
        instances: list["FakeClient"] = []

        def __init__(self, token: str, base_url: str):
            self.posts: list[tuple[str, object]] = []
            self.instances.append(self)

        def get(self, path: str, game_id: str):
            if path == "/game/board":
                return {"game_id": "practice:team", "is_practice": True, "no_reset": False}
            if path == "/game/config":
                return config
            if path == "/game/state":
                return {"status": "selecting_agents", "day": -1}
            raise AssertionError(path)

        def post(self, path: str, payload: object):
            self.posts.append((path, payload))
            raise AssertionError("dry-run must not post")

        def close(self) -> None:
            pass

    monkeypatch.setattr(api, "GameClient", FakeClient)
    monkeypatch.setattr(api, "load_token", lambda _: "redacted-test-token")
    api.deploy(
        "practice",
        "greedy",
        tmp_path / ".env",
        tmp_path / "state",
        dry_run=True,
        once=True,
        deadline_margin=2,
        poll_interval=0.01,
        binary_path=str(api.find_binary()),
        base_url="https://example.invalid/api",
    )
    assert FakeClient.instances[0].posts == []


def test_real_deploy_uses_deadline_governed_online_alns_search(
    monkeypatch, tmp_path: Path
) -> None:
    config = generate_scenario(10, "medium", "small", "single")["config"]
    captured: list[dict] = []
    progress_events: list[dict] = []

    class FakeClient:
        def __init__(self, token: str, base_url: str):
            pass

        def get(self, path: str, game_id: str):
            if path == "/game/board":
                return {"game_id": "real:team", "is_practice": False}
            if path == "/game/config":
                return config
            if path == "/game/state":
                return {"status": "in_progress", "day": 0}
            if path == "/game/day":
                return {
                    "day": 0,
                    "endsAt": time.time() + 60,
                    "agents": [
                        {"kind": 0, "pos": pos, "fuel": config["fuelLimits"]}
                        for pos in config["agents"]
                    ],
                    "others": [],
                    "traffics": [
                        {"pos": row * config["map"]["width"] + column, "status": 0}
                        for row, cells in enumerate(config["map"]["cells"])
                        for column, terrain in enumerate(cells)
                        if terrain == 1
                    ],
                }
            raise AssertionError(path)

        def post(self, path: str, payload: object):
            raise AssertionError("dry-run must not post")

        def close(self) -> None:
            pass

    def fake_run_core(command, method, payload, **kwargs):
        if command == "plan":
            captured.append(payload["search"])
            return [[-config["daySteps"][0]] for _ in config["agents"]]
        if command == "check":
            return {"valid": True}
        raise AssertionError(command)

    monkeypatch.setattr(api, "GameClient", FakeClient)
    monkeypatch.setattr(api, "load_token", lambda _: "redacted-test-token")
    monkeypatch.setattr(api, "run_core", fake_run_core)
    api.deploy(
        "real",
        "alns",
        tmp_path / ".env",
        tmp_path / "state",
        dry_run=True,
        once=True,
        deadline_margin=2,
        poll_interval=0.01,
        binary_path=str(api.find_binary()),
        base_url="https://example.invalid/api",
        progress=progress_events.append,
    )
    assert len(captured) == 1
    assert captured[0]["timeLimitMs"] > 0
    assert captured[0]["minIterations"] == 32
    assert captured[0]["maxIterations"] == 10_000_000
    assert captured[0]["stagnationIterations"] == 0
    planning = next(event for event in progress_events if event["status"] == "planning")
    assert planning["day"] == 0
    assert planning["budget_seconds"] > 0


def test_real_deploy_alns_iterations_activates_untimed_exact_search(
    monkeypatch, tmp_path: Path
) -> None:
    config = generate_scenario(10, "medium", "small", "single")["config"]
    captured: list[dict] = []
    progress_events: list[dict] = []

    class FakeClient:
        def __init__(self, token: str, base_url: str):
            pass

        def get(self, path: str, game_id: str):
            if path == "/game/board":
                return {"game_id": "practice:team", "is_practice": True}
            if path == "/game/config":
                return config
            if path == "/game/state":
                return {"status": "in_progress", "day": 0}
            if path == "/game/day":
                return {
                    "day": 0,
                    "agents": [
                        {"kind": 0, "pos": pos, "fuel": config["fuelLimits"]}
                        for pos in config["agents"]
                    ],
                    "others": [],
                    "traffics": [],
                }
            raise AssertionError(path)

        def close(self) -> None:
            pass

    def fake_run_core(command, method, payload, **kwargs):
        if command == "plan":
            captured.append(payload["search"])
            return [[-config["daySteps"][0]] for _ in config["agents"]]
        if command == "check":
            return {"valid": True}
        raise AssertionError(command)

    monkeypatch.setattr(api, "GameClient", FakeClient)
    monkeypatch.setattr(api, "load_token", lambda _: "redacted-test-token")
    monkeypatch.setattr(api, "run_core", fake_run_core)
    api.deploy(
        "practice",
        "alns",
        tmp_path / ".env",
        tmp_path / "state",
        dry_run=True,
        once=True,
        deadline_margin=2,
        poll_interval=0.01,
        binary_path=str(api.find_binary()),
        base_url="https://example.invalid/api",
        method_hyperparameters={"alns_iterations": 10_000},
        progress=progress_events.append,
    )

    assert captured == [
        {
            "minIterations": 2048,
            "maxIterations": 10_000,
            "stagnationIterations": 10_000,
        }
    ]
    planning = next(event for event in progress_events if event["status"] == "planning")
    assert planning["alns_iterations"] == 10_000
    assert "budget_seconds" not in planning


def test_practice_benchmark_resets_ranks_and_leaves_best(monkeypatch, tmp_path: Path) -> None:
    completed: list[str] = []
    progress_events: list[dict] = []

    class FakeClient:
        instances: list["FakeClient"] = []

        def __init__(self, token: str, base_url: str):
            self.posts: list[tuple[str, object]] = []
            self.instances.append(self)

        def get(self, path: str, game_id: str):
            if path == "/game/board":
                return {"game_id": "practice:13", "is_practice": True, "no_reset": False}
            if path == "/game/config":
                return {
                    "daySteps": [10, 10, 10, 10],
                    "spots": [
                        {"brand": 0, "stocks": 2},
                        {"brand": 1, "stocks": 1},
                        {"brand": 2, "stocks": 1},
                        {"brand": 3, "stocks": 1},
                    ],
                }
            if path == "/game/practice/score":
                policy = completed[-1]
                servings = 100 if policy == "local_search" else 60
                daily = 28 if policy == "local_search" else 20
                return {
                    "ranking": ["13"],
                    "detail": {
                        "13": {
                            "distinct_types": 4,
                            "cumulative_daily_types": daily,
                            "total_servings": servings,
                            "cumulative_response_time": 0.0,
                        }
                    },
                }
            raise AssertionError(path)

        def post(self, path: str, payload: object):
            assert path == "/game/practice/reset"
            self.posts.append((path, payload))
            return {"ok": True}

        def close(self) -> None:
            pass

    def fake_deploy(game_id: str, method: str, *args, **kwargs):
        callback = kwargs.get("progress")
        assert callback is not None
        callback(
            {
                "game_id": "practice:13",
                "day": 0,
                "status": "planning",
                "budget_seconds": 58.0,
            }
        )
        completed.append(method)
        return {"status": "finished"}

    monkeypatch.setattr(api, "GameClient", FakeClient)
    monkeypatch.setattr(api, "load_token", lambda _: "redacted-test-token")
    monkeypatch.setattr(api, "deploy", fake_deploy)
    report = api.practice_benchmark(
        "practice",
        ["greedy", "local_search"],
        tmp_path / ".env",
        tmp_path / "state",
        tmp_path / "report",
        peer_team_ids=[],
        quiet=True,
        progress=progress_events.append,
    )
    assert report["best_policy"] == "local_search"
    assert report["final_policy"] == "local_search"
    assert [row["policy"] for row in report["results"]] == [
        "local_search",
        "greedy",
    ]
    assert len(FakeClient.instances[0].posts) == 2
    assert {
        "policy": "greedy",
        "game_id": "practice:13",
        "day": 0,
        "status": "planning",
        "budget_seconds": 58.0,
    } in progress_events
    assert {"policy": "greedy", "status": "reset_complete"} in progress_events
    assert (tmp_path / "report" / "report.json").exists()


def test_practice_benchmark_refuses_no_reset_game(monkeypatch, tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self, token: str, base_url: str):
            self.posts = []

        def get(self, path: str, game_id: str):
            return {"game_id": "practice:13", "is_practice": True, "no_reset": True}

        def post(self, path: str, payload: object):
            self.posts.append((path, payload))

        def close(self) -> None:
            pass

    monkeypatch.setattr(api, "GameClient", FakeClient)
    monkeypatch.setattr(api, "load_token", lambda _: "redacted-test-token")
    import pytest

    with pytest.raises(RuntimeError, match="no_reset"):
        api.practice_benchmark(
            "practice",
            ["greedy"],
            tmp_path / ".env",
            tmp_path / "state",
            tmp_path / "report",
            quiet=True,
        )


def test_peer_baselines_distinguish_completed_partial_and_unstarted() -> None:
    class FakeClient:
        def get(self, path: str, game_id: str):
            team_id = game_id.rsplit(":", 1)[-1]
            submitted = {"18": 7, "8": 1, "7": 0}[team_id]
            if path == "/game/practice/peer":
                return {
                    "days": [
                        {
                            "day": day,
                            "teams": [
                                {"team_id": team_id, "submitted": True}
                            ],
                        }
                        for day in range(submitted)
                    ]
                }
            if path == "/game/practice/score":
                return {
                    "detail": {
                        team_id: {
                            "distinct_types": 4 if submitted else 0,
                            "cumulative_daily_types": submitted * 3,
                            "total_servings": submitted * 6,
                            "cumulative_response_time": 0.0,
                        }
                    }
                }
            raise AssertionError(path)

    peers = api.collect_peer_baselines(
        FakeClient(), "question", "13", ["18", "8", "7", "13"], 7
    )
    assert {row["team_id"]: row["status"] for row in peers} == {
        "18": "completed",
        "8": "partial",
        "7": "not_started",
    }


def test_discover_practice_questions_only_returns_resettable_maps(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self):
            def row(question_id: str, practice: bool, no_reset: bool):
                return {
                    "id": question_id,
                    "name": question_id,
                    "question_data": json.dumps(
                        {
                            "is_practice": practice,
                            "no_reset": no_reset,
                            "map": {"width": 12, "height": 13},
                            "daySteps": [10, 10],
                            "teams": [{"team_id": "13"}, {"team_id": "18"}],
                        }
                    ),
                }

            return {
                "data": [
                    row("safe", True, False),
                    row("real", False, False),
                    row("locked", True, True),
                ]
            }

    monkeypatch.setattr(api.httpx, "get", lambda *args, **kwargs: FakeResponse())
    questions = api.discover_practice_questions("not-a-jwt", "https://example.invalid/api")
    assert questions == [
        {
            "question_id": "safe",
            "name": "safe",
            "width": 12,
            "height": 13,
            "total_days": 2,
            "team_ids": ["13", "18"],
        }
    ]


def test_practice_suite_reports_completed_and_provisional_ranks(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(api, "load_token", lambda _: "redacted-test-token")
    monkeypatch.setattr(
        api,
        "discover_practice_questions",
        lambda token, base_url: [
            {
                "question_id": "question",
                "name": "Q01",
                "width": 12,
                "height": 12,
                "total_days": 7,
                "team_ids": ["13", "18", "19", "20", "21"],
            }
        ],
    )

    def score(team_id: str, status: str, distinct: int, daily: int, servings: int):
        return {
            "team_id": team_id,
            "status": status,
            "submitted_days": 7 if status == "completed" else 1,
            "total_days": 7,
            "distinct_types": distinct,
            "cumulative_daily_types": daily,
            "total_servings": servings,
            "cumulative_response_time": 0.0,
        }

    def fake_benchmark(*args, **kwargs):
        return {
            "game_id": "question:13",
            "best_policy": "local_search",
            "final_policy": "local_search",
            "results": [
                {
                    "policy": "local_search",
                    "distinct_types": 4,
                    "cumulative_daily_types": 28,
                    "total_servings": 100,
                    "cumulative_response_time": 0.0,
                }
            ],
            "peer_baselines": [
                score("18", "completed", 4, 29, 20),
                score("19", "completed", 4, 20, 200),
                score("20", "partial", 5, 5, 5),
                score("21", "not_started", 0, 0, 0),
            ],
            "completed_peer_comparison": {"wins": 1, "ties": 0, "losses": 1},
        }

    monkeypatch.setattr(api, "practice_benchmark", fake_benchmark)
    summary = api.practice_suite(
        ["local_search"],
        tmp_path / ".env",
        tmp_path / "state",
        tmp_path / "reports",
        quiet=True,
    )
    row = summary["maps"][0]
    assert (row["completed_rank"], row["completed_players"]) == (2, 3)
    assert (row["provisional_rank"], row["configured_players"]) == (3, 5)
    assert summary["completed_peer_comparison"] == {"wins": 1, "ties": 0, "losses": 1}
    assert (tmp_path / "reports" / "summary.json").exists()
    assert (tmp_path / "reports" / "summary.md").exists()
