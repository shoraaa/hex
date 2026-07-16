from __future__ import annotations

import json
from itertools import combinations, product
from pathlib import Path

from hexbench.generator import SUITE_FACTORS, generate_scenario, generate_suite
from hexbench.models import is_connected, validate_config
from hexbench.runner import (
    find_binary,
    grade_suite,
    normalized_performance,
    run_core,
    structural_optimum,
)


def test_seeded_generation_is_deterministic_and_valid() -> None:
    for seed in range(12):
        first = generate_scenario(seed, "medium", "small", "single")
        second = generate_scenario(seed, "medium", "small", "single")
        assert first == second
        validate_config(first["config"])
        assert is_connected(first["config"]["map"]["cells"])


def test_quick_manifest_and_core_evaluation(tmp_path) -> None:
    manifest_path = generate_suite("quick", tmp_path)
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["cases"]) == 30
    scenario = json.loads((tmp_path / manifest["cases"][0]["path"]).read_text())
    result = run_core("eval", "greedy", scenario, binary=find_binary())
    assert result["invalid_days"] == 0
    assert result["valid_days"] == len(scenario["config"]["daySteps"])


def test_full_suite_covers_independent_initial_map_and_match_factors(tmp_path) -> None:
    manifest_path = generate_suite("full", tmp_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["design_version"] == 2
    assert manifest["coverage"] == "balanced-pairwise"
    assert manifest["players"] == 16
    assert len(manifest["cases"]) == 1_000

    designs = [case["design"] for case in manifest["cases"]]
    for left, right in combinations(SUITE_FACTORS, 2):
        expected = set(product(SUITE_FACTORS[left], SUITE_FACTORS[right]))
        observed = {(row[left], row[right]) for row in designs}
        assert observed == expected, (left, right, expected - observed)

    scenarios = [
        json.loads((tmp_path / case["path"]).read_text())
        for case in manifest["cases"]
    ]
    assert all(scenario["config"]["players"] == 16 for scenario in scenarios)
    assert all(len(scenario["opponents"]) == 15 for scenario in scenarios)
    assert all(
        set(scenario["opponents"]) <= {"wait", "greedy", "hotspot"}
        for scenario in scenarios
    )
    assert {len(scenario["config"]["agents"]) for scenario in scenarios} == set(
        range(3, 9)
    )
    assert {len(scenario["config"]["daySteps"]) for scenario in scenarios} == set(
        range(4, 11)
    )
    normalized_steps = [
        steps / (scenario["config"]["map"]["height"] + scenario["config"]["map"]["width"])
        for scenario in scenarios
        for steps in scenario["config"]["daySteps"]
    ]
    assert min(normalized_steps) == 1
    assert max(normalized_steps) == 4

    dimensions = [
        (scenario["config"]["map"]["height"], scenario["config"]["map"]["width"])
        for scenario in scenarios
    ]
    assert min(height for height, _ in dimensions) == 8
    assert max(height for height, _ in dimensions) == 32
    assert min(width for _, width in dimensions) == 8
    assert max(width for _, width in dimensions) == 32
    assert any(width / height >= 3 for height, width in dimensions)
    assert any(height / width >= 3 for height, width in dimensions)


def test_converted_policy_registry_runs_same_scenario() -> None:
    scenario = generate_scenario(31, "medium", "small", "single")
    for policy in (
        "greedy",
        "utility_greedy",
        "fuel_aware",
        "stock_maximiser",
        "coordinated",
        "local_search",
        "lns",
        "alns",
        "aco",
        "aco_ls",
    ):
        result = run_core("eval", policy, scenario, binary=find_binary())
        assert result["invalid_days"] == 0, (policy, result["errors"])


def test_parallel_core_and_grader_are_score_deterministic(tmp_path) -> None:
    scenario = generate_scenario(41, "medium", "small", "single")
    binary = find_binary()
    serial = run_core(
        "eval", "local_search", scenario, binary=binary, core_threads=1
    )
    parallel = run_core(
        "eval", "local_search", scenario, binary=binary, core_threads=4
    )
    assert serial == parallel

    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(scenario))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite": "parallel-test",
                "cases": [{"path": case_path.name, "seed": 41}],
            }
        )
    )
    serial_report = grade_suite(
        manifest_path,
        "local_search",
        ["coordinated"],
        tmp_path / "serial",
        jobs=1,
    )
    parallel_report = grade_suite(
        manifest_path,
        "local_search",
        ["coordinated"],
        tmp_path / "parallel",
        jobs=4,
    )
    assert serial_report["comparisons"] == parallel_report["comparisons"]
    assert serial_report["ranking"] == parallel_report["ranking"]
    for method in ("local_search", "coordinated"):
        assert (
            serial_report["cases"][0]["results"][method]["score"]
            == parallel_report["cases"][0]["results"][method]["score"]
        )


def test_normalized_grade_uses_equal_case_weight_and_invalid_zero() -> None:
    scenario = {
        "config": {
            "daySteps": [10, 10],
            "agents": [0, 1, 2],
            "spots": [
                {"brand": 0, "stocks": 2},
                {"brand": 1, "stocks": 5},
                {"brand": 1, "stocks": 1},
            ],
        }
    }
    optimum = structural_optimum(scenario)
    assert optimum == {
        "distinct_types": 2,
        "cumulative_daily_types": 4,
        "total_servings": 12,
    }
    valid = normalized_performance(
        {
            "invalid_days": 0,
            "score": {
                "distinct_types": 2,
                "cumulative_daily_types": 2,
                "total_servings": 6,
            },
        },
        optimum,
    )
    assert valid["objective_percentages"] == {
        "distinct_types": 100.0,
        "cumulative_daily_types": 50.0,
        "total_servings": 50.0,
    }
    invalid = normalized_performance(
        {
            "invalid_days": 1,
            "score": {
                "distinct_types": 2,
                "cumulative_daily_types": 4,
                "total_servings": 12,
            },
        },
        optimum,
    )
    assert invalid["objective_percentages"] == {
        "distinct_types": 0.0,
        "cumulative_daily_types": 0.0,
        "total_servings": 0.0,
    }


def test_grader_ranks_average_percentages_lexicographically(
    monkeypatch, tmp_path: Path
) -> None:
    scenario = {
        "config": {
            "daySteps": [1],
            "agents": [0],
            "spots": [
                {"brand": 0, "stocks": 1},
                {"brand": 1, "stocks": 1},
            ],
        }
    }
    (tmp_path / "case.json").write_text(json.dumps(scenario))
    manifest = {
        "schema_version": 1,
        "suite": "lexicographic-average-test",
        "cases": [{"path": "case.json", "seed": 1}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    def fake_run_core(command, policy, payload, **kwargs):
        assert command == "eval"
        score = (
            {
                "distinct_types": 2,
                "cumulative_daily_types": 0,
                "total_servings": 0,
            }
            if policy == "priority"
            else {
                "distinct_types": 1,
                "cumulative_daily_types": 2,
                "total_servings": 2,
            }
        )
        return {"invalid_days": 0, "score": score}

    monkeypatch.setattr("hexbench.runner.run_core", fake_run_core)
    report = grade_suite(
        tmp_path / "manifest.json",
        "priority",
        ["lower"],
        tmp_path / "report",
        binary_path=str(find_binary()),
        jobs=1,
    )
    assert report["ranking"] == ["priority", "lower"]
    assert report["comparisons"]["lower"]["lexicographic_result"] == "win"


def test_lns_is_deterministic_across_core_thread_counts() -> None:
    scenario = generate_scenario(57, "hard", "small", "scripted")
    binary = find_binary()
    serial = run_core("eval", "lns", scenario, binary=binary, core_threads=1)
    parallel = run_core("eval", "lns", scenario, binary=binary, core_threads=4)
    assert serial == parallel
    assert serial["invalid_days"] == 0

    serial_alns = run_core("eval", "alns", scenario, binary=binary, core_threads=1)
    parallel_alns = run_core("eval", "alns", scenario, binary=binary, core_threads=4)
    assert serial_alns == parallel_alns
    assert serial_alns["invalid_days"] == 0


def test_alns_static_split_preserves_sparse_map_support() -> None:
    binary = find_binary()
    compact = generate_scenario(20106, "hard", "small", "multi")
    compact_types = run_core("types", "alns", compact["config"], binary=binary)
    assert compact_types.count(1) == 1

    sparse = generate_scenario(22107, "hard", "large", "multi")
    sparse_types = run_core("types", "alns", sparse["config"], binary=binary)
    assert sparse_types.count(1) == 2


def test_alns_type_selection_accepts_the_published_day_schedule() -> None:
    binary = find_binary()
    # Exercise both the published variable schedule and a legal all-minimum
    # schedule through the one-time role-selection path.
    scenario = generate_scenario(13, "hard", "large", "multi")
    config = scenario["config"]
    expected = run_core("types", "alns", config, binary=binary)

    altered = json.loads(json.dumps(config))
    minimum = altered["map"]["width"] + altered["map"]["height"]
    altered["daySteps"] = [minimum] * len(altered["daySteps"])
    altered["daySeconds"] = [60.0] * len(altered["daySteps"])
    changed = run_core("types", "alns", altered, binary=binary)
    assert len(expected) == len(config["agents"])
    assert len(changed) == len(config["agents"])
    assert set(expected) <= {0, 1}
    assert set(changed) <= {0, 1}


def test_q01_online_alns_uses_the_protected_aco_ls_type_assignment() -> None:
    scenario_path = (
        Path(__file__).resolve().parents[1]
        / "reports/fuel-stress-current/cases"
        / "d2d87157-9158-484f-be37-814a0cf44524-server.json"
    )
    config = json.loads(scenario_path.read_text())["config"]
    binary = find_binary()
    aco_types = run_core("types", "aco_ls", config, binary=binary)
    alns_types = run_core("types", "alns", config, binary=binary)
    assert aco_types == [0, 0, 1, 0]
    assert alns_types == aco_types


def test_q01_alns_seed_ignores_refuel_agent_fuel() -> None:
    scenario_path = (
        Path(__file__).resolve().parents[1]
        / "reports/fuel-stress-current/cases"
        / "d2d87157-9158-484f-be37-814a0cf44524-server.json"
    )
    config = json.loads(scenario_path.read_text())["config"]
    types = [0, 0, 1, 0]
    roads = [
        {"pos": row * config["map"]["width"] + column, "status": 0}
        for row, cells in enumerate(config["map"]["cells"])
        for column, terrain in enumerate(cells)
        if terrain == 1
    ]
    day_info = {
        "day": 0,
        "agents": [
            {"kind": kind, "pos": pos, "fuel": config["fuelLimits"]}
            for kind, pos in zip(types, config["agents"], strict=True)
        ],
        "others": [],
        "traffics": roads,
    }
    search = {
        "minIterations": 96,
        "maxIterations": 96,
        "stagnationIterations": 96,
    }
    payload = {
        "config": config,
        "day_info": day_info,
        "history": {},
        "types": types,
        "search": search,
    }
    binary = find_binary()
    local_plan = run_core("plan", "alns", payload, binary=binary)
    payload["day_info"]["agents"][2]["fuel"] = 0
    online_plan = run_core("plan", "alns", payload, binary=binary)
    assert online_plan == local_plan


def test_q01_alns_recommended_profile_preserves_saved_online_serving_target() -> None:
    scenario_path = (
        Path(__file__).resolve().parents[1]
        / "reports/fuel-stress-current/cases"
        / "d2d87157-9158-484f-be37-814a0cf44524-server.json"
    )
    scenario = json.loads(scenario_path.read_text())
    scenario["hyperparameters"] = {
        "alns_iterations": 1536,
        "final_alns_iterations": 1024,
        "min_iterations": 1536,
        "stagnation_iterations": 2048,
        "seed_iterations": 2048,
        "exact_nodes": 512,
        "final_exact_nodes": 1024,
    }
    result = run_core("eval", "alns", scenario, binary=find_binary())
    assert result["score"] == {
        "distinct_types": 4,
        "cumulative_daily_types": 28,
        "total_servings": 126,
    }


def test_new_question_alns_reaches_the_online_219_serving_baseline() -> None:
    scenario_path = (
        Path(__file__).resolve().parents[1]
        / "reports/fuel-stress-current/cases"
        / "52962f8f-4ac3-4587-9493-9c45ae947243-server.json"
    )
    scenario = json.loads(scenario_path.read_text())
    scenario["hyperparameters"] = {
        "alns_iterations": 1536,
        "final_alns_iterations": 1024,
        "min_iterations": 1536,
        "stagnation_iterations": 2048,
        "seed_iterations": 2048,
        "exact_nodes": 512,
        "final_exact_nodes": 1024,
    }
    result = run_core("eval", "alns", scenario, binary=find_binary())
    assert result["score"] == {
        "distinct_types": 5,
        "cumulative_daily_types": 35,
        "total_servings": 219,
    }


def test_all_planners_accept_published_future_day_lengths() -> None:
    scenario_path = (
        Path(__file__).resolve().parents[1]
        / "reports/fuel-stress-current/cases"
        / "d2d87157-9158-484f-be37-814a0cf44524-server.json"
    )
    config = json.loads(scenario_path.read_text())["config"]
    binary = find_binary()
    policies = (
        "greedy",
        "utility_greedy",
        "fuel_aware",
        "stock_maximiser",
        "coordinated",
        "local_search",
        "lns",
        "alns",
        "aco",
        "aco_ls",
    )
    altered = json.loads(json.dumps(config))
    minimum = config["map"]["width"] + config["map"]["height"]
    maximum = 4 * minimum
    altered["daySteps"][1:] = [
        maximum if index % 2 == 0 else minimum
        for index in range(len(altered["daySteps"]) - 1)
    ]
    altered["daySeconds"][1:] = [60.0] * (len(altered["daySteps"]) - 1)
    for policy in policies:
        types = run_core("types", policy, config, binary=binary)
        day = {
            "day": 0,
            "agents": [
                {"kind": kind, "pos": pos, "fuel": config["fuelLimits"]}
                for kind, pos in zip(types, config["agents"], strict=True)
            ],
            "others": [],
            "traffics": [
                {
                    "pos": row * config["map"]["width"] + column,
                    "status": 0,
                }
                for row, cells in enumerate(config["map"]["cells"])
                for column, terrain in enumerate(cells)
                if terrain == 1
            ],
        }
        def payload(view):
            return {
                "config": view,
                "day_info": day,
                "history": {"distinct_brands": [], "submitted_actions": []},
                "types": types,
                "search": {
                    "minIterations": 32,
                    "maxIterations": 96,
                    "stagnationIterations": 96,
                },
            }
        original = run_core("plan", policy, payload(config), binary=binary)
        changed = run_core("plan", policy, payload(altered), binary=binary)
        # Day 0 remains the same horizon, so both plans must stay legal even
        # when future horizons are very different. A policy may or may not
        # change its first-day route depending on whether continuation search
        # finds a different ending state.
        assert original and changed


def test_aco_is_deterministic_across_core_thread_counts() -> None:
    scenario = generate_scenario(73, "hard", "small", "scripted")
    binary = find_binary()
    config = scenario["config"]
    types = run_core("types", "aco", config, binary=binary)
    day_info = {
        "day": 0,
        "agents": [
            {"kind": kind, "pos": pos, "fuel": config["fuelLimits"]}
            for kind, pos in zip(types, config["agents"], strict=True)
        ],
        "others": [],
        "traffics": [
            {"pos": row * config["map"]["width"] + column, "status": 0}
            for row, cells in enumerate(config["map"]["cells"])
            for column, terrain in enumerate(cells)
            if terrain == 1
        ],
    }
    payload = {
        "config": config,
        "day_info": day_info,
        "history": {"distinct_brands": []},
        "types": types,
    }
    serial_plan = run_core("plan", "aco", payload, binary=binary, core_threads=1)
    parallel_plan = run_core("plan", "aco", payload, binary=binary, core_threads=4)
    assert serial_plan == parallel_plan
    serial_ls_plan = run_core("plan", "aco_ls", payload, binary=binary, core_threads=1)
    parallel_ls_plan = run_core("plan", "aco_ls", payload, binary=binary, core_threads=4)
    assert serial_ls_plan == parallel_ls_plan
    serial = run_core("eval", "aco", scenario, binary=binary, core_threads=1)
    parallel = run_core("eval", "aco", scenario, binary=binary, core_threads=4)
    assert serial == parallel
    assert serial["invalid_days"] == 0
