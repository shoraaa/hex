from __future__ import annotations

import json
from itertools import combinations, product
from pathlib import Path

from hexbench.generator import SUITE_FACTORS, generate_scenario, generate_suite
from hexbench.models import is_connected, validate_config
from hexbench.runner import find_binary, grade_suite, run_core


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
    assert len(manifest["cases"]) == 192

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
    for method in ("local_search", "coordinated"):
        assert (
            serial_report["cases"][0]["results"][method]["score"]
            == parallel_report["cases"][0]["results"][method]["score"]
        )


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


def test_alns_type_selection_does_not_depend_on_day_schedule() -> None:
    binary = find_binary()
    scenario = generate_scenario(20106, "hard", "small", "multi")
    config = scenario["config"]
    expected = run_core("types", "alns", config, binary=binary)

    altered = json.loads(json.dumps(config))
    altered["daySteps"] = [1, 10_000]
    altered["daySeconds"] = [0.1, 600.0]
    assert run_core("types", "alns", altered, binary=binary) == expected


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


def test_all_planners_ignore_unrevealed_future_day_lengths() -> None:
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
    altered["daySteps"][1:] = [9999, 1, 8888, 2, 7777, 3]
    altered["daySeconds"][1:] = [0.1, 600, 0.2, 500, 0.3, 400]
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
        assert run_core("plan", policy, payload(config), binary=binary) == run_core(
            "plan", policy, payload(altered), binary=binary
        )


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
