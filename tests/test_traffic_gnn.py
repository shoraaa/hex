from __future__ import annotations

import torch
from torch.nn import functional as F

import pytest

import hexbench.traffic_gnn as traffic_gnn
from hexbench.cli import build_parser
from hexbench.traffic_gnn import (
    FEATURE_NAMES,
    TRAFFIC_CLASSES,
    TrafficGNN,
    generate_traffic_dataset,
    graph_samples_from_replay,
    load_traffic_dataset,
    make_traffic_scenario,
    predict_future_traffic,
    predict_traffic,
    train_traffic_gnn,
)


def _replay_fixture() -> tuple[dict, dict]:
    scenario = {
        "seed": 7,
        "config": {
            "daySteps": [8, 8, 8],
            "map": {"height": 2, "width": 2, "cells": [[1, 0], [2, 1]]},
            "spots": [{"brand": 0, "pos": 1, "stocks": 1}],
            "agents": [1, 2],
            "fuelLimits": 8,
            "players": 16,
            "busyThreshold": 2,
            "jammedThreshold": 5,
        },
    }

    def day(index: int, left: int, right: int) -> dict:
        return {
            "day": index,
            "road_condition": {"0": left, "3": right},
            "teams": [
                {
                    "frames": [
                        {
                            "step": 0,
                            "agents": [
                                {"cell": 1, "fuel": 8, "type": 0},
                                {"cell": 2, "fuel": 8, "type": 1},
                            ],
                        }
                    ]
                }
            ],
        }

    replay = {"replay": {"days": [day(0, 0, 0), day(1, 1, 0), day(2, 2, 1)]}}
    return scenario, replay


def test_replay_becomes_history_conditioned_road_examples() -> None:
    scenario, replay = _replay_fixture()
    samples = graph_samples_from_replay(scenario, replay)

    assert len(samples) == 2  # Day zero is fixed smooth and is not a target.
    assert samples[0].features.shape == (4, len(FEATURE_NAMES))
    assert samples[0].road_mask.tolist() == [True, False, False, True]
    assert samples[0].labels[samples[0].road_mask].tolist() == [1, 0]
    previous_smooth = FEATURE_NAMES.index("previous_status_smooth")
    previous_busy = FEATURE_NAMES.index("previous_status_busy")
    assert samples[0].features[0, previous_smooth] == 1
    assert samples[1].features[0, previous_busy] == 1
    assert samples[0].edge_index.shape[0] == 2
    assert samples[0].edge_index.shape[1] == samples[0].edge_direction.shape[0]


def test_minimal_gnn_cross_entropy_backward() -> None:
    scenario, replay = _replay_fixture()
    sample = graph_samples_from_replay(scenario, replay)[0]
    model = TrafficGNN(len(FEATURE_NAMES), hidden_size=16, layers=2)

    logits = model(sample.features, sample.edge_index, sample.edge_direction)
    loss = F.cross_entropy(logits[sample.road_mask], sample.labels[sample.road_mask])
    loss.backward()

    assert logits.shape == (4, 3)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_online_scenario_uses_sixteen_default_policy_players() -> None:
    scenario = make_traffic_scenario(11, alns_iterations=3, tier="easy")

    assert scenario["config"]["players"] == 16
    assert scenario["opponents"] == ["lns"] * 15
    assert scenario["searchForAllPlayers"] is True
    assert len(set(scenario["playerSeeds"])) == 16
    assert scenario["search"]["maxIterations"] == 3
    assert scenario["tier"] == "easy"


def test_online_scenario_can_force_alns_policy() -> None:
    scenario = make_traffic_scenario(11, alns_iterations=3, tier="easy", policy="alns")

    assert scenario["opponents"] == ["alns"] * 15
    assert scenario["traffic_mode"] == "alns16"


def test_offline_dataset_roundtrip_and_training_without_simulation(
    tmp_path, monkeypatch
) -> None:
    scenario, replay = _replay_fixture()
    samples = graph_samples_from_replay(scenario, replay)

    def fake_simulation(seeds, **_kwargs):
        summaries = [{"seed": seed, "class_counts": [1, 1, 1]} for seed in seeds]
        return samples, summaries

    monkeypatch.setattr(traffic_gnn, "simulate_online_samples", fake_simulation)
    monkeypatch.setattr(traffic_gnn, "find_binary", lambda _path=None: tmp_path / "core")
    manifest = generate_traffic_dataset(
        output_dir=tmp_path / "dataset",
        train_cases=1,
        validation_cases=1,
        seed=10,
        alns_iterations=2,
    )

    dataset_path = tmp_path / "dataset" / "dataset.pt"
    train, validation, metadata = load_traffic_dataset(dataset_path)
    assert len(train) == len(validation) == 2
    assert manifest["dataset_sha256"] == metadata["dataset_sha256"]

    def simulation_must_not_run(*_args, **_kwargs):
        raise AssertionError("offline training invoked the simulator")

    monkeypatch.setattr(traffic_gnn, "simulate_online_samples", simulation_must_not_run)
    report = train_traffic_gnn(
        dataset_path=dataset_path,
        epochs=1,
        seed=11,
        hidden_size=8,
        layers=1,
        learning_rate=1e-3,
        batch_size=2,
        patience=0,
        minimum_epochs=1,
        device_name="cpu",
        report_dir=tmp_path / "report",
    )
    assert report["kind"] == "offline-lns16-traffic-gnn"
    assert report["train_samples"] == 2
    assert report["best_epoch"] == 1


def test_cli_exposes_offline_generation_and_training() -> None:
    generate = build_parser().parse_args(
        ["traffic-generate", "--train-cases", "2", "--alns-iterations", "3"]
    )
    train = build_parser().parse_args(
        ["traffic-train", "--dataset", "saved.pt", "--epochs", "3"]
    )

    assert generate.command == "traffic-generate"
    assert generate.train_cases == 2
    assert generate.alns_iterations == 3
    assert generate.policy == "lns"
    assert train.command == "traffic-train"
    assert str(train.dataset) == "saved.pt"
    assert train.epochs == 3


def test_predict_traffic_reports_per_day_model_guesses(tmp_path) -> None:
    scenario, replay = _replay_fixture()
    checkpoint_path = tmp_path / "model.pt"
    model = TrafficGNN(len(FEATURE_NAMES), hidden_size=8, layers=1)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "hidden_size": 8,
            "layers": 1,
            "classes": TRAFFIC_CLASSES,
            "dataset_sha256": "test",
            "best_epoch": 1,
        },
        checkpoint_path,
    )

    result = predict_traffic(scenario, replay, checkpoint_path)

    assert result["classes"] == ["smooth", "busy", "jammed"]
    # Day zero is fixed to smooth and is skipped; days 1 and 2 are predicted.
    assert [day["day"] for day in result["days"]] == [1, 2]
    first = result["days"][0]
    # The 2x2 fixture has road cells at positions 0 and 3 only.
    assert first["road_count"] == 2
    assert {cell["pos"] for cell in first["cells"]} == {0, 3}
    assert all(cell["predicted"] in {0, 1, 2} for cell in first["cells"])
    assert all(0.0 <= cell["probability"] <= 1.0 for cell in first["cells"])
    # Ground-truth labels for day 1 are {"0": 1, "3": 0}.
    pos0 = next(cell for cell in first["cells"] if cell["pos"] == 0)
    assert pos0["actual"] == 1
    assert result["confusion"][1][pos0["predicted"]] >= 1
    assert result["road_count"] == 4
    assert result["matched"] + result["confusion"][1][0] + result["confusion"][1][2] >= 1


def test_future_prediction_starts_after_revealed_live_day(tmp_path) -> None:
    scenario, _ = _replay_fixture()
    scenario["day_info"] = {
        "day": 1,
        "agents": [
            {"pos": 1, "fuel": 8, "kind": 0},
            {"pos": 2, "fuel": 8, "kind": 1},
        ],
    }
    checkpoint_path = tmp_path / "model.pt"
    model = TrafficGNN(len(FEATURE_NAMES), hidden_size=8, layers=1)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "hidden_size": 8,
            "layers": 1,
            "classes": TRAFFIC_CLASSES,
        },
        checkpoint_path,
    )

    result = predict_future_traffic(
        scenario,
        checkpoint_path,
        known_day=1,
        known_traffic={0: 2, 3: 1},
    )

    assert [day["day"] for day in result] == [2]


def test_predict_traffic_rejects_incompatible_checkpoint_schema(tmp_path) -> None:
    scenario, replay = _replay_fixture()
    checkpoint_path = tmp_path / "bad.pt"
    torch.save(
        {
            "state_dict": TrafficGNN(len(FEATURE_NAMES), hidden_size=8, layers=1).state_dict(),
            "feature_names": tuple("different") + FEATURE_NAMES,
            "hidden_size": 8,
            "layers": 1,
            "classes": TRAFFIC_CLASSES,
        },
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="feature schema"):
        predict_traffic(scenario, replay, checkpoint_path)
