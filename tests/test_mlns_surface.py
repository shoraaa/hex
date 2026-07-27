from hexbench.api import (
    MLNS_TUNED_DEFAULTS,
    normalize_hyperparameters,
    solver_time_limit_ms,
)
from hexbench.runner import _traffic_gnn_checkpoint


def test_mlns_defaults_are_deadline_governed() -> None:
    normalized = normalize_hyperparameters(["mlns"], {})["mlns"]
    assert normalized == MLNS_TUNED_DEFAULTS
    assert normalized["stagnation_iterations"] == 0


def test_mlns_uses_complete_safe_day_window() -> None:
    assert solver_time_limit_ms("mlns", 58.0) == 55_100
    assert solver_time_limit_ms("alns", 58.0) == 49_300


def test_mlns_accepts_optional_gnn_prediction() -> None:
    normalized = normalize_hyperparameters(
        ["mlns"], {"mlns": {"use_traffic_gnn": True}}
    )["mlns"]

    assert normalized["use_traffic_gnn"] is True


def test_mlns_uses_bundled_pretrained_gnn_by_default(monkeypatch) -> None:
    monkeypatch.delenv("HEXUDON_TRAFFIC_GNN_CHECKPOINT", raising=False)

    checkpoint = _traffic_gnn_checkpoint({})

    assert checkpoint.as_posix().endswith("reports/traffic-1k-test/model.pt")
