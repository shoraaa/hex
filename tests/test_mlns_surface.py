from hexbench.api import (
    MLNS_TUNED_DEFAULTS,
    normalize_hyperparameters,
    solver_time_limit_ms,
)


def test_mlns_defaults_are_deadline_governed() -> None:
    normalized = normalize_hyperparameters(["mlns"], {})["mlns"]
    assert normalized == MLNS_TUNED_DEFAULTS
    assert normalized["stagnation_iterations"] == 0


def test_mlns_uses_complete_safe_day_window() -> None:
    assert solver_time_limit_ms("mlns", 58.0) == 58_000
    assert solver_time_limit_ms("alns", 58.0) == 49_300
