from hexbench.api import STATEFUL_POLICIES, normalize_hyperparameters
from hexbench.web import POLICIES


def test_simple_lns_is_public_and_stateful() -> None:
    assert "simple_lns" in POLICIES
    assert "simple_lns" in STATEFUL_POLICIES


def test_simple_lns_uses_rolling_search_defaults() -> None:
    normalized = normalize_hyperparameters(["simple_lns"], {})["simple_lns"]
    assert normalized["future_discount_percent"] == 90
    assert normalized["min_iterations"] == 32
    assert normalized["max_iterations"] == 128
    assert normalized["stagnation_iterations"] == 0
