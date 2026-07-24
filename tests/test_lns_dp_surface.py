from hexbench.api import (
    LNS_DP_TUNED_DEFAULTS,
    POLICY_HYPERPARAMETERS,
    STATEFUL_POLICIES,
    normalize_hyperparameters,
)
from hexbench.web import POLICIES


def test_lns_dp_is_a_stateful_full_surface_policy() -> None:
    assert "lns_dp" in POLICIES
    assert "lns_dp" in STATEFUL_POLICIES
    assert "lns_dp" in POLICY_HYPERPARAMETERS


def test_lns_dp_defaults_and_small_tuning_surface() -> None:
    normalized = normalize_hyperparameters(["lns_dp"], None)
    assert normalized["lns_dp"] == LNS_DP_TUNED_DEFAULTS
    assert {field["key"] for field in POLICY_HYPERPARAMETERS["lns_dp"]} == {
        "time_limit_ms",
        "min_iterations",
        "max_iterations",
        "stagnation_iterations",
        "future_discount_percent",
    }


def test_lns_dp_validates_iteration_bounds() -> None:
    try:
        normalize_hyperparameters(
            ["lns_dp"],
            {"lns_dp": {"min_iterations": 5, "max_iterations": 4}},
        )
    except ValueError as error:
        assert "min_iterations cannot exceed max_iterations" in str(error)
    else:
        raise AssertionError("invalid LNS-DP iteration bounds were accepted")


def test_lns_dp_proposal_toggle_is_exposed_for_hybrid_solvers() -> None:
    for method in ("alns", "palns", "mlns"):
        fields = {
            field["key"]: field for field in POLICY_HYPERPARAMETERS[method]
        }
        assert fields["use_lns_dp_proposals"]["type"] == "boolean"
        normalized = normalize_hyperparameters(
            [method], {method: {"use_lns_dp_proposals": True}}
        )
        assert normalized[method]["use_lns_dp_proposals"] is True


def test_lns_dp_proposal_toggle_rejects_numeric_substitutes() -> None:
    try:
        normalize_hyperparameters(
            ["alns"], {"alns": {"use_lns_dp_proposals": 1}}
        )
    except ValueError as error:
        assert "must be boolean" in str(error)
    else:
        raise AssertionError("numeric LNS-DP proposal toggle was accepted")
