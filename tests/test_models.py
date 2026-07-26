from __future__ import annotations

from copy import deepcopy

from hexbench.generator import generate_scenario
from hexbench.models import validate_config


def test_live_agent_selection_time_limit_is_a_valid_config_extension() -> None:
    config = deepcopy(generate_scenario(1, "easy", "small", "single")["config"])
    config["agent_selection_time_limit"] = 30.0

    validate_config(config)
