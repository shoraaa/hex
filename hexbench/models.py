from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"


def load_schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text())


MAP_CONFIG_VALIDATOR = Draft202012Validator(load_schema("map_config.schema.json"))
DAY_INFO_VALIDATOR = Draft202012Validator(load_schema("day_info.schema.json"))
AGENT_TYPES_VALIDATOR = Draft202012Validator(load_schema("agent_types.schema.json"))
ACTION_PLAN_VALIDATOR = Draft202012Validator(load_schema("action_plan.schema.json"))


def neighbors(height: int, width: int, pos: int) -> Iterable[tuple[int, int]]:
    row, column = divmod(pos, width)
    offsets = (
        ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (0, -1))
        if row % 2 == 0
        else ((-1, -1), (-1, 0), (0, 1), (1, 0), (1, -1), (0, -1))
    )
    for direction, (dr, dc) in enumerate(offsets):
        nr, nc = row + dr, column + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield direction, nr * width + nc


def is_connected(cells: list[list[int]]) -> bool:
    height, width = len(cells), len(cells[0])
    traversable = {
        row * width + column
        for row in range(height)
        for column in range(width)
        if cells[row][column] != 3
    }
    if not traversable:
        return False
    reached = {next(iter(traversable))}
    queue = deque(reached)
    while queue:
        pos = queue.popleft()
        for _, nxt in neighbors(height, width, pos):
            if nxt in traversable and nxt not in reached:
                reached.add(nxt)
                queue.append(nxt)
    return reached == traversable


def validate_config(config: dict[str, Any]) -> None:
    MAP_CONFIG_VALIDATOR.validate(config)
    game_map = config["map"]
    height, width, cells = game_map["height"], game_map["width"], game_map["cells"]
    if len(cells) != height or any(len(row) != width for row in cells):
        raise ValueError("map cell dimensions do not match height/width")
    if len(config["daySteps"]) != len(config["daySeconds"]):
        raise ValueError("daySteps and daySeconds lengths differ")
    if not 4 <= len(config["daySteps"]) <= 10:
        raise ValueError("official matches require 4 to 10 days")
    for steps in config["daySteps"]:
        if not width + height <= steps <= 4 * (width + height):
            raise ValueError("daySteps outside official bounds")
    agents = config["agents"]
    spots = config["spots"]
    if not len(agents) <= len(spots) <= max(width, height):
        raise ValueError("spot count outside official bounds")
    spot_positions = [spot["pos"] for spot in spots]
    if len(spot_positions) != len(set(spot_positions)):
        raise ValueError("spot positions must be unique")
    if len(agents) != len(set(agents)):
        raise ValueError("generated agent positions must be unique")
    for pos in agents:
        row, column = divmod(pos, width)
        if cells[row][column] != 0 or pos in spot_positions:
            raise ValueError("agent must start on a spot-free plain")
    for spot in spots:
        row, column = divmod(spot["pos"], width)
        if cells[row][column] != 0 or spot["stocks"] > len(agents):
            raise ValueError("invalid spot terrain or stock")
    if not 1 <= config["fuelLimits"] <= 3 * config["daySteps"][0]:
        raise ValueError("fuel limit outside official bounds")
    if not config["busyThreshold"] < config["jammedThreshold"]:
        raise ValueError("traffic thresholds must be ordered")
    if not is_connected(cells):
        raise ValueError("traversable cells are disconnected")


def validate_day_info(info: dict[str, Any]) -> None:
    DAY_INFO_VALIDATOR.validate(info)


def validate_agent_types(types: list[int], expected_agents: int) -> None:
    AGENT_TYPES_VALIDATOR.validate(types)
    if len(types) != expected_agents:
        raise ValueError("agent type answer length mismatch")


def validate_action_shape(actions: list[list[int]], expected_agents: int) -> None:
    ACTION_PLAN_VALIDATOR.validate(actions)
    if len(actions) != expected_agents:
        raise ValueError("action answer row count mismatch")

