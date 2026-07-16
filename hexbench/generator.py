from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from .models import is_connected, validate_config


@dataclass(frozen=True)
class Profile:
    plain: float
    road: float
    mountain: float
    pond: float
    fuel_low: float
    fuel_high: float


PROFILES = {
    "easy": Profile(0.55, 0.30, 0.10, 0.05, 2.0, 3.0),
    "medium": Profile(0.40, 0.25, 0.20, 0.15, 1.0, 2.0),
    "hard": Profile(0.30, 0.20, 0.25, 0.25, 1 / 3, 1.0),
}
SIZE_RANGES = {"small": (8, 12), "medium": (13, 22), "large": (23, 32)}
OPPONENTS = ["wait", "greedy", "hotspot"]


@dataclass(frozen=True)
class TerrainMix:
    plain: float
    road: float
    mountain: float
    pond: float


TERRAIN_MIXES = {
    "plain": TerrainMix(0.60, 0.20, 0.10, 0.10),
    "road": TerrainMix(0.25, 0.50, 0.10, 0.15),
    "mountain": TerrainMix(0.25, 0.15, 0.45, 0.15),
    "pond": TerrainMix(0.35, 0.15, 0.15, 0.35),
    "balanced": TerrainMix(0.30, 0.25, 0.25, 0.20),
    "practice": TerrainMix(0.45, 0.25, 0.15, 0.15),
}
FUEL_RATIOS = (0.10, 0.25, 0.50, 1.0, 2.0, 3.0)
SHAPE_RANGES = {
    "small_square": ((8, 12), (8, 12)),
    "medium_square": ((13, 22), (13, 22)),
    "large_square": ((23, 32), (23, 32)),
    "wide": ((13, 20), (24, 32)),
    "tall": ((24, 32), (13, 20)),
    "extreme_wide": ((8, 10), (30, 32)),
    "extreme_tall": ((30, 32), (8, 10)),
}
DAY_STEP_PATTERNS = (
    "minimum",
    "short",
    "medium",
    "long",
    "maximum",
    "alternating",
    "variable",
)
TRAFFIC_PATTERNS = {
    "balanced16": tuple(["wait"] * 5 + ["greedy"] * 5 + ["hotspot"] * 5),
    "mobile16": tuple(["greedy"] * 8 + ["hotspot"] * 7),
    "greedy16": tuple(["greedy"] * 15),
    "hotspot16": tuple(["hotspot"] * 15),
}
TRAFFIC_THRESHOLDS = {
    "sensitive": (1, 2),
    "practice": (2, 5),
    "tolerant": (5, 10),
}
TOPOLOGIES = ("scattered", "clustered", "corridor")
SPOT_DENSITIES = ("sparse", "medium", "dense")

SUITE_FACTORS: dict[str, tuple[Any, ...]] = {
    "terrain": tuple(TERRAIN_MIXES),
    "fuel_ratio": FUEL_RATIOS,
    "agents": tuple(range(3, 9)),
    "shape": tuple(SHAPE_RANGES),
    "days": tuple(range(4, 11)),
    "day_steps": DAY_STEP_PATTERNS,
    "traffic": tuple(TRAFFIC_PATTERNS),
    "traffic_thresholds": tuple(TRAFFIC_THRESHOLDS),
    "topology": TOPOLOGIES,
    "spots": SPOT_DENSITIES,
}
SUITE_CASE_COUNTS = {"quick": 30, "full": 1_000}


def _remove_ponds(
    rng: random.Random, cells: list[list[int]], target: int, minimum_open: int
) -> None:
    height, width = len(cells), len(cells[0])
    candidates = list(range(height * width))
    rng.shuffle(candidates)
    removed = 0
    for pos in candidates:
        if removed >= target or height * width - removed <= minimum_open:
            break
        row, column = divmod(pos, width)
        cells[row][column] = 3
        if is_connected(cells):
            removed += 1
        else:
            cells[row][column] = 0


def generate_scenario(
    seed: int,
    profile_name: str,
    size_name: str,
    traffic_mode: str,
) -> dict[str, Any]:
    rng = random.Random(seed)
    profile = PROFILES[profile_name]
    low, high = SIZE_RANGES[size_name]
    height, width = rng.randint(low, high), rng.randint(low, high)
    agents_count = rng.randint(3, 8)
    spots_count = rng.randint(agents_count, max(width, height))
    cells = [[0 for _ in range(width)] for _ in range(height)]
    pond_target = round(height * width * profile.pond)
    _remove_ponds(rng, cells, pond_target, agents_count + spots_count + 1)

    open_positions = [
        pos
        for pos in range(height * width)
        if cells[pos // width][pos % width] != 3
    ]
    rng.shuffle(open_positions)
    reserved = open_positions[: agents_count + spots_count]
    agent_positions = sorted(reserved[:agents_count])
    spot_positions = sorted(reserved[agents_count:])
    reserved_set = set(reserved)
    traversable_weights = [profile.plain, profile.road, profile.mountain]
    for pos in open_positions:
        row, column = divmod(pos, width)
        if pos not in reserved_set:
            cells[row][column] = rng.choices([0, 1, 2], traversable_weights)[0]

    brand_count = rng.randint(1, spots_count)
    brands = list(range(brand_count))
    brands.extend(rng.randrange(brand_count) for _ in range(spots_count - brand_count))
    rng.shuffle(brands)
    spots = [
        {"brand": brands[index], "pos": pos, "stocks": rng.randint(1, agents_count)}
        for index, pos in enumerate(spot_positions)
    ]
    days = rng.randint(4, 10)
    min_steps, max_steps = width + height, 4 * (width + height)
    day_steps = [rng.randint(min_steps, max_steps) for _ in range(days)]
    fuel_low = max(1, round(profile.fuel_low * day_steps[0]))
    fuel_high = min(3 * day_steps[0], max(fuel_low, round(profile.fuel_high * day_steps[0])))
    busy = rng.randint(1, 5)
    jammed = rng.randint(max(2, busy + 1), 10)
    players = 1 if traffic_mode == "single" else 4
    config = {
        "startsAt": 1_700_000_000,
        "daySeconds": [60 for _ in range(days)],
        "daySteps": day_steps,
        "map": {"height": height, "width": width, "cells": cells},
        "spots": spots,
        "agents": agent_positions,
        "fuelLimits": rng.randint(fuel_low, fuel_high),
        "players": players,
        "busyThreshold": busy,
        "jammedThreshold": jammed,
    }
    validate_config(config)
    return {
        "schema_version": 1,
        "seed": seed,
        "profile": profile_name,
        "size": size_name,
        "traffic_mode": traffic_mode,
        "config": config,
        "opponents": [] if players == 1 else OPPONENTS,
    }


def _balanced_covering_design(count: int, seed: int) -> list[dict[str, Any]]:
    """Build balanced rows and, for the full suite, cover every factor pair."""
    names = tuple(SUITE_FACTORS)
    universe = {
        (left, left_value, right, right_value)
        for left, right in combinations(names, 2)
        for left_value in SUITE_FACTORS[left]
        for right_value in SUITE_FACTORS[right]
    }
    best_rows: list[dict[str, Any]] = []
    best_missing = len(universe)
    attempts = 2_000 if count >= 192 else 200
    for attempt in range(attempts):
        rng = random.Random(seed + attempt)
        columns: dict[str, list[Any]] = {}
        for name, levels in SUITE_FACTORS.items():
            repeats = math.ceil(count / len(levels))
            column = list(levels * repeats)[:count]
            rng.shuffle(column)
            columns[name] = column
        rows = [
            {name: columns[name][index] for name in names} for index in range(count)
        ]
        covered = {
            (left, row[left], right, row[right])
            for row in rows
            for left, right in combinations(names, 2)
        }
        missing = len(universe - covered)
        if missing < best_missing:
            best_rows, best_missing = rows, missing
            if missing == 0:
                break
    if count >= 192 and best_missing:
        raise RuntimeError(f"could not construct pairwise suite; {best_missing} pairs missing")
    return best_rows


def _suite_dimensions(rng: random.Random, shape: str) -> tuple[int, int]:
    (height_low, height_high), (width_low, width_high) = SHAPE_RANGES[shape]
    return rng.randint(height_low, height_high), rng.randint(width_low, width_high)


def _suite_pond_candidates(
    rng: random.Random, height: int, width: int, topology: str
) -> list[int]:
    candidates = list(range(height * width))
    if topology == "scattered":
        rng.shuffle(candidates)
        return candidates
    if topology == "clustered":
        centers = [
            (rng.randrange(height), rng.randrange(width)),
            (rng.randrange(height), rng.randrange(width)),
        ]
        noise = {pos: rng.random() for pos in candidates}
        candidates.sort(
            key=lambda pos: min(
                (pos // width - row) ** 2 + (pos % width - column) ** 2
                for row, column in centers
            )
            + noise[pos]
        )
        return candidates
    horizontal = width >= height
    midpoint = (height - 1) / 2 if horizontal else (width - 1) / 2
    noise = {pos: rng.random() for pos in candidates}
    candidates.sort(
        key=lambda pos: (
            abs((pos // width if horizontal else pos % width) - midpoint),
            noise[pos],
        )
    )
    return candidates


def _remove_suite_ponds(
    rng: random.Random,
    cells: list[list[int]],
    target: int,
    minimum_open: int,
    topology: str,
) -> None:
    height, width = len(cells), len(cells[0])
    removed = 0
    for pos in _suite_pond_candidates(rng, height, width, topology):
        if removed >= target or height * width - removed <= minimum_open:
            break
        row, column = divmod(pos, width)
        cells[row][column] = 3
        if is_connected(cells):
            removed += 1
        else:
            cells[row][column] = 0


def _terrain_counts(total: int, mix: TerrainMix) -> dict[int, int]:
    weights = (mix.plain, mix.road, mix.mountain)
    weight_sum = sum(weights)
    raw = [total * weight / weight_sum for weight in weights]
    counts = [math.floor(value) for value in raw]
    for index in sorted(
        range(3), key=lambda item: raw[item] - counts[item], reverse=True
    )[: total - sum(counts)]:
        counts[index] += 1
    return dict(enumerate(counts))


def _assign_suite_terrain(
    rng: random.Random,
    cells: list[list[int]],
    reserved: set[int],
    mix: TerrainMix,
    topology: str,
) -> None:
    height, width = len(cells), len(cells[0])
    positions = [
        pos
        for pos in range(height * width)
        if cells[pos // width][pos % width] != 3 and pos not in reserved
    ]
    counts = _terrain_counts(len(positions), mix)
    if topology == "scattered":
        labels = [terrain for terrain, count in counts.items() for _ in range(count)]
        rng.shuffle(labels)
        for pos, terrain in zip(positions, labels, strict=True):
            cells[pos // width][pos % width] = terrain
        return

    if topology == "clustered":
        centers = {
            terrain: (rng.randrange(height), rng.randrange(width)) for terrain in range(3)
        }
        rng.shuffle(positions)
        for pos in positions:
            row, column = divmod(pos, width)
            available = [terrain for terrain in range(3) if counts[terrain]]
            terrain = min(
                available,
                key=lambda item: (row - centers[item][0]) ** 2
                + (column - centers[item][1]) ** 2,
            )
            cells[row][column] = terrain
            counts[terrain] -= 1
        return

    horizontal = width >= height
    midpoint = (height - 1) / 2 if horizontal else (width - 1) / 2
    rng.shuffle(positions)
    positions.sort(
        key=lambda pos: abs(
            (pos // width if horizontal else pos % width) - midpoint
        )
    )
    roads = set(positions[: counts[1]])
    remaining = [pos for pos in positions if pos not in roads]
    remaining.sort(
        key=lambda pos: abs(
            (pos // width if horizontal else pos % width) - midpoint
        ),
        reverse=True,
    )
    mountains = set(remaining[: counts[2]])
    for pos in positions:
        terrain = 1 if pos in roads else (2 if pos in mountains else 0)
        cells[pos // width][pos % width] = terrain


def _suite_day_steps(
    rng: random.Random, days: int, minimum: int, pattern: str
) -> list[int]:
    maximum = 4 * minimum
    if pattern == "minimum":
        return [minimum] * days
    if pattern == "maximum":
        return [maximum] * days
    if pattern == "alternating":
        return [minimum if day % 2 == 0 else maximum for day in range(days)]
    if pattern == "variable":
        ratios = [1 + 3 * day / (days - 1) for day in range(days)]
        rng.shuffle(ratios)
        return [round(minimum * ratio) for ratio in ratios]
    bounds = {
        "short": (1.0, 1.5),
        "medium": (1.5, 2.5),
        "long": (2.5, 3.5),
    }
    low, high = bounds[pattern]
    return [round(minimum * rng.uniform(low, high)) for _ in range(days)]


def _suite_spot_count(density: str, agents: int, maximum: int) -> int:
    if density == "sparse":
        return agents
    if density == "dense":
        return maximum
    return round((agents + maximum) / 2)


def _generate_suite_scenario(seed: int, factors: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(seed)
    height, width = _suite_dimensions(rng, factors["shape"])
    agents_count = factors["agents"]
    spots_count = _suite_spot_count(factors["spots"], agents_count, max(width, height))
    mix = TERRAIN_MIXES[factors["terrain"]]
    cells = [[0 for _ in range(width)] for _ in range(height)]
    _remove_suite_ponds(
        rng,
        cells,
        round(height * width * mix.pond),
        agents_count + spots_count + 1,
        factors["topology"],
    )
    open_positions = [
        pos
        for pos in range(height * width)
        if cells[pos // width][pos % width] != 3
    ]
    rng.shuffle(open_positions)
    reserved = open_positions[: agents_count + spots_count]
    agent_positions = sorted(reserved[:agents_count])
    spot_positions = sorted(reserved[agents_count:])
    _assign_suite_terrain(rng, cells, set(reserved), mix, factors["topology"])

    brand_count = rng.randint(1, spots_count)
    brands = list(range(brand_count))
    brands.extend(rng.randrange(brand_count) for _ in range(spots_count - brand_count))
    rng.shuffle(brands)
    spots = [
        {"brand": brands[index], "pos": pos, "stocks": rng.randint(1, agents_count)}
        for index, pos in enumerate(spot_positions)
    ]
    days = factors["days"]
    day_steps = _suite_day_steps(
        rng, days, width + height, factors["day_steps"]
    )
    fuel_limit = max(
        1, min(3 * day_steps[0], round(factors["fuel_ratio"] * day_steps[0]))
    )
    busy, jammed = TRAFFIC_THRESHOLDS[factors["traffic_thresholds"]]
    opponents = list(TRAFFIC_PATTERNS[factors["traffic"]])
    rng.shuffle(opponents)
    config = {
        "startsAt": 1_700_000_000,
        "daySeconds": [60 for _ in range(days)],
        "daySteps": day_steps,
        "map": {"height": height, "width": width, "cells": cells},
        "spots": spots,
        "agents": agent_positions,
        "fuelLimits": fuel_limit,
        "players": 16,
        "busyThreshold": busy,
        "jammedThreshold": jammed,
    }
    validate_config(config)
    return {
        "schema_version": 1,
        "design_version": 2,
        "seed": seed,
        "profile": factors["terrain"],
        "size": factors["shape"],
        "traffic_mode": factors["traffic"],
        "design": factors,
        "config": config,
        "opponents": opponents,
    }


def generate_suite(name: str, output: Path) -> Path:
    if name not in SUITE_CASE_COUNTS:
        raise ValueError("suite must be quick or full")
    output.mkdir(parents=True, exist_ok=True)
    case_count = SUITE_CASE_COUNTS[name]
    design_seed = 0x484558 + (0 if name == "quick" else 100_000)
    design = _balanced_covering_design(case_count, design_seed)
    cases = [
        _generate_suite_scenario(1_000_000 + index, factors)
        for index, factors in enumerate(design)
    ]

    manifest_cases = []
    for index, scenario in enumerate(cases):
        filename = f"case-{index:04d}.json"
        (output / filename).write_text(json.dumps(scenario, indent=2) + "\n")
        manifest_cases.append(
            {
                "path": filename,
                "seed": scenario["seed"],
                "profile": scenario["profile"],
                "size": scenario["size"],
                "traffic_mode": scenario["traffic_mode"],
                "design": scenario["design"],
            }
        )
    manifest = {
        "schema_version": 1,
        "design_version": 2,
        "suite": name,
        "coverage": "balanced-pairwise" if name == "full" else "balanced-smoke",
        "players": 16,
        "cases": manifest_cases,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path
