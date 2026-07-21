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
VALIDATION_PROFILES = ("hard", "medium", "easy")
VALIDATION_CASE_COUNT = 32
VALIDATION_PROFILE_TIERS = {
    "hard": "brutal",
    "medium": "steady",
    "easy": "easy",
}


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


def generate_validation_suite(
    output: Path,
    per_profile: int = VALIDATION_CASE_COUNT,
    profiles: tuple[str, ...] = VALIDATION_PROFILES,
) -> Path:
    """Generate a deterministic, stratified ALNS validation suite.

    The public hard/medium/easy labels map to the existing curated
    brutal/steady/easy recipes. It is intended for component ablations: every
    profile has the same number of cases and no case is selected by its ALNS
    score.
    """
    if per_profile < 1:
        raise ValueError("per_profile must be positive")
    unknown = set(profiles) - set(VALIDATION_PROFILES)
    if unknown:
        raise ValueError(f"unknown validation profile: {sorted(unknown)}")
    output.mkdir(parents=True, exist_ok=True)
    combined_cases: list[dict[str, Any]] = []
    for profile in profiles:
        tier = VALIDATION_PROFILE_TIERS[profile]
        recipe = HARD_RECIPES[tier]
        profile_dir = output / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        cases: list[dict[str, Any]] = []
        for index in range(per_profile):
            scenario = _generate_hard_scenario(recipe.base_seed + index, tier)
            scenario["validation"] = {
                "profile": profile,
                "source_tier": tier,
                "index": index,
                "suite": "alns-validation",
            }
            filename = f"case-{index:04d}.json"
            (profile_dir / filename).write_text(
                json.dumps(scenario, indent=2) + "\n"
            )
            entry = {
                "path": filename,
                "seed": scenario["seed"],
                "profile": profile,
                "source_tier": tier,
                "target": recipe.target,
                "design": scenario["design"],
                "validation": scenario["validation"],
            }
            cases.append(entry)
            combined_cases.append({**entry, "path": f"{profile}/{filename}"})
        (profile_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "suite": f"alns-validation-{profile}",
                    "profile": profile,
                    "source_tier": tier,
                    "target": recipe.target,
                    "case_count": per_profile,
                    "cases": cases,
                },
                indent=2,
            )
            + "\n"
        )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite": "alns-validation",
                "profiles": list(profiles),
                "source_tiers": {
                    profile: VALIDATION_PROFILE_TIERS[profile]
                    for profile in profiles
                },
                "cases_per_profile": per_profile,
                "cases": combined_cases,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Hard curated suite
#
# The random `quick`/`full` suites are too easy: the reference solver saturates
# the structural optimum of every objective, so the grade cannot separate a
# strong policy from a weak one. The hard suite fixes that by constructing three
# graded tiers, each engineered so that exactly one objective stays below 100%
# while every higher-priority objective is fully reachable:
#
#   * "brutal"  -> distinct types unreachable (the hardest tier). Large map,
#                  every spot a distinct brand spread to far corners, minimal
#                  fuel/agents/steps: even one bowl of every type is impossible.
#   * "steady"  -> distinct reachable across the match, but the cumulative daily
#                  types cannot be maxed: no single day can touch every brand.
#   * "easy"    -> every brand every day is reachable, but total servings cannot
#                  be drained: high stocks and many spots outrun the fleet.
#
# Each candidate is verified with the strongest policy (ALNS): only cases whose
# measured percentages land in the tier's target band are kept, so the suite is
# genuinely discriminating rather than merely nominally "hard".
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardRecipe:
    target: str  # objective (runner.OBJECTIVES key) engineered to stay < 100%
    height: tuple[int, int]
    width: tuple[int, int]
    agents: tuple[int, int]
    days: tuple[int, int]
    spots: tuple[int, int]
    brands: tuple[int, int] | None  # None => one distinct brand per spot
    stocks: str  # "one" or "max"
    fuel: tuple[int, int]  # absolute fuel-limit range (clamped to legal bounds)
    day_step_mult: tuple[float, float]  # multiples of (width + height)
    terrain: TerrainMix
    band: tuple[float, float]  # inclusive [low, high] percent band for `target`
    base_seed: int


# The bands are strict-below-100 with a floor so a case still rewards better
# policies instead of collapsing to noise. Terrain is mountain/road/pond heavy
# for the travel-bound tiers and gentle for the servings tier (whose difficulty
# comes from stock volume, not distance).
HARD_RECIPES: dict[str, HardRecipe] = {
    "brutal": HardRecipe(
        target="distinct_types",
        height=(28, 32),
        width=(28, 32),
        agents=(3, 4),
        days=(4, 5),
        spots=(10, 14),
        brands=None,
        stocks="one",
        fuel=(12, 20),
        day_step_mult=(1.0, 1.0),
        terrain=TerrainMix(0.05, 0.20, 0.60, 0.15),
        band=(15.0, 92.0),
        base_seed=3_000_000,
    ),
    "steady": HardRecipe(
        target="cumulative_daily_types",
        height=(24, 30),
        width=(24, 30),
        agents=(4, 5),
        days=(7, 9),
        spots=(12, 16),
        brands=(5, 7),
        stocks="one",
        fuel=(12, 16),
        day_step_mult=(1.1, 1.4),
        terrain=TerrainMix(0.10, 0.25, 0.50, 0.15),
        band=(30.0, 92.0),
        base_seed=4_000_000,
    ),
    "easy": HardRecipe(
        target="total_servings",
        height=(14, 18),
        width=(14, 18),
        agents=(6, 8),
        days=(5, 7),
        spots=(10, 14),
        brands=(2, 4),
        stocks="max",
        fuel=(40, 80),
        day_step_mult=(2.8, 3.6),
        terrain=TerrainMix(0.45, 0.30, 0.15, 0.10),
        band=(30.0, 92.0),
        base_seed=5_000_000,
    ),
}
HARD_TIERS = tuple(HARD_RECIPES)
# Priority order of objectives (mirrors runner.OBJECTIVES) used to decide which
# objectives must be fully saturated for a tier to be accepted.
HARD_OBJECTIVES = ("distinct_types", "cumulative_daily_types", "total_servings")


def _remove_ponds_avoiding(
    rng: random.Random,
    cells: list[list[int]],
    reserved: set[int],
    target: int,
    minimum_open: int,
) -> None:
    height, width = len(cells), len(cells[0])
    candidates = [pos for pos in range(height * width) if pos not in reserved]
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


def _farthest_first(
    rng: random.Random, positions: list[int], count: int, width: int
) -> list[int]:
    """Greedy max-min sampling: pick `count` positions spread far apart."""
    pool = list(positions)
    rng.shuffle(pool)
    if count >= len(pool):
        return sorted(pool)
    chosen = [pool.pop(rng.randrange(len(pool)))]

    def distance(a: int, b: int) -> int:
        return (a // width - b // width) ** 2 + (a % width - b % width) ** 2

    nearest = {pos: distance(pos, chosen[0]) for pos in pool}
    while len(chosen) < count:
        pick = max(pool, key=lambda pos: nearest[pos])
        pool.remove(pick)
        chosen.append(pick)
        for pos in pool:
            nearest[pos] = min(nearest[pos], distance(pos, pick))
    return sorted(chosen)


def _hard_terrain(
    rng: random.Random, cells: list[list[int]], reserved: set[int], mix: TerrainMix
) -> None:
    """Fill every non-reserved open cell with road/mountain/plain by weight."""
    height, width = len(cells), len(cells[0])
    weights = (mix.plain, mix.road, mix.mountain)
    for pos in range(height * width):
        if pos in reserved:
            continue
        row, column = divmod(pos, width)
        if cells[row][column] == 3:
            continue
        cells[row][column] = rng.choices([0, 1, 2], weights)[0]


def _generate_hard_scenario(seed: int, tier: str) -> dict[str, Any]:
    recipe = HARD_RECIPES[tier]
    rng = random.Random(seed)
    height = rng.randint(*recipe.height)
    width = rng.randint(*recipe.width)
    agents_count = rng.randint(*recipe.agents)
    days = rng.randint(*recipe.days)
    spot_ceiling = min(recipe.spots[1], max(width, height))
    spots_count = rng.randint(min(recipe.spots[0], spot_ceiling), spot_ceiling)
    spots_count = max(spots_count, agents_count)

    cells = [[0 for _ in range(width)] for _ in range(height)]
    reserved_budget = agents_count + spots_count + 1
    _remove_ponds_avoiding(
        rng, cells, set(), round(height * width * recipe.terrain.pond), reserved_budget
    )
    open_positions = [
        pos
        for pos in range(height * width)
        if cells[pos // width][pos % width] != 3
    ]
    # Spots spread to far corners; agents drawn from what remains.
    spot_positions = _farthest_first(rng, open_positions, spots_count, width)
    spot_set = set(spot_positions)
    remaining = [pos for pos in open_positions if pos not in spot_set]
    rng.shuffle(remaining)
    agent_positions = sorted(remaining[:agents_count])
    reserved = spot_set | set(agent_positions)
    _hard_terrain(rng, cells, reserved, recipe.terrain)
    for pos in reserved:  # agents and spots must sit on spot-free plains
        cells[pos // width][pos % width] = 0

    if recipe.brands is None:
        brand_count = spots_count
    else:
        brand_count = rng.randint(recipe.brands[0], min(recipe.brands[1], spots_count))
    brands = list(range(brand_count))
    brands.extend(rng.randrange(brand_count) for _ in range(spots_count - brand_count))
    rng.shuffle(brands)
    stock_value = agents_count if recipe.stocks == "max" else 1
    spots = [
        {"brand": brands[index], "pos": pos, "stocks": min(stock_value, agents_count)}
        for index, pos in enumerate(spot_positions)
    ]

    minimum = width + height
    day_steps = [
        max(
            minimum,
            min(4 * minimum, round(minimum * rng.uniform(*recipe.day_step_mult))),
        )
        for _ in range(days)
    ]
    fuel = max(1, min(3 * day_steps[0], rng.randint(*recipe.fuel)))
    config = {
        "startsAt": 1_700_000_000,
        "daySeconds": [60 for _ in range(days)],
        "daySteps": day_steps,
        "map": {"height": height, "width": width, "cells": cells},
        "spots": spots,
        "agents": agent_positions,
        "fuelLimits": fuel,
        "players": 1,
        "busyThreshold": 1,
        "jammedThreshold": 2,
    }
    validate_config(config)
    return {
        "schema_version": 1,
        "seed": seed,
        "tier": tier,
        "target": recipe.target,
        "profile": "hard",
        "size": f"{height}x{width}",
        "traffic_mode": "single",
        "design": {
            "tier": tier,
            "target": recipe.target,
            "agents": agents_count,
            "days": days,
            "spots": spots_count,
            "brands": brand_count,
            "fuel": fuel,
            "day_steps": day_steps,
        },
        "config": config,
        "opponents": [],
    }


def _objective_percentages(
    scenario: dict[str, Any], binary: Path, policy: str
) -> tuple[dict[str, float], dict[str, Any]]:
    from .runner import run_core, structural_optimum

    result = run_core("eval", policy, scenario, binary=binary, timeout=180)
    optimum = structural_optimum(scenario)
    if result["invalid_days"] != 0:
        return {name: -1.0 for name in HARD_OBJECTIVES}, result
    percentages = {
        name: (
            100.0
            if optimum[name] == 0
            else 100.0 * result["score"][name] / optimum[name]
        )
        for name in HARD_OBJECTIVES
    }
    return percentages, result


def _accepts_band(percentages: dict[str, float], recipe: HardRecipe) -> bool:
    if any(value < 0 for value in percentages.values()):  # invalid days
        return False
    for name in HARD_OBJECTIVES:
        if name == recipe.target:
            low, high = recipe.band
            return low <= percentages[name] <= high
        # Every higher-priority objective must be fully reachable.
        if percentages[name] < 99.999:
            return False
    return False  # target not found (should not happen)


def _write_tier(
    output: Path, tier: str, scenarios: list[dict[str, Any]]
) -> dict[str, Any]:
    tier_dir = output / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    manifest_cases = []
    for index, scenario in enumerate(scenarios):
        filename = f"case-{index:04d}.json"
        (tier_dir / filename).write_text(json.dumps(scenario, indent=2) + "\n")
        manifest_cases.append(
            {
                "path": filename,
                "seed": scenario["seed"],
                "tier": tier,
                "target": scenario["target"],
                "design": scenario["design"],
                "verification": scenario.get("verification"),
            }
        )
    manifest = {
        "schema_version": 1,
        "suite": f"hard-{tier}",
        "tier": tier,
        "target": HARD_RECIPES[tier].target,
        "players": 1,
        "cases": manifest_cases,
    }
    (tier_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def generate_hard_suite(
    output: Path,
    per_tier: int = 6,
    tiers: tuple[str, ...] = HARD_TIERS,
    binary_path: str | None = None,
    verify: bool = True,
    verify_policy: str = "alns",
    max_attempts: int = 80,
) -> Path:
    """Construct the graded hard suite under `output`.

    Writes one sub-suite per tier (`<output>/<tier>/manifest.json`) plus a
    combined `<output>/manifest.json` that grades every tier at once. When
    `verify` is true each candidate is run through `verify_policy` (ALNS by
    default) and only kept if its measured objective percentages fall in the
    tier's target band, guaranteeing the intended objective is unreachable.
    """
    for tier in tiers:
        if tier not in HARD_RECIPES:
            raise ValueError(f"unknown hard tier: {tier}")
    output.mkdir(parents=True, exist_ok=True)
    binary = None
    if verify:
        from .runner import find_binary

        binary = find_binary(binary_path)

    combined_cases: list[dict[str, Any]] = []
    for tier in tiers:
        recipe = HARD_RECIPES[tier]
        accepted: list[dict[str, Any]] = []
        seed = recipe.base_seed
        attempts = 0
        while len(accepted) < per_tier and attempts < max_attempts:
            scenario = _generate_hard_scenario(seed, tier)
            seed += 1
            attempts += 1
            if verify:
                assert binary is not None
                percentages, result = _objective_percentages(
                    scenario, binary, verify_policy
                )
                if not _accepts_band(percentages, recipe):
                    continue
                scenario["verification"] = {
                    "policy": verify_policy,
                    "score": result["score"],
                    "percentages": {
                        name: round(value, 3)
                        for name, value in percentages.items()
                    },
                }
            accepted.append(scenario)
        if len(accepted) < per_tier:
            raise RuntimeError(
                f"hard tier '{tier}': only found {len(accepted)}/{per_tier} "
                f"cases in {attempts} attempts; widen the recipe band"
            )
        _write_tier(output, tier, accepted)
        for index, scenario in enumerate(accepted):
            combined_cases.append(
                {
                    "path": f"{tier}/case-{index:04d}.json",
                    "seed": scenario["seed"],
                    "tier": tier,
                    "target": scenario["target"],
                    "design": scenario["design"],
                    "verification": scenario.get("verification"),
                }
            )

    manifest = {
        "schema_version": 1,
        "suite": "hard",
        "tiers": list(tiers),
        "targets": {tier: HARD_RECIPES[tier].target for tier in tiers},
        "verified_with": verify_policy if verify else None,
        "players": 1,
        "cases": combined_cases,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path
