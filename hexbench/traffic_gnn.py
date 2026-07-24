from __future__ import annotations

import json
import hashlib
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .generator import HARD_TIERS, _generate_hard_scenario
from .models import neighbors
from .runner import find_binary, run_core


FEATURE_NAMES = (
    "terrain_plain",
    "terrain_road",
    "terrain_mountain",
    "terrain_pond",
    "axial_q",
    "axial_r",
    "axial_s",
    "is_spot",
    "spot_stock",
    "same_brand_spots",
    "traversable_degree",
    "previous_status_smooth",
    "previous_status_busy",
    "previous_status_jammed",
    "two_days_ago_status_smooth",
    "two_days_ago_status_busy",
    "two_days_ago_status_jammed",
    "patrol_agents_here",
    "refuel_agents_here",
    "patrol_fuel_here",
    "day_fraction",
    "day_steps",
    "busy_threshold",
    "jammed_threshold",
    "players",
)
DATASET_SCHEMA_VERSION = 1
TRAFFIC_CLASSES = ("smooth", "busy", "jammed")


@dataclass(frozen=True)
class TrafficGraphSample:
    """One map at the beginning of one day.

    ``labels`` are the simulator-provided road conditions for this day.  Loss is
    evaluated only where ``road_mask`` is true; keeping all map cells as nodes
    lets ponds and surrounding terrain remain available as positional context.
    """

    features: Tensor
    edge_index: Tensor
    edge_direction: Tensor
    labels: Tensor
    road_mask: Tensor
    instance_seed: int
    day: int

    def to(self, device: torch.device) -> TrafficGraphSample:
        return TrafficGraphSample(
            features=self.features.to(device),
            edge_index=self.edge_index.to(device),
            edge_direction=self.edge_direction.to(device),
            labels=self.labels.to(device),
            road_mask=self.road_mask.to(device),
            instance_seed=self.instance_seed,
            day=self.day,
        )


def _road_status(day: dict[str, Any]) -> dict[int, int]:
    return {int(pos): int(status) for pos, status in day["road_condition"].items()}


def _edge_tensors(config: dict[str, Any]) -> tuple[Tensor, Tensor]:
    game_map = config["map"]
    height, width = int(game_map["height"]), int(game_map["width"])
    terrain = [int(value) for row in game_map["cells"] for value in row]
    sources: list[int] = []
    destinations: list[int] = []
    directions: list[int] = []
    for source, source_terrain in enumerate(terrain):
        if source_terrain == 3:
            continue
        for direction, destination in neighbors(height, width, source):
            if terrain[destination] == 3:
                continue
            sources.append(source)
            destinations.append(destination)
            directions.append(direction)
    return (
        torch.tensor([sources, destinations], dtype=torch.long),
        torch.tensor(directions, dtype=torch.long),
    )


def _start_agents(day: dict[str, Any]) -> list[dict[str, Any]]:
    all_team_starts = day.get("all_team_starts", [])
    if all_team_starts:
        return [
            agent
            for team in all_team_starts
            for agent in team.get("agents", [])
        ]
    teams = day.get("teams", [])
    if not teams:
        return []
    frames = teams[0].get("frames", [])
    if not frames:
        return []
    return list(frames[0].get("agents", []))


def graph_samples_from_replay(
    scenario: dict[str, Any], replay: dict[str, Any]
) -> list[TrafficGraphSample]:
    """Convert an authoritative simulator replay into day-ahead examples.

    Day zero is omitted because its status is fixed to smooth.  For day ``t``,
    features include statuses through ``t - 1`` and the agents' start-of-day
    state, while the real day-``t`` road status is the class label.
    """

    config = scenario["config"]
    game_map = config["map"]
    height, width = int(game_map["height"]), int(game_map["width"])
    node_count = height * width
    terrain = [int(value) for row in game_map["cells"] for value in row]
    road_mask = torch.tensor([value == 1 for value in terrain], dtype=torch.bool)
    edge_index, edge_direction = _edge_tensors(config)
    spots = {int(spot["pos"]): spot for spot in config.get("spots", [])}
    brand_counts: dict[int, int] = {}
    for spot in config.get("spots", []):
        brand = int(spot["brand"])
        brand_counts[brand] = brand_counts.get(brand, 0) + 1

    degree = [0] * node_count
    for source in edge_index[0].tolist():
        degree[source] += 1

    replay_days = list(replay["replay"]["days"])
    statuses = [_road_status(day) for day in replay_days]
    maximum_extent = float(max(1, width - 1, height - 1))
    agent_denominator = float(max(1, len(config["agents"])))
    stock_denominator = agent_denominator
    spot_denominator = float(max(1, len(spots)))
    fuel_limit = float(max(1, int(config["fuelLimits"])))
    total_days = len(config["daySteps"])
    seed = int(scenario.get("seed", 0))

    samples: list[TrafficGraphSample] = []
    for day_index, day in enumerate(replay_days):
        if day_index == 0:
            continue
        previous = statuses[day_index - 1]
        two_days_ago = statuses[day_index - 2] if day_index >= 2 else {}
        patrol_count = [0.0] * node_count
        refuel_count = [0.0] * node_count
        patrol_fuel = [0.0] * node_count
        start_agents = _start_agents(day)
        represented_players = max(1, len(day.get("all_team_starts", [])))
        dynamic_agent_denominator = agent_denominator * represented_players
        for agent in start_agents:
            position = int(agent["cell"])
            if int(agent["type"]) == 0:
                patrol_count[position] += 1.0
                patrol_fuel[position] += float(agent["fuel"]) / fuel_limit
            else:
                refuel_count[position] += 1.0

        features: list[list[float]] = []
        for position in range(node_count):
            row, column = divmod(position, width)
            # Even rows are shifted right.  This is the corresponding even-r
            # offset-to-axial conversion; s makes all six directions explicit.
            q = column - ((row + (row & 1)) // 2)
            axial_r = row
            axial_s = -q - axial_r
            spot = spots.get(position)
            terrain_one_hot = [float(terrain[position] == value) for value in range(4)]
            previous_one_hot = [
                float(previous.get(position, -1) == value) for value in range(3)
            ]
            older_one_hot = [
                float(two_days_ago.get(position, -1) == value) for value in range(3)
            ]
            features.append(
                terrain_one_hot
                + [q / maximum_extent, axial_r / maximum_extent, axial_s / maximum_extent]
                + [
                    float(spot is not None),
                    0.0 if spot is None else float(spot["stocks"]) / stock_denominator,
                    0.0
                    if spot is None
                    else brand_counts[int(spot["brand"])] / spot_denominator,
                    degree[position] / 6.0,
                ]
                + previous_one_hot
                + older_one_hot
                + [
                    patrol_count[position] / dynamic_agent_denominator,
                    refuel_count[position] / dynamic_agent_denominator,
                    patrol_fuel[position] / dynamic_agent_denominator,
                    day_index / float(max(1, total_days - 1)),
                    float(config["daySteps"][day_index])
                    / float(max(1, 4 * (width + height))),
                    float(config["busyThreshold"]) / 10.0,
                    float(config["jammedThreshold"]) / 10.0,
                    float(config["players"]) / 16.0,
                ]
            )

        labels = torch.zeros(node_count, dtype=torch.long)
        for position, status in statuses[day_index].items():
            labels[position] = status
        samples.append(
            TrafficGraphSample(
                features=torch.tensor(features, dtype=torch.float32),
                edge_index=edge_index,
                edge_direction=edge_direction,
                labels=labels,
                road_mask=road_mask,
                instance_seed=seed,
                day=day_index,
            )
        )
    return samples


def make_traffic_scenario(
    seed: int, alns_iterations: int, tier: str = "easy", *, policy: str = "lns"
) -> dict[str, Any]:
    """Generate a graded-tier online instance with exactly 16 search-policy players.

    ``tier`` selects one of the hard-suite recipes (``brutal``/``steady``/
    ``easy``) so the traffic dataset spans discriminating maps instead of the
    too-easy random profiles. ``policy`` is the planner used by every player
    (``lns`` by default; ``alns`` runs the heavier adaptive search). The
    16-player traffic simulation is layered on top of the constructed scenario.
    """

    scenario = _generate_hard_scenario(seed, tier)
    config = scenario["config"]
    cells = config["map"]["cells"]
    if not any(value == 1 for row in cells for value in row):
        reserved = set(config["agents"]) | {
            int(spot["pos"]) for spot in config.get("spots", [])
        }
        width = int(config["map"]["width"])
        replacement = next(
            position
            for position in range(int(config["map"]["height"]) * width)
            if position not in reserved and cells[position // width][position % width] != 3
        )
        cells[replacement // width][replacement % width] = 1
    scenario["traffic_mode"] = f"{policy}16"
    config["players"] = 16
    scenario["opponents"] = [policy] * 15
    seed_random = random.Random(seed ^ 0x414C4E533136)
    player_seeds: list[int] = []
    while len(player_seeds) < 16:
        candidate = seed_random.getrandbits(63)
        if candidate not in player_seeds:
            player_seeds.append(candidate)
    scenario["playerSeeds"] = player_seeds
    scenario["searchForAllPlayers"] = True
    scenario["search"] = {
        "minIterations": 0,
        "maxIterations": alns_iterations,
        "stagnationIterations": 0,
        "seedIterations": 0,
        "finalAlnsIterations": alns_iterations,
        "exactNodes": 0,
        "finalExactNodes": 0,
    }
    return scenario


def simulate_online_samples(
    seeds: Iterable[int],
    *,
    alns_iterations: int,
    binary: Path | None = None,
    timeout: float = 180.0,
    core_threads: int = 1,
    policy: str = "lns",
) -> tuple[list[TrafficGraphSample], list[dict[str, Any]]]:
    binary = binary or find_binary()
    samples: list[TrafficGraphSample] = []
    instance_summaries: list[dict[str, Any]] = []
    for seed in seeds:
        # Balance the three graded tiers across consecutive seeds so the
        # dataset gets an even mix of brutal/steady/easy maps.
        tier = HARD_TIERS[seed % len(HARD_TIERS)]
        scenario = make_traffic_scenario(seed, alns_iterations, tier, policy=policy)
        replay = run_core(
            "visualize",
            policy,
            scenario,
            binary=binary,
            timeout=timeout,
            core_threads=core_threads,
        )
        generated = graph_samples_from_replay(scenario, replay)
        samples.extend(generated)
        class_counts = [0, 0, 0]
        for sample in generated:
            for label in sample.labels[sample.road_mask].tolist():
                class_counts[label] += 1
        instance_summaries.append(
            {
                "seed": seed,
                "tier": tier,
                "profile": scenario["profile"],
                "days": len(scenario["config"]["daySteps"]),
                "samples": len(generated),
                "nodes": scenario["config"]["map"]["height"]
                * scenario["config"]["map"]["width"],
                "roads": sum(
                    value == 1
                    for row in scenario["config"]["map"]["cells"]
                    for value in row
                ),
                "class_counts": class_counts,
                "invalid_days": int(replay["invalid_days"]),
                "player_seeds": list(scenario["playerSeeds"]),
                "unique_player_plans_by_day": [
                    len(
                        {
                            json.dumps(team["actions"], separators=(",", ":"))
                            for team in day.get("all_team_actions", [])
                        }
                    )
                    for day in replay["replay"]["days"]
                ],
            }
        )
    return samples, instance_summaries


def _sample_payload(sample: TrafficGraphSample) -> dict[str, Any]:
    return {
        "features": sample.features.cpu(),
        "edge_index": sample.edge_index.cpu(),
        "edge_direction": sample.edge_direction.cpu(),
        "labels": sample.labels.cpu(),
        "road_mask": sample.road_mask.cpu(),
        "instance_seed": sample.instance_seed,
        "day": sample.day,
    }


def _sample_from_payload(payload: dict[str, Any]) -> TrafficGraphSample:
    sample = TrafficGraphSample(
        features=payload["features"].to(dtype=torch.float32),
        edge_index=payload["edge_index"].to(dtype=torch.long),
        edge_direction=payload["edge_direction"].to(dtype=torch.long),
        labels=payload["labels"].to(dtype=torch.long),
        road_mask=payload["road_mask"].to(dtype=torch.bool),
        instance_seed=int(payload["instance_seed"]),
        day=int(payload["day"]),
    )
    node_count = sample.features.shape[0]
    if sample.features.ndim != 2 or sample.features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("traffic dataset has an incompatible feature tensor")
    if sample.edge_index.ndim != 2 or sample.edge_index.shape[0] != 2:
        raise ValueError("traffic dataset has an invalid edge index")
    if sample.edge_index.shape[1] != sample.edge_direction.numel():
        raise ValueError("traffic dataset edge directions do not match its edges")
    if sample.labels.shape != (node_count,) or sample.road_mask.shape != (node_count,):
        raise ValueError("traffic dataset labels or road mask do not match its nodes")
    if not bool(sample.road_mask.any()):
        raise ValueError("traffic dataset sample has no road labels")
    road_labels = sample.labels[sample.road_mask]
    if int(road_labels.min()) < 0 or int(road_labels.max()) >= len(TRAFFIC_CLASSES):
        raise ValueError("traffic dataset contains an unknown road-status label")
    return sample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_instance_shard(
    path: Path,
    *,
    split: str,
    seed: int,
    alns_iterations: int,
    policy: str,
    samples: list[TrafficGraphSample],
    summary: dict[str, Any],
) -> None:
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "classes": list(TRAFFIC_CLASSES),
        "split": split,
        "seed": seed,
        "alns_iterations": alns_iterations,
        "policy": policy,
        "summary": summary,
        "samples": [_sample_payload(sample) for sample in samples],
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_instance_shard(
    path: Path, *, split: str, seed: int, alns_iterations: int, policy: str
) -> tuple[list[TrafficGraphSample], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or int(payload.get("schema_version", -1)) != DATASET_SCHEMA_VERSION
        or tuple(payload.get("feature_names", ())) != FEATURE_NAMES
        or tuple(payload.get("classes", ())) != TRAFFIC_CLASSES
        or payload.get("split") != split
        or int(payload.get("seed", -1)) != seed
        or int(payload.get("alns_iterations", -1)) != alns_iterations
        or str(payload.get("policy")) != policy
    ):
        raise ValueError(f"incompatible traffic shard: {path}")
    samples = [_sample_from_payload(item) for item in payload.get("samples", [])]
    if not samples:
        raise ValueError(f"traffic shard has no samples: {path}")
    return samples, dict(payload.get("summary", {}))


def generate_traffic_dataset(
    *,
    output_dir: Path,
    train_cases: int,
    validation_cases: int,
    seed: int,
    alns_iterations: int,
    binary_path: str | None = None,
    simulation_timeout: float = 180.0,
    core_threads: int = 1,
    jobs: int = 1,
    overwrite: bool = False,
    policy: str = "lns",
) -> dict[str, Any]:
    """Run expensive simulations once and persist reusable CPU tensors."""

    if train_cases < 1 or validation_cases < 1:
        raise ValueError("train_cases and validation_cases must both be positive")
    if alns_iterations < 0 or jobs < 1 or core_threads < 1:
        raise ValueError("alns_iterations must be nonnegative")
    dataset_path = output_dir / "dataset.pt"
    manifest_path = output_dir / "manifest.json"
    if not overwrite and dataset_path.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        expected = (train_cases, validation_cases, seed, alns_iterations, policy)
        observed = (
            len(existing.get("train_instances", [])),
            len(existing.get("validation_instances", [])),
            int(existing.get("seed", -1)),
            int(existing.get("alns_iterations", -1)),
            str(existing.get("policy")),
        )
        if observed == expected and existing.get("dataset_sha256") == _sha256(dataset_path):
            return existing
        raise FileExistsError(
            f"a different traffic dataset exists under {output_dir}; pass --overwrite to replace it"
        )

    binary = find_binary(binary_path)
    train_seeds = list(range(seed, seed + train_cases))
    validation_seeds = list(
        range(seed + train_cases, seed + train_cases + validation_cases)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    requested = [
        (split, instance_seed)
        for split, seeds in (("train", train_seeds), ("validation", validation_seeds))
        for instance_seed in seeds
    ]

    def shard_path(split: str, instance_seed: int) -> Path:
        return shard_dir / f"{split}-{instance_seed}-{policy}.pt"

    started_at: dict[tuple[str, int], float] = {}

    def generate_one(split: str, instance_seed: int) -> tuple[Path, dict[str, Any]]:
        # Record when the worker actually begins; the outer elapsed metric
        # would otherwise include ThreadPoolExecutor queue wait, which grows
        # linearly as later futures sit behind earlier ones.
        started_at[(split, instance_seed)] = monotonic()
        path = shard_path(split, instance_seed)
        if path.exists() and not overwrite:
            _, cached_summary = _load_instance_shard(
                path,
                split=split,
                seed=instance_seed,
                alns_iterations=alns_iterations,
                policy=policy,
            )
            return path, cached_summary
        samples, summaries = simulate_online_samples(
            [instance_seed],
            alns_iterations=alns_iterations,
            binary=binary,
            timeout=simulation_timeout,
            core_threads=core_threads,
            policy=policy,
        )
        _save_instance_shard(
            path,
            split=split,
            seed=instance_seed,
            alns_iterations=alns_iterations,
            policy=policy,
            samples=samples,
            summary=summaries[0],
        )
        return path, summaries[0]

    class_totals = [0, 0, 0]
    total_samples = 0
    last_status: str = ""
    failed: list[dict[str, Any]] = []
    succeeded: list[tuple[str, int]] = []
    monotonic = time.monotonic
    progress = tqdm(
        total=len(requested),
        desc="traffic generation",
        unit="inst",
        dynamic_ncols=True,
        file=sys.stderr,
    )
    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures: dict[Any, tuple[str, int]] = {}
            for split, instance_seed in requested:
                future = executor.submit(generate_one, split, instance_seed)
                futures[future] = (split, instance_seed)
            try:
                for future in as_completed(futures):
                    split, instance_seed = futures[future]
                    # Prefer the worker-recorded start time so the reported
                    # elapsed reflects actual work, not pool queue wait.
                    anchor = started_at.get((split, instance_seed))
                    elapsed = monotonic() - anchor if anchor else 0.0
                    try:
                        _, summary = future.result()
                    except Exception as exc:
                        failed.append(
                            {
                                "split": split,
                                "seed": instance_seed,
                                "elapsed_seconds": round(elapsed, 3),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        progress.write(
                            f"FAILED {split} seed={instance_seed} "
                            f"({elapsed:.1f}s): {type(exc).__name__}: {exc}"
                        )
                        progress.update(1)
                        continue
                    class_counts = summary.get("class_counts", [0, 0, 0])
                    for index, value in enumerate(class_counts):
                        class_totals[index] += int(value)
                    total_samples += int(summary.get("samples", 0))
                    succeeded.append((split, instance_seed))
                    last_status = (
                        f"{split[0]}{instance_seed} {elapsed:.1f}s "
                        f"d={summary.get('days', '?')} "
                        f"s={summary.get('samples', 0)} "
                        f"roads={summary.get('roads', '?')} "
                        f"S/B/J={class_counts[0]}/{class_counts[1]}/{class_counts[2]}"
                    )
                    postfix = (
                        f"last[{last_status}] tot={total_samples} "
                        f"S/B/J={class_totals[0]}/{class_totals[1]}/{class_totals[2]}"
                    )
                    if failed:
                        postfix += f" failed={len(failed)}"
                    progress.set_postfix_str(postfix, refresh=True)
                    progress.update(1)
            except KeyboardInterrupt:
                # Ctrl-C: cancel pending submissions so the pool exits without
                # waiting for every in-flight subprocess. Already-succeeded
                # shards are still merged below.
                interrupted = True
                progress.write("interrupted; cancelling pending instances")
                for pending in futures:
                    pending.cancel()
    finally:
        progress.close()

    if not succeeded:
        raise RuntimeError(
            "no traffic instances succeeded"
            + (f"; {len(failed)} failed" if failed else "")
        )

    train_samples: list[TrafficGraphSample] = []
    validation_samples: list[TrafficGraphSample] = []
    train_instances: list[dict[str, Any]] = []
    validation_instances: list[dict[str, Any]] = []
    for split, instance_seed in succeeded:
        samples, summary = _load_instance_shard(
            shard_path(split, instance_seed),
            split=split,
            seed=instance_seed,
            alns_iterations=alns_iterations,
            policy=policy,
        )
        if split == "train":
            train_samples.extend(samples)
            train_instances.append(summary)
        else:
            validation_samples.extend(samples)
            validation_instances.append(summary)

    metadata = {
        "kind": f"offline-{policy}16-traffic-dataset",
        "target": "simulator road status for day t from state/history through t-1",
        "players": 16,
        "policy": policy,
        "alns_iterations": alns_iterations,
        "seed": seed,
        "train_instances": train_instances,
        "validation_instances": validation_instances,
        "requested_instances": len(requested),
        "succeeded_instances": len(succeeded),
        "failed_instances": failed,
        "interrupted": interrupted,
    }
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "classes": list(TRAFFIC_CLASSES),
        "metadata": metadata,
        "splits": {
            "train": [_sample_payload(sample) for sample in train_samples],
            "validation": [_sample_payload(sample) for sample in validation_samples],
        },
    }
    temporary_path = output_dir / "dataset.pt.tmp"
    torch.save(payload, temporary_path)
    temporary_path.replace(dataset_path)
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        **metadata,
        "feature_names": list(FEATURE_NAMES),
        "classes": list(TRAFFIC_CLASSES),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "dataset": str(dataset_path),
        "dataset_bytes": dataset_path.stat().st_size,
        "dataset_sha256": _sha256(dataset_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_traffic_dataset(
    dataset_path: Path,
) -> tuple[list[TrafficGraphSample], list[TrafficGraphSample], dict[str, Any]]:
    """Load a tensor-only dataset without invoking the match simulator."""

    payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("traffic dataset root must be a mapping")
    if int(payload.get("schema_version", -1)) != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported traffic dataset schema version")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("traffic dataset feature schema does not match this model")
    if tuple(payload.get("classes", ())) != TRAFFIC_CLASSES:
        raise ValueError("traffic dataset class schema does not match this model")
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("traffic dataset has no split mapping")
    train_samples = [_sample_from_payload(item) for item in splits.get("train", [])]
    validation_samples = [
        _sample_from_payload(item) for item in splits.get("validation", [])
    ]
    if not train_samples or not validation_samples:
        raise ValueError("traffic dataset requires nonempty train and validation splits")
    metadata = dict(payload.get("metadata", {}))
    metadata["dataset"] = str(dataset_path)
    metadata["dataset_bytes"] = dataset_path.stat().st_size
    metadata["dataset_sha256"] = _sha256(dataset_path)
    return train_samples, validation_samples, metadata


class HexGraphLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(hidden_size, hidden_size)
        self.neighbor_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.direction = nn.Embedding(6, hidden_size)
        self.normalization = nn.LayerNorm(hidden_size)

    def forward(self, nodes: Tensor, edge_index: Tensor, edge_direction: Tensor) -> Tensor:
        source, destination = edge_index
        messages = self.neighbor_projection(nodes[source]) + self.direction(edge_direction)
        aggregate = torch.zeros_like(nodes)
        aggregate.index_add_(0, destination, messages)
        counts = torch.zeros(nodes.shape[0], device=nodes.device, dtype=nodes.dtype)
        counts.index_add_(0, destination, torch.ones_like(destination, dtype=nodes.dtype))
        aggregate = aggregate / counts.clamp_min(1.0).unsqueeze(1)
        updated = self.self_projection(nodes) + aggregate
        return F.gelu(self.normalization(updated))


class TrafficGNN(nn.Module):
    def __init__(self, feature_count: int, hidden_size: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_count, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.layers = nn.ModuleList(HexGraphLayer(hidden_size) for _ in range(layers))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 3)
        )

    def forward(self, features: Tensor, edge_index: Tensor, edge_direction: Tensor) -> Tensor:
        hidden = self.encoder(features)
        for layer in self.layers:
            hidden = hidden + layer(hidden, edge_index, edge_direction)
        return self.classifier(hidden)


def _batch_samples(samples: list[TrafficGraphSample]) -> TrafficGraphSample:
    if not samples:
        raise ValueError("cannot batch an empty traffic sample list")
    features: list[Tensor] = []
    edges: list[Tensor] = []
    directions: list[Tensor] = []
    labels: list[Tensor] = []
    road_masks: list[Tensor] = []
    offset = 0
    for sample in samples:
        features.append(sample.features)
        edges.append(sample.edge_index + offset)
        directions.append(sample.edge_direction)
        labels.append(sample.labels)
        road_masks.append(sample.road_mask)
        offset += sample.features.shape[0]
    return TrafficGraphSample(
        features=torch.cat(features),
        edge_index=torch.cat(edges, dim=1),
        edge_direction=torch.cat(directions),
        labels=torch.cat(labels),
        road_mask=torch.cat(road_masks),
        instance_seed=-1,
        day=-1,
    )


def _sample_batches(
    samples: list[TrafficGraphSample], batch_size: int
) -> Iterable[TrafficGraphSample]:
    for begin in range(0, len(samples), batch_size):
        yield _batch_samples(samples[begin : begin + batch_size])


def _metrics(
    model: TrafficGNN,
    samples: list[TrafficGraphSample],
    device: torch.device,
    batch_size: int,
    *,
    desc: str = "eval",
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    count = 0
    confusion = torch.zeros((3, 3), dtype=torch.long)
    total_batches = (len(samples) + batch_size - 1) // batch_size
    with torch.no_grad():
        for raw_sample in tqdm(
            _sample_batches(samples, batch_size),
            total=total_batches,
            desc=desc,
            unit="batch",
            leave=False,
            file=sys.stderr,
        ):
            sample = raw_sample.to(device)
            logits = model(sample.features, sample.edge_index, sample.edge_direction)
            road_logits = logits[sample.road_mask]
            road_labels = sample.labels[sample.road_mask]
            loss_sum += float(F.cross_entropy(road_logits, road_labels, reduction="sum"))
            count += int(road_labels.numel())
            predictions = road_logits.argmax(dim=1).cpu()
            labels = road_labels.cpu()
            for label, prediction in zip(labels.tolist(), predictions.tolist(), strict=True):
                confusion[label, prediction] += 1
    f1: list[float] = []
    for label in range(3):
        true_positive = int(confusion[label, label])
        false_positive = int(confusion[:, label].sum()) - true_positive
        false_negative = int(confusion[label, :].sum()) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    correct = int(confusion.diagonal().sum())
    return {
        "loss": math.nan if count == 0 else loss_sum / count,
        "accuracy": math.nan if count == 0 else correct / count,
        "macro_f1": sum(f1) / len(f1),
        "class_f1": f1,
        "confusion": confusion.tolist(),
        "road_nodes": count,
    }


def train_traffic_gnn(
    *,
    dataset_path: Path,
    epochs: int,
    seed: int,
    hidden_size: int,
    layers: int,
    learning_rate: float,
    batch_size: int,
    patience: int,
    minimum_epochs: int,
    device_name: str,
    report_dir: Path,
    warmup_epochs: int = 0,
) -> dict[str, Any]:
    if (
        epochs < 1
        or hidden_size < 1
        or layers < 1
        or learning_rate <= 0
        or batch_size < 1
        or patience < 0
        or minimum_epochs < 1
        or minimum_epochs > epochs
        or warmup_epochs < 0
        or warmup_epochs >= epochs
    ):
        raise ValueError("invalid traffic training hyperparameters")
    random.seed(seed)
    torch.manual_seed(seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    train_samples, validation_samples, dataset_metadata = load_traffic_dataset(
        dataset_path
    )

    model = TrafficGNN(len(FEATURE_NAMES), hidden_size=hidden_size, layers=layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs)
    )
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler: torch.optim.lr_scheduler.LRScheduler = (
            torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        )
    else:
        scheduler = cosine
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = report_dir / "model.pt"
    history: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    epoch_bar = tqdm(
        range(1, epochs + 1), desc="epochs", unit="epoch", file=sys.stderr
    )
    for epoch in epoch_bar:
        model.train()
        random.shuffle(train_samples)
        train_total = (len(train_samples) + batch_size - 1) // batch_size
        for raw_sample in tqdm(
            _sample_batches(train_samples, batch_size),
            total=train_total,
            desc=f"ep{epoch} train",
            unit="batch",
            leave=False,
            file=sys.stderr,
        ):
            sample = raw_sample.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(sample.features, sample.edge_index, sample.edge_direction)
            loss = F.cross_entropy(logits[sample.road_mask], sample.labels[sample.road_mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        train_metrics = _metrics(
            model, train_samples, device, batch_size, desc=f"ep{epoch} train-eval"
        )
        validation_metrics = _metrics(
            model, validation_samples, device, batch_size, desc=f"ep{epoch} val-eval"
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        if validation_metrics["loss"] < best_validation_loss - 1e-8:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_names": FEATURE_NAMES,
                    "hidden_size": hidden_size,
                    "layers": layers,
                    "classes": TRAFFIC_CLASSES,
                    "dataset_sha256": dataset_metadata["dataset_sha256"],
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        epoch_bar.set_postfix(
            train_loss=f"{train_metrics['loss']:.4f}",
            val_loss=f"{validation_metrics['loss']:.4f}",
            val_acc=f"{validation_metrics['accuracy']:.3f}",
            val_f1=f"{validation_metrics['macro_f1']:.3f}",
            best=best_epoch,
            stale=stale_epochs,
            refresh=True,
        )
        if patience > 0 and epoch >= minimum_epochs and stale_epochs >= patience:
            break
    epoch_bar.close()

    best_record = history[best_epoch - 1]
    report = {
        "kind": f"offline-{dataset_metadata.get('policy', 'alns')}16-traffic-gnn",
        "target": "simulator road status for day t from state/history through t-1",
        "loss": "unweighted road-node cross entropy",
        "training_seed": seed,
        "device": str(device),
        "features": FEATURE_NAMES,
        "dataset": dataset_metadata,
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "batch_size": batch_size,
        "maximum_epochs": epochs,
        "minimum_epochs": minimum_epochs,
        "patience": patience,
        "warmup_epochs": warmup_epochs,
        "scheduler": "linear-warmup+cosine" if warmup_epochs > 0 else "cosine",
        "best_epoch": best_epoch,
        "best_train": best_record["train"],
        "best_validation": best_record["validation"],
        "epochs": history,
        "checkpoint": str(checkpoint_path),
    }
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


@lru_cache(maxsize=8)
def load_traffic_model(checkpoint_path: Path) -> TrafficGNN:
    """Reconstruct a trained GNN from a saved checkpoint.

    The checkpoint stores the schema (``feature_names``/``classes``) alongside
    the architecture hyperparameters so a prediction caller does not need to
    remember the training configuration.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("traffic checkpoint is missing a state_dict")
    feature_names = tuple(checkpoint.get("feature_names") or ())
    if feature_names and feature_names != FEATURE_NAMES:
        raise ValueError("traffic checkpoint feature schema does not match the model")
    classes = tuple(checkpoint.get("classes") or ())
    if classes and classes != TRAFFIC_CLASSES:
        raise ValueError("traffic checkpoint class schema does not match the model")
    hidden_size = int(checkpoint.get("hidden_size") or 64)
    layers = int(checkpoint.get("layers") or 3)
    model = TrafficGNN(len(FEATURE_NAMES), hidden_size=hidden_size, layers=layers)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict_traffic(
    scenario: dict[str, Any], replay: dict[str, Any], checkpoint_path: Path
) -> dict[str, Any]:
    """Run a trained traffic GNN over an authoritative replay.

    For every predicted day (``day >= 1``) each road cell reports the model's
    guessed status, the simulator ground truth, the predicted probability, and
    whether the guess matched. Day zero is fixed to smooth and is skipped.
    """

    samples = graph_samples_from_replay(scenario, replay)
    if not samples:
        return {
            "classes": list(TRAFFIC_CLASSES),
            "days": [],
            "road_count": 0,
            "matched": 0,
            "accuracy": 0.0,
            "confusion": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        }
    model = load_traffic_model(checkpoint_path)
    days_out: list[dict[str, Any]] = []
    total_roads = 0
    matched = 0
    confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    with torch.no_grad():
        for sample in samples:
            logits = model(sample.features, sample.edge_index, sample.edge_direction)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            road_positions = sample.road_mask.nonzero(as_tuple=True)[0].tolist()
            cells: list[dict[str, Any]] = []
            day_matched = 0
            for pos in road_positions:
                predicted = int(predictions[pos].item())
                actual = int(sample.labels[pos].item())
                confusion[actual][predicted] += 1
                total_roads += 1
                correct = predicted == actual
                if correct:
                    matched += 1
                    day_matched += 1
                cells.append(
                    {
                        "pos": pos,
                        "predicted": predicted,
                        "actual": actual,
                        "probability": round(
                            float(probabilities[pos, predicted].item()), 4
                        ),
                        "correct": correct,
                    }
                )
            days_out.append(
                {
                    "day": int(sample.day),
                    "road_count": len(cells),
                    "matched": day_matched,
                    "accuracy": (day_matched / len(cells)) if cells else 0.0,
                    "cells": cells,
                }
            )
    return {
        "classes": list(TRAFFIC_CLASSES),
        "days": days_out,
        "road_count": total_roads,
        "matched": matched,
        "accuracy": (matched / total_roads) if total_roads else 0.0,
        "confusion": confusion,
    }


def predict_future_traffic(
    scenario: dict[str, Any],
    checkpoint_path: Path,
    *,
    known_day: int | None = None,
    known_traffic: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Forecast road statuses for an MLNS suffix without future labels.

    The predictor is deliberately autoregressive: only day-zero smooth traffic
    and an optionally supplied currently revealed day are seeded; every later
    day's model output becomes the next day's history.  This makes the payload
    safe for local replay and live planning, where future opponent actions are
    unavailable.  The C++ planner consumes the returned maps only for future
    suffix simulation; official day evaluation still uses authoritative roads.
    """

    config = scenario["config"]
    total_days = len(config["daySteps"])
    if total_days <= 1:
        return []
    width = int(config["map"]["width"])
    height = int(config["map"]["height"])
    roads = [
        position
        for position, value in enumerate(
            value for row in config["map"]["cells"] for value in row
        )
        if int(value) == 1
    ]
    statuses: list[dict[int, int]] = [{position: 0 for position in roads}]
    if known_day is not None and known_traffic:
        while len(statuses) <= known_day:
            statuses.append({position: 0 for position in roads})
        statuses[known_day] = {
            position: int(known_traffic.get(position, 0)) for position in roads
        }
    model = load_traffic_model(checkpoint_path)
    agent_starts = [
        {
            "cell": int(position),
            "type": 0,
            "fuel": int(config["fuelLimits"]),
        }
        for position in config.get("agents", [])
    ]
    if known_day is not None and known_day >= 0 and known_day < total_days:
        # A live day carries the actual agent positions/types; retaining those
        # as the forecast context is better than resetting to map starts.
        day_agents = scenario.get("day_info", {}).get("agents", [])
        if day_agents:
            agent_starts = [
                {
                    "cell": int(agent["pos"]),
                    "type": int(agent.get("kind", 0)),
                    "fuel": int(agent.get("fuel") or config["fuelLimits"]),
                }
                for agent in day_agents
            ]

    def replay_for(current: list[dict[int, int]]) -> dict[str, Any]:
        days = []
        for day, road_status in enumerate(current):
            days.append(
                {
                    "day": day,
                    "road_condition": {
                        str(position): int(road_status.get(position, 0))
                        for position in roads
                    },
                    "all_team_starts": [
                        {"agents": [dict(agent) for agent in agent_starts]}
                    ],
                }
            )
        return {"replay": {"days": days}}

    output: list[dict[str, Any]] = []
    first_forecast_day = (
        max(1, known_day + 1) if known_day is not None else 1
    )
    with torch.no_grad():
        for day in range(first_forecast_day, total_days):
            if len(statuses) <= day:
                statuses.append({position: 0 for position in roads})
            samples = graph_samples_from_replay(scenario, replay_for(statuses))
            sample = next((item for item in samples if item.day == day), None)
            if sample is None:
                break
            logits = model(sample.features, sample.edge_index, sample.edge_direction)
            predictions = logits.argmax(dim=1)
            predicted = {
                position: int(predictions[position].item()) for position in roads
            }
            statuses[day] = predicted
            output.append(
                {
                    "day": day,
                    "traffics": [
                        {"pos": position, "status": status}
                        for position, status in predicted.items()
                    ],
                }
            )
    return output
