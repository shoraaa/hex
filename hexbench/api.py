from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import httpx

from .models import neighbors, validate_action_shape, validate_agent_types, validate_config
from .runner import find_binary, run_core

BASE_URL = "https://hexudon.hairbui76.id.vn/api"

# These are deliberately client-side controls. They are validated before a
# practice reset starts and are passed to the C++ planner only for the named
# policy. Empty entries preserve each policy's compiled defaults.
POLICY_HYPERPARAMETERS: dict[str, tuple[dict[str, Any], ...]] = {
    "wait": (),
    "hotspot": (),
    "greedy": (
        {"key": "max_targets", "label": "Maximum targets", "type": "integer", "min": 1, "max": 64, "step": 1},
    ),
    "utility_greedy": (
        {"key": "max_targets", "label": "Maximum targets", "type": "integer", "min": 1, "max": 64, "step": 1},
    ),
    "fuel_aware": (
        {"key": "max_targets", "label": "Maximum targets", "type": "integer", "min": 1, "max": 64, "step": 1},
        {"key": "fuel_reserve", "label": "Fuel reserve", "type": "integer", "min": 1, "max": 256, "step": 1},
    ),
    "stock_maximiser": (
        {"key": "max_targets", "label": "Maximum targets", "type": "integer", "min": 1, "max": 64, "step": 1},
    ),
    "coordinated": (
        {"key": "max_targets", "label": "Maximum targets", "type": "integer", "min": 1, "max": 64, "step": 1},
    ),
    "local_search": (
        {"key": "passes", "label": "Route-search passes", "type": "integer", "min": 1, "max": 32, "step": 1},
    ),
    "lns": (
        {"key": "time_limit_ms", "label": "Time limit", "unit": "ms", "type": "integer", "min": 50, "step": 50, "ui_max": 10_000, "help": "Wall-clock mode; leave blank when using fixed iterations."},
        {"key": "fixed_iterations", "label": "Fixed iteration budget", "unit": "iterations", "type": "integer", "min": 1, "step": 1, "ui_min": 32, "ui_max": 12_000, "ui_step": 32, "recommended": 2_048, "presets": [{"label": "Fast", "value": 256}, {"label": "Balanced", "value": 1_024}, {"label": "Recommended", "value": 2_048}, {"label": "Deep", "value": 6_000}], "help": "Deterministic mode. The recommended budget preserves Q01's 126 and reaches 219 on New Question."},
        {"key": "min_iterations", "label": "Minimum iterations", "type": "integer", "min": 1, "max": 2048, "step": 1},
        {"key": "max_iterations", "label": "Maximum iterations", "type": "integer", "min": 1, "step": 1, "ui_max": 10_000},
        {"key": "stagnation_iterations", "label": "Stagnation limit", "type": "integer", "min": 0, "step": 1, "ui_max": 10_000, "help": "Use 0 to disable early stopping."},
    ),
    "alns": (
        {"key": "time_limit_ms", "label": "Time limit", "unit": "ms", "type": "integer", "min": 50, "step": 50, "ui_max": 10_000, "help": "Wall-clock mode; leave blank when using fixed iterations."},
        {"key": "fixed_iterations", "label": "Fixed iteration budget", "unit": "iterations", "type": "integer", "min": 1, "step": 1, "ui_min": 32, "ui_max": 12_000, "ui_step": 32, "recommended": 2_048, "presets": [{"label": "Fast", "value": 256}, {"label": "Balanced", "value": 1_024}, {"label": "Recommended", "value": 2_048}, {"label": "Deep", "value": 6_000}], "help": "Deterministic mode. The recommended budget preserves Q01's 126 and reaches 219 on New Question."},
        {"key": "min_iterations", "label": "Minimum iterations", "type": "integer", "min": 1, "max": 2048, "step": 1},
        {"key": "max_iterations", "label": "Maximum iterations", "type": "integer", "min": 1, "step": 1, "ui_max": 10_000},
        {"key": "stagnation_iterations", "label": "Stagnation limit", "type": "integer", "min": 0, "step": 1, "ui_max": 10_000, "help": "Use 0 to disable early stopping."},
    ),
    "aco": (
        {"key": "ants", "label": "Ant count", "type": "integer", "min": 1, "step": 1, "ui_max": 128, "recommended": 20},
        {"key": "iterations", "label": "ACO iterations", "type": "integer", "min": 1, "step": 1, "ui_max": 100, "recommended": 20},
        {"key": "evaporation", "label": "Pheromone retention", "type": "number", "exclusive_min": 0, "exclusive_max": 1, "step": 0.01, "ui_min": 0.05, "ui_max": 0.99, "recommended": 0.85},
    ),
    "aco_ls": (
        {"key": "ants", "label": "Ant count", "type": "integer", "min": 1, "step": 1, "ui_max": 128, "recommended": 20},
        {"key": "iterations", "label": "ACO iterations", "type": "integer", "min": 1, "step": 1, "ui_max": 100, "recommended": 20},
        {"key": "evaporation", "label": "Pheromone retention", "type": "number", "exclusive_min": 0, "exclusive_max": 1, "step": 0.01, "ui_min": 0.05, "ui_max": 0.99, "recommended": 0.85},
    ),
}


def normalize_hyperparameters(
    methods: list[str], values: Any | None,
) -> dict[str, dict[str, int | float]]:
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError("hyperparameters must be an object keyed by policy")
    allowed_methods = set(methods)
    unknown_methods = sorted(set(values) - allowed_methods)
    if unknown_methods:
        raise ValueError(
            "hyperparameters supplied for unselected policies: "
            + ", ".join(unknown_methods)
        )
    result: dict[str, dict[str, int | float]] = {}
    for method, raw in values.items():
        if method not in POLICY_HYPERPARAMETERS:
            raise ValueError(f"unknown policy: {method}")
        if not isinstance(raw, dict):
            raise ValueError(f"hyperparameters for {method} must be an object")
        fields = {field["key"]: field for field in POLICY_HYPERPARAMETERS[method]}
        unknown = sorted(set(raw) - set(fields))
        if unknown:
            raise ValueError(
                f"unknown hyperparameters for {method}: {', '.join(unknown)}"
            )
        normalized: dict[str, int | float] = {}
        for key, value in raw.items():
            field = fields[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{method}.{key} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{method}.{key} must be finite")
            if "min" in field and value < field["min"]:
                raise ValueError(f"{method}.{key} must be at least {field['min']}")
            if "max" in field and value > field["max"]:
                raise ValueError(f"{method}.{key} must be at most {field['max']}")
            if "exclusive_min" in field and value <= field["exclusive_min"]:
                raise ValueError(
                    f"{method}.{key} must be greater than {field['exclusive_min']}"
                )
            if "exclusive_max" in field and value >= field["exclusive_max"]:
                raise ValueError(
                    f"{method}.{key} must be less than {field['exclusive_max']}"
                )
            if field["type"] == "integer" and int(value) != value:
                raise ValueError(f"{method}.{key} must be an integer")
            if field["type"] == "integer" and not -(2**31) <= value < 2**31:
                raise ValueError(f"{method}.{key} exceeds the C++ integer range")
            normalized[key] = int(value) if field["type"] == "integer" else float(value)
        if method in {"lns", "alns"}:
            if "fixed_iterations" in normalized and "time_limit_ms" in normalized:
                raise ValueError(
                    f"{method}.fixed_iterations cannot be combined with time_limit_ms"
                )
            effective_min = normalized.get("min_iterations", 32)
            effective_max = normalized.get(
                "fixed_iterations",
                normalized.get("max_iterations", 10_000_000),
            )
            if effective_min > effective_max:
                raise ValueError(f"{method}.min_iterations cannot exceed max_iterations")
        result[method] = normalized
    return result


def maximum_score(config: dict[str, Any]) -> dict[str, int]:
    """Return map-level theoretical maxima, ignoring travel feasibility."""
    days = len(config["daySteps"])
    distinct = len({int(spot["brand"]) for spot in config["spots"]})
    daily_servings = sum(int(spot["stocks"]) for spot in config["spots"])
    return {
        "distinct_types": distinct,
        "cumulative_daily_types": distinct * days,
        "total_servings": daily_servings * days,
    }


def structural_maximum_score(
    config: dict[str, Any], patrol_agents: int
) -> dict[str, int]:
    """Return the stock ceiling after accounting for per-patrol spot visits."""
    days = len(config["daySteps"])
    distinct = len({int(spot["brand"]) for spot in config["spots"]})
    daily_servings = sum(
        min(int(spot["stocks"]), max(0, patrol_agents))
        for spot in config["spots"]
    )
    return {
        "distinct_types": distinct if patrol_agents > 0 else 0,
        "cumulative_daily_types": distinct * days if patrol_agents > 0 else 0,
        "total_servings": daily_servings * days,
    }


def load_token(env_path: Path) -> str:
    if token := os.environ.get("TOKEN"):
        return token
    if not env_path.exists():
        raise RuntimeError(f"TOKEN is not set and {env_path} does not exist")
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("TOKEN="):
            token = line.split("=", 1)[1].strip().strip('"').strip("'")
            if token:
                return token
    raise RuntimeError(f"TOKEN is missing from {env_path}")


class GameClient:
    def __init__(self, token: str, base_url: str = BASE_URL, timeout: float = 15):
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def get(self, path: str, game_id: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.get(path, params={"game_id": game_id})
                response.raise_for_status()
                return response.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                last_error = error
                if isinstance(error, httpx.HTTPStatusError) or attempt == 2:
                    break
                time.sleep(0.25 * (attempt + 1))
        status = (
            last_error.response.status_code
            if isinstance(last_error, httpx.HTTPStatusError)
            else "network"
        )
        raise RuntimeError(f"GET {path} failed ({status})") from None

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        try:
            response = self._client.post(path, json=payload)
            response.raise_for_status()
            return response.json() if response.content else {}
        except httpx.HTTPStatusError as error:
            try:
                body = error.response.json()
                detail = body.get("detail", "request rejected") if isinstance(body, dict) else "request rejected"
            except ValueError:
                detail = "request rejected"
            raise RuntimeError(f"POST {path} failed ({error.response.status_code}): {detail}") from None
        except httpx.TransportError:
            raise RuntimeError(f"POST {path} failed (ambiguous network error; not retried)") from None


def fetch_game_snapshot(
    token: str, game_id: str, base_url: str = BASE_URL
) -> dict[str, Any]:
    """Fetch the read-only map/state/day bundle used by both web modes."""
    client = GameClient(token, base_url)
    try:
        board = client.get("/game/board", game_id)
        resolved_id = str(board.get("game_id", game_id))
        config = client.get("/game/config", resolved_id)
        state = client.get("/game/state", resolved_id)
        day: dict[str, Any] | None = None
        if state.get("status") == "in_progress":
            day = client.get("/game/day", resolved_id)
        return {
            "requested_game_id": game_id,
            "game_id": resolved_id,
            "board": board,
            "config": config,
            "state": state,
            "day": day,
            "fetched_at": datetime.now(UTC).isoformat(),
        }
    finally:
        client.close()


def _state_path(state_dir: Path, game_id: str) -> Path:
    key = hashlib.sha256(game_id.encode()).hexdigest()[:24]
    return state_dir / f"{key}.json"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "distinct_brands": [],
            "submitted_days": {},
            "day_snapshots": {},
            "types": None,
        }
    state = json.loads(path.read_text())
    state.setdefault("day_snapshots", {})
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def _terrain_time(config: dict[str, Any], pos: int, roads: dict[int, int]) -> int:
    width = config["map"]["width"]
    terrain = config["map"]["cells"][pos // width][pos % width]
    if terrain == 0:
        return 2
    if terrain == 2:
        return 3
    if terrain == 1:
        return (1, 2, 4)[roads.get(pos, 0)]
    raise ValueError("action starts from pond")


def planning_budget(
    config: dict[str, Any],
    day_info: dict[str, Any],
    *,
    is_practice: bool,
    deadline_margin: float,
) -> float:
    """Return solver time without applying stale timed deadlines to practice."""
    day_index = day_info["day"]
    budget = max(0.1, config["daySeconds"][day_index] - deadline_margin)
    if not is_practice and day_info.get("endsAt") is not None:
        budget = max(
            0.1,
            min(budget, day_info["endsAt"] - time.time() - deadline_margin),
        )
    return budget


def trace_action_plan(
    config: dict[str, Any],
    day_info: dict[str, Any],
    history: dict[str, Any] | None,
    actions: list[list[int]],
    *,
    binary_path: str | None = None,
) -> dict[str, Any]:
    """Replay a proposed day plan with the authoritative C++ simulator."""
    binary = find_binary(binary_path)
    payload: dict[str, Any] = {
        "config": config,
        "day_info": day_info,
        "actions": actions,
    }
    if history:
        payload["history"] = history
    return run_core("trace", "greedy", payload, binary=binary)


def predict_acquired_brands(
    config: dict[str, Any], day_info: dict[str, Any], actions: list[list[int]]
) -> set[int]:
    """Replay accepted own-team actions to maintain observable brand history."""
    width, height = config["map"]["width"], config["map"]["height"]
    roads = {item["pos"]: item["status"] for item in day_info["traffics"]}
    spot_by_pos = {item["pos"]: item for item in config["spots"]}
    stock = {item["pos"]: item["stocks"] for item in config["spots"]}
    positions = [item["pos"] for item in day_info["agents"]]
    kinds = [item["kind"] for item in day_info["agents"]]
    visited = [set() for _ in positions]
    cursor = [0 for _ in positions]
    pending: list[dict[str, Any] | None] = [None for _ in positions]

    def schedule(index: int) -> None:
        action = actions[index][cursor[index]]
        cursor[index] += 1
        if action < 0:
            pending[index] = {"remaining": -action, "destination": positions[index]}
            return
        destination = dict(neighbors(height, width, positions[index])).get(action)
        if destination is None:
            raise ValueError("invalid action during history replay")
        pending[index] = {
            "remaining": _terrain_time(config, positions[index], roads),
            "destination": destination,
        }

    for index in range(len(positions)):
        schedule(index)
    acquired: set[int] = set()
    horizon = config["daySteps"][day_info["day"]]
    for step in range(1, horizon + 1):
        for index, action in enumerate(pending):
            assert action is not None
            action["remaining"] -= 1
            if action["remaining"] == 0:
                positions[index] = action["destination"]
                pending[index] = None
        for index, pos in enumerate(positions):
            spot = spot_by_pos.get(pos)
            if kinds[index] == 0 and spot and pos not in visited[index] and stock[pos] > 0:
                visited[index].add(pos)
                stock[pos] -= 1
                acquired.add(spot["brand"])
        if step < horizon:
            for index, action in enumerate(pending):
                if action is None:
                    schedule(index)
    return acquired


def fetch_fixture(
    game_id: str, output: Path, env_path: Path, base_url: str = BASE_URL
) -> dict[str, Any]:
    client = GameClient(load_token(env_path), base_url)
    try:
        payload = {
            "board": client.get("/game/board", game_id),
            "config": client.get("/game/config", game_id),
            "state": client.get("/game/state", game_id),
            "day": client.get("/game/day", game_id),
        }
    finally:
        client.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def deploy(
    game_id: str,
    method: str,
    env_path: Path,
    state_dir: Path,
    *,
    dry_run: bool,
    once: bool,
    deadline_margin: float,
    poll_interval: float,
    binary_path: str | None = None,
    base_url: str = BASE_URL,
    quiet: bool = False,
    method_hyperparameters: dict[str, int | float] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    def emit(payload: dict[str, Any]) -> dict[str, Any]:
        if progress is not None:
            progress(payload)
        if not quiet:
            print(json.dumps(payload))
        return payload

    binary = find_binary(binary_path)
    normalized_hyperparameters = normalize_hyperparameters(
        [method], {method: method_hyperparameters or {}}
    ).get(method, {})
    client = GameClient(load_token(env_path), base_url)
    try:
        board = client.get("/game/board", game_id)
        config = client.get("/game/config", game_id)
        validate_config(config)
        resolved_id = board.get("game_id", game_id)
        is_practice = bool(board.get("is_practice"))
        state_path = _state_path(state_dir, resolved_id)
        journal = _load_state(state_path)
        status = client.get("/game/state", resolved_id)

        if status["status"] == "selecting_agents":
            # A resettable practice game may reuse the same composite id. Its
            # old local journal must not suppress submissions in the replay.
            journal = {
                "distinct_brands": [],
                "submitted_days": {},
                "day_snapshots": {},
                "types": None,
            }
            emit({"game_id": resolved_id, "status": "selecting_agent_types"})
            types = run_core("types", method, config, binary=binary)
            validate_agent_types(types, len(config["agents"]))
            if dry_run:
                return emit({"dry_run": True, "game_id": resolved_id, "types": types})
            client.post("/game/agent-types", {"game_id": resolved_id, "types": types})
            journal["types"] = types
            _save_state(state_path, journal)

        while True:
            status = client.get("/game/state", resolved_id)
            if status["status"] == "finished":
                result_endpoint = "/game/practice/score" if is_practice else "/game/result"
                result = client.get(result_endpoint, resolved_id)
                return emit({"game_id": resolved_id, "status": "finished", "result": result})
            if status["status"] != "in_progress":
                if once:
                    return emit({"game_id": resolved_id, "status": status["status"]})
                time.sleep(poll_interval)
                continue
            day_index = int(status["day"])
            if day_index >= len(config["daySteps"]):
                if is_practice:
                    result = client.get("/game/practice/score", resolved_id)
                    return emit({"game_id": resolved_id, "status": "finished", "result": result})
                time.sleep(poll_interval)
                continue
            day_key = str(day_index)
            if day_key in journal["submitted_days"]:
                if once:
                    return emit(
                        {
                            "game_id": resolved_id,
                            "day": day_index,
                            "status": "already_submitted",
                        }
                    )
                time.sleep(poll_interval)
                continue
            day_info = client.get("/game/day", resolved_id)
            types = [agent["kind"] for agent in day_info["agents"]]
            submitted = journal.get("submitted_days", {})
            history = {
                "distinct_brands": journal.get("distinct_brands", []),
                "submitted_actions": [
                    submitted[str(index)]
                    for index in range(day_index)
                    if str(index) in submitted
                ],
            }
            close_to_deadline = (
                not is_practice
                and day_info.get("endsAt") is not None
                and day_info["endsAt"] - time.time() <= deadline_margin
            )
            if close_to_deadline:
                emit(
                    {
                        "game_id": resolved_id,
                        "day": day_index,
                        "status": "deadline_fallback",
                    }
                )
                actions = [[-config["daySteps"][day_index]] for _ in types]
            else:
                budget = planning_budget(
                    config,
                    day_info,
                    is_practice=is_practice,
                    deadline_margin=deadline_margin,
                )
                safe_time_limit_ms = max(50, int(budget * 0.85 * 1000))
                fixed_iterations = normalized_hyperparameters.get("fixed_iterations")
                planner_hyperparameters = {
                    key: value
                    for key, value in normalized_hyperparameters.items()
                    if key != "fixed_iterations"
                }
                if fixed_iterations is not None:
                    fixed_iterations = int(fixed_iterations)
                    search = {
                        "minIterations": int(
                            normalized_hyperparameters.get(
                                "min_iterations", min(2048, fixed_iterations)
                            )
                        ),
                        "maxIterations": fixed_iterations,
                        "stagnationIterations": int(
                            normalized_hyperparameters.get(
                                "stagnation_iterations", fixed_iterations
                            )
                        ),
                    }
                else:
                    requested_time_limit_ms = int(
                        normalized_hyperparameters.get(
                            "time_limit_ms", safe_time_limit_ms
                        )
                    )
                    solver_time_limit_ms = min(
                        requested_time_limit_ms, safe_time_limit_ms
                    )
                    deadline_governed_search = method in {"lns", "alns"}
                    search = {
                        "timeLimitMs": solver_time_limit_ms,
                        "minIterations": int(
                            normalized_hyperparameters.get("min_iterations", 32)
                        ),
                        "maxIterations": int(
                            normalized_hyperparameters.get(
                                "max_iterations",
                                10_000_000 if deadline_governed_search else 2048,
                            )
                        ),
                        "stagnationIterations": int(
                            normalized_hyperparameters.get(
                                "stagnation_iterations",
                                0 if deadline_governed_search else 96,
                            )
                        ),
                    }
                progress_event = {
                    "game_id": resolved_id,
                    "day": day_index,
                    "status": "planning",
                    "response_window_seconds": budget,
                }
                if fixed_iterations is not None:
                    progress_event["iteration_limit"] = fixed_iterations
                else:
                    progress_event["budget_seconds"] = solver_time_limit_ms / 1000
                emit(
                    progress_event
                )
                try:
                    # Keep validation, fallback construction, and submission outside
                    # the solver budget. The C++ ALNS clock includes preprocessing.
                    # The server deadline is the authoritative stopping rule for
                    # LNS/ALNS. Keep validation and submission headroom outside
                    # this budget, but do not pair a long wall-clock allowance
                    # with small iteration/stagnation caps that make the solver
                    # converge prematurely. Explicit limits remain available for
                    # controlled benchmarks and debugging.
                    actions = run_core(
                        "plan",
                        method,
                        {
                            "config": config,
                            "day_info": day_info,
                            "history": history,
                            "types": types,
                            "search": search,
                            "hyperparameters": planner_hyperparameters,
                        },
                        binary=binary,
                        timeout=budget,
                    )
                except (subprocess.TimeoutExpired, RuntimeError):
                    actions = [[-config["daySteps"][day_index]] for _ in types]
            validate_action_shape(actions, len(types))
            check = run_core(
                "check",
                method,
                {"config": config, "day_info": day_info, "actions": actions},
                binary=binary,
            )
            if not check["valid"]:
                actions = [[-config["daySteps"][day_index]] for _ in types]
                fallback_check = run_core(
                    "check",
                    method,
                    {"config": config, "day_info": day_info, "actions": actions},
                    binary=binary,
                )
                if not fallback_check["valid"]:
                    raise RuntimeError(f"all-wait fallback failed validation: {fallback_check['error']}")
            if dry_run:
                return emit(
                    {
                        "dry_run": True,
                        "game_id": resolved_id,
                        "day": day_index,
                        "actions": actions,
                    }
                )
            emit(
                {
                    "game_id": resolved_id,
                    "day": day_index,
                    "status": "submitting",
                }
            )
            endpoint = "/game/practice/actions" if is_practice else "/game/actions"
            client.post(
                endpoint,
                {"game_id": resolved_id, "day": day_index, "actions": actions},
            )
            acquired = predict_acquired_brands(config, day_info, actions)
            journal["distinct_brands"] = sorted(
                set(journal.get("distinct_brands", [])) | acquired
            )
            journal["submitted_days"][day_key] = actions
            journal["day_snapshots"][day_key] = {
                "day_info": day_info,
                "actions": actions,
                "validation": check,
                "submitted_at": datetime.now(UTC).isoformat(),
            }
            _save_state(state_path, journal)
            submitted = {
                "game_id": resolved_id,
                "day": day_index,
                "status": "submitted",
            }
            emit(submitted)
            if is_practice and day_index + 1 >= len(config["daySteps"]):
                result = client.get("/game/practice/score", resolved_id)
                return emit({"game_id": resolved_id, "status": "finished", "result": result})
            if once:
                return submitted
            time.sleep(poll_interval)
    finally:
        client.close()


def _score_detail(response: dict[str, Any], resolved_id: str) -> dict[str, Any]:
    detail = response.get("detail", {})
    team_id = resolved_id.rsplit(":", 1)[-1]
    score = detail.get(team_id)
    if score is None and len(detail) == 1:
        score = next(iter(detail.values()))
    if not isinstance(score, dict):
        raise RuntimeError("practice score response does not contain this team")
    return score


def _official_score_key(row: dict[str, Any]) -> tuple[int, int, int, float]:
    return (
        int(row["distinct_types"]),
        int(row["cumulative_daily_types"]),
        int(row["total_servings"]),
        -float(row.get("cumulative_response_time", 0.0)),
    )


def _manager_authority(base_url: str) -> str:
    parsed = httpx.URL(base_url)
    authority = f"{parsed.scheme}://{parsed.host}"
    if parsed.port is not None:
        authority += f":{parsed.port}"
    return authority


def _token_team_id(token: str) -> str | None:
    """Read the non-secret team id from the manager JWT when available."""
    try:
        encoded = token.removeprefix("Bearer ").split(".")[1]
        encoded += "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(encoded))
        team_id = payload.get("id")
        return str(team_id) if team_id is not None else None
    except (IndexError, ValueError, TypeError, binascii.Error, json.JSONDecodeError):
        return None


def discover_practice_questions(
    token: str, base_url: str = BASE_URL
) -> list[dict[str, Any]]:
    """List practice questions that are safe to reset for this team."""
    try:
        response = httpx.get(
            f"{_manager_authority(base_url)}/manager/api/question",
            headers={"Authorization": token},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        questions = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(questions, list):
            raise RuntimeError("manager question response is not a list")
    except (httpx.HTTPError, ValueError) as error:
        raise RuntimeError(f"practice question discovery failed: {error}") from None

    own_team_id = _token_team_id(token)
    discovered: list[dict[str, Any]] = []
    for question in questions:
        try:
            question_data = question.get("question_data", {})
            if isinstance(question_data, str):
                question_data = json.loads(question_data)
            if not question_data.get("is_practice") or question_data.get("no_reset"):
                continue
            teams = [str(team["team_id"]) for team in question_data.get("teams", [])]
            if own_team_id is not None and teams and own_team_id not in teams:
                continue
            game_map = question_data.get("map", {})
            discovered.append(
                {
                    "question_id": str(question["id"]),
                    "name": str(question.get("name") or question["id"]),
                    "width": game_map.get("width"),
                    "height": game_map.get("height"),
                    "total_days": len(question_data.get("daySteps", [])),
                    "team_ids": teams,
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return discovered


def _question_data(question: dict[str, Any]) -> dict[str, Any]:
    """Decode manager question metadata without leaking it to callers."""
    value = question.get("question_data", {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _game_descriptor(
    question: dict[str, Any], data: dict[str, Any], own_team_id: str | None
) -> dict[str, Any] | None:
    """Normalize a manager question into a safe UI-facing game descriptor.

    Resettable practice maps are deliberately distinguished from non-reset
    practice competitions.  The latter use the practice action endpoint but
    must be operated exactly like a live competition.
    """
    teams = [str(team.get("team_id")) for team in data.get("teams", []) if isinstance(team, dict)]
    if own_team_id is not None and teams and own_team_id not in teams:
        return None
    game_map = data.get("map", {}) if isinstance(data.get("map"), dict) else {}
    is_practice = bool(data.get("is_practice"))
    no_reset = bool(data.get("no_reset"))
    practice_lab = is_practice and not no_reset
    question_id = str(question.get("id", ""))
    if not question_id:
        return None
    return {
        "question_id": question_id,
        "name": str(question.get("name") or question_id),
        "mode": "practice" if practice_lab else "competition",
        "competition_kind": "practice_competition" if is_practice else "competition",
        "is_practice": is_practice,
        "no_reset": no_reset,
        "width": game_map.get("width"),
        "height": game_map.get("height"),
        "total_days": len(data.get("daySteps", [])),
        "day_seconds": list(data.get("daySeconds", [])),
        "day_steps": list(data.get("daySteps", [])),
        "players": data.get("players"),
        "team_ids": teams,
        "starts_at": data.get("startsAt"),
        "capabilities": {
            "reset": practice_lab,
            "benchmark": practice_lab,
            "local_evaluation": practice_lab,
            "submit": True,
            "peer_rank": is_practice,
            "replay": is_practice,
        },
    }


def discover_assigned_games(
    token: str, base_url: str = BASE_URL
) -> list[dict[str, Any]]:
    """Return every assigned question classified for Practice or Competition.

    Unlike :func:`discover_practice_questions`, this includes real matches and
    ``no_reset`` practice competitions.  The manager endpoint is the source of
    assignment truth; game state is fetched lazily by the web layer.
    """
    try:
        response = httpx.get(
            f"{_manager_authority(base_url)}/manager/api/question",
            headers={"Authorization": token},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        questions = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(questions, list):
            raise RuntimeError("manager question response is not a list")
    except (httpx.HTTPError, ValueError) as error:
        raise RuntimeError(f"question discovery failed: {error}") from None

    own_team_id = _token_team_id(token)
    discovered: list[dict[str, Any]] = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        descriptor = _game_descriptor(question, _question_data(question), own_team_id)
        if descriptor is not None:
            discovered.append(descriptor)
    return discovered


def build_fuel_stress_variants(
    question: dict[str, Any],
    config: dict[str, Any],
    fuel_multipliers: tuple[float, ...] = (1.0, 0.5, 0.25),
) -> list[dict[str, Any]]:
    """Create local scenarios from one authoritative map by changing only fuel."""
    validate_config(config)
    day_one_steps = int(config["daySteps"][0])
    original_fuel = int(config["fuelLimits"])
    candidates: list[tuple[str, int, float]] = [
        ("server", original_fuel, original_fuel / day_one_steps)
    ]
    for multiplier in fuel_multipliers:
        if not math.isfinite(multiplier) or multiplier <= 0 or multiplier > 3:
            raise ValueError("fuel multipliers must be finite values in (0, 3]")
        fuel = max(1, min(3 * day_one_steps, round(day_one_steps * multiplier)))
        candidates.append((f"{multiplier:g}x", fuel, multiplier))

    opponents = [
        ("wait", "greedy", "hotspot")[index % 3]
        for index in range(max(0, int(config["players"]) - 1))
    ]
    variants: list[dict[str, Any]] = []
    used_fuel: set[int] = set()
    for label, fuel, multiplier in candidates:
        if fuel in used_fuel:
            continue
        used_fuel.add(fuel)
        variant_config = copy.deepcopy(config)
        variant_config["fuelLimits"] = fuel
        validate_config(variant_config)
        variants.append(
            {
                "question": {
                    key: question.get(key)
                    for key in (
                        "question_id",
                        "name",
                        "width",
                        "height",
                        "total_days",
                    )
                },
                "fuel_label": label,
                "fuel_multiplier": multiplier,
                "original_fuel": original_fuel,
                "fuel_limit": fuel,
                "maximum_score": maximum_score(variant_config),
                "scenario": {
                    "schema_version": 1,
                    "source": "authoritative-practice-fuel-stress",
                    "config": variant_config,
                    "opponents": opponents,
                },
            }
        )
    return variants


def fuel_stress_benchmark(
    methods: list[str],
    env_path: Path,
    report_dir: Path,
    *,
    game_ids: list[str] | None = None,
    fuel_multipliers: tuple[float, ...] = (1.0, 0.5, 0.25),
    hyperparameters: dict[str, dict[str, int | float]] | None = None,
    jobs: int = 0,
    binary_path: str | None = None,
    base_url: str = BASE_URL,
) -> dict[str, Any]:
    """Read server practice maps and grade lower-fuel variants locally."""
    methods = list(dict.fromkeys(methods))
    if not methods:
        raise ValueError("at least one method is required")
    normalized_hyperparameters = normalize_hyperparameters(methods, hyperparameters)
    token = load_token(env_path)
    questions = discover_practice_questions(token, base_url)
    by_id = {row["question_id"]: row for row in questions}
    if game_ids is not None:
        selected: list[dict[str, Any]] = []
        for game_id in dict.fromkeys(game_ids):
            question_id = game_id.split(":", 1)[0]
            if question_id not in by_id:
                raise ValueError(f"unknown resettable practice game: {question_id}")
            selected.append(by_id[question_id])
        questions = selected
    if not questions:
        raise RuntimeError("no resettable practice questions were discovered")

    client = GameClient(token, base_url)
    variants: list[dict[str, Any]] = []
    try:
        for question in questions:
            config = client.get("/game/config", question["question_id"])
            variants.extend(
                build_fuel_stress_variants(question, config, fuel_multipliers)
            )
    finally:
        client.close()

    report_dir.mkdir(parents=True, exist_ok=True)
    case_dir = report_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        question_id = variant["question"]["question_id"]
        label = variant["fuel_label"].replace(".", "_")
        path = case_dir / f"{question_id}-{label}.json"
        path.write_text(json.dumps(variant["scenario"], indent=2) + "\n")
        variant["scenario_path"] = str(path)

    binary = find_binary(binary_path)
    worker_count = max(1, jobs or min(os.cpu_count() or 1, 8))
    tasks = [
        (variant_index, method)
        for variant_index in range(len(variants))
        for method in methods
    ]

    def evaluate(task: tuple[int, str]) -> tuple[int, str, dict[str, Any]]:
        variant_index, method = task
        payload = copy.deepcopy(variants[variant_index]["scenario"])
        if parameters := normalized_hyperparameters.get(method):
            payload["hyperparameters"] = parameters
        started = time.perf_counter()
        result = run_core(
            "eval",
            method,
            payload,
            binary=binary,
            core_threads=1 if worker_count > 1 else None,
        )
        result["runtime_seconds"] = time.perf_counter() - started
        return variant_index, method, result

    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        evaluated = list(executor.map(evaluate, tasks))
    wall_seconds = time.perf_counter() - wall_started
    for variant in variants:
        variant["results"] = {}
        variant.pop("scenario", None)
    for variant_index, method, result in evaluated:
        variants[variant_index]["results"][method] = result

    def result_key(result: dict[str, Any]) -> tuple[int, int, int, int]:
        score = result["score"]
        return (
            int(result["invalid_days"] == 0),
            int(score["distinct_types"]),
            int(score["cumulative_daily_types"]),
            int(score["total_servings"]),
        )

    aggregates: dict[str, dict[str, dict[str, float | int]]] = {}
    for variant in variants:
        ordered = sorted(
            methods,
            key=lambda method: result_key(variant["results"][method]),
            reverse=True,
        )
        for rank, method in enumerate(ordered, 1):
            variant["results"][method]["rank"] = rank
        label = variant["fuel_label"]
        aggregates.setdefault(label, {})
        for method in methods:
            result = variant["results"][method]
            score = result["score"]
            aggregate = aggregates[label].setdefault(
                method,
                {
                    "cases": 0,
                    "valid_cases": 0,
                    "distinct_types": 0,
                    "cumulative_daily_types": 0,
                    "total_servings": 0,
                    "maximum_distinct_types": 0,
                    "maximum_cumulative_daily_types": 0,
                    "maximum_total_servings": 0,
                    "refuel_events": 0,
                    "refuel_agents": 0,
                    "runtime_seconds": 0.0,
                },
            )
            aggregate["cases"] += 1
            aggregate["valid_cases"] += int(result["invalid_days"] == 0)
            aggregate["distinct_types"] += int(score["distinct_types"])
            aggregate["cumulative_daily_types"] += int(
                score["cumulative_daily_types"]
            )
            aggregate["total_servings"] += int(score["total_servings"])
            aggregate["maximum_distinct_types"] += int(
                variant["maximum_score"]["distinct_types"]
            )
            aggregate["maximum_cumulative_daily_types"] += int(
                variant["maximum_score"]["cumulative_daily_types"]
            )
            aggregate["maximum_total_servings"] += int(
                variant["maximum_score"]["total_servings"]
            )
            aggregate["refuel_events"] += int(result.get("refuel_events", 0))
            aggregate["refuel_agents"] += int(result.get("refuel_agents", 0))
            aggregate["runtime_seconds"] += float(result["runtime_seconds"])

    report = {
        "schema_version": 1,
        "kind": "authoritative-practice-fuel-stress",
        "methods": methods,
        "fuel_multipliers": list(fuel_multipliers),
        "map_count": len(questions),
        "case_count": len(variants),
        "jobs": worker_count,
        "wall_seconds": wall_seconds,
        "hyperparameters": normalized_hyperparameters,
        "aggregates": aggregates,
        "cases": variants,
    }
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# HEXUDON authoritative-map fuel stress benchmark",
        "",
        "Each case preserves the server map, days, terrain, spots, stocks, and "
        "agent starts; only `fuelLimits` changes. Evaluation is local and does "
        "not reset or submit to the server.",
        "",
        f"Maps: `{len(questions)}` · variants: `{len(variants)}` · wall time: `{wall_seconds:.3f}s`",
        "",
        "## Aggregate by fuel level",
        "",
        "| Fuel | Method | Valid | Distinct | Daily | Servings | Refuel events | Refuel agents | Runtime (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, by_method in aggregates.items():
        for method in methods:
            row = by_method[method]
            lines.append(
                f"| `{label}` | `{method}` | {row['valid_cases']}/{row['cases']} | "
                f"{row['distinct_types']} | {row['cumulative_daily_types']} | "
                f"{row['total_servings']} | {row['refuel_events']} | "
                f"{row['refuel_agents']} | {row['runtime_seconds']:.3f} |"
            )
    lines.extend(
        (
            "",
            "## Per-map results",
            "",
            "| Map | Fuel | Limit | Rank | Method | Score D/D/S | Refuels | Types P/R | Valid |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|",
        )
    )
    for variant in variants:
        for method in sorted(
            methods, key=lambda item: variant["results"][item]["rank"]
        ):
            result = variant["results"][method]
            score = result["score"]
            lines.append(
                f"| {variant['question']['name']} | `{variant['fuel_label']}` | "
                f"{variant['fuel_limit']} | {result['rank']} | `{method}` | "
                f"{score['distinct_types']}/{score['cumulative_daily_types']}/{score['total_servings']} | "
                f"{result.get('refuel_events', 0)} | "
                f"{result.get('patrol_agents', 0)}/{result.get('refuel_agents', 0)} | "
                f"{result['valid_days']}/{result['valid_days'] + result['invalid_days']} |"
            )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n")
    return report


def lns_time_benchmark(
    env_path: Path,
    report_dir: Path,
    *,
    method: str = "lns",
    game_ids: list[str] | None = None,
    fuel_multiplier: float = 0.5,
    time_limits_ms: tuple[int, ...] = (25, 100, 500, 2000, 10000),
    jobs: int = 0,
    binary_path: str | None = None,
    base_url: str = BASE_URL,
) -> dict[str, Any]:
    """Measure an LNS-family score curve as its daily budget grows."""
    if method not in {"lns", "alns"}:
        raise ValueError("time benchmark method must be lns or alns")
    if (
        not math.isfinite(fuel_multiplier)
        or fuel_multiplier <= 0
        or fuel_multiplier > 3
    ):
        raise ValueError("fuel multiplier must be in (0, 3]")
    budgets = tuple(dict.fromkeys(int(value) for value in time_limits_ms))
    if not budgets or any(value < 1 or value > 60_000 for value in budgets):
        raise ValueError("time limits must be integers from 1 to 60000 ms")

    token = load_token(env_path)
    questions = discover_practice_questions(token, base_url)
    by_id = {row["question_id"]: row for row in questions}
    if game_ids is not None:
        questions = []
        for game_id in dict.fromkeys(game_ids):
            question_id = game_id.split(":", 1)[0]
            if question_id not in by_id:
                raise ValueError(f"unknown resettable practice game: {question_id}")
            questions.append(by_id[question_id])
    if not questions:
        raise RuntimeError("no resettable practice questions were discovered")

    client = GameClient(token, base_url)
    cases: list[dict[str, Any]] = []
    try:
        for question in questions:
            config = client.get("/game/config", question["question_id"])
            variants = build_fuel_stress_variants(
                question, config, (fuel_multiplier,)
            )
            target_fuel = max(
                1,
                min(
                    3 * int(config["daySteps"][0]),
                    round(int(config["daySteps"][0]) * fuel_multiplier),
                ),
            )
            variant = next(row for row in variants if row["fuel_limit"] == target_fuel)
            cases.append(variant)
    finally:
        client.close()

    binary = find_binary(binary_path)
    worker_count = max(1, jobs or min(os.cpu_count() or 1, 8))
    tasks = [
        (case_index, budget)
        for case_index in range(len(cases))
        for budget in budgets
    ]

    def evaluate(task: tuple[int, int]) -> tuple[int, int, dict[str, Any]]:
        case_index, budget = task
        payload = copy.deepcopy(cases[case_index]["scenario"])
        payload["hyperparameters"] = {
            "time_limit_ms": budget,
            "min_iterations": 1,
            "max_iterations": 10_000_000,
            "stagnation_iterations": 0,
        }
        started = time.perf_counter()
        result = run_core(
            "eval",
            method,
            payload,
            binary=binary,
            timeout=max(60, budget / 1000 * len(payload["config"]["daySteps"]) + 30),
            core_threads=1 if worker_count > 1 else None,
        )
        result["runtime_seconds"] = time.perf_counter() - started
        return case_index, budget, result

    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        evaluated = list(executor.map(evaluate, tasks))
    wall_seconds = time.perf_counter() - wall_started
    for case in cases:
        case["results"] = {}
        case.pop("scenario", None)
    for case_index, budget, result in evaluated:
        cases[case_index]["results"][str(budget)] = result

    aggregates: dict[str, dict[str, float | int]] = {}
    for budget in budgets:
        aggregate: dict[str, float | int] = {
            "maps": len(cases),
            "valid_maps": 0,
            "distinct_types": 0,
            "cumulative_daily_types": 0,
            "total_servings": 0,
            "maximum_distinct_types": 0,
            "maximum_cumulative_daily_types": 0,
            "maximum_total_servings": 0,
            "refuel_events": 0,
            "runtime_seconds": 0.0,
        }
        for case in cases:
            result = case["results"][str(budget)]
            score = result["score"]
            aggregate["valid_maps"] += int(result["invalid_days"] == 0)
            aggregate["distinct_types"] += int(score["distinct_types"])
            aggregate["cumulative_daily_types"] += int(
                score["cumulative_daily_types"]
            )
            aggregate["total_servings"] += int(score["total_servings"])
            aggregate["maximum_distinct_types"] += int(
                case["maximum_score"]["distinct_types"]
            )
            aggregate["maximum_cumulative_daily_types"] += int(
                case["maximum_score"]["cumulative_daily_types"]
            )
            aggregate["maximum_total_servings"] += int(
                case["maximum_score"]["total_servings"]
            )
            aggregate["refuel_events"] += int(result["refuel_events"])
            aggregate["runtime_seconds"] += float(result["runtime_seconds"])
        aggregates[str(budget)] = aggregate

    report = {
        "schema_version": 1,
        "kind": "lns-time-curve",
        "method": method,
        "fuel_multiplier": fuel_multiplier,
        "time_limits_ms": list(budgets),
        "map_count": len(cases),
        "jobs": worker_count,
        "wall_seconds": wall_seconds,
        "aggregates": aggregates,
        "cases": cases,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        f"# HEXUDON {method.upper()} time-budget curve",
        "",
        f"Fuel: `{fuel_multiplier:g}x` Day-1 steps · maps: `{len(cases)}` · wall time: `{wall_seconds:.3f}s`",
        "",
        "The budget is applied independently to every match day. Stagnation "
        "stopping is disabled and the iteration cap is intentionally unreachable.",
        "",
        "## Aggregate curve",
        "",
        "| Budget/day | Valid | Distinct | Daily | Servings | Refuels | Actual total runtime |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in budgets:
        row = aggregates[str(budget)]
        lines.append(
            f"| {budget} ms | {row['valid_maps']}/{row['maps']} | "
            f"{row['distinct_types']} | {row['cumulative_daily_types']} | "
            f"{row['total_servings']} | {row['refuel_events']} | "
            f"{row['runtime_seconds']:.3f}s |"
        )
    lines.extend(
        (
            "",
            "## Per-map curve",
            "",
            "| Map | Fuel | Budget/day | Score D/D/S | Refuels | Types P/R | Runtime |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for case in cases:
        for budget in budgets:
            result = case["results"][str(budget)]
            score = result["score"]
            lines.append(
                f"| {case['question']['name']} | {case['fuel_limit']} | "
                f"{budget} ms | {score['distinct_types']}/{score['cumulative_daily_types']}/"
                f"{score['total_servings']} | {result['refuel_events']} | "
                f"{result['patrol_agents']}/{result['refuel_agents']} | "
                f"{result['runtime_seconds']:.3f}s |"
            )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n")
    return report


def discover_peer_team_ids(
    token: str, question_id: str, base_url: str = BASE_URL
) -> list[str]:
    """Discover configured practice peers from the manager question metadata.

    The gameplay API intentionally has no list endpoint. Discovery is therefore
    best-effort; callers may supply explicit ids when the manager service is not
    present on a competition network.
    """
    try:
        response = httpx.get(
            f"{_manager_authority(base_url)}/manager/api/question/{question_id}",
            headers={"Authorization": token},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        question_data = payload.get("question_data", {})
        if isinstance(question_data, str):
            question_data = json.loads(question_data)
        return [str(team["team_id"]) for team in question_data.get("teams", [])]
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return []


def collect_peer_baselines(
    client: GameClient,
    question_id: str,
    own_team_id: str,
    peer_team_ids: list[str],
    total_days: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for team_id in dict.fromkeys(str(value) for value in peer_team_ids):
        if team_id == own_team_id:
            continue
        composite_id = f"{question_id}:{team_id}"
        try:
            replay = client.get("/game/practice/peer", composite_id)
            score_response = client.get("/game/practice/score", composite_id)
            score = _score_detail(score_response, composite_id)
        except RuntimeError as error:
            rows.append(
                {
                    "team_id": team_id,
                    "status": "unavailable",
                    "submitted_days": 0,
                    "total_days": total_days,
                    "error": str(error),
                }
            )
            continue
        submitted_days = 0
        for day in replay.get("days", []):
            if any(
                str(team.get("team_id")) == team_id and team.get("submitted")
                for team in day.get("teams", [])
            ):
                submitted_days += 1
        status = (
            "completed"
            if submitted_days >= total_days
            else ("partial" if submitted_days else "not_started")
        )
        rows.append(
            {
                "team_id": team_id,
                "status": status,
                "submitted_days": submitted_days,
                "total_days": total_days,
                **score,
            }
        )
    return rows


def practice_benchmark(
    game_id: str,
    methods: list[str],
    env_path: Path,
    state_dir: Path,
    report_dir: Path,
    *,
    leave_best: bool = True,
    peer_team_ids: list[str] | None = None,
    poll_interval: float = 0.05,
    binary_path: str | None = None,
    base_url: str = BASE_URL,
    quiet: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
    hyperparameters: dict[str, dict[str, int | float]] | None = None,
) -> dict[str, Any]:
    if not methods:
        raise ValueError("at least one practice method is required")
    normalized_hyperparameters = normalize_hyperparameters(methods, hyperparameters)
    token = load_token(env_path)
    client = GameClient(token, base_url)
    try:
        board = client.get("/game/board", game_id)
        if not board.get("is_practice"):
            raise RuntimeError("practice benchmark refuses to reset a non-practice game")
        if board.get("no_reset"):
            raise RuntimeError("practice benchmark refuses a no_reset practice game")
        resolved_id = board.get("game_id", game_id)
        question_id, own_team_id = resolved_id.rsplit(":", 1)
        config = client.get("/game/config", game_id)
        if peer_team_ids is None:
            peer_team_ids = discover_peer_team_ids(token, question_id, base_url)
        rows: list[dict[str, Any]] = []
        for method in methods:
            def deploy_progress(event: dict[str, Any], policy: str = method) -> None:
                if progress is not None:
                    progress({"policy": policy, **event})

            if progress is not None:
                progress({"policy": method, "status": "resetting"})
            if not quiet:
                print(json.dumps({"policy": method, "status": "resetting"}))
            client.post("/game/practice/reset", {"game_id": resolved_id})
            if progress is not None:
                progress({"policy": method, "status": "reset_complete"})
            started = time.perf_counter()
            deploy(
                game_id,
                method,
                env_path,
                state_dir,
                dry_run=False,
                once=False,
                deadline_margin=2.0,
                poll_interval=poll_interval,
                binary_path=binary_path,
                base_url=base_url,
                quiet=True,
                method_hyperparameters=normalized_hyperparameters.get(method),
                progress=deploy_progress,
            )
            elapsed = time.perf_counter() - started
            response = client.get("/game/practice/score", resolved_id)
            score = _score_detail(response, resolved_id)
            row = {"policy": method, **score, "wall_seconds": elapsed}
            journal = _load_state(_state_path(state_dir, resolved_id))
            selected_types = journal.get("types")
            if isinstance(selected_types, list):
                patrol_agents = sum(int(kind) == 0 for kind in selected_types)
                row["structural_maximum_score"] = structural_maximum_score(
                    config, patrol_agents
                )
            rows.append(row)
            if progress is not None:
                progress({"policy": method, "status": "finished", "score": score})
            if not quiet:
                print(json.dumps({"policy": method, "status": "finished", "score": score}))

        ranked = sorted(rows, key=_official_score_key, reverse=True)
        previous_key: tuple[int, int, int, float] | None = None
        previous_rank = 0
        for index, row in enumerate(ranked, 1):
            key = _official_score_key(row)
            if key != previous_key:
                previous_rank = index
                previous_key = key
            row["rank"] = previous_rank

        best_policy = ranked[0]["policy"]
        final_policy = methods[-1]
        if leave_best and final_policy != best_policy:
            if progress is not None:
                progress({"policy": best_policy, "status": "restoring_best"})
            client.post("/game/practice/reset", {"game_id": resolved_id})
            def restore_progress(event: dict[str, Any]) -> None:
                if progress is not None:
                    progress({"policy": best_policy, "restoring": True, **event})

            deploy(
                game_id,
                best_policy,
                env_path,
                state_dir,
                dry_run=False,
                once=False,
                deadline_margin=2.0,
                poll_interval=poll_interval,
                binary_path=binary_path,
                base_url=base_url,
                quiet=True,
                method_hyperparameters=normalized_hyperparameters.get(best_policy),
                progress=restore_progress,
            )
            final_policy = best_policy

        peers = collect_peer_baselines(
            client,
            question_id,
            own_team_id,
            peer_team_ids,
            len(config["daySteps"]),
        )
        combined = [
            {"source": "our_policy", "name": row["policy"], **row}
            for row in ranked
        ]
        combined.extend(
            {"source": "peer_team", "name": f"team:{peer['team_id']}", **peer}
            for peer in peers
            if peer["status"] == "completed"
        )
        combined.sort(key=_official_score_key, reverse=True)
        for index, row in enumerate(combined, 1):
            row["combined_rank"] = index
        best_row = next(row for row in ranked if row["policy"] == best_policy)
        completed_peers = [peer for peer in peers if peer["status"] == "completed"]
        peer_comparison = {"wins": 0, "ties": 0, "losses": 0}
        for peer in completed_peers:
            own_key, peer_key = _official_score_key(best_row), _official_score_key(peer)
            bucket = "wins" if own_key > peer_key else ("losses" if own_key < peer_key else "ties")
            peer_comparison[bucket] += 1

        report = {
            "schema_version": 1,
            "game_id": resolved_id,
            "maximum_score": maximum_score(config),
            "structural_maximum_score": best_row.get(
                "structural_maximum_score", maximum_score(config)
            ),
            "best_policy": best_policy,
            "final_policy": final_policy,
            "results": ranked,
            "peer_baselines": peers,
            "completed_peer_comparison": peer_comparison,
            "combined_ranking": combined,
        }
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        lines = [
            "# HEXUDON practice policy benchmark",
            "",
            f"Game: `{resolved_id}`",
            "",
            "| Rank | Policy | Distinct types | Daily types | Servings / structural / raw | Wall seconds |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for row in ranked:
            lines.append(
                f"| {row['rank']} | `{row['policy']}` | {row['distinct_types']} | "
                f"{row['cumulative_daily_types']} | {row['total_servings']}/"
                f"{row.get('structural_maximum_score', maximum_score(config))['total_servings']}/"
                f"{maximum_score(config)['total_servings']} | "
                f"{row['wall_seconds']:.3f} |"
            )
        lines.extend(
            (
                "",
                "## Peer baselines",
                "",
                "| Team | Status | Submitted days | Distinct types | Daily types | Servings |",
                "|---|---|---:|---:|---:|---:|",
            )
        )
        for peer in peers:
            lines.append(
                f"| `{peer['team_id']}` | {peer['status']} | "
                f"{peer['submitted_days']}/{peer['total_days']} | "
                f"{peer.get('distinct_types', '-')} | "
                f"{peer.get('cumulative_daily_types', '-')} | "
                f"{peer.get('total_servings', '-')} |"
            )
        lines.extend(
            (
                "",
                "## Combined completed ranking",
                "",
                "| Rank | Entry | Source | Distinct types | Daily types | Servings |",
                "|---:|---|---|---:|---:|---:|",
            )
        )
        for row in combined:
            lines.append(
                f"| {row['combined_rank']} | `{row['name']}` | {row['source']} | "
                f"{row['distinct_types']} | {row['cumulative_daily_types']} | "
                f"{row['total_servings']} |"
            )
        lines.extend(("", f"Best policy left on server: `{final_policy}`", ""))
        (report_dir / "report.md").write_text("\n".join(lines))
        return report
    finally:
        client.close()


def _safe_report_name(question: dict[str, Any]) -> str:
    name = re.sub(r"[^a-z0-9]+", "-", question["name"].lower()).strip("-")
    return f"{name or 'practice'}-{question['question_id'][:8]}"


def _suite_map_summary(
    question: dict[str, Any], report: dict[str, Any], report_path: Path
) -> dict[str, Any]:
    best = next(
        row for row in report["results"] if row["policy"] == report["best_policy"]
    )
    peers = report["peer_baselines"]
    completed = [row for row in peers if row["status"] == "completed"]
    scored = [row for row in peers if "distinct_types" in row]
    own_key = _official_score_key(best)
    status_counts = {
        status: sum(row["status"] == status for row in peers)
        for status in ("completed", "partial", "not_started", "unavailable")
    }
    completed_rank = 1 + sum(_official_score_key(row) > own_key for row in completed)
    provisional_rank = 1 + sum(_official_score_key(row) > own_key for row in scored)
    return {
        **{key: question.get(key) for key in (
            "question_id", "name", "width", "height", "total_days"
        )},
        "game_id": report["game_id"],
        "best_policy": report["best_policy"],
        "final_policy": report["final_policy"],
        "maximum_score": report.get(
            "maximum_score",
            {
                "distinct_types": best["distinct_types"],
                "cumulative_daily_types": best["cumulative_daily_types"],
                "total_servings": best["total_servings"],
            },
        ),
        "structural_maximum_score": report.get(
            "structural_maximum_score", report.get("maximum_score")
        ),
        "score": {
            key: best[key]
            for key in (
                "distinct_types",
                "cumulative_daily_types",
                "total_servings",
                "cumulative_response_time",
            )
        },
        "completed_rank": completed_rank,
        "completed_players": len(completed) + 1,
        "provisional_rank": provisional_rank,
        "configured_players": len(peers) + 1,
        "peer_status_counts": status_counts,
        "completed_peer_comparison": report["completed_peer_comparison"],
        "report": str(report_path),
    }


def practice_suite(
    methods: list[str],
    env_path: Path,
    state_dir: Path,
    report_dir: Path,
    *,
    game_ids: list[str] | None = None,
    peer_team_ids: list[str] | None = None,
    poll_interval: float = 0.05,
    binary_path: str | None = None,
    base_url: str = BASE_URL,
    quiet: bool = False,
    hyperparameters: dict[str, dict[str, int | float]] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Benchmark policies on every resettable practice map and rank the result."""
    if not methods:
        raise ValueError("at least one practice method is required")
    token = load_token(env_path)
    try:
        discovered = discover_practice_questions(token, base_url)
    except RuntimeError:
        if game_ids is None:
            raise
        discovered = []
    by_id = {row["question_id"]: row for row in discovered}
    if game_ids is None:
        questions = discovered
    else:
        questions = []
        for game_id in dict.fromkeys(game_ids):
            question_id = game_id.split(":", 1)[0]
            questions.append(
                by_id.get(
                    question_id,
                    {
                        "question_id": question_id,
                        "name": question_id,
                        "width": None,
                        "height": None,
                        "total_days": None,
                        "team_ids": [],
                    },
                )
            )
    if not questions:
        raise RuntimeError("no resettable practice questions were discovered")

    report_dir.mkdir(parents=True, exist_ok=True)
    maps: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, question in enumerate(questions, 1):
        question_id = question["question_id"]
        destination = report_dir / _safe_report_name(question)
        if not quiet:
            print(
                json.dumps(
                    {
                        "map": f"{index}/{len(questions)}",
                        "name": question["name"],
                        "game_id": question_id,
                        "status": "benchmarking",
                    }
                )
            )
        automatic_peers = question.get("team_ids") or None
        selected_peers = automatic_peers if peer_team_ids is None else peer_team_ids
        if progress is not None:
            progress(
                {
                    "status": "benchmarking",
                    "map": index,
                    "maps": len(questions),
                    "game_id": question_id,
                    "name": question["name"],
                }
            )
        try:
            report = practice_benchmark(
                question_id,
                methods,
                env_path,
                state_dir,
                destination,
                leave_best=True,
                peer_team_ids=selected_peers,
                poll_interval=poll_interval,
                binary_path=binary_path,
                base_url=base_url,
                quiet=True,
                hyperparameters=hyperparameters,
            )
            row = _suite_map_summary(question, report, destination / "report.json")
            maps.append(row)
            if progress is not None:
                progress(
                    {
                        "status": "finished_map",
                        "map": index,
                        "maps": len(questions),
                        "game_id": question_id,
                        "name": question["name"],
                        "score": row["score"],
                    }
                )
            if not quiet:
                print(
                    json.dumps(
                        {
                            "name": row["name"],
                            "status": "finished",
                            "score": row["score"],
                            "completed_rank": (
                                f"{row['completed_rank']}/{row['completed_players']}"
                            ),
                            "provisional_rank": (
                                f"{row['provisional_rank']}/{row['configured_players']}"
                            ),
                        }
                    )
                )
        except (RuntimeError, ValueError) as error:
            errors.append(
                {"question_id": question_id, "name": question["name"], "error": str(error)}
            )
            if progress is not None:
                progress(
                    {
                        "status": "map_failed",
                        "map": index,
                        "maps": len(questions),
                        "game_id": question_id,
                        "name": question["name"],
                        "error": str(error),
                    }
                )
            if not quiet:
                print(json.dumps({"name": question["name"], "status": "error", "error": str(error)}))

    aggregate = {"wins": 0, "ties": 0, "losses": 0}
    for row in maps:
        for key in aggregate:
            aggregate[key] += row["completed_peer_comparison"][key]
    summary = {
        "schema_version": 1,
        "methods": methods,
        "map_count": len(maps),
        "maps": maps,
        "completed_peer_comparison": aggregate,
        "errors": errors,
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# HEXUDON practice suite",
        "",
        "Completed rank includes only teams that submitted every day. Provisional "
        "rank includes current partial and unstarted scores for all available teams.",
        "",
        "| Map | Size | Policy | Score (distinct/daily/servings) | Completed rank | Provisional rank | Peer status (C/P/N/U) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in maps:
        score = row["score"]
        counts = row["peer_status_counts"]
        lines.append(
            f"| {row['name']} | {row['width']}x{row['height']} | `{row['best_policy']}` | "
            f"{score['distinct_types']}/{score['cumulative_daily_types']}/{score['total_servings']} | "
            f"{row['completed_rank']}/{row['completed_players']} | "
            f"{row['provisional_rank']}/{row['configured_players']} | "
            f"{counts['completed']}/{counts['partial']}/{counts['not_started']}/{counts['unavailable']} |"
        )
    lines.extend(
        (
            "",
            f"Completed-peer head-to-head: **{aggregate['wins']} wins, "
            f"{aggregate['ties']} ties, {aggregate['losses']} losses**.",
            "",
        )
    )
    if errors:
        lines.extend(("## Errors", ""))
        lines.extend(f"- `{row['name']}`: {row['error']}" for row in errors)
        lines.append("")
    (report_dir / "summary.md").write_text("\n".join(lines))
    return summary
