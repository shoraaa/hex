"""Run one map under many RNG seeds and report whether the plans diverge.

This is the experiment the task asks for: pin a single map, sweep
``search.randomSeed`` over N values, and compare the action plans the ALNS
planner returns. If different seeds yield different (still valid) plans, the
seed knob reaches the search; if every seed is identical, the planner is
effectively deterministic regardless of the seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from hexbench.runner import find_binary, run_core

ROOT = Path(__file__).resolve().parents[1]


def _day_zero_info(config: dict[str, Any], types: list[int]) -> dict[str, Any]:
    """Build a day-0 DayInfo for a single-team scenario.

    Every agent starts at its configured cell with a full tank; day 0 has no
    prior traffic so every road cell reports status 0.
    """
    fuel_limit = int(config["fuelLimits"])
    agents = [
        {"kind": int(kind), "pos": int(pos), "fuel": fuel_limit}
        for kind, pos in zip(types, config["agents"])
    ]
    cells = config["map"]["cells"]
    width = int(config["map"]["width"])
    height = int(config["map"]["height"])
    # cells is a row-major 2D array; flatten to a 1D terrain vector.
    flat = [int(c) for row in cells for c in row]
    traffics = [
        {"pos": pos, "status": 0}
        for pos in range(width * height)
        # Terrain::Road is 1; Pond is 0, House is 2.
        if flat[pos] == 1
    ]
    return {"day": 0, "agents": agents, "traffics": traffics}


def _plan_payload(
    config: dict[str, Any],
    day: dict[str, Any],
    types: list[int],
    *,
    policy: str,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    return {
        "config": config,
        "day_info": day,
        "types": types,
        "search": {
            # Deterministic iteration-only search: no wall-clock deadline so the
            # only thing changing between runs is the seed.
            "timeLimitMs": -1,
            "minIterations": iterations,
            "maxIterations": iterations,
            "stagnationIterations": iterations,
            "randomSeed": seed,
        },
    }


def _fingerprint(plan: list[list[int]]) -> str:
    return hashlib.sha256(
        json.dumps(plan, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _levenshtein(a: list[int], b: list[int]) -> int:
    """Plain edit distance on two integer sequences (insert/delete/substitute)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, bj in enumerate(b, start=1):
            cost = 0 if ai == bj else 1
            current[j] = min(
                current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost
            )
        previous = current
    return previous[-1]


def _normalized_edit(a: list[int], b: list[int]) -> float:
    """Levenshtein normalized to [0, 1] by the longer sequence length."""
    denominator = max(len(a), len(b))
    if denominator == 0:
        return 0.0
    return _levenshtein(a, b) / denominator


def _jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def _skeleton_from_trace(
    trace: dict[str, Any], agent_count: int
) -> list[list[int]]:
    """Per-agent ordered list of spot positions acquired (the route skeleton).

    Refuel/idle agents collect nothing, so their slot is empty. Order is the
    acquisition step, which is the order the patrol actually served the spot.
    """
    per_agent: list[list[int]] = [[] for _ in range(agent_count)]
    events = sorted(
        trace.get("acquisitions", []), key=lambda e: (e["agent"], e["step"])
    )
    for event in events:
        agent = event["agent"]
        if 0 <= agent < agent_count:
            per_agent[agent].append(event["spot"])
    return per_agent


def _route_distance(left: list[list[int]], right: list[list[int]]) -> float:
    """Mean normalized edit distance over agents active in either skeleton.

    An agent is "active" if it collected at least one spot in either plan, so
    refuel cars (always empty) do not dilute the metric. 0 means identical
    skeletons, 1 means disjoint single-spot routes.
    """
    max_agents = max(len(left), len(right))
    distances: list[float] = []
    for agent in range(max_agents):
        la = left[agent] if agent < len(left) else []
        ra = right[agent] if agent < len(right) else []
        if not la and not ra:
            continue
        distances.append(_normalized_edit(la, ra))
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def _spotset_distance(left: list[list[int]], right: list[list[int]]) -> float:
    """Mean Jaccard distance over agents active in either skeleton."""
    max_agents = max(len(left), len(right))
    distances: list[float] = []
    for agent in range(max_agents):
        la = set(left[agent]) if agent < len(left) else set()
        ra = set(right[agent]) if agent < len(right) else set()
        if not la and not ra:
            continue
        distances.append(_jaccard(la, ra))
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def _action_distance(left: list[list[int]], right: list[list[int]]) -> float:
    """Mean normalized edit distance over the raw per-agent action arrays."""
    max_agents = max(len(left), len(right))
    distances = [
        _normalized_edit(
            left[a] if a < len(left) else [],
            right[a] if a < len(right) else [],
        )
        for a in range(max_agents)
    ]
    return sum(distances) / len(distances) if distances else 0.0


def _pairwise_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        nargs="?",
        default=str(ROOT / "cases" / "quick" / "case-0000.json"),
        help="Scenario JSON (defaults to cases/quick/case-0000.json)",
    )
    parser.add_argument(
        "--policy", default="alns", help="Planner policy (default: alns)"
    )
    parser.add_argument(
        "--seeds", type=int, default=16, help="Number of seeds to sweep (default 16)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=384,
        help="ALNS iterations per seed (default 384)",
    )
    parser.add_argument(
        "--binary", default=None, help="Path to the hexudon binary"
    )
    args = parser.parse_args()

    binary = find_binary(args.binary)
    scenario = json.loads(Path(args.case).read_text())
    config = scenario["config"]

    types = run_core("types", args.policy, config, binary=binary)
    day = _day_zero_info(config, types)

    print(
        f"case={Path(args.case).name} policy={args.policy} "
        f"seeds={args.seeds} iterations={args.iterations}"
    )
    print(
        f"agents={len(types)} spots={len(config['spots'])} "
        f"players={config['players']} days={len(config['daySteps'])}"
    )
    print("-" * 72)

    results: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        payload = _plan_payload(
            config,
            day,
            types,
            policy=args.policy,
            seed=seed,
            iterations=args.iterations,
        )
        plan = run_core("plan", args.policy, payload, binary=binary)
        trace = run_core(
            "trace",
            args.policy,
            {"config": config, "day_info": day, "types": types, "actions": plan},
            binary=binary,
        )
        score = trace.get("score", {})
        fingerprint = _fingerprint(plan)
        skeleton = _skeleton_from_trace(trace, len(types))
        results.append(
            {
                "seed": seed,
                "fingerprint": fingerprint,
                "score": score,
                "valid": bool(trace.get("valid")),
                "plan": plan,
                "skeleton": skeleton,
            }
        )
        print(
            f"seed={seed:>2}  fp={fingerprint}  "
            f"distinct={score.get('distinct_types', '-')} "
            f"daily={score.get('daily_types', '-')} "
            f"servings={score.get('servings', '-')}  "
            f"spots={sum(len(r) for r in skeleton)}  "
            f"valid={trace.get('valid')}"
        )

    print("-" * 72)
    fingerprints = {r["fingerprint"] for r in results}
    distinct_plans = len(fingerprints)
    print(f"distinct plans across {args.seeds} seeds: {distinct_plans}")
    if distinct_plans == 1:
        print("All seeds produced the IDENTICAL plan (planner is seed-invariant).")

    # Group seeds by plan fingerprint and show the score each group achieved.
    groups: dict[str, list[int]] = {}
    for r in results:
        groups.setdefault(r["fingerprint"], []).append(r["seed"])
    print(f"Plan groups ({len(groups)} unique):")
    for fp, seeds in sorted(groups.items(), key=lambda kv: kv[1][0]):
        head = results[seeds[0]]
        s = head["score"]
        seeds_str = ", ".join(str(s_) for s_ in seeds)
        print(
            f"  fp={fp}  seeds=[{seeds_str}]  "
            f"distinct={s.get('distinct_types', '-')} "
            f"daily={s.get('daily_types', '-')} "
            f"servings={s.get('servings', '-')}"
        )

    scores = [tuple(r["score"].get(k, 0) for k in ("distinct_types", "daily_types", "servings")) for r in results]
    best = max(scores)
    worst = min(scores)
    if best == worst:
        print("Different plans but all achieve the SAME score.")
    else:
        print(f"Score varies: worst={worst} best={best}")

    print("-" * 72)
    _report_diversity(results)
    return 0


def _report_diversity(results: list[dict[str, Any]]) -> None:
    """Summarise how different the N plans are from one another.

    Three complementary views, each averaged over all C(N,2) seed pairs:

    * skeleton diversity -- normalised edit distance of each agent's ordered
      spot-acquisition list. This is the headline "did the strategy change?"
      measure and is robust to different move encodings of the same route.
    * spot-set diversity -- Jaccard distance of each agent's visited-spot set,
      ignoring visit order.
    * action diversity -- normalised edit distance of the raw move arrays,
      i.e. how different the literal paths are on the grid.
    """
    n = len(results)
    print(f"Solution diversity across {n} seeds")
    skeleton_dists: list[float] = []
    spotset_dists: list[float] = []
    action_dists: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            left, right = results[i], results[j]
            skeleton_dists.append(
                _route_distance(left["skeleton"], right["skeleton"])
            )
            spotset_dists.append(
                _spotset_distance(left["skeleton"], right["skeleton"])
            )
            action_dists.append(
                _action_distance(left["plan"], right["plan"])
            )

    def line(label: str, stats: dict[str, float]) -> None:
        print(
            f"  {label:<20} mean={stats['mean']:.3f}  "
            f"min={stats['min']:.3f}  max={stats['max']:.3f}  "
            f"pairs={stats['n']}"
        )

    line("skeleton diversity", _pairwise_stats(skeleton_dists))
    line("spot-set diversity", _pairwise_stats(spotset_dists))
    line("action diversity", _pairwise_stats(action_dists))

    unique_skeletons = {
        json.dumps(r["skeleton"], separators=(",", ":")) for r in results
    }
    unique_plans = {r["fingerprint"] for r in results}
    print(
        f"  distinct plans={len(unique_plans)}/{n}  "
        f"distinct skeletons={len(unique_skeletons)}/{n}"
    )
    sk = _pairwise_stats(skeleton_dists)
    if sk["mean"] < 1e-9:
        print("  -> zero skeleton diversity: every seed visits the same spots.")
    elif sk["max"] < 1e-9:
        print("  -> at least one identical pair, but routes vary on average.")
    else:
        print(
            "  -> seeds produce measurably different routes "
            f"(mean skeleton distance {sk['mean']:.1%} of the route length)."
        )


if __name__ == "__main__":
    sys.exit(main())
