#!/usr/bin/env python3
"""Offline eval of a policy on the online-q4q11 fixtures at a fixed per-day budget."""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hexbench.runner import find_binary, run_core  # noqa: E402


def run_one(args):
    fixture, policy, tl, core_threads, env = args
    scenario = json.loads(fixture.read_text())
    search = dict(scenario.get("search", {}))
    search.update({"timeLimitMs": tl, "maxIterations": 10_000_000,
                   "stagnationIterations": 0, "useLnsDpProposals": True})
    scenario["search"] = search
    binary = find_binary(None)
    day_count = len(scenario["config"]["daySteps"])
    t0 = time.perf_counter()
    res = run_core("eval", policy, scenario, binary=binary,
                   timeout=day_count * tl / 1000 * 1.5 + 120,
                   core_threads=core_threads, environment_overrides=env)
    wall = time.perf_counter() - t0
    score = res.get("score", {})
    days = res.get("daily_scores") or []
    daily = [d.get("daily_types") for d in days] if days else None
    return scenario["question"]["name"], score, wall, daily, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "cases/online-q4q11/manifest.json")
    ap.add_argument("--policy", default="mlns")
    ap.add_argument("--time-limit-ms", type=int, default=5000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--core-threads", type=int, default=2)
    ap.add_argument("--only", default=None, help="comma-separated case name substrings")
    ap.add_argument("--env-key", action="append", default=[], help="KEY=VALUE core env overrides")
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    base = args.manifest.parent
    env = {}
    for kv in args.env_key:
        k, v = kv.split("=", 1)
        env[k] = v
    cases = manifest["cases"]
    if args.only:
        subs = args.only.split(",")
        cases = [c for c in cases if any(s in c["name"] for s in subs)]
    tasks = [(base / c["path"], args.policy, args.time_limit_ms, args.core_threads, env or None)
             for c in cases]
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for name, score, wall, daily, _ in ex.map(run_one, tasks):
            rows.append((name, score, wall, daily))
            print(json.dumps({"q": name, "score": score, "wall": round(wall, 1),
                              "daily_per_day": daily}), flush=True)
    # totals
    td = sum(int(s.get("distinct_types", 0)) for _, s, _, _ in rows)
    ty = sum(int(s.get("cumulative_daily_types", 0)) for _, s, _, _ in rows)
    tv = sum(int(s.get("total_servings", 0)) for _, s, _, _ in rows)
    print(json.dumps({"policy": args.policy, "tl_ms": args.time_limit_ms,
                      "totals": {"distinct": td, "daily": ty, "servings": tv}}), flush=True)


if __name__ == "__main__":
    main()
