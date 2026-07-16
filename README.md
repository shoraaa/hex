# HEXUDON benchmark and deployment client

This repository contains a deterministic local HEXUDON environment and a
guarded client for the live server. The C++20 core owns simulation, scoring,
and policies; Python owns generation, reporting, and HTTP orchestration. Both
local grading and live deployment invoke the same compiled policy code.

## Build and test

```sh
uv sync --extra test
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
uv run pytest -q
```

The build uses the system Boost.JSON library plus the already-vendored `fmt`
and `cxxopts` headers. No token is needed for local generation or grading.

## Generate and grade locally

```sh
uv run hexbench generate --suite quick --out cases/quick
uv run hexbench grade \
  --cases cases/quick/manifest.json \
  --method greedy \
  --baselines wait,greedy \
  --report reports/quick-greedy \
  --jobs 8
```

`quick` contains 30 balanced smoke cases. `full` contains 192 cases from a
deterministic pairwise covering design. Fuel pressure, terrain composition,
agent count, map shape, match-day count, per-day step horizon, traffic mix,
traffic thresholds, terrain topology, and spot density vary independently.
The full suite includes 8-32-cell square, wide, tall, and 4:1 maps; 3-8 agents;
4-10 days; legal horizons from `(width + height)` through
`4 * (width + height)` steps; and fuel levels from 0.1x through 3x Day-1
steps. Every suite case simulates 16 teams (the candidate plus 15 scripted
opponents), so road state is derived from multi-team traffic on every match.
Every emitted `config` uses the official map configuration shape;
benchmark-only design metadata is kept outside it.

## Official-flow policy benchmark

All policy benchmarks use the competition information boundary from
[`qa.md`](/home/shora/Research/hex/qa.md:98):
agent types are selected from the initial map without reading even Day-1's
step count, and each daily planner receives only the current revealed step
horizon, current traffic, current positions/fuel, and accepted history. The
local evaluator replaces unrevealed future day lengths with neutral legal
placeholders before calling every policy, so local scores cannot benefit from
future-schedule leakage. The online suite resets each practice map between
policies and restores the best policy afterward.

The current all-policy snapshots are saved in
[local-full-all](/home/shora/Research/hex/reports/local-full-all/report.md) and
[online-practice-all](/home/shora/Research/hex/reports/online-practice-all/summary.md).

Local full suite: 192 cases, 16-team traffic, 100% valid for every policy.
Values below are means per case; runtime is the aggregate policy runtime
divided by 192.

| Policy | Distinct | Daily | Servings | Runtime (s) |
|---|---:|---:|---:|---:|
| `greedy` | 7.55 | 31.52 | 70.40 | 1.02 |
| `utility_greedy` | 8.27 | 39.07 | 98.24 | 1.02 |
| `fuel_aware` | 7.27 | 37.70 | 87.01 | 1.05 |
| `stock_maximiser` | 6.21 | 27.48 | 71.85 | 1.04 |
| `coordinated` | 8.02 | 34.03 | 77.78 | 1.13 |
| `local_search` | 8.15 | 42.09 | 108.96 | 1.34 |
| `lns` | 8.46 | 56.22 | 218.71 | 5.21 |
| `alns` | 8.46 | 56.22 | 218.71 | 5.27 |
| `aco` | 8.47 | 55.09 | 195.26 | 1.32 |
| `aco_ls` | 8.47 | 55.04 | 196.99 | 2.82 |

On the local full suite, ALNS versus ACO+LS is `108 wins / 57 ties / 27
losses`; versus local search it is `175 / 14 / 3`. These are official
lexicographic comparisons, not a weighted score.

Online practice suite: four authoritative maps (Q01, Q02, Q03, and New
Question), all ten policies valid on every map. Values below are means across
the four maps; `rank-1` counts official-score wins, including ties.

| Policy | Distinct | Daily | Servings | Runtime (s) | Rank-1 |
|---|---:|---:|---:|---:|---:|
| `greedy` | 7.25 | 32.50 | 87.75 | 3.97 | 0 |
| `utility_greedy` | 7.25 | 35.00 | 107.75 | 3.58 | 0 |
| `fuel_aware` | 7.25 | 33.00 | 100.25 | 3.74 | 0 |
| `stock_maximiser` | 7.00 | 30.25 | 95.50 | 3.16 | 0 |
| `coordinated` | 7.25 | 34.75 | 103.75 | 3.28 | 0 |
| `local_search` | 7.25 | 34.50 | 112.00 | 3.31 | 0 |
| `lns` | 7.25 | 35.75 | 152.00 | 6.62 | 4 |
| `alns` | 7.25 | 35.75 | 152.00 | 5.26 | 4 |
| `aco` | 7.25 | 35.75 | 150.25 | 4.14 | 1 |
| `aco_ls` | 7.25 | 35.75 | 151.25 | 3.65 | 2 |

On Q01, the fixed online ALNS ties ACO+LS at `4/28/124`; on Q02 all four
search policies reach `10/40/132`; on Q03 ALNS/LNS reach `10/40/136`; and on
New Question ALNS/LNS reach `5/35/216`.

The report records validity first and then the official deterministic score
tuple: distinct brands, cumulative daily brands, and servings. Runtime is
reported but is not combined into an invented scalar score.

Grading parallelizes independent case/method processes (`--jobs 0` selects up
to eight workers). Each worker pins the C++ core to one thread to avoid nested
oversubscription. Live deployment instead uses bounded C++ parallelism for
coordinated resource paths and local-search candidate scoring. Set
`HEXUDON_THREADS=N` to control that inner limit; it defaults to at most eight.

## Live reads and deployment

Put the team JWT in an untracked `.env` file:

```text
TOKEN=...
```

Fetch a read-only contract fixture:

```sh
uv run hexbench fetch --game-id QUESTION_ID --out fixtures/live.json
```

Exercise the deployment path without any writes:

```sh
uv run hexbench deploy --game-id QUESTION_ID --method greedy --dry-run --once
```

Remove `--dry-run` only when the process should select agent types and submit
plans. The client detects practice versus timed games, validates every plan in
the C++ simulator, uses an all-wait fallback if necessary, and journals
accepted days under `.hexbench-state/`. It never resets, copies, generates, or
deletes a server game. `--once` stops after the current state/day; without it,
the client polls until the match finishes.

`play` is an alias for `deploy`. The same command automatically selects
`/game/practice/actions` for practice games and `/game/actions` for real timed
games:

```sh
uv run hexbench play --game-id QUESTION_ID --method local_search
```

To compare policies on one resettable practice game without any manual HTTP
commands, run:

```sh
uv run hexbench practice-benchmark \
  --game-id QUESTION_ID \
  --methods greedy,utility_greedy,fuel_aware,stock_maximiser,coordinated,local_search,lns,alns,aco,aco_ls \
  --report reports/practice
```

The command verifies `is_practice=true` and `no_reset=false` before writing,
resets and completes the same game for every policy, ranks the authoritative
server scores, reads other configured teams through `practice/peer` and
`practice/score`, and leaves the winning policy's completed run on the server.
Use `--leave-last` only when the final listed method should remain instead.
Peer ids are discovered automatically when the manager service is available;
use `--peer-team-ids 6,8,18` to supply them explicitly or
`--peer-team-ids none` to skip peer baselines.

To run that workflow across every practice map that is safe to reset, use:

```sh
uv run hexbench practice-suite \
  --methods local_search \
  --report reports/practice-suite
```

This discovers the team's assigned questions, excludes real games and
`no_reset` practice games, completes each selected policy on every remaining
map, and leaves the best run on the server. `summary.md` shows two ranks per
map: the rank among players who completed every day, and a provisional rank
against all currently available partial/unstarted scores. Per-map score and
peer evidence is retained in subdirectories. Use `--game-ids ID1,ID2` to run a
subset; multiple policies can be supplied as a comma-separated `--methods`
list.

## Fuel-stress benchmark

To measure patrol/refueling behavior on the same authoritative practice maps
without modifying or resetting the server games, run:

```sh
uv run hexbench fuel-benchmark \
  --methods lns,alns,local_search,aco,coordinated,fuel_aware \
  --fuel-multipliers 1.0,0.5,0.25 \
  --report reports/fuel-stress
```

The server fuel limit is retained as a baseline. Additional cases preserve the
map, days, terrain, spots, stocks, and initial positions while changing only
`fuelLimits` to the requested multiple of Day-1 steps. Evaluation is entirely
local, and the report includes selected patrol/refueling counts and actual
refueling events.

To measure how LNS quality changes with a larger per-day response budget, run:

```sh
uv run hexbench lns-time-benchmark \
  --method alns \
  --fuel-multiplier 0.5 \
  --time-limits-ms 25,100,500,2000,10000,60000 \
  --report reports/lns-time-curve
```

Timed runs disable stagnation stopping and use an intentionally unreachable
iteration cap, so the soft deadline becomes the stopping rule. A 60-second
point is applied independently to every day and can therefore take several
minutes per map. The web dashboard exposes the same curve controls.

## Web dashboard

Start the browser interface on the competition machine with:

```sh
uv run hexbench web --host 0.0.0.0 --port 5678
```

Open `http://localhost:5678` locally, or use the machine's network address from
another device. The dashboard lists only assigned practice games that are safe
to reset, lets you select one or more policies, and offers two modes: an
authoritative server benchmark and a read-only local fuel-stress benchmark with
editable fuel multipliers. It displays policy scores plus completed and
provisional peer rankings or fuel/refueling telemetry. Jobs are serialized to
prevent concurrent resets from corrupting a run. The token stays in the server
process; generated reports are stored under `reports/web/`.

## Adding a method

Implement the `select_agent_types` and `plan_day` policy branches declared in
`include/hexudon/core.hpp`, register the same policy name in the evaluator,
and grade it against `wait,greedy`. A policy receives only the official map and
daily information plus history reconstructible from its own accepted plans;
future road states and simulator internals are not exposed.

`lns` and `alns` also accept an internal `search` object in the C++ `plan` command
(including `timeLimitMs`, `minIterations`, `maxIterations`, and
`stagnationIterations`). This client-side budgeting hint is not part of the
official action protocol; omitting it selects deterministic fixed-iteration
search for local grading. Live practice and competition deployment default to
the same bounded online search (`2048` maximum iterations, `96` stagnant
iterations); a user-supplied time limit explicitly opts into exhaustive timed
benchmarking.

Budgets above 96 iterations activate systematic exact branch-and-bound after
the ALNS warm start. A completed final-day enumeration proves the official
daily optimum. On earlier days, exact candidates preserve the incumbent's end
positions, acquired new-brand set, traffic, and patrol fuel so additional
search cannot damage the observable continuation state. Timed searches enable
the exact phase at five seconds and always retain the best valid incumbent.
The promoted exact phase uses admissible reachability, per-patrol serving,
stock-allocation, and no-refuel fuel bounds. They only prune branches whose
best possible lexicographic score cannot beat the incumbent, so they improve
search throughput without weakening eventual optimality.

Production ALNS is online: planning reads only the current day's revealed step
horizon, traffic, positions, fuel, and accepted history. It does not use future
`daySteps`, `daySeconds`, or predicted future traffic. Every day begins with a
protected ACO+LS incumbent under the same fixed agent types; ALNS returns only
a lexicographically better current-day score and preserves the ACO+LS plan on
ties instead of guessing which ending state a hidden future schedule will
favor. Short timed budgets use a reduced ACO seed.
This removes the practice-only match beam whose assumed future schedule could
make longer searches non-monotone.

Maps with more than 15 spots reuse the base rendezvous cache instead of adding
transit nodes and rebuilding the quartic meeting table. Agent types are also
selected without reading `daySteps`, `daySeconds`, or daily traffic: every
assignment with zero to three refuelers is tested over seven synthetic cycles,
map-derived 1x/2x/4x horizons, and smooth/jammed uncertainty. Static fuel and
service-pressure guards prevent unsafe zero-refuel and over-supplied splits
when the real schedule is unknown. The rollout chooses the number of
refuelers; a one-refuel assignment uses the same map-only remote-start choice
as ACO+LS, and three-agent teams retain at least two patrol cars. This makes the
protected daily ACO+LS incumbent use the same fixed types.

The heuristic-rule ablation is retained under
`reports/alns-heuristic-ablation/`. Nine repair-ranking rules (scarcity,
last-chance, reachability slack, stock, rendezvous, fuel, opportunity cost,
congestion, and end positioning) were not promoted: each lost one of 30 quick
cases to the frozen control and won none. Exact branch ordering and dominance
caching were also rejected due to runtime overhead. The four promoted bounds
tied all 30 high-budget scores and reduced serial runtime by 6.4% as the final
production configuration; the 192-case default-budget gate had 192 ties and
zero invalid plans.

The web console exposes the supported controls per selected policy. Values are
validated server-side, retained in the job record, and forwarded only to that
policy; leaving a field blank uses the compiled default.

ACO and ACO-LS do not impose artificial upper limits on ant count or iteration
count. Positive values are accepted up to the C++ integer representation, and
pheromone retention may be any value strictly between 0 and 1.

## Converted policies

The policy registry includes the methods ported from the adjacent historical
Python implementation. All now emit complete official action arrays and run
through this repository's C++ validator/simulator:

| Policy | C++ adaptation |
|---|---|
| `greedy` | Prioritise unseen match/day brands, then travel cost and stock. |
| `utility_greedy` | Weighted new-brand, daily-brand, portion, and travel-time utility. |
| `fuel_aware` | Require a round-trip fuel reserve and retreat toward refueling cars. |
| `stock_maximiser` | Prefer maximum unreserved stock, then new brands and distance. |
| `coordinated` | Resource-constrained paths, joint type assignment, and refuel staging. |
| `local_search` | Start from coordinated and test valid whole-plan/route substitutions using the official objectives. |
| `lns` | Promoted ALNS with stable rendezvous choices, shared preprocessing, and admissible exact-search resource bounds. |
| `alns` | Explicit alias for the promoted ALNS, retained for benchmark and tuning workflows. |
| `aco` | Pure ant-colony search over complete patrol routes and synchronized mobile-refueling rendezvous. |
| `aco_ls` | ACO whose every ant is refined by one-pass route-substitution local search before ranking and selection. |

For example:

```sh
uv run hexbench grade --cases cases/quick/manifest.json \
  --method aco \
  --baselines greedy,coordinated,local_search
```
