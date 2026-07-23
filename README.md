# HEXUDON benchmark and deployment client

This repository contains a deterministic local HEXUDON environment and a
guarded client for the live server. The C++20 core owns simulation, scoring,
and policies; Python owns generation, reporting, and HTTP orchestration. Both
local grading and live deployment invoke the same compiled policy code.

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The build uses the system Boost.JSON library plus the already-vendored `fmt`
and `cxxopts` headers. It produces `build/hexudon`; no regression suite is run
as part of the build. No token is needed for local generation or grading.

## Generate and grade locally

```sh
uv run hexbench generate --suite quick --out cases/quick
uv run hexbench generate --suite full --out cases/full
uv run hexbench grade \
  --cases cases/quick/manifest.json \
  --method greedy \
  --baselines wait,greedy \
  --report reports/quick-greedy \
  --jobs 8
```

### Hard curated suite

The random `quick`/`full` suites are too easy: the reference solver saturates
the structural optimum of every objective, so the grade cannot separate a
strong policy from a weak one. `generate-hard` builds a small, solver-verified
suite with three graded tiers, each engineered so that exactly one objective
stays below 100% while every higher-priority objective is fully reachable:

| Tier | Target objective held below 100% | How it is made hard |
| :--- | :--- | :--- |
| `brutal` | distinct types | large map, every spot a distinct brand spread to far corners, mountain/pond terrain, minimal fuel/agents/steps — even one bowl of every type is impossible |
| `steady` | cumulative daily types | brands reachable across the whole match, but no single day can touch every brand |
| `easy` | total servings | every brand every day is reachable, but high stocks and many spots outrun the fleet |

```sh
uv run hexbench generate-hard --out cases/hard --per-tier 6
uv run hexbench grade \
  --cases cases/hard/manifest.json \
  --method alns --baselines greedy,local_search \
  --report reports/hard --jobs 8
```

Each candidate is verified with the strongest policy (ALNS by default): only
cases whose measured percentages land in the tier's target band are kept, so the
suite is genuinely discriminating rather than merely nominally "hard". The
generator writes one grade-ready sub-suite per tier
(`cases/hard/<tier>/manifest.json`) plus a combined `cases/hard/manifest.json`
that grades every tier at once; each case records its verified percentages under
`verification`. Pass `--tiers brutal,steady` to build a subset or `--no-verify`
for a fast constructive-only draft.

To tune ALNS locally, evaluate a deterministic grid of explicit ALNS iteration
budgets on the same manifest:

```sh
uv run hexbench alns-tune \
  --cases cases/quick/manifest.json \
  --alns-iterations 128,256,512,1024,2048,3072,4096,6000 \
  --min-iterations 32 \
  --stagnation-iterations 0 \
  --report reports/alns-tuning
```

The tuner ranks candidates by the same lexicographic macro-average as the
grader (distinct brands, cumulative daily brands, then servings); runtime is
only a tie-breaker. It uses explicit ALNS iteration counts rather than wall-clock
limits so a saved report can be reproduced on another machine. The selected
configuration and every candidate result are written to `report.json` and
`report.md`.

Seed construction can be ablated on the same deterministic grid with
`--seed-profiles`. Profiles isolate the protected ACO seed, the legacy-LNS
diversification seed, and the standalone local-search seed without requiring
separate binaries:

```sh
uv run hexbench alns-tune \
  --cases cases/quick/manifest.json \
  --alns-iterations 512 \
  --seed-profiles production,no_local,no_legacy,no_local_legacy,reduced_aco,reduced_no_local,reduced_no_legacy,reduced_minimal,no_aco \
  --report reports/alns-seed-ablation
```

The July 2026 ablation found no seed phase that was globally redundant.
Reduced ACO without the local-search seed helped Q04 and the 30-case high-depth
sample, but production remained lexicographically better on the complete
1,000-case suite and a 100-case stratified 3,072-iteration check. Production
therefore retains all three seed phases.

Production `alns` runs three independent same-day ALNS chains. Restarts zero
and one are the protected production searches with the established operator
portfolio and independent deterministic random streams. Restart two is a
reduced-seed diversification chain with a HEX-adapted SISR recreate arm: it
samples random, official-brand-tier, close-to-start, KNN-2, and KNN-5 visit
orderings, uses Shaw-biased selection, and occasionally blinks the cheapest
feasible insertion. All chains start from the identical revealed day state. A
non-final alternative is accepted only when it strictly improves the official
score while preserving ending positions, fuel, generated traffic, and
resulting distinct-brand history. On the final day, any strict official
improvement is accepted. Official ties always retain restart zero.
The 1,000-case 512-iteration promotion gate produced 27 wins, 973 ties, and no
losses versus one restart, improving mean normalized servings from 67.4378% to
67.4468% without changing normalized distinct or daily coverage. The restart
count is fixed: `alns` always runs all three chains and `lns` remains
single-restart.

The SISR chain passed the complete 1,000-case 512-iteration promotion gate
against the protected two-chain solver with `29` wins, `971` ties, and no
losses. Mean normalized distinct coverage stayed at `97.55990%`; daily coverage
improved from `96.26825%` to `96.27004%`, and servings improved from
`67.44677%` to `67.45301%`. Aggregate benchmark compute increased by `20.2%`.

`quick` contains 30 balanced smoke cases. `full` contains 1,000 cases from a
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

For ALNS component ablations, generate an equal-size profile-stratified suite:

```sh
uv run --no-sync hexbench generate-validation \
  --out cases/alns-validation --per-profile 32
```

This writes `hard`, `medium`, and `easy` manifests with 32 deterministic cases
each, mapped to the existing curated recipes as `brutal`, `steady`, and `easy`.
Cases are selected by seed position, never by the current ALNS score, so the
suite does not inherit the component behavior it is meant to test.

Local grading is based on average normalized performance, not a requirement
to beat every case. For each case, the structural optimum is all brands, all
brands on every day, and all stock collectable under the one-visit-per-patrol
rule, ignoring travel, fuel, and congestion feasibility. Each objective is
macro-averaged across cases, so every map has equal weight. Policies are then
ranked lexicographically by `(average distinct %, average daily %,
average servings %)`: daily coverage matters only when distinct coverage ties,
and servings matter only when both earlier components tie. Invalid cases
contribute 0% to every component.

## Official-flow policy benchmark

All policy benchmarks use the competition information boundary from
[`qa.md`](/home/shora/Research/hex/qa.md:55): the complete `daySteps` and
`daySeconds` schedule is available in the initial map configuration, while
daily planning additionally receives only the current day's traffic,
positions/fuel, and accepted history. The planner may use the known schedule
for role selection and continuation-aware search, but it does not assume
future traffic or future agent actions. The online suite resets each practice
map between policies and restores the best policy afterward.

The current all-policy snapshots are saved in
[local-full-all](/home/shora/Research/hex/reports/local-full-all/report.md) and
[online-practice-all](/home/shora/Research/hex/reports/online-practice-all/summary.md).

The table below is the historical 192-case snapshot retained for reference;
new local acceptance decisions use the 1,000-case normalized grade described
above. Values below are means per case; runtime is the aggregate policy
runtime divided by 192.

| Rank | Variant | Valid | Brutal | Steady | Easy | Overall | Runtime (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `mlns+dp` | 300/300 | 73.138 / 35.398 / 35.398 | 100.000 / 87.070 / 48.175 | 100.000 / 100.000 / 61.433 | 91.046 / 74.156 / 48.335 | 9678.649 |
| 2 | `alns+dp` | 300/300 | 72.202 / 35.277 / 35.277 | 100.000 / 85.241 / 44.960 | 100.000 / 100.000 / 56.576 | 90.734 / 73.506 / 45.604 | 1870.564 |
| 3 | `palns+dp` | 300/300 | 71.670 / 34.890 / 34.890 | 100.000 / 86.539 / 47.557 | 100.000 / 100.000 / 55.352 | 90.557 / 73.810 / 45.933 | 1884.412 |
| 4 | `mlns` | 300/300 | 63.715 / 30.682 / 30.682 | 100.000 / 82.816 / 43.017 | 100.000 / 100.000 / 61.374 | 87.905 / 71.166 / 45.024 | 6752.330 |
| 5 | `palns` | 300/300 | 60.781 / 30.617 / 30.617 | 99.800 / 80.922 / 42.350 | 100.000 / 100.000 / 55.469 | 86.860 / 70.513 / 42.812 | 1813.108 |
| 6 | `alns` | 300/300 | 60.285 / 31.338 / 31.338 | 99.657 / 83.127 / 43.475 | 100.000 / 100.000 / 57.306 | 86.647 / 71.488 / 44.040 | 1932.732 |

The historical snapshot used pairwise lexicographic counts. New reports rank
policies by their average percentage of the structural optimum and report the
percentage-point difference between policies.

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

On Q01, congestion-aware ALNS with 1,024 ALNS iterations reaches
`4/28/126`, ahead of the saved ACO+LS result `4/28/124`; on Q02 all four
search policies reach `10/40/132`; on Q03 ALNS/LNS reach `10/40/136`; and the
explicit recommended ALNS profile reaches `5/35/219` on New Question. That
profile runs 1,536 normal-day and 1,024 final-day ALNS iterations, a 2,048
iteration LNS warm start, and 512 normal-day / 1,024 final-day exact nodes.

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

## Traffic GNN prototype

Generate the reusable dataset once. Maps are drawn from the graded hard suite
and balanced evenly across the `brutal`, `steady`, and `easy` tiers, so the
model sees road-condition labels from discriminating scenarios rather than the
saturated random profiles. Every simulated player uses LNS (pass `--policy alns`
for the heavier adaptive search) with a distinct deterministic seed:

```sh
uv run hexbench traffic-generate \
  --train-cases 800 \
  --validation-cases 200 \
  --alns-iterations 16 \
  --jobs 8 \
  --out datasets/traffic-gnn
```

This is the only stage that runs the authoritative C++ simulator. It writes a
versioned `dataset.pt` containing CPU tensors and a `manifest.json` containing
the instance split, label counts, player seeds, distinct-plan counts, byte size,
and SHA-256 digest. Each completed instance is checkpointed under `shards/`, so
rerunning the same command resumes interrupted generation. Existing complete
datasets are protected unless `--overwrite` is passed.

Train repeatedly from those saved tensors without running any simulation:

```sh
uv run hexbench traffic-train \
  --dataset datasets/traffic-gnn/dataset.pt \
  --epochs 200 \
  --batch-size 64 \
  --patience 30 \
  --report reports/traffic-gnn
```

The model predicts each road's day-`t` smooth/busy/jammed label from map
features, normalized axial coordinates, all players' start-of-day agents, and
road statuses through day `t-1`. The loss is unweighted cross entropy over road
nodes only. The report directory receives `report.json` with per-epoch metrics
and `model.pt` with the best-validation checkpoint, dataset digest, and feature schema.
Use `--device cpu` for a CPU-only run or leave `auto` to use an available GPU.

PyTorch is intentionally resolved from the normal Python package index rather than
being pinned to an AMD ROCm wheel. On Linux this provides the standard CUDA-enabled
PyTorch build; on other platforms uv selects the compatible published wheel. If an
AMD machine needs ROCm, install the matching official ROCm wheels after `uv sync`,
for example:

```sh
uv pip install \\
  --index-url https://download.pytorch.org/whl/rocm6.3 \\
  torch torchvision torchaudio pytorch-triton-rocm
```

## Web dashboard

Start the browser interface on the competition machine with:

```sh
uv run hexbench web --host 0.0.0.0 --port 5678
```

Open `http://localhost:5678` locally, or use the machine's network address from
another device. The offline, bilingual browser interface follows the official
HEXUDON game layout: choose an assigned game, inspect its configuration and
map, select agent types, edit day plans, view answers, and play resolved days
back step by step. Reset and earlier-day resubmission controls are offered only
for resettable practice games; real matches and `no_reset` practice
competitions never expose reset.

The `LOCAL` tab does not require a server token. It discovers the checked-in
manifest suites (`brutal`, `steady`, `easy`, `quick`, and `full`), runs a
selected compiled policy on an exact case, and displays achieved/structural
maximum scores plus authoritative day-by-day simulator playback. The timeline
shows route trails, agent fuel, collected spots, traffic, and the generated
action plan for every day.

The planner panel beside the official-style answer box supports three modes:

- **Manual** generates a validated plan with any registered policy, fills the
  editable JSON box, and submits only when `Submit day plan` is pressed.
- **Auto** validates and submits normal role/day proposals until the match is
  complete. It pauses for review instead of silently submitting an all-wait
  fallback after a planner failure.
- **curl** validates the current editor value and explicitly reveals an
  executable command. This command contains the real bearer token, so reveal
  and copy it only on a trusted machine; the token is excluded from normal UI
  responses, logs, and durable session files.

Play sessions and proposal traces are stored under `reports/web/` and reloaded
after a dashboard restart. The static UI uses no CDN or internet-hosted assets.
The benchmark, fuel-stress, and time-curve workflows remain available through
their CLI commands even though they are no longer separate dashboard screens.
Ordinary dashboard server benchmarks use 2,048 ALNS iterations per day for
LNS/ALNS when neither a time nor ALNS-iteration limit is supplied. This is the
best full-match ALNS setting found by the local quality sweep: larger iteration
can improve a day's isolated score while leaving a worse continuation state.
Enter a wall-clock limit for a deeper anytime run, use the time-curve mode to
compare budgets, or use `deploy` for the full safe competition window.

## Adding a method

Implement the `select_agent_types` and `plan_day` policy branches declared in
`include/hexudon/core.hpp`, register the same policy name in the evaluator,
and grade it against `wait,greedy`. A policy receives only the official map and
daily information plus history reconstructible from its own accepted plans;
future road states and simulator internals are not exposed.

`lns` and `alns` accept an internal `search` object in the C++ `plan` command.
The quality/runtime controls are explicit:

- `timeLimitMs`: wall-clock limit; mutually exclusive with untimed ALNS iterations.
- `maxIterations`: ALNS iteration cap in timed mode; `minIterations` and
  `stagnationIterations` control the lower bound and early stopping.
- `seedIterations`: bounded legacy-LNS seed work before ALNS.
- `exactNodes`: final-day exact-search node allowance; `0` disables it and `-1`
  selects the automatic cap.
- `duplicateWarmupIterations`: how long repeated candidates are still decoded.
- `annealingEpoch` and `coolingSteps`: deterministic reheating/cooling schedule.
- `acoAnts`, `acoIterations`, and `acoEvaporation`: ant count, rounds, and
  pheromone retention for the protected ACO seed.

For untimed search, use `alns_iterations` in the web/API hyperparameter map;
the planner converts it to `maxIterations`. These controls are not part of the
official action protocol. Omitting them selects compiled defaults. Live practice
and competition deployment make the safe wall-clock deadline authoritative for
LNS/ALNS, while explicit controls remain available for controlled benchmarks.

ALNS iteration counts are direct loop counts. Counts above 96 also enable a
bounded exact branch-and-bound add-on on the final day, after the ALNS warm
start; a completed enumeration proves the official final-day optimum. Exact
candidates are not constrained to reproduce an incumbent's arbitrary ending
positions: the final day is optimized against its revealed traffic, and earlier
days retain their full ALNS budget so continuation state is not replaced by a
daily-only exact solution. An exact candidate replaces the incumbent only when
it strictly improves the three official objectives; ties do not exchange one
unscored continuation state for another. Timed final-day searches reserve the
configured exact-search percentage only when a positive exact-node budget is
configured; otherwise ALNS uses the complete deadline. Every phase
always retains the best valid incumbent.
The promoted exact phase uses admissible reachability, per-patrol serving,
stock-allocation, and no-refuel fuel bounds. They only prune branches whose
best possible lexicographic score cannot beat the incumbent, so they improve
search throughput without weakening eventual optimality.

Production ALNS is online: planning reads the current day's revealed traffic,
positions, fuel, and accepted history, plus the complete schedule supplied in
the initial config. It never reads future traffic or opponent actions. Its
continuation look-ahead simulates future days with a symmetric self-traffic
surrogate, shares the reserved wall-clock window across candidate roots and
future days, and stops at the live solver deadline so the selected plan can be
streamed before submission closes.
Every day begins with a protected ACO+LS incumbent under the same fixed agent
types. The main ALNS loop retains only a lexicographically better current-day
incumbent; continuation look-ahead may choose an equal-realized-distinct plan
when its simulated remaining-match rank is strictly better. Ending positions
are free decision variables rather than equality constraints. Timed seed
construction receives a proportional slice of the current-day ALNS budget.

`palns` is the projection-guided ALNS variant. It preserves the realized
current-day official triple as the primary objective and uses a simulated final
match triple only to break exact ties. Every unique best-tier continuation
state is projected immediately; current-day and nested future-day ALNS loops
draw from one `total_iterations` ledger. Predicted ending patrol fuel breaks
ties between equal projected official triples, while projection-worse states
remain eligible for ordinary ALNS acceptance so the noisy forecast does not
turn plateau exploration into greedy hill climbing. Non-final days therefore
have no separate continuation-time share. A timed final day is split only when
exact completion is enabled. Evaluation JSON includes `palns_diagnostics`,
including outer/projection iteration counts and the percentage of projection
requests that safely fell back because the remaining iteration or time budget
could not complete every future day.

`mlns` is the simpler rolling whole-match alternative. One candidate contains
route skeletons for every remaining day. The current day is replayed with the
road conditions supplied by the server; later days are replayed recursively
with the existing symmetric-opponent traffic assumption. A mutation ruins and
repairs a geometrically selected pivot or contiguous multi-day block, then
re-decodes downstream skeletons from the resulting positions, fuel, brand
history, and predicted roads. Untimed full-match runs initialize a bounded beam
of distinct partial-match states, retaining equal-scoring daily plans when they
end in different positions, fuel, or traffic. ACO, escort, local-search, and
bounded fixed-type ALNS plans are proposal generators for that beam; only the
weighted complete-match objective selects the winner. Neighborhood iterations evaluate
both a fully re-decoded suffix and a still-legal retained-action suffix, and can
mutate contiguous multi-day blocks. Short timed requests keep the lightweight
single-day/retained-suffix path so initialization cannot consume the deadline.
There are no projection caches, adaptive MCTS operators, or multi-restarts.
Beam states, independent proposal generators, proposal simulations, and the two
suffix-decoding alternatives run in parallel. Nested worker accounting divides
`HEXUDON_THREADS` across those levels, preventing outer beam workers from each
spawning a full inner pool. The Web server still serializes practice jobs to
protect reset state, but each active C++ MLNS process can use multiple cores.

MLNS compares the exact lexicographic vector of discounted marginal distinct
brands, daily types, and servings. The default horizon weights are `1`, `0.5`,
`0.5²`, and so on; `future_discount_percent` changes the ratio. This permits a
better weighted match plan to collect less on the current day. The unexecuted
suffix is serialized in the local controller journal and re-rooted against the
next day's actual traffic and agent state. Alongside spot-index route skeletons,
the state stores exact suffix actions and their predicted starting signature;
an action plan is replayed only when the revealed agents and traffic match that
signature. The state is never sent to the competition server; stale state whose
map, types, source day, or committed action does not match accepted history is
discarded automatically.

`simple_lns` is a separate rolling SISR policy with a classical route genome.
Each patrol stores only its ordered udon spots. Each refuel vehicle stores
semantic `{target patrol, escort tiles}` tasks; the decoder chooses the earliest
feasible interception cell on that patrol's predicted path and sends the
refueler there by shortest path. Unknown future roads are held at today's
revealed status, and the saved suffix is repaired when the next day is revealed.
The search adapts string/split-string removal, Shaw-biased recreate orderings,
blink insertion, and simulated annealing from the Apache-2.0
`open-source-sisr-routing` implementation by Martin Pajersky, Vaclav Sobotka,
and Hana Rudova. SISR-specific constants are fixed; the public controls are the
ordinary deadline/iteration/seed controls and `future_discount_percent`.
The untimed default performs 128 destroy-and-repair moves with stagnation
stopping disabled; the final official lexicographic match score ranks solutions,
while the 90% future discount only breaks official-score ties.
Patrol strings are repaired first. Later neighborhoods also remove contiguous
strings from the current day's refuel task routes and repair task allocation and
order across refuel vehicles; rendezvous cells remain decoder-derived rather
than stored coordinates.

MLNS remains experimental. The full deterministic 1,000-case design completed
with `1,000/1,000` valid cases and zero invalid days when exercising cold/warm
construction without additional search iterations. A 12-case hard-profile
screen at `1 s/day` produced four lexicographic wins, four ties, and four losses
against ALNS. MLNS collected 90 total distinct brands versus ALNS's 86, while
ALNS retained the secondary cumulative-daily advantage (`211` versus `201`).
This is encouraging but is not the complete 96-case promotion gate, so MLNS
remains opt-in. A subsequent 32-trial Bayesian search at `1 s/day` completed
all 96 hard/medium/easy validation cases for every candidate with zero invalid
days. It selected `min_iterations=32`, `stagnation_iterations=16`, and
`future_discount_percent=90`. Against the previous `32/0/50` default, the
normalized macro score changed from `88.7708 / 71.2049 / 45.3228` to
`88.8546 / 71.0422 / 45.3994`; the primary distinct-coverage gain therefore
wins lexicographically. Total/max iterations were not tuned and remain a high
safety ceiling under the Web deadline.

Optimize MLNS on the complete deterministic 96-case hard/medium/easy suite
with the resumable Tree-structured Parzen (Bayesian) search:

```sh
uv run --no-sync hexbench mlns-tune \
  --cases cases/alns-validation/manifest.json \
  --trials 32 \
  --time-limit-ms 1000 \
  --report reports/mlns-validation
```

Every trial sees all 96 cases under the same per-day wall-clock budget. The
optimizer searches minimum iterations, stagnation stopping, and the future-day
discount; the total/max iteration ceiling is intentionally fixed outside the
search so the Web UI can remain deadline-governed. Trials are ranked by the
official lexicographic normalized distinct/daily/servings vector; validity is
a hard gate and runtime only breaks an exact quality tie. `state.json` is
checkpointed after every case and the same command resumes it. Use `--no-resume`
to start over. The final directory contains `report.json`, `report.md`, and a
Web-UI-ready `best-search.json` that omits total iterations.

Tune the fixed projection depth and restart count, then report the public total
iteration curve on the 96-case brutal/steady/easy validation suite with:

```sh
uv run --no-sync hexbench palns-tune \
  --cases cases/alns-validation/manifest.json \
  --report reports/palns-validation
```

The production anytime allocation is scale-free and uniform across CLI and Web
Competition runs:

- On every non-final day, current-day ALNS and continuation simulation each
  receive 50% by default. Both phases are deadline-governed;
  there is no hidden iteration or stagnation cap below the explicit controls.
- Continuation time is divided fairly across the incumbent and distinct elite
  roots, then across every remaining simulated day.
- On the final day, ALNS uses 100% unless exact search is enabled with a
  positive node budget. With exact search, ALNS uses 70% and exact completion
  uses the remaining 30%.
- Explicit iteration and exact-node limits remain safety ceilings for controlled
  benchmarks. The Web UI supplies a high iteration ceiling and zero stagnation,
  making its requested wall-clock value authoritative.
- `continuation_time_percent` and `exact_time_percent` are manual tuning
  controls. Changing them preserves the same percentage rule for every
  `daySeconds` value rather than introducing duration-specific branches.

Maps with more than 15 spots reuse the base rendezvous cache instead of adding
transit nodes and rebuilding the quartic meeting table. Agent types are also
selected from the published `daySteps`/`daySeconds` schedule without reading
daily traffic: every assignment with zero to three refuelers is tested over
the actual schedule plus three seven-cycle synthetic schedules with
map-derived 1x/2x/4x horizons and smooth/jammed uncertainty. Static fuel and
service-pressure guards prevent unsafe zero-refuel and over-supplied splits.
The rollout chooses the number of
refuelers; a one-refuel assignment uses the same map-only remote-start choice
as ACO+LS, and three-agent teams retain at least two patrol cars. This makes the
protected daily ACO+LS incumbent use the same fixed types.

Historical heuristic-rule results remain under
`reports/alns-heuristic-ablation/`, but the unpromoted repair-ranking rules,
lookahead/match-beam variants, exact-repair variant, and experimental policy
aliases are no longer part of the supported C++ policy surface. Each lost one
of 30 quick cases to the frozen control or added runtime. Production keeps the
four validated exact bounds, stable travel, shared preprocessing, exact
completion, and the ACO seed. The promoted bounds tied all 30 high-budget
scores and reduced serial runtime by 6.4%; the 192-case default-budget gate
had 192 ties and zero invalid plans.

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

ALNS uses a small Monte-Carlo tree over `(destroy, repair, travel)` operator
tuples. Each validated candidate is a rollout: its bounded score signal is
backpropagated through the selected path, and subsequent choices use UCB1 to
balance unexplored operator combinations against combinations that have
performed well. The seeded generator keeps this adaptive policy reproducible.

| Policy | C++ adaptation |
|---|---|
| `greedy` | Prioritise unseen match/day brands, then travel cost and stock. |
| `utility_greedy` | Weighted new-brand, daily-brand, portion, and travel-time utility. |
| `fuel_aware` | Require a round-trip fuel reserve and retreat toward refueling cars. |
| `stock_maximiser` | Prefer maximum unreserved stock, then new brands and distance. |
| `coordinated` | Resource-constrained paths, joint type assignment, and refuel staging. |
| `local_search` | Start from coordinated and test valid whole-plan/route substitutions using the official objectives. |
| `lns` | Promoted ALNS with stable rendezvous choices, shared preprocessing, MCTS operator selection, and admissible exact-search resource bounds. |
| `alns` | Explicit alias for the promoted ALNS; its destroy/repair/travel operators are selected by a deterministic UCB tree that backpropagates validated rollout quality. |
| `simple_lns` | Rolling whole-match SISR over spot-only patrol routes and shortest-path refuel interception tasks; its state is repaired against each newly revealed road map. |
| `lns_dp` | Independent request-bank LNS with full-recharge fuel DP, global rendezvous scheduling, traffic-filtered terminal value, scenario finalist grading, and two-day adaptive recourse. |
| `aco` | Pure ant-colony search over complete patrol routes and synchronized mobile-refueling rendezvous. |
| `aco_ls` | ACO whose every ant is refined by one-pass route-substitution local search before ranking and selection. |
| `stop_bp` (`bp`) | Branch-and-price over fuel-relaxed patrol routes: a set-partition master LP solved with an in-house revised simplex, label-setting pricing with dominance, and lambda-branching. Warm-started from a short ALNS run, which is also returned when the instance exceeds the spot/brand cap or the deadline expires. |

LNS-DP can also supply additive route proposals to `alns`, `palns`, and `mlns`.
Enable **Use experimental LNS-DP proposals** in the Web UI, pass the
`use_lns_dp_proposals: true` hyperparameter, or set
`HEXUDON_ENABLE_LNS_DP_PROPOSALS=1`. This remains experimental: it improves
aggregate validation quality but has not passed the per-case no-regression
promotion gate. The proposal contains a stock-expanded patrol
visit order, a simulator-valid action seed, and residual-fuel annotations.
ALNS/PALNS rank the direct seed with their existing official/projection
objectives and can re-decode the visit order through the established rendezvous
graph; MLNS evaluates it as another whole-match root. The existing ALNS
rendezvous decoder, PALNS projection logic, and MLNS continuation beam remain
authoritative.

For example:

```sh
uv run hexbench grade --cases cases/quick/manifest.json \
  --method aco \
  --baselines greedy,coordinated,local_search
```
