// Adapted from the SISR control structure in open-source-sisr-routing
// (Copyright 2025 Martin Pajersky, Vaclav Sobotka, Hana Rudova), licensed
// under Apache-2.0.  This is a HEX-specific reimplementation: depots, time
// windows and scalar prizes are replaced by rolling daily routes, the official
// lexicographic objective and mobile-refuel synchronization.

#include "hexudon/core.hpp"
#include "hexudon/internal.hpp"

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

namespace hexudon {
namespace {

using Wide = boost::multiprecision::uint128_t;

struct SimpleRefuelTask {
  int target_patrol{-1};
  int escort_tiles{};

  bool operator==(const SimpleRefuelTask&) const = default;
};

struct SimpleDayGenome {
  std::vector<std::vector<int>> patrol_routes;
  std::vector<std::vector<SimpleRefuelTask>> refuel_routes;

  bool operator==(const SimpleDayGenome&) const = default;
};

struct SimpleGenome {
  std::vector<SimpleDayGenome> days;

  bool operator==(const SimpleGenome&) const = default;
};

struct SimpleRank {
  std::array<Wide, 3> weighted{};
  std::tuple<int, int, int> projected{};
  int ending_fuel{};
  int travel{};
  std::size_t hash{};
};

struct SimpleEvaluation {
  SimpleGenome genome;
  SimpleRank rank;
  std::vector<ActionPlan> plans;
  std::vector<CandidateEvaluation> evaluations;
  std::vector<DayInfo> day_infos;
  std::vector<PolicyHistory> histories_before;
};

std::uint64_t simple_mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::uint64_t simple_seed(const MapConfig& config, const DayInfo& day,
                          const PolicyHistory& history,
                          std::uint64_t salt = 0) {
  std::uint64_t seed = 0x53494d504c4e5353ULL ^ salt;
  auto add = [&](std::uint64_t value) { seed = simple_mix(seed ^ value); };
  add(config.width);
  add(config.height);
  add(config.fuel_limit);
  add(day.day);
  for (const auto& agent : day.agents) {
    add(static_cast<std::uint64_t>(agent.pos + 1));
    add(static_cast<std::uint64_t>(agent.fuel + 1));
    add(static_cast<std::uint64_t>(agent.kind));
  }
  for (int brand : history.distinct_brands) add(brand + 1);
  return seed;
}

Wide simple_power(int base, int exponent) {
  Wide result = 1;
  for (int index = 0; index < exponent; ++index) {
    result *= static_cast<unsigned>(base);
  }
  return result;
}

bool rank_better(const SimpleRank& left, const SimpleRank& right) {
  if (left.projected != right.projected) return left.projected > right.projected;
  if (left.weighted != right.weighted) return left.weighted > right.weighted;
  if (left.ending_fuel != right.ending_fuel) {
    return left.ending_fuel > right.ending_fuel;
  }
  if (left.travel != right.travel) return left.travel < right.travel;
  return left.hash < right.hash;
}

bool rank_equal(const SimpleRank& left, const SimpleRank& right) {
  return left.weighted == right.weighted &&
         left.projected == right.projected &&
         left.ending_fuel == right.ending_fuel &&
         left.travel == right.travel && left.hash == right.hash;
}

std::size_t genome_hash(const SimpleGenome& genome) {
  std::size_t hash = 0xcbf29ce484222325ULL;
  auto add = [&](std::size_t value) {
    hash ^= value + 1U;
    hash *= 0x100000001b3ULL;
  };
  for (const auto& day : genome.days) {
    for (std::size_t agent = 0; agent < day.patrol_routes.size(); ++agent) {
      add(agent);
      for (int spot : day.patrol_routes[agent]) add(spot + 17U);
    }
    for (std::size_t agent = 0; agent < day.refuel_routes.size(); ++agent) {
      add(agent + 101U);
      for (const auto& task : day.refuel_routes[agent]) {
        add(task.target_patrol + 257U);
        add(task.escort_tiles + 4099U);
      }
    }
  }
  return hash;
}

std::string config_fingerprint(const MapConfig& config) {
  std::uint64_t value = 0x53494d504c455354ULL;
  auto add = [&](std::uint64_t item) { value = simple_mix(value ^ item); };
  add(config.width);
  add(config.height);
  add(config.fuel_limit);
  add(config.players);
  for (int steps : config.day_steps) add(steps);
  for (Terrain terrain : config.cells) add(static_cast<int>(terrain));
  for (const auto& spot : config.spots) {
    add(spot.brand + 1);
    add(spot.pos + 1);
    add(spot.stocks);
  }
  for (int pos : config.agents) add(pos + 1);
  std::ostringstream stream;
  stream << std::hex << value;
  return stream.str();
}

int direction_between(const MapConfig& config, int from, int to) {
  for (int direction = 0; direction < 6; ++direction) {
    if (neighbor(config, from, direction) == to) return direction;
  }
  return -1;
}

int path_fuel(const MapConfig& config, int source,
              const std::vector<int>& directions) {
  int result = 0;
  int cursor = source;
  for (int direction : directions) {
    result += terrain_fuel(config, cursor);
    cursor = *neighbor(config, cursor, direction);
  }
  return result;
}

std::vector<int> route_cells(const MapConfig& config, const DayInfo& day,
                             std::size_t agent,
                             const std::vector<int>& route) {
  std::vector<int> cells{day.agents[agent].pos};
  int cursor = cells.front();
  for (int spot_index : route) {
    if (spot_index < 0 || spot_index >= static_cast<int>(config.spots.size())) {
      continue;
    }
    const int target = config.spots[spot_index].pos;
    const auto path = shortest_path(config, cursor, target, day.traffics);
    if (path.cost >= std::numeric_limits<int>::max() / 8) break;
    if (path.directions.empty()) {
      // A duplicate cell is a one-step service wait. Step zero itself neither
      // collects udon nor refuels.
      cells.push_back(cursor);
      continue;
    }
    for (int direction : path.directions) {
      cursor = *neighbor(config, cursor, direction);
      cells.push_back(cursor);
    }
  }
  return cells;
}

struct DecodePending {
  bool active{};
  bool move{};
  bool advances_route{};
  bool escort_move{};
  int remaining{};
  int destination{};
  int fuel_cost{};
};

struct DecodeAgent {
  int pos{};
  int fuel{};
  std::vector<int> cells;
  std::size_t cursor{};
  DecodePending pending;
  std::vector<int> actions;
};

struct RefuelRuntime {
  std::size_t task_index{};
  bool active{};
  bool met{};
  int target{-1};
  int meeting{-1};
  int escort_remaining{};
  std::vector<int> path;
  std::size_t path_cursor{};
};

ActionPlan compress_waits(ActionPlan plan) {
  for (auto& row : plan) {
    std::vector<int> compressed;
    for (int action : row) {
      if (action < 0 && !compressed.empty() && compressed.back() < 0) {
        compressed.back() += action;
      } else {
        compressed.push_back(action);
      }
    }
    row = std::move(compressed);
  }
  return plan;
}

std::optional<ActionPlan> decode_simple_day(
    const MapConfig& config, const DayInfo& day, const AgentTypes& types,
    const SimpleDayGenome& genome) {
  const int horizon = config.day_steps.at(day.day);
  const std::size_t count = types.size();
  if (day.agents.size() != count || genome.patrol_routes.size() != count ||
      genome.refuel_routes.size() != count) {
    return std::nullopt;
  }

  std::vector<DecodeAgent> agents(count);
  std::vector<RefuelRuntime> refuel_runtime(count);
  for (std::size_t agent = 0; agent < count; ++agent) {
    agents[agent].pos = day.agents[agent].pos;
    agents[agent].fuel = day.agents[agent].fuel;
    if (types[agent] == AgentKind::Patrol) {
      agents[agent].cells =
          route_cells(config, day, agent, genome.patrol_routes[agent]);
    }
  }

  auto schedule_wait = [&](std::size_t agent, int duration) {
    duration = std::max(1, duration);
    agents[agent].actions.push_back(-duration);
    agents[agent].pending = {true, false, false, false, duration,
                             agents[agent].pos, 0};
  };

  auto schedule_move = [&](std::size_t agent, int direction,
                           bool advances_route, bool escort_move) {
    const int duration = terrain_time(config, agents[agent].pos, day.traffics);
    const int destination = *neighbor(config, agents[agent].pos, direction);
    agents[agent].actions.push_back(direction);
    agents[agent].pending =
        {true, true, advances_route, escort_move, duration, destination,
         terrain_fuel(config, agents[agent].pos)};
  };

  auto activate_task = [&](std::size_t refuel, int step) {
    auto& runtime = refuel_runtime[refuel];
    const auto& tasks = genome.refuel_routes[refuel];
    std::size_t next_index = runtime.task_index;
    runtime = {};
    runtime.task_index = next_index;
    while (runtime.task_index < tasks.size()) {
      const auto task = tasks[runtime.task_index];
      if (task.target_patrol < 0 ||
          task.target_patrol >= static_cast<int>(count) ||
          types[task.target_patrol] != AgentKind::Patrol) {
        ++runtime.task_index;
        continue;
      }
      const auto& target = agents[static_cast<std::size_t>(task.target_patrol)];
      struct Meeting {
        std::tuple<int, int, std::size_t, int> rank;
        int cell{};
        std::vector<int> path;
      };
      std::optional<Meeting> best;
      int patrol_time = 0;
      int patrol_fuel = target.fuel;
      int previous = target.pos;
      std::size_t begin = std::min(target.cursor + 1, target.cells.size());
      bool pending_arrival = false;
      if (target.pending.active && target.pending.advances_route) {
        patrol_time = target.pending.remaining;
        if (target.pending.move) {
          patrol_fuel -= target.pending.fuel_cost;
          if (patrol_fuel < 0) {
            ++runtime.task_index;
            continue;
          }
          previous = target.pending.destination;
        }
        pending_arrival = true;
      }
      for (std::size_t index = begin; index < target.cells.size(); ++index) {
        const int cell = target.cells[index];
        if (!(pending_arrival && index == begin)) {
          if (cell == previous) {
            ++patrol_time;
          } else {
            const int cost = terrain_fuel(config, previous);
            if (cost > patrol_fuel) break;
            patrol_fuel -= cost;
            patrol_time += terrain_time(config, previous, day.traffics);
          }
        }
        previous = cell;
        auto refuel_path = shortest_path(config, agents[refuel].pos, cell,
                                         day.traffics);
        if (refuel_path.cost >= std::numeric_limits<int>::max() / 8) continue;
        const int meeting_time =
            step + std::max(patrol_time, refuel_path.cost);
        if (meeting_time > horizon) continue;
        Meeting candidate{{meeting_time, refuel_path.cost, index, cell}, cell,
                          std::move(refuel_path.directions)};
        if (!best || candidate.rank < best->rank) best = std::move(candidate);
      }
      if (!best) {
        ++runtime.task_index;
        continue;
      }
      runtime.active = true;
      runtime.target = task.target_patrol;
      runtime.meeting = best->cell;
      runtime.escort_remaining = std::max(0, task.escort_tiles);
      runtime.path = std::move(best->path);
      return;
    }
  };

  auto target_must_hold = [&](std::size_t patrol) {
    for (std::size_t refuel = 0; refuel < count; ++refuel) {
      const auto& runtime = refuel_runtime[refuel];
      if (runtime.active && !runtime.met && runtime.target == static_cast<int>(patrol) &&
          runtime.meeting == agents[patrol].pos) {
        return true;
      }
    }
    return false;
  };

  auto schedule_patrol = [&](std::size_t patrol, int step) {
    auto& state = agents[patrol];
    if (target_must_hold(patrol)) {
      schedule_wait(patrol, 1);
      return;
    }
    if (state.cursor + 1 >= state.cells.size()) {
      schedule_wait(patrol, horizon - step);
      return;
    }
    const int next = state.cells[state.cursor + 1];
    if (next == state.pos) {
      schedule_wait(patrol, 1);
      state.pending.advances_route = true;
      return;
    }
    const int direction = direction_between(config, state.pos, next);
    if (direction < 0 || terrain_fuel(config, state.pos) > state.fuel) {
      schedule_wait(patrol, 1);
      return;
    }
    const int duration = terrain_time(config, state.pos, day.traffics);
    if (step + duration > horizon) {
      schedule_wait(patrol, horizon - step);
      return;
    }
    schedule_move(patrol, direction, true, false);
  };

  std::function<void(std::size_t, int)> schedule_refuel;
  schedule_refuel = [&](std::size_t refuel, int step) {
    auto& state = agents[refuel];
    auto& runtime = refuel_runtime[refuel];
    if (!runtime.active) activate_task(refuel, step);
    if (!runtime.active) {
      schedule_wait(refuel, horizon - step);
      return;
    }
    if (!runtime.met) {
      if (state.pos == runtime.meeting) {
        schedule_wait(refuel, 1);
        return;
      }
      if (runtime.path_cursor >= runtime.path.size()) {
        // The committed path became inconsistent; let the next reflection
        // expose the failed task and then advance.
        ++runtime.task_index;
        runtime.active = false;
        schedule_refuel(refuel, step);
        return;
      }
      const int direction = runtime.path[runtime.path_cursor++];
      const int duration = terrain_time(config, state.pos, day.traffics);
      if (step + duration > horizon) {
        schedule_wait(refuel, horizon - step);
        return;
      }
      schedule_move(refuel, direction, false, false);
      return;
    }
    if (runtime.escort_remaining <= 0) {
      ++runtime.task_index;
      runtime.active = false;
      schedule_refuel(refuel, step);
      return;
    }
    const auto target_index = static_cast<std::size_t>(runtime.target);
    auto& target = agents[target_index];
    if (state.pos != target.pos || !target.pending.active) {
      schedule_wait(refuel, 1);
      return;
    }
    if (!target.pending.move) {
      schedule_wait(refuel, std::min(1, horizon - step));
      return;
    }
    const int direction = direction_between(config, state.pos,
                                            target.pending.destination);
    if (direction < 0 || step + target.pending.remaining > horizon) {
      schedule_wait(refuel, 1);
      return;
    }
    // Both vehicles leave the same cell on the same reflection schedule.
    schedule_move(refuel, direction, false, true);
  };

  // Schedule step zero.
  for (std::size_t refuel = 0; refuel < count; ++refuel) {
    if (types[refuel] == AgentKind::Refuel) activate_task(refuel, 0);
  }
  for (std::size_t agent = 0; agent < count; ++agent) {
    if (types[agent] == AgentKind::Patrol) schedule_patrol(agent, 0);
  }
  for (std::size_t agent = 0; agent < count; ++agent) {
    if (types[agent] == AgentKind::Refuel) schedule_refuel(agent, 0);
  }

  for (int step = 1; step <= horizon; ++step) {
    for (std::size_t agent = 0; agent < count; ++agent) {
      auto& pending = agents[agent].pending;
      if (!pending.active) continue;
      --pending.remaining;
      if (pending.remaining != 0) continue;
      if (pending.move) {
        if (types[agent] == AgentKind::Patrol) {
          if (pending.fuel_cost > agents[agent].fuel) return std::nullopt;
          agents[agent].fuel -= pending.fuel_cost;
        }
        agents[agent].pos = pending.destination;
      }
      if (pending.advances_route) ++agents[agent].cursor;
      if (pending.escort_move) {
        auto& runtime = refuel_runtime[agent];
        if (runtime.escort_remaining > 0) --runtime.escort_remaining;
      }
      pending.active = false;
    }

    // Official reflection: co-location refuels after movement.
    for (std::size_t refuel = 0; refuel < count; ++refuel) {
      if (types[refuel] != AgentKind::Refuel) continue;
      for (std::size_t patrol = 0; patrol < count; ++patrol) {
        if (types[patrol] == AgentKind::Patrol &&
            agents[patrol].pos == agents[refuel].pos) {
          agents[patrol].fuel = config.fuel_limit;
        }
      }
      auto& runtime = refuel_runtime[refuel];
      if (runtime.active && runtime.target >= 0 &&
          agents[static_cast<std::size_t>(runtime.target)].pos ==
              agents[refuel].pos) {
        runtime.met = true;
        if (runtime.escort_remaining <= 0) {
          ++runtime.task_index;
          runtime.active = false;
        }
      }
    }

    if (step == horizon) break;
    for (std::size_t agent = 0; agent < count; ++agent) {
      if (agents[agent].pending.active) continue;
      if (types[agent] == AgentKind::Patrol) schedule_patrol(agent, step);
    }
    for (std::size_t agent = 0; agent < count; ++agent) {
      if (agents[agent].pending.active) continue;
      if (types[agent] == AgentKind::Refuel) schedule_refuel(agent, step);
    }
  }

  ActionPlan result(count);
  for (std::size_t agent = 0; agent < count; ++agent) {
    result[agent] = std::move(agents[agent].actions);
    if (result[agent].empty()) result[agent].push_back(-horizon);
  }
  result = compress_waits(std::move(result));
  if (validate_action_plan(config, day, result)) return std::nullopt;
  return result;
}

void update_history(const MapConfig& config, const ActionPlan& plan,
                    const CandidateEvaluation& evaluation,
                    PolicyHistory& history) {
  history.submitted_actions.push_back(plan);
  for (const auto& event : evaluation.trace.acquisitions) {
    if (const Spot* spot = spot_at(config, event.spot_pos)) {
      history.distinct_brands.insert(spot->brand);
    }
  }
  history.cumulative_daily_types += std::get<1>(evaluation.value);
  history.total_servings += std::get<2>(evaluation.value);
}

std::optional<SimpleEvaluation> evaluate_genome(
    const MapConfig& config, const DayInfo& root_day,
    const PolicyHistory& root_history, const AgentTypes& types,
    SimpleGenome genome, int discount_percent) {
  const std::size_t remaining = config.day_steps.size() -
                                static_cast<std::size_t>(root_day.day);
  if (genome.days.size() != remaining) return std::nullopt;
  SimpleEvaluation result;
  result.genome = std::move(genome);
  PolicyHistory history = root_history;
  std::vector<int> positions;
  std::vector<int> fuel;
  std::vector<std::array<int, 3>> rewards;

  for (std::size_t offset = 0; offset < remaining; ++offset) {
    DayInfo info;
    if (offset == 0) {
      info = root_day;
    } else {
      info.day = root_day.day + static_cast<int>(offset);
      info.traffics = root_day.traffics;  // fixed revealed-road forecast
      for (std::size_t agent = 0; agent < types.size(); ++agent) {
        info.agents.push_back({types[agent], positions[agent], fuel[agent]});
      }
    }
    result.histories_before.push_back(history);
    auto plan = decode_simple_day(config, info, types, result.genome.days[offset]);
    if (!plan) return std::nullopt;
    auto evaluation = evaluate_candidate(config, info, history, *plan);
    if (!evaluation) return std::nullopt;
    for (std::size_t agent = 0; agent < types.size(); ++agent) {
      if (types[agent] == AgentKind::Patrol &&
          spot_at(config, evaluation->ending_positions[agent]) == nullptr) {
        return std::nullopt;
      }
    }
    const int before_distinct = static_cast<int>(history.distinct_brands.size());
    update_history(config, *plan, *evaluation, history);
    rewards.push_back(
        {static_cast<int>(history.distinct_brands.size()) - before_distinct,
         std::get<1>(evaluation->value), std::get<2>(evaluation->value)});
    positions = evaluation->ending_positions;
    fuel = evaluation->ending_fuel;
    result.plans.push_back(std::move(*plan));
    result.evaluations.push_back(std::move(*evaluation));
    result.day_infos.push_back(std::move(info));
  }

  for (std::size_t offset = 0; offset < rewards.size(); ++offset) {
    const int exponent = static_cast<int>(offset);
    const int inverse = static_cast<int>(rewards.size() - offset - 1);
    const Wide weight = simple_power(discount_percent, exponent) *
                        simple_power(100, inverse);
    for (std::size_t objective = 0; objective < 3; ++objective) {
      result.rank.weighted[objective] +=
          static_cast<unsigned>(rewards[offset][objective]) * weight;
    }
  }
  result.rank.projected =
      {static_cast<int>(history.distinct_brands.size()),
       history.cumulative_daily_types, history.total_servings};
  if (!result.evaluations.empty()) {
    const auto& tail = result.evaluations.back();
    for (std::size_t agent = 0; agent < types.size(); ++agent) {
      if (types[agent] == AgentKind::Patrol) {
        result.rank.ending_fuel += tail.ending_fuel[agent];
      }
    }
  }
  for (const auto& plan : result.plans) {
    for (const auto& row : plan) {
      result.rank.travel +=
          static_cast<int>(std::count_if(row.begin(), row.end(),
                                         [](int action) { return action >= 0; }));
    }
  }
  result.rank.hash = genome_hash(result.genome);
  return result;
}

int route_time(const MapConfig& config, const DayInfo& day, int start,
               const std::vector<int>& route) {
  int total = 0;
  int cursor = start;
  for (int spot : route) {
    const auto path = shortest_path(config, cursor, config.spots[spot].pos,
                                    day.traffics);
    if (path.cost >= std::numeric_limits<int>::max() / 8) return path.cost;
    total += path.directions.empty() ? 1 : path.cost;
    cursor = config.spots[spot].pos;
  }
  return total;
}

int route_fuel_cost(const MapConfig& config, const DayInfo& day, int start,
                    const std::vector<int>& route) {
  int total = 0;
  int cursor = start;
  for (int spot : route) {
    const auto path = shortest_path(config, cursor, config.spots[spot].pos,
                                    day.traffics);
    if (path.cost >= std::numeric_limits<int>::max() / 8) return path.cost;
    total += path_fuel(config, cursor, path.directions);
    cursor = config.spots[spot].pos;
  }
  return total;
}

void rebuild_refuel_tasks(const MapConfig& config, const DayInfo& day,
                          const AgentTypes& types, SimpleDayGenome& genome) {
  for (auto& tasks : genome.refuel_routes) tasks.clear();
  std::vector<std::size_t> refuels;
  std::vector<std::tuple<int, int, int>> patrols;
  for (std::size_t agent = 0; agent < types.size(); ++agent) {
    if (types[agent] == AgentKind::Refuel) {
      refuels.push_back(agent);
    } else {
      int fuel_need = 0;
      int cursor = day.agents[agent].pos;
      int tiles = 0;
      for (int spot : genome.patrol_routes[agent]) {
        auto path = shortest_path(config, cursor, config.spots[spot].pos,
                                  day.traffics);
        fuel_need += path_fuel(config, cursor, path.directions);
        tiles += static_cast<int>(path.directions.size());
        cursor = config.spots[spot].pos;
      }
      patrols.emplace_back(day.agents[agent].fuel - fuel_need,
                           static_cast<int>(agent), std::max(1, tiles));
    }
  }
  if (refuels.empty()) return;
  std::sort(patrols.begin(), patrols.end());
  for (std::size_t index = 0; index < patrols.size(); ++index) {
    auto [pressure, patrol, tiles] = patrols[index];
    (void)pressure;
    const std::size_t refuel = refuels[index % refuels.size()];
    genome.refuel_routes[refuel].push_back(
        {patrol, std::min(config.day_steps[day.day], tiles)});
  }
}

SimpleGenome construct_genome(const MapConfig& config, const DayInfo& root_day,
                              const AgentTypes& types) {
  const std::size_t remaining = config.day_steps.size() -
                                static_cast<std::size_t>(root_day.day);
  SimpleGenome genome;
  genome.days.resize(remaining);
  for (std::size_t offset = 0; offset < remaining; ++offset) {
    auto& day_genome = genome.days[offset];
    day_genome.patrol_routes.resize(types.size());
    day_genome.refuel_routes.resize(types.size());
    DayInfo info = root_day;
    info.day = root_day.day + static_cast<int>(offset);
    info.traffics = root_day.traffics;
    if (offset > 0) {
      info.agents.clear();
      for (std::size_t agent = 0; agent < types.size(); ++agent) {
        info.agents.push_back(
            {types[agent], config.agents[agent], config.fuel_limit});
      }
    }
    std::vector<int> assigned(config.spots.size());
    for (std::size_t agent = 0; agent < types.size(); ++agent) {
      if (types[agent] != AgentKind::Patrol) continue;
      int cursor = info.agents[agent].pos;
      int elapsed = 0;
      while (true) {
        int best = -1;
        std::tuple<int, int, int> best_rank{
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max()};
        for (std::size_t spot = 0; spot < config.spots.size(); ++spot) {
          if (assigned[spot] >= config.spots[spot].stocks ||
              std::find(day_genome.patrol_routes[agent].begin(),
                        day_genome.patrol_routes[agent].end(),
                        static_cast<int>(spot)) !=
                  day_genome.patrol_routes[agent].end()) {
            continue;
          }
          auto path = shortest_path(config, cursor, config.spots[spot].pos,
                                    info.traffics);
          const int added = path.directions.empty() ? 1 : path.cost;
          if (elapsed + added > config.day_steps[info.day] * 3 / 4) continue;
          const auto rank = std::tuple{assigned[spot], added,
                                       static_cast<int>(spot)};
          if (rank < best_rank) {
            best_rank = rank;
            best = static_cast<int>(spot);
          }
        }
        if (best < 0) break;
        day_genome.patrol_routes[agent].push_back(best);
        ++assigned[best];
        auto path = shortest_path(config, cursor, config.spots[best].pos,
                                  info.traffics);
        elapsed += path.directions.empty() ? 1 : path.cost;
        cursor = config.spots[best].pos;
        // Start from the smallest robust TOP solution: one reachable udon per
        // patrol and day. SISR recreate grows these routes. Filling to 75% here
        // can create a genome whose first evaluation strands every patrol, so
        // neither role selection nor ruin/recreate gets a valid incumbent.
        break;
      }
    }
    rebuild_refuel_tasks(config, info, types, day_genome);
  }
  return genome;
}

enum class RecreateOrder { Random, Score, CloseStart, Knn2, Knn5 };

void sisr_ruin(const MapConfig& config, const DayInfo& day,
               const AgentTypes& types, SimpleDayGenome& genome,
               std::mt19937_64& random) {
  struct Visit {
    std::size_t agent{};
    std::size_t position{};
    int spot{};
  };
  std::vector<Visit> visits;
  int longest = 0;
  int patrols = 0;
  for (std::size_t agent = 0; agent < types.size(); ++agent) {
    if (types[agent] != AgentKind::Patrol) continue;
    ++patrols;
    longest = std::max(longest,
                       static_cast<int>(genome.patrol_routes[agent].size()));
    for (std::size_t position = 0;
         position < genome.patrol_routes[agent].size(); ++position) {
      visits.push_back({agent, position, genome.patrol_routes[agent][position]});
    }
  }
  if (visits.empty() || patrols == 0) return;
  const Visit seed = visits[static_cast<std::size_t>(random() % visits.size())];
  std::sort(visits.begin(), visits.end(), [&](const Visit& left,
                                               const Visit& right) {
    const int origin = config.spots[seed.spot].pos;
    const int left_distance = shortest_path(
        config, origin, config.spots[left.spot].pos, day.traffics).cost;
    const int right_distance = shortest_path(
        config, origin, config.spots[right.spot].pos, day.traffics).cost;
    return std::tie(left_distance, left.agent, left.position) <
           std::tie(right_distance, right.agent, right.position);
  });
  const int max_cardinality = std::max(1, std::min(10, longest));
  const int average_removed =
      std::max(1, std::min(10, static_cast<int>((visits.size() + 4) / 5)));
  const int cardinality = std::max(
      1, std::min(max_cardinality,
                  static_cast<int>(visits.size()) / std::max(1, patrols)));
  const int maximum_strings = std::max(1, 4 * average_removed / (1 + cardinality));
  const int strings = 1 + static_cast<int>(random() % maximum_strings);
  std::set<std::size_t> ruined;
  std::map<std::size_t, std::vector<std::size_t>> remove_positions;
  for (const Visit& related : visits) {
    if (static_cast<int>(ruined.size()) >= strings) break;
    if (!ruined.insert(related.agent).second) continue;
    const auto& route = genome.patrol_routes[related.agent];
    if (route.empty()) continue;
    const int remove_count =
        1 + static_cast<int>(random() % std::min<int>(cardinality, route.size()));
    const int min_start = std::max(0, static_cast<int>(related.position) -
                                         remove_count + 1);
    const int max_start = std::min(static_cast<int>(route.size()) - remove_count,
                                   static_cast<int>(related.position));
    const int start = min_start + static_cast<int>(
                                      random() % (max_start - min_start + 1));
    const bool split = remove_count > 1 &&
                       remove_count < static_cast<int>(route.size()) &&
                       random() % 100U < 50U;
    if (!split) {
      for (int index = 0; index < remove_count; ++index) {
        remove_positions[related.agent].push_back(
            static_cast<std::size_t>(start + index));
      }
    } else {
      int gap = 1;
      while (remove_count + gap < static_cast<int>(route.size()) &&
             random() % 100U >= 1U) {
        ++gap;
      }
      const int whole = std::min<int>(route.size(), remove_count + gap);
      const int whole_min = std::max(
          0, static_cast<int>(related.position) - whole + 1);
      const int whole_max = std::min(static_cast<int>(route.size()) - whole,
                                     static_cast<int>(related.position));
      const int whole_start = whole_min + static_cast<int>(
                                            random() % (whole_max - whole_min + 1));
      const int keep_start = static_cast<int>(random() % (whole - gap + 1));
      for (int index = 0; index < whole; ++index) {
        if (index >= keep_start && index < keep_start + gap) continue;
        remove_positions[related.agent].push_back(
            static_cast<std::size_t>(whole_start + index));
      }
    }
  }
  for (auto& [agent, positions] : remove_positions) {
    std::sort(positions.begin(), positions.end(), std::greater<>());
    positions.erase(std::unique(positions.begin(), positions.end()),
                    positions.end());
    for (std::size_t position : positions) {
      if (position < genome.patrol_routes[agent].size()) {
        genome.patrol_routes[agent].erase(
            genome.patrol_routes[agent].begin() +
            static_cast<std::ptrdiff_t>(position));
      }
    }
  }
}

void plain_ruin(const AgentTypes& types, SimpleDayGenome& genome,
                std::mt19937_64& random) {
  std::vector<std::pair<std::size_t, std::size_t>> visits;
  for (std::size_t agent = 0; agent < types.size(); ++agent) {
    if (types[agent] != AgentKind::Patrol) continue;
    for (std::size_t position = 0;
         position < genome.patrol_routes[agent].size(); ++position) {
      visits.emplace_back(agent, position);
    }
  }
  if (visits.empty()) return;
  std::shuffle(visits.begin(), visits.end(), random);
  visits.resize(std::max<std::size_t>(1, (visits.size() + 4) / 5));
  std::map<std::size_t, std::vector<std::size_t>> removals;
  for (const auto [agent, position] : visits) {
    removals[agent].push_back(position);
  }
  for (auto& [agent, positions] : removals) {
    std::sort(positions.begin(), positions.end(), std::greater<>());
    for (std::size_t position : positions) {
      genome.patrol_routes[agent].erase(
          genome.patrol_routes[agent].begin() +
          static_cast<std::ptrdiff_t>(position));
    }
  }
}

void sisr_recreate(const MapConfig& config, const DayInfo& day,
                   const PolicyHistory& history, const AgentTypes& types,
                   SimpleDayGenome& genome, std::mt19937_64& random,
                   bool sisr_enabled = true) {
  std::vector<int> assigned(config.spots.size());
  const bool has_refuel =
      std::find(types.begin(), types.end(), AgentKind::Refuel) != types.end();
  std::set<int> planned_brands;
  for (std::size_t agent = 0; agent < types.size(); ++agent) {
    for (int spot : genome.patrol_routes[agent]) {
      ++assigned[spot];
      planned_brands.insert(config.spots[spot].brand);
    }
  }
  std::vector<int> absent;
  for (std::size_t spot = 0; spot < config.spots.size(); ++spot) {
    for (int copy = assigned[spot]; copy < config.spots[spot].stocks; ++copy) {
      absent.push_back(static_cast<int>(spot));
    }
  }
  static constexpr std::array<int, 5> weights{4, 4, 1, 1, 1};
  int draw = sisr_enabled ? static_cast<int>(random() % 11U) : 4;
  int choice = 0;
  for (std::size_t index = 0; index < weights.size(); ++index) {
    if (draw < weights[index]) {
      choice = static_cast<int>(index);
      break;
    }
    draw -= weights[index];
  }
  const auto order = sisr_enabled ? static_cast<RecreateOrder>(choice)
                                  : RecreateOrder::Score;
  auto density = [&](int spot, std::size_t neighbors) {
    std::vector<int> distances;
    for (std::size_t other = 0; other < config.spots.size(); ++other) {
      if (static_cast<int>(other) == spot) continue;
      distances.push_back(shortest_path(config, config.spots[spot].pos,
                                        config.spots[other].pos,
                                        day.traffics).cost);
    }
    std::sort(distances.begin(), distances.end());
    return std::accumulate(
        distances.begin(),
        distances.begin() + static_cast<std::ptrdiff_t>(
                                std::min(neighbors, distances.size())),
        0);
  };
  if (order == RecreateOrder::Random) {
    std::shuffle(absent.begin(), absent.end(), random);
  } else {
    std::sort(absent.begin(), absent.end(), [&](int left, int right) {
      auto tier = [&](int spot) {
        const int brand = config.spots[spot].brand;
        return std::tuple{history.distinct_brands.contains(brand) ? 0 : 1,
                          planned_brands.contains(brand) ? 0 : 1};
      };
      if (order == RecreateOrder::Score) {
        return std::tuple{tier(left), -assigned[left], -left} >
               std::tuple{tier(right), -assigned[right], -right};
      }
      if (order == RecreateOrder::CloseStart) {
        auto closest = [&](int spot) {
          int best = std::numeric_limits<int>::max();
          for (std::size_t agent = 0; agent < types.size(); ++agent) {
            if (types[agent] == AgentKind::Patrol) {
              best = std::min(best, shortest_path(
                  config, day.agents[agent].pos, config.spots[spot].pos,
                  day.traffics).cost);
            }
          }
          return best;
        };
        return std::tuple{closest(left), left} <
               std::tuple{closest(right), right};
      }
      const std::size_t k = order == RecreateOrder::Knn2 ? 2U : 5U;
      return std::tuple{tier(left), -density(left, k), -left} >
             std::tuple{tier(right), -density(right, k), -right};
    });
  }
  const double shaw = sisr_enabled
                          ? 0.3 * static_cast<double>(random() % 10001U) /
                                10000.0
                          : 0.0;
  std::vector<int> queue = absent;
  if (order != RecreateOrder::Random) std::reverse(queue.begin(), queue.end());
  while (!queue.empty()) {
    std::size_t index = queue.size() - 1;
    if (order != RecreateOrder::Random && shaw > 0.0001) {
      const double unit = (static_cast<double>(random() % 1000001U) + 0.5) /
                          1000001.0;
      index = std::min(queue.size() - 1,
                       static_cast<std::size_t>(queue.size() *
                                                std::pow(unit, shaw)));
    }
    const int spot = queue[index];
    queue.erase(queue.begin() + static_cast<std::ptrdiff_t>(index));
    struct Insertion {
      int delta{};
      std::size_t agent{};
      std::size_t position{};
    };
    std::vector<Insertion> insertions;
    std::vector<Insertion> blinked;
    for (std::size_t agent = 0; agent < types.size(); ++agent) {
      if (types[agent] != AgentKind::Patrol) {
        continue;
      }
      const auto& route = genome.patrol_routes[agent];
      const int current_time =
          route_time(config, day, day.agents[agent].pos, route);
      for (std::size_t position = 0; position <= route.size(); ++position) {
        const int previous = position == 0
                                 ? day.agents[agent].pos
                                 : config.spots[route[position - 1]].pos;
        const int next = position == route.size()
                             ? -1
                             : config.spots[route[position]].pos;
        const auto leg_time = [&](int from, int to) {
          const auto path = shortest_path(config, from, to, day.traffics);
          return path.directions.empty() ? 1 : path.cost;
        };
        const int direct = next < 0 ? 0 : leg_time(previous, next);
        const int first = leg_time(previous, config.spots[spot].pos);
        const int second = next < 0
                               ? 0
                               : leg_time(config.spots[spot].pos, next);
        const int delta = first + second - direct;
        if (first >= std::numeric_limits<int>::max() / 8 ||
            second >= std::numeric_limits<int>::max() / 8 ||
            current_time + delta > config.day_steps[day.day]) {
          continue;
        }
        if (!has_refuel) {
          std::vector<int> candidate_route;
          candidate_route.reserve(route.size() + 1);
          for (std::size_t index = 0; index <= route.size(); ++index) {
            if (index == position) candidate_route.push_back(spot);
            if (index < route.size()) candidate_route.push_back(route[index]);
          }
          if (route_fuel_cost(config, day, day.agents[agent].pos,
                              candidate_route) > day.agents[agent].fuel) {
            continue;
          }
        }
        if (sisr_enabled && random() % 100U == 0U) {
          blinked.push_back({delta, agent, position});
        } else {
          insertions.push_back({delta, agent, position});
        }
      }
    }
    if (insertions.empty()) insertions = std::move(blinked);
    if (insertions.empty()) continue;
    std::sort(insertions.begin(), insertions.end(), [](const auto& left,
                                                       const auto& right) {
      return std::tie(left.delta, left.agent, left.position) <
             std::tie(right.delta, right.agent, right.position);
    });
    const auto insertion = insertions.front();
    auto& route = genome.patrol_routes[insertion.agent];
    route.insert(route.begin() + static_cast<std::ptrdiff_t>(insertion.position),
                 spot);
    ++assigned[spot];
    planned_brands.insert(config.spots[spot].brand);
  }
}

void refuel_task_destroy_repair(const MapConfig& config, const DayInfo& day,
                                const AgentTypes& types,
                                SimpleDayGenome& genome,
                                std::mt19937_64& random) {
  std::vector<std::size_t> refuelers;
  std::vector<int> patrols;
  for (std::size_t agent = 0; agent < types.size(); ++agent) {
    if (types[agent] == AgentKind::Refuel) {
      refuelers.push_back(agent);
    } else {
      patrols.push_back(static_cast<int>(agent));
    }
  }
  if (refuelers.empty() || patrols.empty()) return;

  // SISR-style task-string removal. The refuel route is a short semantic
  // sequence, so removing one to three adjacent tasks is already a large
  // neighborhood.
  std::vector<std::size_t> nonempty;
  for (std::size_t refuel : refuelers) {
    if (!genome.refuel_routes[refuel].empty()) nonempty.push_back(refuel);
  }
  if (!nonempty.empty()) {
    const std::size_t refuel =
        nonempty[static_cast<std::size_t>(random() % nonempty.size())];
    auto& route = genome.refuel_routes[refuel];
    const std::size_t count =
        1U + static_cast<std::size_t>(random() % std::min<std::size_t>(3, route.size()));
    const std::size_t start = static_cast<std::size_t>(
        random() % (route.size() - count + 1U));
    route.erase(route.begin() + static_cast<std::ptrdiff_t>(start),
                route.begin() + static_cast<std::ptrdiff_t>(start + count));
  }

  // Keep at most one task per patrol. Repair then restores every missing
  // target while freely changing refuel assignment, order and escort length.
  std::set<int> assigned;
  for (std::size_t refuel : refuelers) {
    auto& route = genome.refuel_routes[refuel];
    route.erase(std::remove_if(route.begin(), route.end(), [&](const auto& task) {
                  if (std::find(patrols.begin(), patrols.end(),
                                task.target_patrol) == patrols.end() ||
                      !assigned.insert(task.target_patrol).second) {
                    return true;
                  }
                  return false;
                }),
                route.end());
  }

  std::vector<int> missing;
  for (int patrol : patrols) {
    if (!assigned.contains(patrol)) missing.push_back(patrol);
  }
  std::sort(missing.begin(), missing.end(), [&](int left, int right) {
    const int left_fuel = route_fuel_cost(
        config, day, day.agents[left].pos, genome.patrol_routes[left]);
    const int right_fuel = route_fuel_cost(
        config, day, day.agents[right].pos, genome.patrol_routes[right]);
    return std::tie(left_fuel, left) > std::tie(right_fuel, right);
  });
  if (missing.size() > 1 && random() % 100U < 30U) {
    std::shuffle(missing.begin(), missing.end(), random);
  }

  for (int patrol : missing) {
    std::size_t refuel = refuelers.front();
    for (std::size_t candidate : refuelers) {
      if (genome.refuel_routes[candidate].size() <
          genome.refuel_routes[refuel].size()) {
        refuel = candidate;
      }
    }
    if (refuelers.size() > 1 && random() % 100U < 25U) {
      refuel = refuelers[static_cast<std::size_t>(random() % refuelers.size())];
    }
    auto& route = genome.refuel_routes[refuel];
    const std::size_t position = static_cast<std::size_t>(
        random() % (route.size() + 1U));
    const auto cells = route_cells(config, day, static_cast<std::size_t>(patrol),
                                   genome.patrol_routes[patrol]);
    const int tiles = std::max(0, static_cast<int>(cells.size()) - 1);
    route.insert(route.begin() + static_cast<std::ptrdiff_t>(position),
                 {patrol, std::min(config.day_steps[day.day], tiles)});
  }
}

long double rank_loss(const SimpleRank& current, const SimpleRank& candidate) {
  const std::array<int, 3> current_projected{
      std::get<0>(current.projected), std::get<1>(current.projected),
      std::get<2>(current.projected)};
  const std::array<int, 3> candidate_projected{
      std::get<0>(candidate.projected), std::get<1>(candidate.projected),
      std::get<2>(candidate.projected)};
  for (std::size_t index = 0; index < current_projected.size(); ++index) {
    if (candidate_projected[index] < current_projected[index]) {
      return std::max(
          1e-12L,
          static_cast<long double>(current_projected[index] -
                                   candidate_projected[index]) /
              std::max(1, current_projected[index]));
    }
    if (candidate_projected[index] > current_projected[index]) return 0.0L;
  }
  for (std::size_t index = 0; index < 3; ++index) {
    if (candidate.weighted[index] < current.weighted[index]) {
      const long double incumbent = current.weighted[index].convert_to<long double>();
      const long double neighbor = candidate.weighted[index].convert_to<long double>();
      return std::max(1e-12L, (incumbent - neighbor) /
                                  std::max(1.0L, incumbent));
    }
    if (candidate.weighted[index] > current.weighted[index]) return 0.0L;
  }
  return 1e-9L;
}

std::optional<SimpleEvaluation> search_simple(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& types,
    const SearchLimits& limits, SimpleGenome initial,
    const ImprovementSink* on_improve) {
  auto best = evaluate_genome(config, day, history, types, std::move(initial),
                              limits.future_discount_percent);
  if (!best) return std::nullopt;
  SimpleEvaluation current = *best;
  auto emit = [&] {
    if (on_improve == nullptr || best->plans.empty()) return;
    const auto official = alns_official_value(best->evaluations.front().value);
    IncumbentRank rank;
    rank.available = true;
    rank.objective_mode = "simple_lns";
    rank.predicted_final_available = true;
    rank.predicted_final = {std::get<0>(best->rank.projected),
                            std::get<1>(best->rank.projected),
                            std::get<2>(best->rank.projected)};
    rank.predicted_ending_patrol_fuel = best->rank.ending_fuel;
    rank.patrol_fuel = std::get<3>(best->evaluations.front().value);
    rank.future_discount_percent = limits.future_discount_percent;
    for (std::size_t index = 0; index < 3; ++index) {
      rank.weighted_match[index] = best->rank.weighted[index].convert_to<std::string>();
    }
    (*on_improve)(best->plans.front(),
                  {std::get<0>(official), std::get<1>(official),
                   std::get<2>(official)},
                  rank);
  };
  emit();
  std::mt19937_64 random(simple_seed(config, day, history,
                                     limits.random_seed));
  std::uniform_real_distribution<long double> unit(0.0L, 1.0L);
  const auto started = std::chrono::steady_clock::now();
  const auto deadline = started + std::chrono::milliseconds(
                                      std::max(0, limits.time_limit_ms));
  const int maximum = std::max(0, limits.max_iterations);
  const int minimum = std::min(std::max(0, limits.min_iterations), maximum);
  std::vector<long double> losses;
  const bool sisr_ruin_enabled =
      std::getenv("HEXUDON_SIMPLE_LNS_DISABLE_SISR_RUIN") == nullptr;
  const bool sisr_recreate_enabled =
      std::getenv("HEXUDON_SIMPLE_LNS_DISABLE_SISR_RECREATE") == nullptr;
  const bool annealing_enabled =
      std::getenv("HEXUDON_SIMPLE_LNS_DISABLE_SA") == nullptr;
  int stagnation = 0;
  for (int iteration = 0; iteration < maximum; ++iteration) {
    if (limits.time_limit_ms >= 0 && iteration > 0 &&
        std::chrono::steady_clock::now() >= deadline) {
      break;
    }
    SimpleGenome neighbor = current.genome;
    std::size_t pivot = 0;
    if (neighbor.days.size() > 1) {
      std::vector<std::uint64_t> weights;
      std::uint64_t total = 0;
      for (std::size_t offset = 0; offset < neighbor.days.size(); ++offset) {
        std::uint64_t weight = 1'000'000;
        for (std::size_t power = 0; power < offset; ++power) {
          weight *= static_cast<std::uint64_t>(limits.future_discount_percent);
          weight /= 100U;
          if (weight == 0) break;
        }
        weight = std::max<std::uint64_t>(1, weight);
        weights.push_back(weight);
        total += weight;
      }
      std::uint64_t draw = random() % total;
      while (pivot + 1 < weights.size() && draw >= weights[pivot]) {
        draw -= weights[pivot++];
      }
    }
    const auto& info = current.day_infos[pivot];
    if (sisr_ruin_enabled) {
      sisr_ruin(config, info, types, neighbor.days[pivot], random);
    } else {
      plain_ruin(types, neighbor.days[pivot], random);
    }
    sisr_recreate(config, info, current.histories_before[pivot], types,
                  neighbor.days[pivot], random, sisr_recreate_enabled);
    rebuild_refuel_tasks(config, info, types, neighbor.days[pivot]);
    auto evaluated = evaluate_genome(config, day, history, types,
                                     std::move(neighbor),
                                     limits.future_discount_percent);
    if (!evaluated) {
      ++stagnation;
      continue;
    }
    const bool improves = rank_better(evaluated->rank, current.rank) ||
                          rank_equal(evaluated->rank, current.rank);
    bool accepted = improves;
    if (!accepted && annealing_enabled) {
      const long double loss = rank_loss(current.rank, evaluated->rank);
      if (losses.size() < 12U) losses.push_back(loss);
      long double probability = 0.05L;
      if (losses.size() >= 3U) {
        auto sorted = losses;
        std::sort(sorted.begin(), sorted.end());
        const long double median = sorted[sorted.size() / 2U];
        const long double initial = std::max(1e-12L, -median / std::log(0.5L));
        const long double final = std::max(1e-15L, -median / std::log(0.01L));
        // Keep the ruin-recreate prefix stable when callers extend the
        // iteration budget. Scaling temperature by `maximum` made a 128-move
        // run follow different first moves than a 32-move run and could return
        // a worse official score despite doing more work.
        long double progress = std::clamp(
            static_cast<long double>(iteration) / 127.0L, 0.0L, 1.0L);
        if (limits.time_limit_ms > 0) {
          const auto elapsed =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - started)
                  .count();
          progress = std::clamp(
              static_cast<long double>(elapsed) / limits.time_limit_ms,
              0.0L, 1.0L);
        }
        const long double temperature =
            initial * std::pow(final / initial, progress);
        probability = std::exp(-loss / temperature);
      }
      accepted = unit(random) < probability;
    }
    if (rank_better(evaluated->rank, best->rank)) {
      best = *evaluated;
      stagnation = 0;
      emit();
    } else {
      ++stagnation;
    }
    if (accepted) current = std::move(*evaluated);
    if (iteration + 1 >= minimum && limits.stagnation_iterations > 0 &&
        stagnation >= limits.stagnation_iterations) {
      break;
    }
  }

  const bool has_refuel_route =
      std::find(types.begin(), types.end(), AgentKind::Refuel) != types.end();
  if (has_refuel_route && maximum >= 64) {
    SimpleEvaluation task_current = *best;
    std::mt19937_64 task_random(simple_seed(
        config, day, history, limits.random_seed ^ 0x52454655454c4c4eULL));
    const int task_moves = std::min(16, std::max(1, maximum / 8));
    for (int move = 0; move < task_moves; ++move) {
      if (limits.time_limit_ms >= 0 &&
          std::chrono::steady_clock::now() >= deadline) {
        break;
      }
      SimpleGenome neighbor = task_current.genome;
      refuel_task_destroy_repair(config, task_current.day_infos.front(), types,
                                 neighbor.days.front(), task_random);
      auto evaluated = evaluate_genome(config, day, history, types,
                                       std::move(neighbor),
                                       limits.future_discount_percent);
      if (!evaluated) continue;
      const auto candidate_today =
          alns_official_value(evaluated->evaluations.front().value);
      const auto current_today =
          alns_official_value(task_current.evaluations.front().value);
      if (candidate_today >= current_today) task_current = *evaluated;
      const auto best_today =
          alns_official_value(best->evaluations.front().value);
      if (candidate_today > best_today ||
          (candidate_today == best_today &&
           rank_better(evaluated->rank, best->rank))) {
        best = std::move(evaluated);
        emit();
      }
    }
  }
  return best;
}

ActionPlan parse_actions(const json::value& value) {
  ActionPlan result;
  for (const auto& row : value.as_array()) {
    std::vector<int> actions;
    for (const auto& item : row.as_array()) actions.push_back(item.to_number<int>());
    result.push_back(std::move(actions));
  }
  return result;
}

std::optional<SimpleGenome> parse_state(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& types,
    int discount_percent, const json::value* planner_state) {
  if (planner_state == nullptr || !planner_state->is_object() || day.day == 0 ||
      history.submitted_actions.empty()) {
    return std::nullopt;
  }
  try {
    const auto& root = planner_state->as_object();
    if (root.at("schema_version").to_number<int>() != 1 ||
        root.at("policy").as_string() != "simple_lns" ||
        root.at("config_fingerprint").as_string() != config_fingerprint(config) ||
        root.at("source_day").to_number<int>() != day.day - 1 ||
        root.at("future_discount_percent").to_number<int>() != discount_percent ||
        parse_actions(root.at("committed_actions")) !=
            history.submitted_actions.back()) {
      return std::nullopt;
    }
    const auto& encoded_types = root.at("types").as_array();
    if (encoded_types.size() != types.size()) return std::nullopt;
    for (std::size_t agent = 0; agent < types.size(); ++agent) {
      if (encoded_types[agent].to_number<int>() != static_cast<int>(types[agent])) {
        return std::nullopt;
      }
    }
    const auto& suffix = root.at("suffix").as_array();
    const std::size_t expected = config.day_steps.size() -
                                 static_cast<std::size_t>(day.day);
    if (suffix.size() != expected) return std::nullopt;
    SimpleGenome genome;
    for (const auto& day_value : suffix) {
      const auto& encoded_day = day_value.as_object();
      SimpleDayGenome decoded;
      decoded.patrol_routes.resize(types.size());
      decoded.refuel_routes.resize(types.size());
      const auto& patrol_routes = encoded_day.at("patrol_routes").as_array();
      const auto& refuel_routes = encoded_day.at("refuel_routes").as_array();
      if (patrol_routes.size() != types.size() ||
          refuel_routes.size() != types.size()) {
        return std::nullopt;
      }
      for (std::size_t agent = 0; agent < types.size(); ++agent) {
        for (const auto& spot_value : patrol_routes[agent].as_array()) {
          const int spot = spot_value.to_number<int>();
          if (spot < 0 || spot >= static_cast<int>(config.spots.size())) {
            return std::nullopt;
          }
          decoded.patrol_routes[agent].push_back(spot);
        }
        for (const auto& task_value : refuel_routes[agent].as_array()) {
          const auto& task = task_value.as_object();
          decoded.refuel_routes[agent].push_back(
              {task.at("target_patrol").to_number<int>(),
               task.at("escort_tiles").to_number<int>()});
        }
      }
      genome.days.push_back(std::move(decoded));
    }
    return genome;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

json::object serialize_state(const MapConfig& config, const DayInfo& day,
                             const AgentTypes& types, int discount_percent,
                             const SimpleEvaluation& solution) {
  json::array encoded_types;
  for (auto type : types) encoded_types.push_back(static_cast<int>(type));
  json::array suffix;
  for (std::size_t offset = 1; offset < solution.genome.days.size(); ++offset) {
    json::array patrol_routes;
    json::array refuel_routes;
    for (const auto& route : solution.genome.days[offset].patrol_routes) {
      json::array encoded;
      for (int spot : route) encoded.push_back(spot);
      patrol_routes.push_back(std::move(encoded));
    }
    for (const auto& route : solution.genome.days[offset].refuel_routes) {
      json::array encoded;
      for (const auto& task : route) {
        encoded.push_back(json::object{{"target_patrol", task.target_patrol},
                                       {"escort_tiles", task.escort_tiles}});
      }
      refuel_routes.push_back(std::move(encoded));
    }
    suffix.push_back(json::object{
        {"day", day.day + static_cast<int>(offset)},
        {"patrol_routes", std::move(patrol_routes)},
        {"refuel_routes", std::move(refuel_routes)}});
  }
  return json::object{
      {"schema_version", 1},
      {"policy", "simple_lns"},
      {"config_fingerprint", config_fingerprint(config)},
      {"source_day", day.day},
      {"future_discount_percent", discount_percent},
      {"types", std::move(encoded_types)},
      {"committed_actions", to_json(solution.plans.front())},
      {"suffix", std::move(suffix)}};
}

DayInfo initial_day(const MapConfig& config, const AgentTypes& types) {
  DayInfo day;
  day.day = 0;
  for (std::size_t agent = 0; agent < types.size(); ++agent) {
    day.agents.push_back({types[agent], config.agents[agent], config.fuel_limit});
  }
  for (int pos = 0; pos < config.width * config.height; ++pos) {
    if (config.cells[pos] == Terrain::Road) day.traffics[pos] = 0;
  }
  return day;
}

}  // namespace

AgentTypes select_simple_lns_agent_types(const MapConfig& config,
                                         const SearchLimits& limits) {
  const std::size_t count = config.agents.size();
  AgentTypes fallback(count, AgentKind::Patrol);
  std::optional<SimpleEvaluation> best;
  AgentTypes best_types = fallback;
  const std::uint64_t masks = std::uint64_t{1} << count;
  struct RoleCandidate {
    AgentTypes types;
    SimpleGenome genome;
    SimpleEvaluation evaluation;
  };
  struct RoleSeed {
    AgentTypes types;
    SimpleGenome genome;
    std::tuple<int, int, int, int, std::uint64_t> rank;
  };
  std::vector<RoleSeed> seeds;
  for (std::uint64_t mask = 0; mask < masks; ++mask) {
    AgentTypes types(count, AgentKind::Patrol);
    int patrols = 0;
    for (std::size_t agent = 0; agent < count; ++agent) {
      if ((mask & (std::uint64_t{1} << agent)) != 0) {
        types[agent] = AgentKind::Refuel;
      } else {
        ++patrols;
      }
    }
    if (patrols == 0) continue;
    DayInfo day = initial_day(config, types);
    auto genome = construct_genome(config, day, types);
    std::set<int> brands;
    int visits = 0;
    for (const auto& candidate_day : genome.days) {
      for (const auto& route : candidate_day.patrol_routes) {
        visits += static_cast<int>(route.size());
        for (int spot : route) brands.insert(config.spots[spot].brand);
      }
    }
    const int refuelers = static_cast<int>(count) - patrols;
    const bool severe_pressure =
        2 * config.fuel_limit <=
        *std::max_element(config.day_steps.begin(), config.day_steps.end());
    const int desired_refuelers =
        severe_pressure ? std::max(1, static_cast<int>(count) / 4) : 0;
    seeds.push_back(
        {types, std::move(genome),
         {static_cast<int>(brands.size()), visits,
          -std::abs(refuelers - desired_refuelers), patrols,
          std::numeric_limits<std::uint64_t>::max() - mask}});
  }
  std::sort(seeds.begin(), seeds.end(), [](const auto& left,
                                           const auto& right) {
    return left.rank > right.rank;
  });
  std::vector<RoleCandidate> candidates;
  const std::size_t authoritative = std::min<std::size_t>(32, seeds.size());
  for (std::size_t index = 0; index < authoritative; ++index) {
    DayInfo day = initial_day(config, seeds[index].types);
    auto evaluated = evaluate_genome(config, day, {}, seeds[index].types,
                                     seeds[index].genome,
                                     limits.future_discount_percent);
    if (!evaluated) continue;
    candidates.push_back(
        {seeds[index].types, std::move(seeds[index].genome),
         std::move(*evaluated)});
  }
  if (candidates.empty()) return fallback;
  std::sort(candidates.begin(), candidates.end(), [](const auto& left,
                                                     const auto& right) {
    return rank_better(left.evaluation.rank, right.evaluation.rank);
  });
  const std::size_t refine = std::min<std::size_t>(8, candidates.size());
  SearchLimits role_limits = limits;
  role_limits.min_iterations = 0;
  // Type selection happens once per match and must see enough repaired route
  // genomes for refueling to reveal its multi-day value. Reusing the small
  // per-day default here gave each shortlisted mask only four moves, strongly
  // biasing the comparison toward the initially fuller all-patrol genome.
  const int role_iteration_budget =
      std::clamp(std::max(limits.max_iterations, 256), 256, 1024);
  role_limits.max_iterations = std::max(
      1, role_iteration_budget / static_cast<int>(refine));
  role_limits.stagnation_iterations = 0;
  role_limits.time_limit_ms = -1;
  for (std::size_t index = 0; index < refine; ++index) {
    DayInfo day = initial_day(config, candidates[index].types);
    auto refined = search_simple(config, day, {}, candidates[index].types,
                                 role_limits,
                                 candidates[index].genome, nullptr);
    if (refined && (!best || rank_better(refined->rank, best->rank))) {
      best = std::move(refined);
      best_types = candidates[index].types;
    }
  }
  return best ? best_types : candidates.front().types;
}

PlannerResult build_simple_lns_plan(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& types,
    const SearchLimits& limits, const json::value* planner_state,
    const ImprovementSink* on_improve) {
  SimpleGenome genome;
  if (auto warm = parse_state(config, day, history, types,
                              limits.future_discount_percent, planner_state)) {
    genome = std::move(*warm);
  } else {
    genome = construct_genome(config, day, types);
  }
  auto solved = search_simple(config, day, history, types, limits,
                              std::move(genome), on_improve);
  if (!solved || solved->plans.empty()) {
    return {wait_plan(types.size(), config.day_steps.at(day.day)), std::nullopt};
  }
  return {solved->plans.front(),
          serialize_state(config, day, types,
                          limits.future_discount_percent, *solved)};
}

}  // namespace hexudon
