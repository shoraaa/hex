#include "hexudon/core.hpp"
#include "hexudon/internal.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace hexudon {
struct AcoAgentState {
  int pos{};
  int node{};
  int elapsed{};
  int fuel{};
  ActionPlan::value_type actions;
  std::uint64_t visited_spots{};
};

struct AcoCandidate {
  int patrol{};
  int target_node{};
  int refuel{-1};
  int meeting_node{-1};
  int meeting_time{};
  int finish_time{};
  int refuel_finish_time{};
  long long reward{};
  long double desirability{};
  AcoPath patrol_first;
  AcoPath refuel_path;
  AcoPath patrol_after;
  std::vector<std::size_t> move_pheromones;
  std::optional<std::size_t> meeting_pheromone;
};

struct AcoAnt {
  ActionPlan plan;
  std::optional<std::tuple<int, int, int, int>> quality;
  std::vector<std::size_t> used_moves;
  std::vector<std::size_t> used_meetings;
  std::size_t index{};
};

std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::uint64_t aco_base_seed(const MapConfig& config, const DayInfo& day,
                            const PolicyHistory& history) {
  std::uint64_t hash = 1469598103934665603ULL;
  auto mix = [&](std::uint64_t value) {
    hash ^= splitmix64(value + hash);
    hash *= 1099511628211ULL;
  };
  mix(config.width);
  mix(config.height);
  mix(config.fuel_limit);
  mix(day.day);
  for (auto cell : config.cells) mix(static_cast<int>(cell));
  for (const auto& spot : config.spots) {
    mix(spot.brand);
    mix(spot.pos);
    mix(spot.stocks);
  }
  for (const auto& agent : day.agents) {
    mix(static_cast<int>(agent.kind));
    mix(agent.pos);
    mix(agent.kind == AgentKind::Refuel ? config.fuel_limit : agent.fuel);
  }
  for (const auto& [pos, status] : day.traffics) {
    mix(pos);
    mix(status);
  }
  for (int brand : history.distinct_brands) mix(brand);
  return hash;
}

ActionPlan build_aco_plan(const MapConfig& config, const DayInfo& day,
                          const PolicyHistory& history,
                           const AgentTypes& types, bool apply_local_search,
                           const SearchLimits& limits) {
  constexpr int path_variants = 7;
  const int horizon = config.day_steps[day.day];
  const auto graph = build_aco_graph(config, day);
  const auto meeting_cache = build_aco_meeting_cache(graph);
  const std::size_t agent_count = day.agents.size();
  const std::size_t node_count = graph.nodes.size();
  const auto move_index = [&](std::size_t agent, int from, int to,
                              int variant) {
    return (((agent * node_count + static_cast<std::size_t>(from)) *
                 node_count +
             static_cast<std::size_t>(to)) *
                path_variants +
            static_cast<std::size_t>(variant));
  };
  const auto meeting_index = [&](std::size_t patrol, std::size_t refuel,
                                 int node) {
    return ((patrol * agent_count + refuel) * node_count +
            static_cast<std::size_t>(node));
  };
  std::vector<double> move_pheromone(agent_count * node_count * node_count *
                                         path_variants,
                                     1.0);
  std::vector<double> meeting_pheromone(agent_count * agent_count * node_count,
                                        1.0);
  std::map<int, int> brand_index;
  for (const auto& spot : config.spots) {
    if (!brand_index.contains(spot.brand)) {
      brand_index[spot.brand] = static_cast<int>(brand_index.size());
    }
  }
  std::vector<int> spot_for_cell(config.cells.size(), -1);
  std::vector<int> spot_brand(config.spots.size());
  int total_stock = 0;
  for (std::size_t index = 0; index < config.spots.size(); ++index) {
    const auto& spot = config.spots[index];
    spot_for_cell[spot.pos] = static_cast<int>(index);
    spot_brand[index] = brand_index.at(spot.brand);
    total_stock += spot.stocks;
  }
  const long long daily_weight = total_stock + 1LL;
  const long long match_weight =
      (static_cast<long long>(brand_index.size()) + 1LL) * daily_weight;
  std::uint64_t initial_match = 0;
  for (int brand : history.distinct_brands) {
    if (auto found = brand_index.find(brand); found != brand_index.end()) {
      initial_match |= std::uint64_t{1} << found->second;
    }
  }
  const std::uint64_t base_seed = aco_base_seed(config, day, history);
  std::vector<ActionPlan> local_search_alternatives;
  if (apply_local_search) {
    const std::array<std::string, 4> alternative_policies{
        "greedy", "utility_greedy", "fuel_aware", "stock_maximiser"};
    local_search_alternatives = parallel_indexed(
        alternative_policies.size(), [&](std::size_t index) {
        return build_routing_plan(alternative_policies[index], config, day,
                                  history, types, {}, limits);
        });
  }

  auto construct_ant = [&](std::size_t ant_index, int iteration) {
    AcoAnt ant;
    ant.index = ant_index;
    std::mt19937_64 random(splitmix64(
        base_seed ^ (static_cast<std::uint64_t>(iteration) << 32U) ^ ant_index));
    std::vector<AcoAgentState> state(agent_count);
    for (std::size_t index = 0; index < agent_count; ++index) {
      state[index].pos = day.agents[index].pos;
      state[index].node = graph.node_for_pos.at(state[index].pos);
      state[index].fuel = day.agents[index].fuel;
    }
    std::vector<int> stock;
    stock.reserve(config.spots.size());
    for (const auto& spot : config.spots) stock.push_back(spot.stocks);
    std::uint64_t planned_match = initial_match;
    std::uint64_t planned_daily = 0;

    auto acquire = [&](std::size_t patrol, int pos) {
      const int spot = spot_for_cell[pos];
      if (spot < 0) return;
      const std::uint64_t spot_bit = std::uint64_t{1} << spot;
      if ((state[patrol].visited_spots & spot_bit) != 0 || stock[spot] <= 0) {
        return;
      }
      state[patrol].visited_spots |= spot_bit;
      --stock[spot];
      const std::uint64_t brand_bit = std::uint64_t{1} << spot_brand[spot];
      planned_match |= brand_bit;
      planned_daily |= brand_bit;
    };
    auto append_wait = [&](std::size_t agent, int duration) {
      if (duration <= 0) return;
      state[agent].actions.push_back(-duration);
      state[agent].elapsed += duration;
      if (types[agent] == AgentKind::Patrol) acquire(agent, state[agent].pos);
    };
    auto append_path = [&](std::size_t agent, const AcoPath& path) {
      for (int direction : path.directions) {
        const int duration =
            terrain_time(config, state[agent].pos, day.traffics);
        if (types[agent] == AgentKind::Patrol && duration > 1) {
          acquire(agent, state[agent].pos);
        }
        state[agent].actions.push_back(direction);
        state[agent].elapsed += duration;
        if (types[agent] == AgentKind::Patrol) {
          state[agent].fuel -= terrain_fuel(config, state[agent].pos);
        }
        state[agent].pos = *neighbor(config, state[agent].pos, direction);
        if (types[agent] == AgentKind::Patrol) acquire(agent, state[agent].pos);
      }
      state[agent].node = graph.node_for_pos.at(state[agent].pos);
    };

    std::vector<bool> finished(agent_count);
    while (true) {
      std::optional<std::size_t> selected_patrol;
      for (std::size_t index = 0; index < agent_count; ++index) {
        if (types[index] != AgentKind::Patrol || finished[index]) continue;
        if (!selected_patrol ||
            std::tie(state[index].elapsed, index) <
                std::tie(state[*selected_patrol].elapsed, *selected_patrol)) {
          selected_patrol = index;
        }
      }
      if (!selected_patrol) break;
      const std::size_t patrol = *selected_patrol;
      std::vector<AcoCandidate> candidates;
      candidates.reserve(config.spots.size() * 4);
      for (std::size_t spot_index = 0; spot_index < config.spots.size();
           ++spot_index) {
        const auto& spot = config.spots[spot_index];
        const std::uint64_t spot_bit = std::uint64_t{1} << spot_index;
        if (stock[spot_index] <= 0 ||
            (state[patrol].visited_spots & spot_bit) != 0) {
          continue;
        }
        const int target_node = graph.node_for_pos.at(spot.pos);
        const std::uint64_t brand_bit =
            std::uint64_t{1} << spot_brand[spot_index];
        const int is_new_match = (planned_match & brand_bit) == 0 ? 1 : 0;
        const int is_new_daily = (planned_daily & brand_bit) == 0 ? 1 : 0;
        const long long reward = is_new_match * match_weight +
                                 is_new_daily * daily_weight + 1;
        bool direct_feasible = false;
        for (const auto& path : graph.paths[state[patrol].node][target_node]) {
          const int added = path.directions.empty() ? 1 : path.time;
          if (path.fuel > state[patrol].fuel ||
              state[patrol].elapsed + added > horizon) {
            continue;
          }
          direct_feasible = true;
          AcoCandidate candidate;
          candidate.patrol = static_cast<int>(patrol);
          candidate.target_node = target_node;
          candidate.finish_time = state[patrol].elapsed + added;
          candidate.reward = reward;
          candidate.patrol_first = path;
          candidate.move_pheromones.push_back(move_index(
              patrol, state[patrol].node, target_node, path.variant));
          const long double eta = static_cast<long double>(reward + 1) /
                                  static_cast<long double>(added + 1);
          candidate.desirability =
              move_pheromone[candidate.move_pheromones.front()] * eta * eta;
          candidates.push_back(std::move(candidate));
        }

        if (direct_feasible && state[patrol].fuel * 3 > config.fuel_limit) {
          continue;
        }
        for (std::size_t refuel = 0; refuel < agent_count; ++refuel) {
          if (types[refuel] != AgentKind::Refuel) continue;
          const std::size_t cache_index =
              ((static_cast<std::size_t>(state[patrol].node) * node_count +
                static_cast<std::size_t>(state[refuel].node)) *
                   node_count +
               static_cast<std::size_t>(target_node));
          for (int meeting : meeting_cache[cache_index]) {
            if (meeting < 0) break;
            const auto& refuel_options =
                graph.paths[state[refuel].node][meeting];
            if (refuel_options.empty()) continue;
            const auto& refuel_path = refuel_options.front();
            std::optional<AcoCandidate> best_meeting;
            for (const auto& first :
                 graph.paths[state[patrol].node][meeting]) {
              if (first.fuel > state[patrol].fuel) continue;
              for (const auto& after : graph.paths[meeting][target_node]) {
                if (after.fuel > config.fuel_limit) continue;
                int meeting_time =
                    std::max(state[patrol].elapsed + first.time,
                             state[refuel].elapsed + refuel_path.time);
                if (first.directions.empty() &&
                    refuel_path.directions.empty() &&
                    state[patrol].elapsed == state[refuel].elapsed) {
                  ++meeting_time;
                }
                const int finish = meeting_time + after.time;
                if (meeting_time > horizon || finish > horizon) continue;
                AcoCandidate candidate;
                candidate.patrol = static_cast<int>(patrol);
                candidate.target_node = target_node;
                candidate.refuel = static_cast<int>(refuel);
                candidate.meeting_node = meeting;
                candidate.meeting_time = meeting_time;
                candidate.finish_time = finish;
                candidate.refuel_finish_time = meeting_time;
                candidate.reward = reward;
                candidate.patrol_first = first;
                candidate.refuel_path = refuel_path;
                candidate.patrol_after = after;
                candidate.move_pheromones = {
                    move_index(patrol, state[patrol].node, meeting,
                               first.variant),
                    move_index(patrol, meeting, target_node, after.variant)};
                candidate.meeting_pheromone =
                    meeting_index(patrol, refuel, meeting);
                const int added =
                    finish - state[patrol].elapsed +
                    meeting_time - state[refuel].elapsed;
                const long double eta =
                    static_cast<long double>(reward + 1) /
                    static_cast<long double>(added + 1);
                long double tau =
                    meeting_pheromone[*candidate.meeting_pheromone];
                for (auto key : candidate.move_pheromones) {
                  tau *= move_pheromone[key];
                }
                candidate.desirability = tau * eta * eta;
                if (!best_meeting ||
                    std::tie(candidate.finish_time, candidate.refuel_finish_time,
                             candidate.patrol_first.fuel,
                             candidate.patrol_after.fuel) <
                        std::tie(best_meeting->finish_time,
                                 best_meeting->refuel_finish_time,
                                 best_meeting->patrol_first.fuel,
                                 best_meeting->patrol_after.fuel)) {
                  best_meeting = std::move(candidate);
                }
              }
            }
            if (best_meeting) candidates.push_back(std::move(*best_meeting));
          }
        }
      }

      if (candidates.empty()) {
        finished[patrol] = true;
        continue;
      }
      long double total = 0;
      for (const auto& candidate : candidates) total += candidate.desirability;
      std::uniform_real_distribution<long double> distribution(0, total);
      long double draw = distribution(random);
      std::size_t selected = candidates.size() - 1;
      for (std::size_t index = 0; index < candidates.size(); ++index) {
        draw -= candidates[index].desirability;
        if (draw <= 0) {
          selected = index;
          break;
        }
      }
      const auto& choice = candidates[selected];
      if (choice.refuel < 0) {
        if (choice.patrol_first.directions.empty()) {
          append_wait(patrol, 1);
        } else {
          append_path(patrol, choice.patrol_first);
        }
      } else {
        const std::size_t refuel = static_cast<std::size_t>(choice.refuel);
        append_path(patrol, choice.patrol_first);
        append_path(refuel, choice.refuel_path);
        append_wait(patrol, choice.meeting_time - state[patrol].elapsed);
        append_wait(refuel, choice.meeting_time - state[refuel].elapsed);
        state[patrol].fuel = config.fuel_limit;
        append_path(patrol, choice.patrol_after);
        ant.used_meetings.push_back(*choice.meeting_pheromone);
      }
      ant.used_moves.insert(ant.used_moves.end(),
                            choice.move_pheromones.begin(),
                            choice.move_pheromones.end());
    }

    ant.plan.resize(agent_count);
    for (std::size_t index = 0; index < agent_count; ++index) {
      append_wait(index, horizon - state[index].elapsed);
      if (state[index].actions.empty()) state[index].actions.push_back(-horizon);
      ant.plan[index] = std::move(state[index].actions);
    }
    if (apply_local_search) {
      ant.plan = improve_with_route_substitutions(
          config, day, history, std::move(ant.plan),
          local_search_alternatives, 1, false);
    }
    ant.quality = candidate_value(config, day, history, ant.plan);
    return ant;
  };

  AcoAnt best;
  best.plan = wait_plan(agent_count, horizon);
  best.quality = candidate_value(config, day, history, best.plan);
  const int default_ant_count =
      std::clamp(static_cast<int>(2 * agent_count + 8), 16, 24);
  const int ant_count =
      limits.aco_ants > 0 ? limits.aco_ants : default_ant_count;
  const int default_iterations = config.day_seconds[day.day] <= 10.0 ? 12 : 20;
  const int iterations =
      limits.aco_iterations > 0 ? limits.aco_iterations : default_iterations;
  const double evaporation = limits.aco_evaporation;
  auto better = [](const AcoAnt& left, const AcoAnt& right) {
    if (left.quality != right.quality) return left.quality > right.quality;
    return left.index < right.index;
  };
  for (int iteration = 0; iteration < iterations; ++iteration) {
    auto ants = parallel_indexed(
        ant_count, [&](std::size_t index) { return construct_ant(index, iteration); });
    std::vector<std::size_t> order(ants.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
      return better(ants[left], ants[right]);
    });
    if (!order.empty() && better(ants[order.front()], best)) {
      best = ants[order.front()];
    }

    for (double& value : move_pheromone) value = std::max(0.05, value * evaporation);
    for (double& value : meeting_pheromone) {
      value = std::max(0.05, value * evaporation);
    }
    const std::size_t elite_count = std::max<std::size_t>(1, ants.size() / 5);
    auto reinforce = [&](const AcoAnt& ant, double amount) {
      const double move_amount = amount / std::max<std::size_t>(1, ant.used_moves.size());
      for (auto key : ant.used_moves) {
        move_pheromone[key] = std::min(5.0, move_pheromone[key] + move_amount);
      }
      const double meeting_amount =
          amount / std::max<std::size_t>(1, ant.used_meetings.size());
      for (auto key : ant.used_meetings) {
        meeting_pheromone[key] =
            std::min(5.0, meeting_pheromone[key] + meeting_amount);
      }
    };
    for (std::size_t rank = 0; rank < elite_count; ++rank) {
      reinforce(ants[order[rank]],
                static_cast<double>(elite_count - rank) / elite_count);
    }
    reinforce(best, 1.0);
  }
  return best.plan;
}
}  // namespace hexudon
