#include "hexudon/core.hpp"
#include "hexudon/internal.hpp"

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace hexudon {
namespace {
thread_local PalnsDiagnostics palns_diagnostics;
thread_local MlnsDiagnostics mlns_diagnostics;
thread_local std::vector<std::map<int, int>> mlns_predicted_traffic;
using PalnsReturnedRank =
    std::tuple<std::tuple<int, int, int>, int>;
thread_local std::map<ActionPlan, PalnsReturnedRank>
    palns_returned_projections;

bool lns_dp_proposals_enabled(const SearchLimits& limits) {
  const char* disabled = std::getenv("HEXUDON_DISABLE_LNS_DP_PROPOSALS");
  if (disabled != nullptr && disabled[0] != '\0' && disabled[0] != '0')
    return false;
  if (limits.use_lns_dp_proposals)
    return true;
  const char* enabled = std::getenv("HEXUDON_ENABLE_LNS_DP_PROPOSALS");
  return enabled != nullptr && enabled[0] != '\0' && enabled[0] != '0';
}

bool stop_bp_proposals_enabled() {
  const char* disabled = std::getenv("HEXUDON_DISABLE_STOP_BP_PROPOSALS");
  if (disabled != nullptr && disabled[0] != '\0' && disabled[0] != '0') {
    return false;
  }
  const char* enabled = std::getenv("HEXUDON_ENABLE_STOP_BP_PROPOSALS");
  return enabled != nullptr && enabled[0] != '\0' && enabled[0] != '0';
}
}

void reset_palns_diagnostics() {
  palns_diagnostics = {};
  palns_returned_projections.clear();
}

PalnsDiagnostics current_palns_diagnostics() { return palns_diagnostics; }

void reset_mlns_diagnostics() { mlns_diagnostics = {}; }

MlnsDiagnostics current_mlns_diagnostics() { return mlns_diagnostics; }

AlnsAnytimeBudget alns_anytime_budget(int total_ms, bool final_day,
                                      bool allow_continuation,
                                      bool exact_enabled,
                                      int continuation_time_percent,
                                      int exact_time_percent) {
  const int budget = std::max(0, total_ms);
  if (final_day && exact_enabled) {
    const int main = budget * (100 - exact_time_percent) / 100;
    return {main, 0, budget - main};
  }
  if (!final_day && allow_continuation) {
    const int main = budget * (100 - continuation_time_percent) / 100;
    return {main, budget - main, 0};
  }
  return {budget, 0, 0};
}

struct LnsAgentState {
  int pos{};
  int node{};
  int elapsed{};
  int fuel{};
  std::vector<int> actions;
};

std::uint64_t lns_mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::uint64_t lns_seed(const MapConfig& config, const DayInfo& day,
                       const PolicyHistory& history) {
  std::uint64_t seed = 0x4c4e535f484558ULL;
  auto add = [&](std::uint64_t value) { seed = lns_mix(seed ^ value); };
  add(static_cast<std::uint64_t>(config.width));
  add(static_cast<std::uint64_t>(config.height));
  add(static_cast<std::uint64_t>(config.fuel_limit));
  add(static_cast<std::uint64_t>(day.day));
  for (const auto& agent : day.agents) {
    add(static_cast<std::uint64_t>(agent.pos + 1));
    // Refuel agents do not consume fuel.  The live server reports their fuel
    // as zero while the local evaluator historically initialized it to the
    // configured limit; canonicalize that irrelevant field so equivalent
    // online/offline states follow the same deterministic ALNS trajectory.
    const int canonical_fuel =
        agent.kind == AgentKind::Refuel ? config.fuel_limit : agent.fuel;
    add(static_cast<std::uint64_t>(canonical_fuel + 1));
    add(static_cast<std::uint64_t>(agent.kind == AgentKind::Refuel));
  }
  for (int brand : history.distinct_brands) {
    add(static_cast<std::uint64_t>(brand + 1));
  }
  return seed;
}

int lns_path_time(const AcoGraph& graph, int from, int to) {
  const auto& paths = graph.paths[from][to];
  return paths.empty() ? std::numeric_limits<int>::max() / 4
                       : paths.front().time;
}

int lns_route_time(const MapConfig& config, const DayInfo& day,
                   const AcoGraph& graph, std::size_t agent,
                   const std::vector<int>& route) {
  int node = graph.node_for_pos.at(day.agents[agent].pos);
  int total = 0;
  for (int spot_index : route) {
    const int next = graph.node_for_pos.at(config.spots[spot_index].pos);
    const int added = lns_path_time(graph, node, next);
    if (added >= std::numeric_limits<int>::max() / 8) return added;
    total += node == next ? 1 : added;
    node = next;
  }
  return total;
}

struct LnsInsertion {
  int spot{};
  std::size_t agent{};
  std::size_t position{};
  int tier{};
  int delta{};
  int route_time{};
  int projected_time{};
  int regret{};
  std::uint64_t tie_break{};
};

void repair_lns_skeleton(const MapConfig& config, const DayInfo& day,
                         const PolicyHistory& history,
                         const AgentTypes& types, const AcoGraph& graph,
                         LnsSkeleton& skeleton, int mode,
                         std::mt19937_64& random) {
  const int horizon = config.day_steps[day.day];
  int maximum_visits = 0;
  int patrol_count = 0;
  for (auto type : types) patrol_count += type == AgentKind::Patrol;
  for (const auto& spot : config.spots) {
    maximum_visits += std::min(spot.stocks, patrol_count);
  }
  auto total_visits = [&] {
    std::size_t total = 0;
    for (const auto& route : skeleton.routes) total += route.size();
    return static_cast<int>(total);
  };

  // SISR recreate samples a customer ordering once per repair, then takes a
  // strongly best-biased random element from that ordering.  HEX has no scalar
  // customer prize, so the score ordering below uses the official brand tier;
  // the density orderings prefer spots that are cheap to connect to two/five
  // other spots.  The 4:4:1:1:1 mix mirrors the useful TOP orderings (random,
  // score, close-to-start, KNN-2, KNN-5) without importing depot assumptions.
  int sisr_sort = 0;
  double sisr_shaw = 0.0;
  if (mode == 4) {
    static constexpr std::array<int, 5> weights{4, 4, 1, 1, 1};
    const int draw = static_cast<int>(random() % 11U);
    int cumulative = 0;
    for (std::size_t index = 0; index < weights.size(); ++index) {
      cumulative += weights[index];
      if (draw < cumulative) {
        sisr_sort = static_cast<int>(index);
        break;
      }
    }
    sisr_shaw = 0.3 * static_cast<double>(random() % 10001U) / 10000.0;
  }

  // Precompute each spot's summed distance to its k nearest spots once. The
  // value depends only on the (unchanging) graph and the fixed neighbour count,
  // so the old per-candidate, per-comparison all-pairs sort recomputed the same
  // numbers thousands of times per repair. Produces identical values.
  std::vector<int> spot_density;
  if (mode == 4 && sisr_sort >= 3) {
    const std::size_t neighbours = sisr_sort == 3 ? 2U : 5U;
    const std::size_t spot_count = config.spots.size();
    spot_density.assign(spot_count, 0);
    std::vector<int> distances;
    for (std::size_t spot = 0; spot < spot_count; ++spot) {
      distances.clear();
      const int node = graph.node_for_pos.at(config.spots[spot].pos);
      for (std::size_t other = 0; other < spot_count; ++other) {
        if (other == spot) continue;
        distances.push_back(lns_path_time(
            graph, node, graph.node_for_pos.at(config.spots[other].pos)));
      }
      std::sort(distances.begin(), distances.end());
      int total = 0;
      for (std::size_t index = 0;
           index < std::min(neighbours, distances.size()); ++index) {
        total += distances[index];
      }
      spot_density[spot] = total;
    }
  }

  while (total_visits() < maximum_visits) {
    std::vector<int> assigned(config.spots.size());
    std::set<int> planned_brands;
    for (const auto& route : skeleton.routes) {
      for (int spot : route) {
        ++assigned[spot];
        planned_brands.insert(config.spots[spot].brand);
      }
    }

    std::vector<LnsInsertion> candidates;
    for (std::size_t spot = 0; spot < config.spots.size(); ++spot) {
      if (assigned[spot] >= config.spots[spot].stocks) continue;
      std::vector<LnsInsertion> options;
      const int brand = config.spots[spot].brand;
      const int tier = !history.distinct_brands.contains(brand) &&
                               !planned_brands.contains(brand)
                           ? 3
                           : (!planned_brands.contains(brand) ? 2 : 1);
      const int target = graph.node_for_pos.at(config.spots[spot].pos);
      for (std::size_t agent = 0; agent < types.size(); ++agent) {
        if (types[agent] != AgentKind::Patrol) continue;
        auto& route = skeleton.routes[agent];
        if (std::find(route.begin(), route.end(), static_cast<int>(spot)) !=
            route.end()) {
          continue;
        }
        const int current_time =
            lns_route_time(config, day, graph, agent, route);
        for (std::size_t position = 0; position <= route.size(); ++position) {
          const int previous =
              position == 0
                  ? graph.node_for_pos.at(day.agents[agent].pos)
                  : graph.node_for_pos.at(
                        config.spots[route[position - 1]].pos);
          const int next =
              position == route.size()
                  ? -1
                  : graph.node_for_pos.at(config.spots[route[position]].pos);
          const int before = next < 0 ? 0 : lns_path_time(graph, previous, next);
          const int first = lns_path_time(graph, previous, target);
          const int second = next < 0 ? 0 : lns_path_time(graph, target, next);
          if (first >= std::numeric_limits<int>::max() / 8 ||
              second >= std::numeric_limits<int>::max() / 8) {
            continue;
          }
          int delta = first + second - before;
          if (previous == target) delta = std::max(1, delta);
          if (current_time + delta > horizon) continue;
          options.push_back({static_cast<int>(spot), agent, position, tier,
                             delta, current_time, current_time + delta, 0,
                             random()});
        }
      }
      if (options.empty()) continue;
      std::sort(options.begin(), options.end(), [](const auto& left,
                                                   const auto& right) {
        return std::tie(left.projected_time, left.delta, left.tie_break,
                        left.position) <
               std::tie(right.projected_time, right.delta, right.tie_break,
                        right.position);
      });
      std::size_t option_index = 0;
      if (mode == 4 && options.size() > 1) {
        // SISR's blink mechanism occasionally ignores the current cheapest
        // feasible insertion.  Keep its one-percent rate, but always retain a
        // feasible fallback so a sequence of blinks cannot truncate repair.
        while (option_index + 1 < options.size() && random() % 100U == 0U) {
          ++option_index;
        }
      }
      options[option_index].regret =
          options.size() > 1
              ? options[1].projected_time - options[0].projected_time
              : horizon;
      candidates.push_back(options[option_index]);
    }
    if (candidates.empty()) break;
    std::sort(candidates.begin(), candidates.end(), [&](const auto& left,
                                                        const auto& right) {
      if (mode == 4) {
        if (sisr_sort == 0) return left.tie_break < right.tie_break;
        if (sisr_sort == 1) {
          if (left.tier != right.tier) return left.tier > right.tier;
          // Within one official-value tier, schedule the difficult insertion
          // first.  Cheap-first leaves high-detour brands until they no longer
          // fit; this HEX-specific regret safeguard was stronger in ablation.
          return std::tie(left.delta, left.tie_break) >
                 std::tie(right.delta, right.tie_break);
        }
        if (sisr_sort == 2) {
          return std::tie(left.projected_time, left.delta, left.tie_break) <
                 std::tie(right.projected_time, right.delta, right.tie_break);
        }
        const int left_density = spot_density[static_cast<std::size_t>(left.spot)];
        const int right_density =
            spot_density[static_cast<std::size_t>(right.spot)];
        return std::tuple{left.tier, -left_density, -left.delta,
                          left.tie_break} >
               std::tuple{right.tier, -right_density, -right.delta,
                          right.tie_break};
      }
      if (mode == 1 || mode == 3) {
        if (left.tier != right.tier) return left.tier > right.tier;
        if (left.regret != right.regret) return left.regret > right.regret;
      }
      return std::tie(left.projected_time, left.delta, left.tie_break,
                      left.position) <
             std::tie(right.projected_time, right.delta, right.tie_break,
                      right.position);
    });
    std::size_t selected = 0;
    if (mode == 4 && sisr_sort != 0 && sisr_shaw > 0.0001) {
      const double unit = (static_cast<double>(random() % 1000001U) + 0.5) /
                          1000001.0;
      const std::size_t reverse_index = std::min<std::size_t>(
          candidates.size() - 1,
          static_cast<std::size_t>(candidates.size() *
                                   std::pow(unit, sisr_shaw)));
      selected = candidates.size() - 1 - reverse_index;
    } else if (mode == 2 || mode == 3) {
      const std::size_t range = std::min<std::size_t>(3, candidates.size());
      selected = static_cast<std::size_t>(random() % range);
    }
    const auto choice = candidates[selected];
    auto& route = skeleton.routes[choice.agent];
    route.insert(route.begin() + static_cast<std::ptrdiff_t>(choice.position),
                 choice.spot);
  }
}

void destroy_lns_skeleton(const MapConfig& config,
                          const PolicyHistory& history,
                          const AcoGraph& graph, LnsSkeleton& skeleton,
                          int mode, int remove_count,
                          std::mt19937_64& random) {
  using Visit = std::pair<std::size_t, std::size_t>;
  std::vector<Visit> visits;
  for (std::size_t agent = 0; agent < skeleton.routes.size(); ++agent) {
    for (std::size_t position = 0; position < skeleton.routes[agent].size();
         ++position) {
      visits.emplace_back(agent, position);
    }
  }
  if (visits.empty()) return;
  remove_count = std::clamp(remove_count, 1, static_cast<int>(visits.size()));
  std::vector<Visit> removed;
  if (mode == 1) {
    std::vector<std::size_t> nonempty;
    for (std::size_t agent = 0; agent < skeleton.routes.size(); ++agent) {
      if (!skeleton.routes[agent].empty()) nonempty.push_back(agent);
    }
    const std::size_t agent = nonempty[random() % nonempty.size()];
    const auto& route = skeleton.routes[agent];
    const std::size_t length =
        std::min<std::size_t>(route.size(), remove_count);
    const std::size_t begin = random() % (route.size() - length + 1);
    for (std::size_t offset = 0; offset < length; ++offset) {
      removed.emplace_back(agent, begin + offset);
    }
  } else if (mode == 2) {
    const Visit seed = visits[random() % visits.size()];
    const int seed_spot = skeleton.routes[seed.first][seed.second];
    const int seed_node =
        graph.node_for_pos.at(config.spots[seed_spot].pos);
    std::sort(visits.begin(), visits.end(), [&](const Visit& left,
                                                const Visit& right) {
      auto rank = [&](const Visit& visit) {
        const int spot = skeleton.routes[visit.first][visit.second];
        const int node = graph.node_for_pos.at(config.spots[spot].pos);
        return std::tuple{
            config.spots[spot].brand == config.spots[seed_spot].brand ? 0 : 1,
            lns_path_time(graph, seed_node, node), visit.first, visit.second};
      };
      return rank(left) < rank(right);
    });
    removed.assign(visits.begin(), visits.begin() + remove_count);
  } else if (mode == 3) {
    std::map<int, int> brand_count;
    for (const auto& route : skeleton.routes) {
      for (int spot : route) ++brand_count[config.spots[spot].brand];
    }
    std::sort(visits.begin(), visits.end(), [&](const Visit& left,
                                                const Visit& right) {
      auto rank = [&](const Visit& visit) {
        const auto& route = skeleton.routes[visit.first];
        const int spot = route[visit.second];
        const int brand = config.spots[spot].brand;
        const int tier = !history.distinct_brands.contains(brand) &&
                                 brand_count[brand] == 1
                             ? 3
                             : (brand_count[brand] == 1 ? 2 : 1);
        int saving = 0;
        const int node = graph.node_for_pos.at(config.spots[spot].pos);
        if (visit.second > 0 && visit.second + 1 < route.size()) {
          const int previous = graph.node_for_pos.at(
              config.spots[route[visit.second - 1]].pos);
          const int next = graph.node_for_pos.at(
              config.spots[route[visit.second + 1]].pos);
          saving = lns_path_time(graph, previous, node) +
                   lns_path_time(graph, node, next) -
                   lns_path_time(graph, previous, next);
        }
        return std::tuple{tier, -saving, visit.first, visit.second};
      };
      return rank(left) < rank(right);
    });
    removed.assign(visits.begin(), visits.begin() + remove_count);
  } else if (mode == 4) {
    std::sort(visits.begin(), visits.end(), [&](const Visit& left,
                                                const Visit& right) {
      const auto left_size = skeleton.routes[left.first].size();
      const auto right_size = skeleton.routes[right.first].size();
      if (left_size != right_size) return left_size > right_size;
      if (left.second != right.second) return left.second > right.second;
      return left.first < right.first;
    });
    removed.assign(visits.begin(), visits.begin() + remove_count);
  } else {
    std::shuffle(visits.begin(), visits.end(), random);
    removed.assign(visits.begin(), visits.begin() + remove_count);
  }
  if (static_cast<int>(removed.size()) < remove_count) {
    std::shuffle(visits.begin(), visits.end(), random);
    for (const auto& visit : visits) {
      if (std::find(removed.begin(), removed.end(), visit) == removed.end()) {
        removed.push_back(visit);
        if (static_cast<int>(removed.size()) == remove_count) break;
      }
    }
  }
  std::sort(removed.begin(), removed.end(), [](const Visit& left,
                                               const Visit& right) {
    if (left.first != right.first) return left.first > right.first;
    return left.second > right.second;
  });
  for (const auto [agent, position] : removed) {
    skeleton.routes[agent].erase(
        skeleton.routes[agent].begin() + static_cast<std::ptrdiff_t>(position));
  }
}

[[maybe_unused]] bool destroy_alns_refuel_bottleneck(
    LnsSkeleton& skeleton, const SimulationTrace& trace, int remove_count,
    std::mt19937_64& random) {
  if (trace.refuels.empty()) return false;
  std::map<std::size_t, int> load;
  for (const auto& event : trace.refuels) ++load[event.refuel];
  const auto busiest = std::max_element(
      load.begin(), load.end(), [](const auto& left, const auto& right) {
        if (left.second != right.second) return left.second < right.second;
        return left.first > right.first;
      });
  if (busiest == load.end()) return false;
  std::map<std::size_t, int> patrol_load;
  for (const auto& event : trace.refuels) {
    if (event.refuel == busiest->first) ++patrol_load[event.patrol];
  }
  using Visit = std::pair<std::size_t, std::size_t>;
  std::vector<Visit> visits;
  for (const auto& [patrol, events] : patrol_load) {
    if (patrol >= skeleton.routes.size()) continue;
    for (std::size_t position = 0;
         position < skeleton.routes[patrol].size(); ++position) {
      visits.emplace_back(patrol, position);
    }
  }
  if (visits.empty()) return false;
  std::shuffle(visits.begin(), visits.end(), random);
  std::stable_sort(visits.begin(), visits.end(), [&](const Visit& left,
                                                      const Visit& right) {
    return patrol_load[left.first] > patrol_load[right.first];
  });
  visits.resize(std::min<std::size_t>(visits.size(),
                                     std::max(1, remove_count)));
  std::sort(visits.begin(), visits.end(), [](const Visit& left,
                                             const Visit& right) {
    return left.first != right.first ? left.first > right.first
                                     : left.second > right.second;
  });
  for (const auto [agent, position] : visits) {
    skeleton.routes[agent].erase(
        skeleton.routes[agent].begin() + static_cast<std::ptrdiff_t>(position));
  }
  return true;
}

std::optional<ActionPlan> decode_lns_skeleton(
    const MapConfig& config, const DayInfo& day, const AgentTypes& types,
    const AcoGraph& graph, const std::vector<AcoMeetingList>& meeting_cache,
    const LnsSkeleton& skeleton,
    const AlnsTravelChoices* travel_choices, bool strict_travel) {
  const int horizon = config.day_steps[day.day];
  const std::size_t count = day.agents.size();
  const std::size_t nodes = graph.nodes.size();
  std::vector<LnsAgentState> state(count);
  std::vector<std::size_t> cursor(count);
  for (std::size_t index = 0; index < count; ++index) {
    state[index].pos = day.agents[index].pos;
    state[index].node = graph.node_for_pos.at(state[index].pos);
    state[index].fuel = day.agents[index].fuel;
  }
  auto append_wait = [&](std::size_t agent, int duration) {
    if (duration <= 0) return;
    state[agent].actions.push_back(-duration);
    state[agent].elapsed += duration;
  };
  auto append_path = [&](std::size_t agent, const AcoPath& path) {
    for (int direction : path.directions) {
      state[agent].actions.push_back(direction);
      state[agent].elapsed +=
          terrain_time(config, state[agent].pos, day.traffics);
      if (types[agent] == AgentKind::Patrol) {
        state[agent].fuel -= terrain_fuel(config, state[agent].pos);
      }
      state[agent].pos = *neighbor(config, state[agent].pos, direction);
    }
    state[agent].node = graph.node_for_pos.at(state[agent].pos);
  };

  while (true) {
    std::optional<std::size_t> selected;
    for (std::size_t agent = 0; agent < count; ++agent) {
      if (types[agent] != AgentKind::Patrol ||
          cursor[agent] >= skeleton.routes[agent].size()) {
        continue;
      }
      if (!selected || std::tie(state[agent].elapsed, agent) <
                           std::tie(state[*selected].elapsed, *selected)) {
        selected = agent;
      }
    }
    if (!selected) break;
    const std::size_t patrol = *selected;
    const std::size_t leg_index = cursor[patrol]++;
    const int spot_index = skeleton.routes[patrol][leg_index];
    const int target =
        graph.node_for_pos.at(config.spots[spot_index].pos);
    const std::uint32_t choice_code =
        travel_choices && patrol < travel_choices->size() &&
                leg_index < (*travel_choices)[patrol].size()
            ? (*travel_choices)[patrol][leg_index]
            : 0U;
    const int strategy = static_cast<int>(choice_code % 4U);
    const std::size_t option_rank =
        static_cast<std::size_t>((choice_code / 4U) % 3U);

    std::vector<AcoPath> direct_options;
    for (const auto& path : graph.paths[state[patrol].node][target]) {
      const int added = path.directions.empty() ? 1 : path.time;
      if (path.fuel <= state[patrol].fuel &&
          state[patrol].elapsed + added <= horizon) {
        direct_options.push_back(path);
      }
    }
    std::sort(direct_options.begin(), direct_options.end(),
              [&](const AcoPath& left, const AcoPath& right) {
                if (strategy == 1 || strategy == 2) {
                  return std::tie(left.fuel, left.time, left.directions) <
                         std::tie(right.fuel, right.time, right.directions);
                }
                if (strategy == 3) {
                  return std::tuple{left.time + left.fuel, left.time,
                                    left.fuel, left.directions} <
                         std::tuple{right.time + right.fuel, right.time,
                                    right.fuel, right.directions};
                }
                return std::tie(left.time, left.fuel, left.directions) <
                       std::tie(right.time, right.fuel, right.directions);
              });
    std::optional<AcoPath> direct;
    if (!direct_options.empty()) {
      direct = direct_options[std::min(option_rank, direct_options.size() - 1)];
    }
    bool prefer_refuel = !direct.has_value();
    if (direct && cursor[patrol] < skeleton.routes[patrol].size()) {
      const int next = graph.node_for_pos.at(
          config.spots[skeleton.routes[patrol][cursor[patrol]]].pos);
      int required = std::numeric_limits<int>::max() / 4;
      for (const auto& path : graph.paths[target][next]) {
        required = std::min(required, path.fuel);
      }
      prefer_refuel = state[patrol].fuel - direct->fuel < required;
    }
    if (travel_choices &&
        (strategy == 2 || ((choice_code / 12U) % 2U) != 0U)) {
      prefer_refuel = true;
    }

    struct MeetingChoice {
      std::size_t refuel{};
      int meeting{};
      int meeting_time{};
      int finish{};
      AcoPath first;
      AcoPath supply;
      AcoPath after;
    };
    std::vector<MeetingChoice> meeting_options;
    if (prefer_refuel) {
      for (std::size_t refuel = 0; refuel < count; ++refuel) {
        if (types[refuel] != AgentKind::Refuel) continue;
        const std::size_t cache_index =
            ((static_cast<std::size_t>(state[patrol].node) * nodes +
              static_cast<std::size_t>(state[refuel].node)) *
                 nodes +
             static_cast<std::size_t>(target));
        for (int meeting : meeting_cache[cache_index]) {
          if (meeting < 0) break;
          for (const auto& supply :
               graph.paths[state[refuel].node][meeting]) {
            for (const auto& first : graph.paths[state[patrol].node][meeting]) {
              if (first.fuel > state[patrol].fuel) continue;
              for (const auto& after : graph.paths[meeting][target]) {
                if (after.fuel > config.fuel_limit) continue;
                int meeting_time =
                    std::max(state[patrol].elapsed + first.time,
                             state[refuel].elapsed + supply.time);
                if (first.directions.empty() && supply.directions.empty() &&
                    state[patrol].elapsed == state[refuel].elapsed) {
                  ++meeting_time;
                }
                const int finish = meeting_time + after.time;
                if (meeting_time > horizon || finish > horizon) continue;
                meeting_options.push_back({refuel, meeting, meeting_time,
                                           finish, first, supply, after});
              }
            }
          }
        }
      }
    }
    std::sort(meeting_options.begin(), meeting_options.end(),
              [&](const MeetingChoice& left, const MeetingChoice& right) {
                if (strategy == 1) {
                  return std::tuple{left.after.fuel, left.finish,
                                    left.meeting_time, left.refuel,
                                    left.meeting} <
                         std::tuple{right.after.fuel, right.finish,
                                    right.meeting_time, right.refuel,
                                    right.meeting};
                }
                if (strategy == 3) {
                  return std::tuple{left.finish + left.first.fuel +
                                        left.after.fuel,
                                    left.finish, left.refuel, left.meeting} <
                         std::tuple{right.finish + right.first.fuel +
                                        right.after.fuel,
                                    right.finish, right.refuel, right.meeting};
                }
                return std::tie(left.finish, left.meeting_time, left.refuel,
                                left.meeting) <
                       std::tie(right.finish, right.meeting_time, right.refuel,
                                right.meeting);
              });
    std::optional<MeetingChoice> meeting_choice;
    if (!meeting_options.empty()) {
      meeting_choice = meeting_options[
          std::min(option_rank, meeting_options.size() - 1)];
    }

    if (meeting_choice) {
      auto& choice = *meeting_choice;
      append_path(patrol, choice.first);
      append_path(choice.refuel, choice.supply);
      append_wait(patrol, choice.meeting_time - state[patrol].elapsed);
      append_wait(choice.refuel,
                  choice.meeting_time - state[choice.refuel].elapsed);
      state[patrol].fuel = config.fuel_limit;
      append_path(patrol, choice.after);
    } else if (direct) {
      if (direct->directions.empty()) {
        append_wait(patrol, 1);
      } else {
        append_path(patrol, *direct);
      }
    } else if (travel_choices && strict_travel) {
      return std::nullopt;
    }
    // When strict_travel is false an unreachable leg is simply skipped: the
    // patrol keeps its position/fuel and the decoder realizes the feasible
    // subsequence of the skeleton instead of discarding the whole candidate.
    // This is what lets the ALNS loop produce valid candidates under tight
    // fuel, where fuel-blind repair routinely over-fills a route.
  }

  ActionPlan plan(count);
  for (std::size_t agent = 0; agent < count; ++agent) {
    append_wait(agent, horizon - state[agent].elapsed);
    if (state[agent].actions.empty()) state[agent].actions.push_back(-horizon);
    plan[agent] = std::move(state[agent].actions);
  }
  if (validate_action_plan(config, day, plan)) return std::nullopt;
  return plan;
}

LnsSkeleton lns_skeleton_from_trace(const MapConfig& config,
                                    std::size_t agent_count,
                                    const SimulationTrace& trace) {
  std::map<int, int> spot_index;
  for (std::size_t index = 0; index < config.spots.size(); ++index) {
    spot_index[config.spots[index].pos] = static_cast<int>(index);
  }
  LnsSkeleton skeleton;
  skeleton.routes.resize(agent_count);
  for (const auto& event : trace.acquisitions) {
    skeleton.routes[event.agent].push_back(spot_index.at(event.spot_pos));
  }
  return skeleton;
}

std::size_t lns_skeleton_hash(const LnsSkeleton& skeleton) {
  std::size_t hash = 0xcbf29ce484222325ULL;
  for (std::size_t agent = 0; agent < skeleton.routes.size(); ++agent) {
    hash ^= agent + 1;
    hash *= 0x100000001b3ULL;
    for (int spot : skeleton.routes[agent]) {
      hash ^= static_cast<std::size_t>(spot + 1);
      hash *= 0x100000001b3ULL;
    }
  }
  return hash;
}

ActionPlan build_lns_plan(const MapConfig& config, const DayInfo& day,
                          const PolicyHistory& history,
                          const AgentTypes& types,
                           const SearchLimits& limits,
                           const AcoGraph* shared_graph,
                           const std::vector<AcoMeetingList>* shared_meetings) {
  std::optional<AcoGraph> owned_graph;
  if (!shared_graph) owned_graph = build_aco_graph(config, day);
  const AcoGraph& graph = shared_graph ? *shared_graph : *owned_graph;
  std::optional<std::vector<AcoMeetingList>> owned_meetings;
  if (!shared_meetings) owned_meetings = build_aco_meeting_cache(graph);
  const auto& meeting_cache =
      shared_meetings ? *shared_meetings : *owned_meetings;
  const auto forced = coordinated_first_targets(config, day, history, types);
  std::vector<ActionPlan> seeds;
  seeds.push_back(build_local_search_plan(config, day, history, types, limits));
  seeds.push_back(build_routing_plan("coordinated", config, day, history,
                                     types, forced, limits));
  for (const std::string policy : {"greedy", "utility_greedy", "fuel_aware",
                                   "stock_maximiser"}) {
    seeds.push_back(
        build_routing_plan(policy, config, day, history, types, {}, limits));
  }

  struct Elite {
    LnsSkeleton skeleton;
    ActionPlan plan;
    CandidateValue value;
    std::size_t hash{};
  };
  std::vector<Elite> elite;
  ActionPlan best = seeds.front();
  auto best_value = *candidate_value(config, day, history, best);
  for (auto& seed : seeds) {
    // One evaluation yields both the value and the trace the skeleton is rebuilt
    // from; the previous code simulated each seed twice (candidate_value returns
    // exactly evaluate_candidate(...).value, so this is behaviour-preserving).
    auto evaluation = evaluate_candidate(config, day, history, seed);
    if (!evaluation) continue;
    const CandidateValue value = evaluation->value;
    LnsSkeleton skeleton =
        lns_skeleton_from_trace(config, day.agents.size(), evaluation->trace);
    if (value > best_value) {
      best = seed;
      best_value = value;
    }
    const auto hash = lns_skeleton_hash(skeleton);
    if (std::none_of(elite.begin(), elite.end(),
                     [&](const Elite& item) { return item.hash == hash; })) {
      elite.push_back({std::move(skeleton), seed, value, hash});
    }
  }
  std::sort(elite.begin(), elite.end(),
            [](const Elite& left, const Elite& right) {
              if (left.value != right.value) return left.value > right.value;
              return left.hash < right.hash;
            });
  if (elite.size() > 8) elite.resize(8);
  if (elite.empty()) return best;

  std::mt19937_64 random(lns_seed(config, day, history) ^ limits.random_seed);
  std::array<double, 4> destroy_weights{1, 1, 1, 1};
  std::array<double, 3> repair_weights{1, 1, 1};
  auto choose = [&](const auto& weights) {
    const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    std::uniform_real_distribution<double> distribution(0.0, sum);
    double draw = distribution(random);
    for (std::size_t index = 0; index < weights.size(); ++index) {
      draw -= weights[index];
      if (draw <= 0) return static_cast<int>(index);
    }
    return static_cast<int>(weights.size() - 1);
  };
  const int minimum = std::max(0, limits.min_iterations);
  const int maximum = std::max(minimum, limits.max_iterations);
  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(std::max(0, limits.time_limit_ms));
  int stagnation = 0;
  for (int iteration = 0; iteration < maximum; ++iteration) {
    if (limits.time_limit_ms >= 0 && iteration > 0 &&
        std::chrono::steady_clock::now() >= deadline) {
      break;
    }
    const std::size_t source =
        static_cast<std::size_t>(random() % std::min<std::size_t>(elite.size(), 4));
    LnsSkeleton candidate = elite[source].skeleton;
    int visits = 0;
    for (const auto& route : candidate.routes) visits += route.size();
    int destroy = -1;
    const int repair = choose(repair_weights);
    if (visits == 0) {
      repair_lns_skeleton(config, day, history, types, graph, candidate,
                          repair, random);
    } else {
      const double fraction = stagnation >= limits.stagnation_iterations
                                  ? 0.50
                                  : 0.15 + (random() % 26) / 100.0;
      const int removed = std::max(1, static_cast<int>(std::ceil(visits * fraction)));
      destroy = choose(destroy_weights);
      destroy_lns_skeleton(config, history, graph, candidate, destroy,
                           removed, random);
      repair_lns_skeleton(config, day, history, types, graph, candidate,
                          repair, random);
    }
    auto plan = decode_lns_skeleton(config, day, types, graph, meeting_cache,
                                    candidate);
    if (!plan) continue;
    auto value = candidate_value(config, day, history, *plan);
    if (!value) continue;
    const auto hash = lns_skeleton_hash(candidate);
    const bool global_improvement = *value > best_value;
    if (global_improvement) {
      best = *plan;
      best_value = *value;
      stagnation = 0;
      if (destroy >= 0) destroy_weights[destroy] += 2.0;
      repair_weights[repair] += 2.0;
    } else {
      ++stagnation;
    }
    auto duplicate = std::find_if(
        elite.begin(), elite.end(),
        [&](const Elite& item) { return item.hash == hash; });
    if (duplicate == elite.end()) {
      elite.push_back({std::move(candidate), std::move(*plan), *value, hash});
      if (destroy >= 0) {
        destroy_weights[destroy] += 0.25;
      }
      repair_weights[repair] += 0.25;
    } else if (*value > duplicate->value) {
      *duplicate = {std::move(candidate), std::move(*plan), *value, hash};
    }
    std::sort(elite.begin(), elite.end(),
              [](const Elite& left, const Elite& right) {
                if (left.value != right.value) return left.value > right.value;
                return left.hash < right.hash;
              });
    if (elite.size() > 8) elite.resize(8);
    if ((iteration + 1) % 16 == 0) {
      for (double& weight : destroy_weights) weight = 0.8 * weight + 0.2;
      for (double& weight : repair_weights) weight = 0.8 * weight + 0.2;
    }
    if (iteration + 1 >= minimum && limits.stagnation_iterations > 0 &&
        stagnation >= limits.stagnation_iterations) {
      break;
    }
  }
  return best;
}

struct AlnsSolution {
  LnsSkeleton skeleton;
  AlnsTravelChoices travel;
};

unsigned alns_features_for_policy(const std::string& policy) {
  if (policy == "lns" || policy == "alns") {
    return kProductionAlnsFeatures;
  }
  if (policy == "palns") {
    return kProductionAlnsFeatures | AlnsProjectedObjective;
  }
  throw std::invalid_argument("unknown ALNS policy: " + policy);
}

void repair_alns_travel(AlnsSolution& solution, int mode,
                        std::mt19937_64& random, bool stable = false) {
  solution.travel.resize(solution.skeleton.routes.size());
  for (std::size_t agent = 0; agent < solution.skeleton.routes.size(); ++agent) {
    auto& choices = solution.travel[agent];
    const std::size_t retained = choices.size();
    choices.resize(solution.skeleton.routes[agent].size(), 0U);
    for (std::size_t index = 0; index < choices.size(); ++index) {
      auto& choice = choices[index];
      if (stable && index < retained &&
          (mode == 0 || random() % 4U != 0U)) {
        continue;
      }
      if (mode == 0) {
        choice = 0U;
      } else if (mode == 1) {
        choice = 1U + 4U * static_cast<std::uint32_t>(random() % 2U);
      } else if (mode == 2) {
        choice = 2U + 4U * static_cast<std::uint32_t>(random() % 3U);
      } else {
        choice = static_cast<std::uint32_t>(random() % 96U);
      }
    }
  }
}

std::size_t alns_solution_hash(const AlnsSolution& solution) {
  std::size_t hash = lns_skeleton_hash(solution.skeleton);
  for (std::size_t agent = 0; agent < solution.travel.size(); ++agent) {
    hash ^= agent + 0x9e3779b9U;
    hash *= 0x100000001b3ULL;
    for (auto choice : solution.travel[agent]) {
      // The decoder observes only the strategy, option rank, and forced-refuel
      // bit. Those repeat every 24 codes; hashing the raw value creates four
      // aliases for the same travel decision and falsely rewards novelty.
      hash ^= static_cast<std::size_t>(choice % 24U + 1U);
      hash *= 0x100000001b3ULL;
    }
  }
  return hash;
}

int alns_solution_distance(const AlnsSolution& left,
                           const AlnsSolution& right) {
  int distance = 0;
  const std::size_t agents =
      std::max(left.skeleton.routes.size(), right.skeleton.routes.size());
  for (std::size_t agent = 0; agent < agents; ++agent) {
    const std::vector<int> empty_route;
    const auto& a = agent < left.skeleton.routes.size()
                        ? left.skeleton.routes[agent]
                        : empty_route;
    const auto& b = agent < right.skeleton.routes.size()
                        ? right.skeleton.routes[agent]
                        : empty_route;
    const std::size_t length = std::max(a.size(), b.size());
    for (std::size_t index = 0; index < length; ++index) {
      if (index >= a.size() || index >= b.size() || a[index] != b[index]) {
        ++distance;
      }
    }
  }
  return distance;
}

struct ExactPendingAction {
  bool active{};
  bool move{};
  int remaining{};
  int destination{};
  int fuel_cost{};
};

struct ExactDayState {
  int step{};
  std::vector<int> positions;
  std::vector<int> fuel;
  std::vector<ExactPendingAction> pending;
  std::vector<std::uint64_t> visited;
  std::vector<int> stock;
  std::uint64_t daily_brands{};
  std::uint64_t new_brands{};
  int servings{};
  std::vector<int> traffic;
  ActionPlan actions;
};

ExactDayResult exact_day_search(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& types,
    const ActionPlan& incumbent_plan, const CandidateValue& incumbent_value,
    std::int64_t node_budget,
    const std::optional<std::chrono::steady_clock::time_point>& deadline,
    unsigned features) {
  ExactDayResult result{incumbent_plan, incumbent_value, 0, true};
  if (node_budget <= 0) {
    result.complete = false;
    return result;
  }
  const int horizon = config.day_steps[day.day];
  const std::size_t agent_count = types.size();
  std::map<int, int> brand_index;
  for (const auto& spot : config.spots) {
    if (!brand_index.contains(spot.brand)) {
      brand_index[spot.brand] = static_cast<int>(brand_index.size());
    }
  }
  std::vector<int> spot_brand(config.spots.size());
  std::unordered_map<int, int> spot_for_pos;
  for (std::size_t index = 0; index < config.spots.size(); ++index) {
    spot_brand[index] = brand_index.at(config.spots[index].brand);
    spot_for_pos[config.spots[index].pos] = static_cast<int>(index);
  }
  const int cell_count = config.width * config.height;
  const int infinity = std::numeric_limits<int>::max() / 4;
  const bool needs_spot_steps =
      (features & (AlnsExactReachableBound | AlnsExactStockBound |
                   AlnsExactFuelBound)) != 0U;
  std::vector<std::vector<int>> spot_steps;
  if (needs_spot_steps) {
    spot_steps.assign(config.spots.size(),
                      std::vector<int>(cell_count, infinity));
    for (std::size_t spot = 0; spot < config.spots.size(); ++spot) {
      std::queue<int> queue;
      spot_steps[spot][config.spots[spot].pos] = 0;
      queue.push(config.spots[spot].pos);
      while (!queue.empty()) {
        const int pos = queue.front();
        queue.pop();
        for (int direction = 0; direction < 6; ++direction) {
          auto next = neighbor(config, pos, direction);
          if (!next || config.cells[*next] == Terrain::Pond ||
              spot_steps[spot][*next] != infinity) {
            continue;
          }
          spot_steps[spot][*next] = spot_steps[spot][pos] + 1;
          queue.push(*next);
        }
      }
    }
  }
  int minimum_move_fuel = 0;
  if ((features & AlnsExactFuelBound) != 0U) {
    minimum_move_fuel = std::numeric_limits<int>::max();
    for (int pos = 0; pos < cell_count; ++pos) {
      if (config.cells[pos] != Terrain::Pond) {
        minimum_move_fuel =
            std::min(minimum_move_fuel, terrain_fuel(config, pos));
      }
    }
    minimum_move_fuel = std::max(0, minimum_move_fuel);
  }
  const bool has_refuel =
      std::find(types.begin(), types.end(), AgentKind::Refuel) != types.end();
  std::uint64_t initially_known = 0;
  for (int brand : history.distinct_brands) {
    if (auto iterator = brand_index.find(brand); iterator != brand_index.end()) {
      initially_known |= std::uint64_t{1} << iterator->second;
    }
  }
  const int initial_distinct = static_cast<int>(history.distinct_brands.size());
  const int maximum_fuel =
      static_cast<int>(std::count(types.begin(), types.end(),
                                  AgentKind::Patrol)) *
      config.fuel_limit;
  int absolute_servings = 0;
  for (const auto& spot : config.spots) absolute_servings += spot.stocks;
  const CandidateValue absolute_upper{
      initial_distinct +
          static_cast<int>(brand_index.size() - std::popcount(initially_known)),
      static_cast<int>(brand_index.size()), absolute_servings, maximum_fuel};
  bool stopped = false;
  bool optimum_reached =
      alns_official_value(result.value) == alns_official_value(absolute_upper);
  auto consume_node = [&] {
    if (optimum_reached) return false;
    if (result.explored_nodes >= node_budget ||
        (deadline && std::chrono::steady_clock::now() >= *deadline)) {
      stopped = true;
      return false;
    }
    ++result.explored_nodes;
    return true;
  };
  auto state_value = [&](const ExactDayState& state) {
    int residual = 0;
    for (std::size_t agent = 0; agent < agent_count; ++agent) {
      if (types[agent] == AgentKind::Patrol) residual += state.fuel[agent];
    }
    return CandidateValue{
        initial_distinct + static_cast<int>(std::popcount(state.new_brands)),
        static_cast<int>(std::popcount(state.daily_brands)), state.servings,
        residual};
  };
  auto can_improve = [&](const ExactDayState& state) {
    std::uint64_t possible_daily = state.daily_brands;
    std::uint64_t possible_new = state.new_brands;
    int remaining_servings = 0;
    const int remaining_steps = horizon - state.step;
    for (std::size_t spot = 0; spot < config.spots.size(); ++spot) {
      if (state.stock[spot] <= 0) continue;
      int reachable_patrols = 0;
      if (needs_spot_steps) {
        for (std::size_t agent = 0; agent < agent_count; ++agent) {
          if (types[agent] != AgentKind::Patrol ||
              (state.visited[agent] & (std::uint64_t{1} << spot)) != 0) {
            continue;
          }
          const auto& pending = state.pending[agent];
          const int origin = pending.active ? pending.destination
                                            : state.positions[agent];
          const int available =
              remaining_steps - (pending.active ? pending.remaining : 0);
          const int distance = spot_steps[spot][origin];
          if (distance > available) continue;
          if ((features & AlnsExactFuelBound) != 0U && !has_refuel &&
              distance * minimum_move_fuel > state.fuel[agent]) {
            continue;
          }
          ++reachable_patrols;
        }
      }
      if (((features & AlnsExactReachableBound) != 0U ||
           (features & AlnsExactFuelBound) != 0U) &&
          reachable_patrols == 0) {
        continue;
      }
      int possible_stock = state.stock[spot];
      if ((features & AlnsExactStockBound) != 0U) {
        possible_stock = std::min(possible_stock, reachable_patrols);
      }
      remaining_servings += possible_stock;
      const auto bit = std::uint64_t{1} << spot_brand[spot];
      possible_daily |= bit;
      if ((initially_known & bit) == 0) possible_new |= bit;
    }
    if ((features & AlnsExactServingBound) != 0U) {
      const int patrols = static_cast<int>(
          std::count(types.begin(), types.end(), AgentKind::Patrol));
      remaining_servings =
          std::min(remaining_servings, patrols * remaining_steps);
    }
    int fuel_upper = maximum_fuel;
    if ((features & AlnsExactFuelBound) != 0U && !has_refuel) {
      fuel_upper = 0;
      for (std::size_t agent = 0; agent < agent_count; ++agent) {
        if (types[agent] == AgentKind::Patrol) fuel_upper += state.fuel[agent];
      }
    }
    const CandidateValue upper{
        initial_distinct + static_cast<int>(std::popcount(possible_new)),
        static_cast<int>(std::popcount(possible_daily)),
        state.servings + remaining_servings, fuel_upper};
    return alns_official_value(upper) > alns_official_value(result.value);
  };

  auto first_visit = [&](const ExactDayState&) { return true; };

  ExactDayState initial;
  initial.positions.reserve(agent_count);
  initial.fuel.reserve(agent_count);
  for (const auto& agent : day.agents) {
    initial.positions.push_back(agent.pos);
    initial.fuel.push_back(agent.fuel);
  }
  initial.pending.resize(agent_count);
  initial.visited.assign(agent_count, 0);
  initial.stock.reserve(config.spots.size());
  for (const auto& spot : config.spots) initial.stock.push_back(spot.stocks);
  initial.actions.resize(agent_count);
  initial.traffic.assign(static_cast<std::size_t>(config.width * config.height),
                         0);

  std::function<void(ExactDayState)> search;
  std::function<void(ExactDayState, std::size_t)> assign_actions;
  auto reflect_step = [&](ExactDayState state) {
    for (std::size_t agent = 0; agent < agent_count; ++agent) {
      auto& pending = state.pending[agent];
      --pending.remaining;
      if (pending.remaining != 0) continue;
      if (pending.move) {
        if (types[agent] == AgentKind::Patrol) {
          if (state.fuel[agent] < pending.fuel_cost) return;
          state.fuel[agent] -= pending.fuel_cost;
        }
        state.positions[agent] = pending.destination;
      }
      pending.active = false;
    }
    for (std::size_t agent = 0; agent < agent_count; ++agent) {
      if (types[agent] != AgentKind::Patrol) continue;
      auto iterator = spot_for_pos.find(state.positions[agent]);
      if (iterator == spot_for_pos.end()) continue;
      const int spot = iterator->second;
      const auto visited_bit = std::uint64_t{1} << spot;
      if ((state.visited[agent] & visited_bit) != 0 ||
          state.stock[spot] <= 0) {
        continue;
      }
      state.visited[agent] |= visited_bit;
      --state.stock[spot];
      ++state.servings;
      const auto brand_bit = std::uint64_t{1} << spot_brand[spot];
      state.daily_brands |= brand_bit;
      if ((initially_known & brand_bit) == 0) state.new_brands |= brand_bit;
    }
    std::unordered_set<int> refuel_cells;
    for (std::size_t agent = 0; agent < agent_count; ++agent) {
      if (types[agent] == AgentKind::Refuel) {
        refuel_cells.insert(state.positions[agent]);
      }
    }
    for (std::size_t agent = 0; agent < agent_count; ++agent) {
      if (types[agent] == AgentKind::Patrol &&
          refuel_cells.contains(state.positions[agent])) {
        state.fuel[agent] = config.fuel_limit;
      }
    }
    for (int pos : state.positions) {
      if (config.cells[pos] == Terrain::Road) ++state.traffic[pos];
    }
    ++state.step;
    if (state.step == horizon) {
      const CandidateValue value = state_value(state);
      // The incumbent endpoint is an implementation artifact, not a game
      // constraint. Exact completion may finish anywhere when it strictly
      // improves today's official score. An official tie retains the incumbent
      // instead of exchanging one unscored continuation state for another.
      if (alns_official_value(value) > alns_official_value(result.value)) {
        result.value = value;
        result.plan = state.actions;
        optimum_reached = alns_official_value(value) ==
                          alns_official_value(absolute_upper);
      }
      return;
    }
    search(std::move(state));
  };
  assign_actions = [&](ExactDayState state, std::size_t agent) {
    if (stopped || optimum_reached) return;
    if (agent == agent_count) {
      reflect_step(std::move(state));
      return;
    }
    if (state.pending[agent].active) {
      assign_actions(std::move(state), agent + 1);
      return;
    }
    std::vector<int> choices;
    for (int direction = 0; direction < 6; ++direction) {
      auto destination = neighbor(config, state.positions[agent], direction);
      if (!destination || config.cells[*destination] == Terrain::Pond) continue;
      const int duration =
          terrain_time(config, state.positions[agent], day.traffics);
      if (state.step + duration <= horizon) choices.push_back(direction);
    }
    choices.push_back(-1);
    const std::size_t decision_index = state.actions[agent].size();
    const int preferred =
        decision_index < incumbent_plan[agent].size()
            ? (incumbent_plan[agent][decision_index] < 0
                   ? -1
                   : incumbent_plan[agent][decision_index])
            : -2;
    if (preferred != -2) {
      auto iterator = std::find(choices.begin(), choices.end(), preferred);
      if (iterator != choices.end()) {
        std::rotate(choices.begin(), iterator, iterator + 1);
      }
    }
    for (int choice : choices) {
      if (!consume_node()) return;
      ExactDayState next = state;
      auto& pending = next.pending[agent];
      pending.active = true;
      if (choice < 0) {
        pending.move = false;
        pending.remaining = 1;
        pending.destination = next.positions[agent];
        pending.fuel_cost = 0;
        next.actions[agent].push_back(-1);
      } else {
        pending.move = true;
        pending.remaining =
            terrain_time(config, next.positions[agent], day.traffics);
        pending.destination = *neighbor(config, next.positions[agent], choice);
        pending.fuel_cost = terrain_fuel(config, next.positions[agent]);
        next.actions[agent].push_back(choice);
      }
      assign_actions(std::move(next), agent + 1);
    }
  };
  search = [&](ExactDayState state) {
    if (stopped || optimum_reached || !consume_node() ||
        !first_visit(state) || !can_improve(state)) {
      return;
    }
    assign_actions(std::move(state), 0);
  };
  search(std::move(initial));
  result.complete = !stopped || optimum_reached;
  return result;
}

struct ExactRouteColumn {
  std::uint64_t mask{};
  std::vector<int> route;
  int time{};
};

ExactDayResult exact_route_search(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& types,
    const AcoGraph& graph, const std::vector<AcoMeetingList>& meeting_cache,
    const ActionPlan& incumbent_plan, const CandidateEvaluation& incumbent,
    std::int64_t node_budget,
    const std::optional<std::chrono::steady_clock::time_point>& deadline,
    bool return_master_tie = false) {
  ExactDayResult result{incumbent_plan, incumbent.value, 0, true};
  if (node_budget <= 0 || config.spots.empty() ||
      config.spots.size() > 15) {
    result.complete = node_budget > 0;
    return result;
  }
  const std::size_t spot_count = config.spots.size();
  const std::size_t mask_count = std::size_t{1} << spot_count;
  const int horizon = config.day_steps[day.day];
  const int infinity = std::numeric_limits<int>::max() / 4;
  std::vector<std::size_t> patrols;
  for (std::size_t agent = 0; agent < types.size(); ++agent) {
    if (types[agent] == AgentKind::Patrol) patrols.push_back(agent);
  }
  if (patrols.empty()) return result;

  struct RouteLabel {
    int time{std::numeric_limits<int>::max() / 4};
    int previous_mask{-1};
    int previous_last{-1};
  };
  std::vector<std::vector<ExactRouteColumn>> columns(types.size());
  for (std::size_t agent : patrols) {
    std::vector<RouteLabel> labels(mask_count * spot_count);
    auto label_at = [&](std::size_t mask, std::size_t last) -> RouteLabel& {
      return labels[mask * spot_count + last];
    };
    const int start = graph.node_for_pos.at(day.agents[agent].pos);
    for (std::size_t spot = 0; spot < spot_count; ++spot) {
      const int target = graph.node_for_pos.at(config.spots[spot].pos);
      const int added = start == target ? 1 : lns_path_time(graph, start, target);
      if (added <= horizon) {
        label_at(std::size_t{1} << spot, spot) = {added, -1, -1};
      }
    }
    for (std::size_t mask = 1; mask < mask_count; ++mask) {
      for (std::size_t last = 0; last < spot_count; ++last) {
        const auto label = label_at(mask, last);
        if (label.time > horizon) continue;
        const int from =
            graph.node_for_pos.at(config.spots[last].pos);
        for (std::size_t next = 0; next < spot_count; ++next) {
          const std::size_t bit = std::size_t{1} << next;
          if ((mask & bit) != 0) continue;
          const int target =
              graph.node_for_pos.at(config.spots[next].pos);
          const int added = lns_path_time(graph, from, target);
          const int candidate = label.time + added;
          auto& destination = label_at(mask | bit, next);
          if (candidate <= horizon && candidate < destination.time) {
            destination = {candidate, static_cast<int>(mask),
                           static_cast<int>(last)};
          }
        }
      }
    }
    auto& agent_columns = columns[agent];
    agent_columns.push_back({0, {}, 0});
    for (std::size_t mask = 1; mask < mask_count; ++mask) {
      int best_last = -1;
      int best_time = infinity;
      for (std::size_t last = 0; last < spot_count; ++last) {
        const int time = label_at(mask, last).time;
        if (time < best_time ||
            (time == best_time &&
             (best_last < 0 || static_cast<int>(last) < best_last))) {
          best_time = time;
          best_last = static_cast<int>(last);
        }
      }
      if (best_last < 0 || best_time > horizon) continue;
      std::vector<int> reverse;
      int current_mask = static_cast<int>(mask);
      int current_last = best_last;
      while (current_last >= 0) {
        reverse.push_back(current_last);
        const auto& label = label_at(
            static_cast<std::size_t>(current_mask),
            static_cast<std::size_t>(current_last));
        current_mask = label.previous_mask;
        current_last = label.previous_last;
      }
      agent_columns.push_back(
          {static_cast<std::uint64_t>(mask),
           std::vector<int>(reverse.rbegin(), reverse.rend()), best_time});
    }
  }

  std::vector<std::uint64_t> incumbent_masks(types.size());
  std::unordered_map<int, int> spot_index;
  for (std::size_t spot = 0; spot < spot_count; ++spot) {
    spot_index[config.spots[spot].pos] = static_cast<int>(spot);
  }
  for (const auto& acquisition : incumbent.trace.acquisitions) {
    if (auto iterator = spot_index.find(acquisition.spot_pos);
        iterator != spot_index.end()) {
      incumbent_masks[acquisition.agent] |=
          std::uint64_t{1} << iterator->second;
    }
  }
  for (std::size_t agent : patrols) {
    auto& agent_columns = columns[agent];
    std::sort(agent_columns.begin(), agent_columns.end(),
              [&](const ExactRouteColumn& left,
                  const ExactRouteColumn& right) {
                auto rank = [&](const ExactRouteColumn& column) {
                  int stock_weight = 0;
                  for (std::size_t spot = 0; spot < spot_count; ++spot) {
                    if ((column.mask & (std::uint64_t{1} << spot)) != 0) {
                      stock_weight += config.spots[spot].stocks;
                    }
                  }
                  return std::tuple{
                      stock_weight, std::popcount(column.mask),
                      std::popcount(column.mask & incumbent_masks[agent]),
                      -column.time, -static_cast<std::int64_t>(column.mask)};
                };
                return rank(left) > rank(right);
              });
  }

  std::set<int> known_brands = history.distinct_brands;
  std::set<int> all_brands = known_brands;
  int absolute_servings = 0;
  for (const auto& spot : config.spots) {
    all_brands.insert(spot.brand);
    absolute_servings += spot.stocks;
  }
  const auto structural_upper =
      std::tuple{static_cast<int>(all_brands.size()),
                 static_cast<int>(all_brands.size()), absolute_servings};
  auto route_upper = structural_upper;

  // Solve the fuel-relaxed route master exactly before branching through
  // concrete rendezvous combinations.  Each spot count occupies four bits;
  // official instances expose at most fifteen agents/spots in this path.
  // This produces a valid upper bound because adding fuel and refuel timing can
  // only remove route combinations.  When the ALNS incumbent meets the bound,
  // the final day is certified without exploring any movement-action tree.
  struct MasterParent {
    std::uint64_t previous{};
    std::size_t column{};
  };
  std::vector<std::vector<std::size_t>> maximal_columns(types.size());
  for (std::size_t agent : patrols) {
    std::vector<bool> feasible(mask_count);
    for (const auto& column : columns[agent]) {
      feasible[static_cast<std::size_t>(column.mask)] = true;
    }
    for (std::size_t index = 0; index < columns[agent].size(); ++index) {
      const auto mask =
          static_cast<std::size_t>(columns[agent][index].mask);
      bool dominated = false;
      for (std::size_t spot = 0; spot < spot_count; ++spot) {
        if ((mask & (std::size_t{1} << spot)) == 0 &&
            feasible[mask | (std::size_t{1} << spot)]) {
          dominated = true;
          break;
        }
      }
      if (!dominated) maximal_columns[agent].push_back(index);
    }
  }
  auto add_mask = [&](std::uint64_t state, std::uint64_t mask) {
    for (std::size_t spot = 0; spot < spot_count; ++spot) {
      if ((mask & (std::uint64_t{1} << spot)) == 0) continue;
      const unsigned shift = static_cast<unsigned>(4 * spot);
      const int count = static_cast<int>((state >> shift) & 0xfU);
      if (count < config.spots[spot].stocks) {
        state += std::uint64_t{1} << shift;
      }
    }
    return state;
  };
  auto master_value = [&](std::uint64_t state) {
    std::set<int> daily;
    std::set<int> distinct = known_brands;
    int servings = 0;
    for (std::size_t spot = 0; spot < spot_count; ++spot) {
      const unsigned shift = static_cast<unsigned>(4 * spot);
      const int count = static_cast<int>((state >> shift) & 0xfU);
      servings += count;
      if (count > 0) {
        daily.insert(config.spots[spot].brand);
        distinct.insert(config.spots[spot].brand);
      }
    }
    return std::tuple{static_cast<int>(distinct.size()),
                      static_cast<int>(daily.size()), servings};
  };
  std::vector<std::uint64_t> master_states{0};
  std::vector<std::unordered_map<std::uint64_t, MasterParent>> master_parents;
  const std::int64_t transition_budget =
      std::max<std::int64_t>(100'000, node_budget * 250'000);
  std::int64_t transitions = 0;
  bool master_complete = true;
  for (std::size_t depth = 0; depth < patrols.size(); ++depth) {
    const std::size_t agent = patrols[depth];
    std::unordered_map<std::uint64_t, MasterParent> next;
    next.reserve(master_states.size() * 2);
    for (const std::uint64_t state : master_states) {
      for (std::size_t column : maximal_columns[agent]) {
        if (++transitions > transition_budget ||
            (deadline && (transitions & 4095) == 0 &&
             std::chrono::steady_clock::now() >= *deadline)) {
          master_complete = false;
          break;
        }
        const std::uint64_t candidate =
            add_mask(state, columns[agent][column].mask);
        next.try_emplace(candidate, MasterParent{state, column});
      }
      if (!master_complete) break;
    }
    if (!master_complete || next.empty()) {
      master_complete = false;
      break;
    }
    master_states.clear();
    master_states.reserve(next.size());
    for (const auto& [state, parent] : next) {
      (void)parent;
      master_states.push_back(state);
    }
    master_parents.push_back(std::move(next));
  }
  if (master_complete) {
    auto best_state = master_states.front();
    route_upper = master_value(best_state);
    for (const std::uint64_t state : master_states) {
      const auto value = master_value(state);
      if (value > route_upper) {
        best_state = state;
        route_upper = value;
      }
    }
    if (route_upper <= alns_official_value(result.value) &&
        !return_master_tie) {
      result.complete = true;
      return result;
    }
    std::vector<const ExactRouteColumn*> master_selected(types.size(), nullptr);
    std::uint64_t state = best_state;
    for (std::size_t depth = patrols.size(); depth > 0; --depth) {
      const auto& parent = master_parents[depth - 1].at(state);
      const std::size_t agent = patrols[depth - 1];
      master_selected[agent] = &columns[agent][parent.column];
      state = parent.previous;
    }
    LnsSkeleton master_skeleton;
    master_skeleton.routes.resize(types.size());
    for (std::size_t agent : patrols) {
      master_skeleton.routes[agent] = master_selected[agent]->route;
    }
    auto plan = decode_lns_skeleton(config, day, types, graph, meeting_cache,
                                    master_skeleton);
    if (plan) {
      auto evaluation = evaluate_candidate(config, day, history, *plan);
      if (evaluation &&
          (alns_official_value(evaluation->value) >
               alns_official_value(result.value) ||
           (return_master_tie &&
            alns_official_value(evaluation->value) ==
                alns_official_value(result.value)))) {
        result.plan = std::move(*plan);
        result.value = evaluation->value;
      }
      if (alns_official_value(result.value) == route_upper) {
        result.complete = true;
        return result;
      }
    }
  }
  bool stopped = false;
  bool optimum_reached =
      alns_official_value(result.value) == route_upper;
  std::vector<int> collected(spot_count);
  std::vector<const ExactRouteColumn*> selected(types.size(), nullptr);

  auto upper_bound = [&](std::size_t depth) {
    std::vector<int> possible = collected;
    for (std::size_t index = depth; index < patrols.size(); ++index) {
      const std::size_t agent = patrols[index];
      std::uint64_t reachable = 0;
      for (const auto& column : columns[agent]) reachable |= column.mask;
      for (std::size_t spot = 0; spot < spot_count; ++spot) {
        if ((reachable & (std::uint64_t{1} << spot)) != 0 &&
            possible[spot] < config.spots[spot].stocks) {
          ++possible[spot];
        }
      }
    }
    std::set<int> daily;
    std::set<int> distinct = known_brands;
    int servings = 0;
    for (std::size_t spot = 0; spot < spot_count; ++spot) {
      const int count = std::min(possible[spot], config.spots[spot].stocks);
      servings += count;
      if (count > 0) {
        daily.insert(config.spots[spot].brand);
        distinct.insert(config.spots[spot].brand);
      }
    }
    return std::tuple{static_cast<int>(distinct.size()),
                      static_cast<int>(daily.size()), servings};
  };

  std::function<void(std::size_t)> search;
  search = [&](std::size_t depth) {
    if (stopped || optimum_reached) return;
    if (result.explored_nodes >= node_budget ||
        (deadline && std::chrono::steady_clock::now() >= *deadline)) {
      stopped = true;
      return;
    }
    ++result.explored_nodes;
    if (upper_bound(depth) <= alns_official_value(result.value)) return;
    if (depth == patrols.size()) {
      LnsSkeleton skeleton;
      skeleton.routes.resize(types.size());
      for (std::size_t agent : patrols) {
        if (selected[agent]) skeleton.routes[agent] = selected[agent]->route;
      }
      auto plan = decode_lns_skeleton(config, day, types, graph, meeting_cache,
                                      skeleton);
      if (!plan) return;
      auto evaluation = evaluate_candidate(config, day, history, *plan);
      if (!evaluation ||
          alns_official_value(evaluation->value) <=
              alns_official_value(result.value)) {
        return;
      }
      result.plan = std::move(*plan);
      result.value = evaluation->value;
      optimum_reached = alns_official_value(result.value) == route_upper;
      return;
    }
    const std::size_t agent = patrols[depth];
    std::vector<const ExactRouteColumn*> ordered;
    ordered.reserve(columns[agent].size());
    for (const auto& column : columns[agent]) ordered.push_back(&column);
    std::stable_sort(
        ordered.begin(), ordered.end(), [&](const auto* left, const auto* right) {
          auto marginal = [&](const ExactRouteColumn* column) {
            int gain = 0;
            int stock_weight = 0;
            for (std::size_t spot = 0; spot < spot_count; ++spot) {
              if ((column->mask & (std::uint64_t{1} << spot)) != 0 &&
                  collected[spot] < config.spots[spot].stocks) {
                ++gain;
                stock_weight += config.spots[spot].stocks;
              }
            }
            return std::tuple{gain, stock_weight,
                              std::popcount(column->mask &
                                            incumbent_masks[agent]),
                              -column->time};
          };
          return marginal(left) > marginal(right);
        });
    for (const auto* column : ordered) {
      selected[agent] = column;
      for (std::size_t spot = 0; spot < spot_count; ++spot) {
        if ((column->mask & (std::uint64_t{1} << spot)) != 0) {
          ++collected[spot];
        }
      }
      search(depth + 1);
      for (std::size_t spot = 0; spot < spot_count; ++spot) {
        if ((column->mask & (std::uint64_t{1} << spot)) != 0) {
          --collected[spot];
        }
      }
      if (stopped || optimum_reached) break;
    }
    selected[agent] = nullptr;
  };
  search(0);
  result.complete = !stopped || optimum_reached;
  return result;
}

struct AlnsRolloutState {
  std::vector<int> positions;
  std::vector<int> fuel;
  std::set<int> distinct_brands;
  std::vector<std::map<int, int>> traffic_history;
  int cumulative_daily{};
  int servings{};
  long double discounted_daily{};
  long double discounted_servings{};
};

std::vector<std::map<int, int>> reconstruct_own_traffic(
    const MapConfig& config, const AgentTypes& types,
    const PolicyHistory& history) {
  TeamState team;
  team.id = "alns-rollout-history";
  for (std::size_t agent = 0; agent < types.size(); ++agent) {
    team.agents.push_back(
        {types[agent], config.agents[agent], config.fuel_limit});
  }
  team.visited_today.resize(types.size());
  std::vector<std::map<int, int>> result;
  for (const auto& actions : history.submitted_actions) {
    const auto roads = road_status_for_day(config, result, 1);
    team.stock.clear();
    team.daily_types.clear();
    for (auto& visited : team.visited_today) visited.clear();
    for (const auto& spot : config.spots) team.stock[spot.pos] = spot.stocks;
    std::map<int, int> traffic;
    if (simulate_team_day(config, team, actions, roads, traffic)) break;
    team.history.submitted_actions.push_back(actions);
    result.push_back(std::move(traffic));
  }
  return result;
}

// Self-congestion tie-break shared by the single instance loop and the
// multi-restart selector so both rank continuations by the identical objective.
// The live single-team practice game derives future traffic entirely from our
// own actions; multi-team benchmark matches include external traffic, so
// self-congestion is not a reliable tie-break there and returns zeros.
std::tuple<int, int, int, int, int, int> alns_self_congestion_rank(
    const MapConfig& config, const DayInfo& day,
    const std::vector<std::map<int, int>>& own_traffic_history,
    const std::map<int, int>& road_traffic) {
  if (config.players != 1) return std::tuple{0, 0, 0, 0, 0, 0};
  const bool exact_rolling_day =
      day.day + 2 == static_cast<int>(config.day_steps.size());
  if (!exact_rolling_day) {
    int jammed_pressure = 0;
    int busy_pressure = 0;
    int squared_traffic = 0;
    int total_traffic = 0;
    for (const auto& [pos, traffic] : road_traffic) {
      (void)pos;
      jammed_pressure += std::max(0, traffic - config.jammed_threshold + 1);
      busy_pressure += std::max(0, traffic - config.busy_threshold + 1);
      squared_traffic += traffic * traffic;
      total_traffic += traffic;
    }
    return std::tuple{-jammed_pressure, -busy_pressure, -squared_traffic,
                      -total_traffic, 0, 0};
  }
  const std::map<int, int>* previous =
      own_traffic_history.empty() ? nullptr : &own_traffic_history.back();
  int jammed_roads = 0;
  int busy_roads = 0;
  int jammed_pressure = 0;
  int busy_pressure = 0;
  int squared_traffic = 0;
  int total_traffic = 0;
  for (int pos = 0; pos < config.width * config.height; ++pos) {
    if (config.cells[pos] != Terrain::Road) continue;
    const int current =
        road_traffic.contains(pos) ? road_traffic.at(pos) : 0;
    const int prior =
        previous && previous->contains(pos) ? previous->at(pos) : 0;
    const int traffic = current + prior;
    jammed_roads += traffic >= config.jammed_threshold;
    busy_roads += traffic >= config.busy_threshold;
    jammed_pressure += std::max(0, traffic - config.jammed_threshold + 1);
    busy_pressure += std::max(0, traffic - config.busy_threshold + 1);
    squared_traffic += traffic * traffic;
    total_traffic += traffic;
  }
  return std::tuple{-jammed_roads, -busy_roads, -jammed_pressure,
                    -busy_pressure, -squared_traffic, -total_traffic};
}

struct AlnsMatchScore {
  std::tuple<int, int, int> worst{};
  std::tuple<int, int, int> total{};
  // Future daily/serving gains are sampled with a geometric discount.  The
  // official match score remains in `worst`/`total`; this is only a
  // continuation tie-break that favours useful progress sooner.
  std::tuple<long double, long double> discounted{};
};

[[maybe_unused]] AlnsMatchScore alns_match_rollout(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& types,
    const CandidateEvaluation& root, int beam_width,
    const std::chrono::steady_clock::time_point& deadline,
    unsigned projection_features,
    const std::vector<std::map<int, int>>* precomputed_own_traffic = nullptr) {
  // The projection must use the same ALNS family as the real continuation;
  // cheap greedy routes are useful fallbacks, but alone they routinely rank a
  // refuel staging move that loses when the actual ALNS resumes next day.
  SearchLimits projection_limits;
  projection_limits.time_limit_ms = -1;
  projection_limits.min_iterations = 64;
  projection_limits.max_iterations = 64;
  projection_limits.stagnation_iterations = 0;
  projection_limits.seed_iterations = 0;
  projection_limits.exact_nodes = 0;
  projection_limits.final_alns_iterations = -1;
  projection_limits.final_exact_nodes = -1;
  projection_limits.use_aco_seed = false;
  projection_limits.use_legacy_seed = false;
  projection_limits.use_local_search_seed = false;
  projection_limits.alns_restarts = 1;
  if (deadline != std::chrono::steady_clock::time_point::max()) {
    // The outer timed look-ahead owns a proportional slice of the live budget;
    // each simulated day receives its share rather than a fixed low-time cap.
    projection_limits.min_iterations = 0;
    projection_limits.max_iterations = 10'000'000;
  }
  AlnsRolloutState initial;
  initial.positions = root.ending_positions;
  initial.fuel = root.ending_fuel;
  initial.distinct_brands = history.distinct_brands;
  for (const auto& acquisition : root.trace.acquisitions) {
    if (const Spot* spot = spot_at(config, acquisition.spot_pos)) {
      initial.distinct_brands.insert(spot->brand);
    }
  }
  if (config.players == 1) {
    initial.traffic_history =
        precomputed_own_traffic != nullptr
            ? *precomputed_own_traffic
            : reconstruct_own_traffic(config, types, history);
  }
  initial.traffic_history.push_back(root.road_traffic);
  initial.cumulative_daily = std::get<1>(root.value);
  initial.servings = std::get<2>(root.value);

  auto rollout_scenario = [&](int scenario) {
    std::vector<AlnsRolloutState> beam{initial};
    for (int next_day = day.day + 1;
         next_day < static_cast<int>(config.day_steps.size()); ++next_day) {
      if (std::chrono::steady_clock::now() >= deadline) break;
      std::vector<AlnsRolloutState> expanded;
      for (const auto& state : beam) {
        DayInfo next;
        next.day = next_day;
        for (std::size_t agent = 0; agent < types.size(); ++agent) {
          next.agents.push_back(
              {types[agent], state.positions[agent], state.fuel[agent]});
        }
        if (config.players == 1) {
          next.traffics = road_status_for_day(config, state.traffic_history, 1);
        } else {
          for (int pos = 0; pos < config.width * config.height; ++pos) {
            if (config.cells[pos] != Terrain::Road) continue;
            next.traffics[pos] = scenario == 1
                                     ? (day.traffics.contains(pos)
                                            ? day.traffics.at(pos)
                                            : 0)
                                     : scenario;
          }
        }
        PolicyHistory next_history;
        next_history.distinct_brands = state.distinct_brands;
        next_history.submitted_actions.resize(next_day);
        std::vector<ActionPlan> plans;
        const auto forced =
            coordinated_first_targets(config, next, next_history, types);
        plans.push_back(build_routing_plan("coordinated", config, next,
                                           next_history, types, forced, {}));
        for (const std::string policy : {"utility_greedy", "fuel_aware",
                                         "stock_maximiser"}) {
          plans.push_back(build_routing_plan(policy, config, next, next_history,
                                             types, {}, {}));
        }
        if (deadline != std::chrono::steady_clock::time_point::max()) {
          const int remaining_days =
              static_cast<int>(config.day_steps.size()) - next_day;
          const auto remaining =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  deadline - std::chrono::steady_clock::now())
                  .count();
          if (remaining <= 0) break;
          projection_limits.time_limit_ms = static_cast<int>(
              std::max<std::int64_t>(1, remaining / remaining_days));
        }
        plans.push_back(build_alns_plan(
            config, next, next_history, types, projection_limits,
            projection_features, false, static_cast<std::uint64_t>(scenario),
            nullptr));
        for (const auto& plan : plans) {
          if (std::chrono::steady_clock::now() >= deadline) break;
          auto evaluation =
              evaluate_candidate(config, next, next_history, plan);
          if (!evaluation) continue;
          AlnsRolloutState child = state;
          child.positions = evaluation->ending_positions;
          child.fuel = evaluation->ending_fuel;
          const long double discount = std::pow(
              0.85L, static_cast<long double>(next_day - day.day - 1));
          child.discounted_daily +=
              static_cast<long double>(std::get<1>(evaluation->value)) *
              discount;
          child.discounted_servings +=
              static_cast<long double>(std::get<2>(evaluation->value)) *
              discount;
          child.cumulative_daily += std::get<1>(evaluation->value);
          child.servings += std::get<2>(evaluation->value);
          for (const auto& acquisition : evaluation->trace.acquisitions) {
            if (const Spot* spot = spot_at(config, acquisition.spot_pos)) {
              child.distinct_brands.insert(spot->brand);
            }
          }
          child.traffic_history.push_back(evaluation->road_traffic);
          if (child.traffic_history.size() > 2) {
            child.traffic_history.erase(child.traffic_history.begin());
          }
          expanded.push_back(std::move(child));
        }
      }
      if (expanded.empty()) break;
      std::sort(expanded.begin(), expanded.end(), [](const auto& left,
                                                     const auto& right) {
        int left_fuel = std::accumulate(left.fuel.begin(), left.fuel.end(), 0);
        int right_fuel =
            std::accumulate(right.fuel.begin(), right.fuel.end(), 0);
        return std::tuple{left.distinct_brands.size(), left.discounted_daily,
                          left.discounted_servings, left.cumulative_daily,
                          left.servings, left_fuel} >
               std::tuple{right.distinct_brands.size(),
                          right.discounted_daily, right.discounted_servings,
                          right.cumulative_daily, right.servings, right_fuel};
      });
      std::set<std::tuple<std::vector<int>, std::vector<int>, std::set<int>>>
          signatures;
      beam.clear();
      for (auto& child : expanded) {
        auto signature =
            std::tuple{child.positions, child.fuel, child.distinct_brands};
        if (!signatures.insert(std::move(signature)).second) continue;
        beam.push_back(std::move(child));
        if (static_cast<int>(beam.size()) >= beam_width) break;
      }
    }
    const auto& best = *std::max_element(
        beam.begin(), beam.end(), [](const auto& left, const auto& right) {
          return std::tuple{left.distinct_brands.size(), left.discounted_daily,
                            left.discounted_servings, left.cumulative_daily,
                            left.servings} <
                 std::tuple{right.distinct_brands.size(),
                            right.discounted_daily, right.discounted_servings,
                            right.cumulative_daily, right.servings};
        });
    return std::tuple{static_cast<int>(best.distinct_brands.size()),
                      best.cumulative_daily, best.servings,
                      best.discounted_daily, best.discounted_servings};
  };

  using RolloutScore = std::tuple<int, int, int, long double, long double>;
  std::vector<RolloutScore> scores;
  if (config.players == 1) {
    scores.push_back(rollout_scenario(0));
  } else {
    for (int scenario = 0; scenario < 3; ++scenario) {
      scores.push_back(rollout_scenario(scenario));
    }
  }
  auto worst = *std::min_element(
      scores.begin(), scores.end(), [](const RolloutScore& left,
                                       const RolloutScore& right) {
        return std::tuple{std::get<0>(left), std::get<1>(left),
                          std::get<2>(left)} <
               std::tuple{std::get<0>(right), std::get<1>(right),
                          std::get<2>(right)};
      });
  std::tuple<int, int, int> total{};
  std::tuple<long double, long double> discounted{};
  for (const auto& score : scores) {
    std::get<0>(total) += std::get<0>(score);
    std::get<1>(total) += std::get<1>(score);
    std::get<2>(total) += std::get<2>(score);
    std::get<0>(discounted) += std::get<3>(score);
    std::get<1>(discounted) += std::get<4>(score);
  }
  return {{std::get<0>(worst), std::get<1>(worst), std::get<2>(worst)},
          total, discounted};
}

ActionPlan build_aco_plan(const MapConfig& config, const DayInfo& day,
                          const PolicyHistory& history,
                          const AgentTypes& types, bool apply_local_search,
                          const SearchLimits& limits);

// Refuel-escort seed. Pair the refuel car with one patrol: the car drives to the
// patrol, then both follow an identical route so the patrol is refuelled to full
// every step and can reach distant new-brand spots that its tight initial fuel
// would otherwise strand. Other agents idle; ALNS refines from here. Returns an
// empty plan when there is no refuel car (the caller then skips this seed).
//
// This closes a real gap: on hard fuel-constrained maps the ordinary seeds strand
// patrols after ~one tank and collect only the 2-3 nearest brands, while an
// escorted patrol can sweep the map. Because refuel cars do not burn fuel, an
// escorted patrol has effectively unlimited range.
ActionPlan build_escort_plan(const MapConfig& config, const DayInfo& day,
                             const PolicyHistory& history,
                             const AgentTypes& types) {
  const int horizon = config.day_steps[day.day];
  const std::size_t n = day.agents.size();
  ActionPlan result(n, std::vector<int>{-horizon});
  int refuel = -1;
  for (std::size_t i = 0; i < n; ++i) {
    if (types[i] == AgentKind::Refuel) {
      refuel = static_cast<int>(i);
      break;
    }
  }
  if (refuel < 0) return {};

  // Escort the patrol closest to an as-yet-unreached brand; otherwise any patrol.
  const std::set<int>& reached = history.distinct_brands;
  int patrol = -1;
  int best_reach = std::numeric_limits<int>::max();
  for (std::size_t i = 0; i < n; ++i) {
    if (types[i] != AgentKind::Patrol) continue;
    for (const auto& spot : config.spots) {
      if (reached.contains(spot.brand)) continue;
      const int cost =
          shortest_path(config, day.agents[i].pos, spot.pos, day.traffics).cost;
      if (cost < best_reach) {
        best_reach = cost;
        patrol = static_cast<int>(i);
      }
    }
  }
  if (patrol < 0) {
    for (std::size_t i = 0; i < n; ++i) {
      if (types[i] == AgentKind::Patrol) {
        patrol = static_cast<int>(i);
        break;
      }
    }
  }
  if (patrol < 0) return {};

  auto leg_time = [&](int from, const std::vector<int>& directions) {
    int time = 0;
    int cursor = from;
    for (int direction : directions) {
      time += terrain_time(config, cursor, day.traffics);
      cursor = *neighbor(config, cursor, direction);
    }
    return time;
  };

  // Rendezvous: the refuel car drives to the patrol while the patrol waits.
  const auto rendezvous =
      shortest_path(config, day.agents[refuel].pos, day.agents[patrol].pos,
                    day.traffics);
  const int rendezvous_time =
      leg_time(day.agents[refuel].pos, rendezvous.directions);
  if (rendezvous_time > horizon) return {};

  // Greedily chain the nearest still-fitting spot, preferring unreached brands.
  const int route_budget = horizon - rendezvous_time;
  std::vector<int> route;
  int cursor = day.agents[patrol].pos;
  int route_time = 0;
  std::set<int> visited;
  std::set<int> chased = reached;
  while (true) {
    const Spot* choice = nullptr;
    std::vector<int> choice_dirs;
    std::pair<int, int> choice_key{2, std::numeric_limits<int>::max()};
    for (const auto& spot : config.spots) {
      if (visited.contains(spot.pos)) continue;
      auto path = shortest_path(config, cursor, spot.pos, day.traffics);
      const int time = leg_time(cursor, path.directions);
      if (route_time + time > route_budget) continue;
      const std::pair<int, int> key{chased.contains(spot.brand) ? 1 : 0, time};
      if (key < choice_key) {
        choice = &spot;
        choice_dirs = std::move(path.directions);
        choice_key = key;
      }
    }
    if (choice == nullptr) break;
    route.insert(route.end(), choice_dirs.begin(), choice_dirs.end());
    route_time += choice_key.second;
    cursor = choice->pos;
    visited.insert(choice->pos);
    chased.insert(choice->brand);
  }

  std::vector<int> patrol_actions;
  if (rendezvous_time > 0) patrol_actions.push_back(-rendezvous_time);
  patrol_actions.insert(patrol_actions.end(), route.begin(), route.end());
  std::vector<int> refuel_actions = rendezvous.directions;
  refuel_actions.insert(refuel_actions.end(), route.begin(), route.end());
  const int remaining = horizon - rendezvous_time - route_time;
  if (remaining > 0) {
    patrol_actions.push_back(-remaining);
    refuel_actions.push_back(-remaining);
  }
  if (patrol_actions.empty()) patrol_actions.push_back(-horizon);
  if (refuel_actions.empty()) refuel_actions.push_back(-horizon);
  result[patrol] = std::move(patrol_actions);
  result[refuel] = std::move(refuel_actions);
  return result;
}

struct PalnsProjection {
  std::tuple<int, int, int> match;
  int ending_fuel{};
};

PalnsReturnedRank palns_projection_rank(const PalnsProjection& projection) {
  return {projection.match, projection.ending_fuel};
}

using PalnsStateSignature =
    std::tuple<std::vector<int>, std::vector<int>, std::map<int, int>,
               std::set<int>, std::tuple<int, int, int>>;

struct PalnsRuntime {
  int remaining_iterations{};
  int projection_iterations_per_day{8};
  std::map<PalnsStateSignature, PalnsProjection> cache;
};

std::set<int> palns_resulting_brands(const MapConfig& config,
                                     const PolicyHistory& history,
                                     const CandidateEvaluation& evaluation) {
  std::set<int> brands = history.distinct_brands;
  for (const auto& acquisition : evaluation.trace.acquisitions) {
    if (const Spot* spot = spot_at(config, acquisition.spot_pos)) {
      brands.insert(spot->brand);
    }
  }
  return brands;
}

std::optional<PalnsProjection> palns_project_final_match(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& types,
    const ActionPlan& current_plan, const CandidateEvaluation& current_eval,
    const SearchLimits& limits, unsigned features, std::uint64_t restart_salt,
    const std::vector<std::map<int, int>>& prior_traffic,
    const std::chrono::steady_clock::time_point& deadline, PalnsRuntime& runtime) {
  const auto official = alns_official_value(current_eval.value);
  auto resulting_brands = palns_resulting_brands(config, history, current_eval);
  PalnsStateSignature signature{current_eval.ending_positions,
                                current_eval.ending_fuel,
                                current_eval.road_traffic, resulting_brands,
                                official};
  if (auto cached = runtime.cache.find(signature); cached != runtime.cache.end()) {
    ++palns_diagnostics.projection_cache_hits;
    return cached->second;
  }

  const int remaining_days =
      static_cast<int>(config.day_steps.size()) - day.day - 1;
  if (remaining_days <= 0) {
    PalnsProjection projection{
        {std::get<0>(official),
         history.cumulative_daily_types + std::get<1>(official),
         history.total_servings + std::get<2>(official)},
        std::get<3>(current_eval.value)};
    runtime.cache.emplace(std::move(signature), projection);
    return projection;
  }

  ++palns_diagnostics.projection_requests;
  const int required =
      remaining_days * runtime.projection_iterations_per_day;
  if (runtime.remaining_iterations < required) {
    ++palns_diagnostics.projection_iteration_fallbacks;
    return std::nullopt;
  }
  if (limits.time_limit_ms >= 0 &&
      std::chrono::steady_clock::now() >= deadline) {
    ++palns_diagnostics.projection_deadline_fallbacks;
    return std::nullopt;
  }

  auto traffic_history = prior_traffic;
  traffic_history.push_back(current_eval.road_traffic);
  PolicyHistory next_history = history;
  next_history.submitted_actions.push_back(current_plan);
  next_history.distinct_brands = resulting_brands;
  next_history.cumulative_daily_types += std::get<1>(official);
  next_history.total_servings += std::get<2>(official);
  std::vector<int> positions = current_eval.ending_positions;
  std::vector<int> fuel = current_eval.ending_fuel;
  int distinct = std::get<0>(official);
  int cumulative_daily = next_history.cumulative_daily_types;
  int servings = next_history.total_servings;

  for (int next_day = day.day + 1;
       next_day < static_cast<int>(config.day_steps.size()); ++next_day) {
    if (limits.time_limit_ms >= 0 &&
        std::chrono::steady_clock::now() >= deadline) {
      ++palns_diagnostics.projection_deadline_fallbacks;
      return std::nullopt;
    }
    const int depth = runtime.projection_iterations_per_day;
    runtime.remaining_iterations -= depth;
    palns_diagnostics.projection_iterations += depth;
    palns_diagnostics.iterations_used += depth;

    DayInfo next;
    next.day = next_day;
    for (std::size_t agent = 0; agent < types.size(); ++agent) {
      next.agents.push_back({types[agent], positions[agent], fuel[agent]});
    }
    next.traffics = road_status_for_day(config, traffic_history, 1);

    SearchLimits projection_limits = limits;
    projection_limits.time_limit_ms = -1;
    projection_limits.min_iterations = depth;
    projection_limits.max_iterations = depth;
    projection_limits.stagnation_iterations = 0;
    projection_limits.seed_iterations = std::min(limits.seed_iterations, depth);
    projection_limits.final_alns_iterations = -1;
    projection_limits.exact_nodes = 0;
    projection_limits.final_exact_nodes = -1;
    projection_limits.alns_restarts = 1;
    projection_limits.total_iterations = -1;
    if (limits.time_limit_ms >= 0) {
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 deadline - std::chrono::steady_clock::now())
                                 .count();
      if (remaining <= 0) {
        ++palns_diagnostics.projection_deadline_fallbacks;
        return std::nullopt;
      }
      projection_limits.time_limit_ms = static_cast<int>(remaining);
      projection_limits.min_iterations = 0;
    }
    auto projected_plan = build_alns_plan(
        config, next, next_history, types, projection_limits,
        features & ~AlnsProjectedObjective, false,
        restart_salt ^ static_cast<std::uint64_t>(next_day + 1), nullptr);
    auto evaluation =
        evaluate_candidate(config, next, next_history, projected_plan);
    if (!evaluation) {
      ++palns_diagnostics.projection_deadline_fallbacks;
      return std::nullopt;
    }
    const auto projected_official = alns_official_value(evaluation->value);
    distinct = std::get<0>(projected_official);
    cumulative_daily += std::get<1>(projected_official);
    servings += std::get<2>(projected_official);
    positions = evaluation->ending_positions;
    fuel = evaluation->ending_fuel;
    traffic_history.push_back(evaluation->road_traffic);
    next_history.submitted_actions.push_back(projected_plan);
    next_history.distinct_brands =
        palns_resulting_brands(config, next_history, *evaluation);
    next_history.cumulative_daily_types = cumulative_daily;
    next_history.total_servings = servings;
  }

  int ending_fuel = 0;
  for (std::size_t agent = 0; agent < types.size(); ++agent) {
    if (types[agent] == AgentKind::Patrol) ending_fuel += fuel[agent];
  }
  PalnsProjection projection{{distinct, cumulative_daily, servings}, ending_fuel};
  runtime.cache.emplace(std::move(signature), projection);
  ++palns_diagnostics.projection_completed;
  return projection;
}

ActionPlan build_alns_plan(const MapConfig& config, const DayInfo& day,
                           const PolicyHistory& history,
                           const AgentTypes& types,
                            const SearchLimits& limits,
                            unsigned features,
                            bool allow_continuation,
                            std::uint64_t restart_salt,
                            const ImprovementSink* on_improve,
                            std::vector<ActionPlan>* elite_plans) {
  const auto started = std::chrono::steady_clock::now();
  const bool timed = limits.time_limit_ms >= 0;
  const auto deadline =
      started + std::chrono::milliseconds(std::max(0, limits.time_limit_ms));
  const bool final_day =
      day.day + 1 >= static_cast<int>(config.day_steps.size());
  const bool palns_enabled = (features & AlnsProjectedObjective) != 0U;
  const int configured_exact_nodes =
      final_day && limits.final_exact_nodes >= 0
          ? limits.final_exact_nodes
          : limits.exact_nodes;
  const bool exact_enabled =
      (features & AlnsExactCompletion) != 0U && configured_exact_nodes > 0;
  const auto anytime_budget = alns_anytime_budget(
      limits.time_limit_ms, final_day,
      allow_continuation && !palns_enabled, exact_enabled,
      limits.continuation_time_percent, limits.exact_time_percent);
  const bool timed_exact = timed && anytime_budget.exact_ms > 0;
  // On non-final days the post-loop remaining-match projection (Tier 2
  // continuation look-ahead) needs its own slice of the budget; otherwise the
  // main loop consumes the whole deadline and the projection is skipped, so the
  // timed path degrades to a myopic search that gets *worse* with more time.
  // The ALNS loop owns only this slice of the wall budget (a shorter slice when
  // a final-day exact-completion phase or a continuation projection must run
  // afterwards). The cooling schedule must anneal across the slice it actually
  // runs in, not the whole request, or temperature never reaches
  // final_temperature.
  const int alns_budget_ms = anytime_budget.main_ms;
  const auto alns_deadline =
      started + std::chrono::milliseconds(alns_budget_ms);
  auto expired = [&] {
    return timed && std::chrono::steady_clock::now() >= alns_deadline;
  };
  std::optional<PalnsRuntime> palns_runtime;
  if (palns_enabled) {
    const int total = limits.total_iterations >= 0
                          ? limits.total_iterations
                          : std::max(1, limits.max_iterations);
    palns_runtime = PalnsRuntime{total, limits.palns_projection_iterations, {}};
    palns_diagnostics.total_iterations += total;
  }

  const auto forced = coordinated_first_targets(config, day, history, types);
  std::vector<ActionPlan> seeds;
  std::vector<LnsDpProposal> dp_proposals;
  seeds.push_back(build_routing_plan("coordinated", config, day, history,
                                     types, forced, limits));
  for (const std::string policy : {"greedy", "utility_greedy", "fuel_aware",
                                   "stock_maximiser"}) {
    if (expired()) break;
    seeds.push_back(
        build_routing_plan(policy, config, day, history, types, {}, limits));
  }
  if (!expired() && limits.use_aco_seed &&
      (features & AlnsAcoSeed) != 0U) {
    SearchLimits aco_limits = limits;
    if (timed) {
      aco_limits.time_limit_ms = std::max(1, alns_budget_ms / 10);
    }
    seeds.insert(
        seeds.begin(),
        build_aco_plan(config, day, history, types, true, aco_limits));
  }
  if (!expired() && (features & AlnsSharedPreprocessing) == 0U) {
    SearchLimits legacy_limits = limits;
    if (timed) {
      legacy_limits.time_limit_ms = std::max(1, alns_budget_ms / 10);
      legacy_limits.min_iterations =
          std::min(32, std::max(1, limits.min_iterations));
      legacy_limits.max_iterations =
          std::min(96, std::max(legacy_limits.min_iterations,
                                limits.max_iterations));
      legacy_limits.stagnation_iterations = legacy_limits.max_iterations;
    }
    seeds.insert(seeds.begin(),
                 build_lns_plan(config, day, history, types, legacy_limits));
  }
  if (!expired() && limits.use_local_search_seed) {
    seeds.push_back(build_local_search_plan(config, day, history, types, limits));
  }
  // LNS-DP contributes an additive request-bank proposal.  Its plan is
  // already simulator-valid, so it can seed the official ALNS/PALNS ranking;
  // the ALNS rendezvous decoder remains authoritative for every neighborhood
  // candidate and for any later skeleton conversion.
  if (lns_dp_proposals_enabled(limits) && !expired() &&
      (!timed || alns_budget_ms >= 250)) {
    SearchLimits dp_limits = limits;
    if (timed) {
      const int remaining_ms = std::max(
          1, static_cast<int>(std::chrono::duration_cast<
                                  std::chrono::milliseconds>(
                                  alns_deadline -
                                  std::chrono::steady_clock::now())
                                  .count()));
      // DP is an additive seed, not the main search. Give it a scale-free
      // tenth of the current-day phase, capped by the actual time remaining.
      dp_limits.time_limit_ms =
          std::min(remaining_ms, std::max(1, alns_budget_ms / 10));
    }
    dp_limits.min_iterations = 0;
    dp_limits.max_iterations = std::min(8, std::max(1, limits.max_iterations));
    dp_limits.stagnation_iterations = 0;
    dp_limits.random_seed ^= restart_salt ^ 0x445053454544ULL;
    const int proposal_count = timed ? 1 : 2;
    dp_proposals = build_lns_dp_route_proposals(
        config, day, history, types, dp_limits, proposal_count);
    for (const auto& proposal : dp_proposals) {
      if (expired())
        break;
      seeds.push_back(proposal.plan);
    }
  }

  // Seed evaluation is one of the most expensive parts of a day. Keep the
  // result alongside each seed so incumbent selection and elite construction
  // do not simulate the same plan twice.
  std::vector<std::optional<CandidateEvaluation>> seed_evaluations;
  seed_evaluations.reserve(seeds.size());
  for (const auto& seed : seeds) {
    seed_evaluations.push_back(evaluate_candidate(config, day, history, seed));
  }
  // Single-team refuel-escort seed. It joins the pool only when it strictly
  // beats every ordinary seed's official score, so it lifts tight-fuel maps
  // (where patrols would otherwise strand) without perturbing the search on
  // maps where fuel is ample and the escort is dominated.
  if (!expired() && config.players == 1) {
    if (auto escort = build_escort_plan(config, day, history, types);
        !escort.empty()) {
      if (auto escort_eval = evaluate_candidate(config, day, history, escort)) {
        std::tuple<int, int, int> best_seed{std::numeric_limits<int>::min(),
                                            std::numeric_limits<int>::min(),
                                            std::numeric_limits<int>::min()};
        for (const auto& evaluation : seed_evaluations) {
          if (evaluation) {
            best_seed = std::max(best_seed,
                                 alns_official_value(evaluation->value));
          }
        }
        if (alns_official_value(escort_eval->value) > best_seed) {
          seeds.push_back(std::move(escort));
          seed_evaluations.push_back(std::move(escort_eval));
        }
      }
    }
  }
  ActionPlan best = seeds.front();
  CandidateEvaluation best_evaluation = *seed_evaluations.front();
  CandidateValue best_value = best_evaluation.value;
  // Our own past road usage. In single-team play this is the exact future
  // traffic source; in multi-team it is the symmetric-opponent surrogate the
  // continuation projection uses to predict future road status (see the
  // road_status_for_day call in the projection). Computed regardless of player
  // count so the look-ahead can run in the real (multi-team) competition too.
  const auto own_traffic_history =
      reconstruct_own_traffic(config, types, history);
  auto congestion_value = [&](const CandidateEvaluation& evaluation) {
    return alns_self_congestion_rank(config, day, own_traffic_history,
                                     evaluation.road_traffic);
  };
  auto online_improves = [&](const CandidateEvaluation& candidate,
                             const CandidateEvaluation& incumbent) {
    const auto candidate_rank =
        std::tuple{alns_official_value(candidate.value),
                   congestion_value(candidate), candidate.workload,
                   std::get<3>(candidate.value)};
    const auto incumbent_rank =
        std::tuple{alns_official_value(incumbent.value),
                   congestion_value(incumbent), incumbent.workload,
                   std::get<3>(incumbent.value)};
    return candidate_rank > incumbent_rank;
  };
  for (std::size_t index = 0; index < seeds.size(); ++index) {
    if (const auto& evaluation = seed_evaluations[index];
        evaluation && online_improves(*evaluation, best_evaluation)) {
      best = seeds[index];
      best_value = evaluation->value;
      best_evaluation = *evaluation;
    }
  }
  std::optional<PalnsProjection> best_projection;
  auto project_candidate = [&](const ActionPlan& plan,
                               const CandidateEvaluation& evaluation) {
    if (!palns_runtime) return std::optional<PalnsProjection>{};
    return palns_project_final_match(
        config, day, history, types, plan, evaluation, limits, features,
        restart_salt, own_traffic_history, alns_deadline, *palns_runtime);
  };
  if (palns_runtime) {
    const auto best_official = alns_official_value(best_evaluation.value);
    for (std::size_t index = 0; index < seeds.size(); ++index) {
      const auto& evaluation = seed_evaluations[index];
      if (!evaluation ||
          alns_official_value(evaluation->value) != best_official) {
        continue;
      }
      auto projection = project_candidate(seeds[index], *evaluation);
      const bool improves_projection =
          projection &&
          (!best_projection ||
           palns_projection_rank(*projection) >
               palns_projection_rank(*best_projection) ||
           (palns_projection_rank(*projection) ==
                palns_projection_rank(*best_projection) &&
            online_improves(*evaluation, best_evaluation)));
      if (improves_projection ||
          (!projection && !best_projection &&
           online_improves(*evaluation, best_evaluation))) {
        best = seeds[index];
        best_value = evaluation->value;
        best_evaluation = *evaluation;
        best_projection = projection;
      } else if (projection && !best_projection) {
        best_projection = projection;
      }
    }
  }
  if (expired()) return best;

  AcoGraph graph = build_aco_graph(config, day);
  if (expired()) return best;
  std::optional<std::vector<AcoMeetingList>> shared_meeting_cache;
  if (limits.use_legacy_seed && limits.seed_iterations > 0 &&
      (features & AlnsSharedPreprocessing) != 0U) {
    shared_meeting_cache = build_aco_meeting_cache(graph);
    SearchLimits legacy_limits = limits;
    // This is a diversification seed, not a second search budget.  Letting it
    // inherit an untimed ALNS maximum made a 6,000-iteration request run 6,000
    // legacy-LNS iterations before ALNS and changed the initial incumbent as a
    // function of the requested depth.  Keep seed construction bounded and
    // identical for every sufficiently large fixed budget.
    legacy_limits.min_iterations = limits.seed_iterations;
    legacy_limits.max_iterations = legacy_limits.min_iterations;
    legacy_limits.stagnation_iterations =
        limits.seed_iterations == 0
            ? 0
            : (limits.stagnation_iterations == 0
                   ? 0
                   : legacy_limits.max_iterations);
    if (timed) {
      legacy_limits.time_limit_ms = std::max(1, alns_budget_ms / 10);
    }
    ActionPlan legacy = build_lns_plan(config, day, history, types,
                                       legacy_limits, &graph,
                                       &*shared_meeting_cache);
    seeds.insert(seeds.begin(), legacy);
    auto evaluation = evaluate_candidate(config, day, history, legacy);
    seed_evaluations.insert(seed_evaluations.begin(), evaluation);
    if (evaluation) {
      if (online_improves(*evaluation, best_evaluation)) {
        best = std::move(legacy);
        best_value = evaluation->value;
        best_evaluation = std::move(*evaluation);
      }
    }
  }
  if (expired()) return best;
  std::vector<int> transit_nodes;
  if (config.spots.size() <= 15) {
    transit_nodes = alns_transit_nodes(config, graph, std::size_t{8}, false);
  }
  if (!transit_nodes.empty() && !expired()) {
    graph = build_aco_graph(config, day, transit_nodes);
    shared_meeting_cache.reset();
  }
  if (expired()) return best;
  auto meeting_cache = shared_meeting_cache
                           ? std::move(*shared_meeting_cache)
                           : build_aco_meeting_cache(graph, std::size_t{6});
  if (expired()) return best;

  // Re-decode the same DP visit proposals through ALNS's richer travel and
  // rendezvous decoder as an alternate seed.  The direct DP plan remains
  // available for PALNS projection, while this conversion tests whether the
  // existing decoder can realize the visit order better.
  for (const auto& proposal : dp_proposals) {
    if (expired())
      break;
    if (auto decoded = decode_lns_skeleton(config, day, types, graph,
                                           meeting_cache, proposal.skeleton,
                                           nullptr, false)) {
      seeds.push_back(*decoded);
      auto evaluation = evaluate_candidate(config, day, history, *decoded);
      seed_evaluations.push_back(evaluation);
      if (evaluation && online_improves(*evaluation, best_evaluation)) {
        best = *decoded;
        best_value = evaluation->value;
        best_evaluation = *evaluation;
      }
    }
  }

  struct Elite {
    AlnsSolution solution;
    ActionPlan plan;
    CandidateValue value;
    CandidateEvaluation evaluation;
    std::optional<PalnsProjection> projection;
    std::size_t hash{};
  };
  std::vector<Elite> elite;
  std::mt19937_64 zero_random(0);
  if (seed_evaluations.size() < seeds.size()) {
    seed_evaluations.resize(seeds.size());
  }
  for (std::size_t index = 0; index < seeds.size(); ++index) {
    const auto& seed = seeds[index];
    if (!seed_evaluations[index]) {
      seed_evaluations[index] =
          evaluate_candidate(config, day, history, seed);
    }
    const auto& evaluation = seed_evaluations[index];
    if (!evaluation) continue;
    // Reuse the seed's evaluation trace instead of simulating it again.
    LnsSkeleton skeleton =
        lns_skeleton_from_trace(config, day.agents.size(), evaluation->trace);
    AlnsSolution solution{std::move(skeleton), {}};
    repair_alns_travel(solution, 0, zero_random);
    const auto hash = alns_solution_hash(solution);
    if (std::none_of(elite.begin(), elite.end(),
                     [&](const Elite& item) { return item.hash == hash; })) {
      std::optional<PalnsProjection> projection;
      if (palns_runtime && alns_official_value(evaluation->value) ==
                               alns_official_value(best_value)) {
        projection = project_candidate(seed, *evaluation);
      }
      elite.push_back({std::move(solution), seed, evaluation->value,
                       *evaluation, projection, hash});
    }
  }
  if (elite.empty()) return best;
  auto elite_order = [&](const Elite& left, const Elite& right) {
    const bool same_official =
        alns_official_value(left.value) == alns_official_value(right.value);
    if (palns_runtime && same_official &&
        static_cast<bool>(left.projection) !=
            static_cast<bool>(right.projection)) {
      return left.projection.has_value();
    }
    if (palns_runtime && same_official && left.projection && right.projection &&
        palns_projection_rank(*left.projection) !=
            palns_projection_rank(*right.projection)) {
      return palns_projection_rank(*left.projection) >
             palns_projection_rank(*right.projection);
    }
    const auto left_rank =
        std::tuple{alns_official_value(left.value),
                   congestion_value(left.evaluation), left.evaluation.workload,
                   std::get<3>(left.value)};
    const auto right_rank =
        std::tuple{alns_official_value(right.value),
                   congestion_value(right.evaluation),
                   right.evaluation.workload, std::get<3>(right.value)};
    if (left_rank != right_rank) return left_rank > right_rank;
    return left.hash < right.hash;
  };
  const std::size_t elite_limit = 12U;
  auto trim_elite = [&] {
    std::sort(elite.begin(), elite.end(), elite_order);
    if (elite.size() <= elite_limit) return;
    if (elite_plans == nullptr) {
      elite.resize(elite_limit);
      return;
    }

    // A whole-match caller needs alternatives that finish today in different
    // places.  Pure daily ranking otherwise fills the bounded pool with nearly
    // identical routes, so a useful refuel/rendezvous boundary disappears
    // before MLNS can evaluate tomorrow. Keep the strongest four unchanged,
    // then reserve the remaining slots for position-distinct routes tied on
    // today's official score. Ordinary ALNS callers retain the original pool.
    std::vector<bool> retained(elite.size(), false);
    std::vector<Elite> trimmed;
    trimmed.reserve(elite_limit);
    const std::size_t protected_count = std::min<std::size_t>(4U, elite.size());
    std::set<std::vector<int>> ending_positions;
    for (std::size_t index = 0; index < protected_count; ++index) {
      ending_positions.insert(elite[index].evaluation.ending_positions);
      retained[index] = true;
      trimmed.push_back(std::move(elite[index]));
    }
    const auto best_official = alns_official_value(trimmed.front().value);
    for (std::size_t index = protected_count;
         index < elite.size() && trimmed.size() < elite_limit; ++index) {
      if (alns_official_value(elite[index].value) == best_official &&
          ending_positions.insert(elite[index].evaluation.ending_positions)
              .second) {
        retained[index] = true;
        trimmed.push_back(std::move(elite[index]));
      }
    }
    for (std::size_t index = protected_count;
         index < elite.size() && trimmed.size() < elite_limit; ++index) {
      if (!retained[index]) trimmed.push_back(std::move(elite[index]));
    }
    elite = std::move(trimmed);
    std::sort(elite.begin(), elite.end(), elite_order);
  };
  trim_elite();
  std::mt19937_64 random(lns_seed(config, day, history) ^ 0x414c4e53ULL ^
                         restart_salt ^ limits.random_seed);
  // Re-initialise seed travel choices with the real deterministic generator.
  for (auto& item : elite) repair_alns_travel(item.solution, 0, random);
  Elite current = elite.front();
  // MCTS operator policy.  The tree keeps separate statistics for destroy,
  // repair, and travel choices, so it can learn useful combinations instead
  // of treating each operator as an independent roulette-wheel arm.
  struct MctsNode {
    std::vector<int> children;
    int visits{};
    double reward{};
  };
  std::vector<MctsNode> mcts(1);  // node zero is the root
  auto add_node = [&](int parent) {
    const int index = static_cast<int>(mcts.size());
    mcts.push_back({});
    mcts[parent].children.push_back(index);
    return index;
  };
  const int repair_operator_count =
      (features & AlnsSisrRecreate) != 0U ? 5 : 4;
  constexpr int destroy_operator_count = 6;
  for (int destroy = 0; destroy < destroy_operator_count; ++destroy) {
    const int destroy_node = add_node(0);
    for (int repair = 0; repair < repair_operator_count; ++repair) {
      const int repair_node = add_node(destroy_node);
      for (int travel = 0; travel < 4; ++travel) {
        (void)add_node(repair_node);
      }
    }
  }
  const double mcts_exploration = std::sqrt(2.0);
  auto select_mcts_child = [&](int parent) {
    const auto& children = mcts[parent].children;
    std::vector<int> unvisited;
    for (const int child : children) {
      if (mcts[child].visits == 0) unvisited.push_back(child);
    }
    if (!unvisited.empty()) {
      return unvisited[static_cast<std::size_t>(random() % unvisited.size())];
    }
    int selected = children.front();
    double best_score = -std::numeric_limits<double>::infinity();
    const double parent_log = std::log(static_cast<double>(mcts[parent].visits) + 1.0);
    for (const int child : children) {
      const auto& node = mcts[child];
      const double mean = node.reward / node.visits;
      const double score = mean +
                           mcts_exploration *
                               std::sqrt(parent_log / node.visits);
      if (score > best_score ||
          (score == best_score && child < selected)) {
        selected = child;
        best_score = score;
      }
    }
    return selected;
  };
  auto backpropagate_mcts = [&](const std::vector<int>& path, double reward) {
    for (const int node : path) {
      ++mcts[node].visits;
      mcts[node].reward += reward;
    }
  };

  const int configured_alns_iterations =
      !timed && final_day && limits.final_alns_iterations >= 0
          ? limits.final_alns_iterations
          : limits.max_iterations;
  const int maximum = std::max(0, configured_alns_iterations);
  const int minimum = std::min(std::max(0, limits.min_iterations), maximum);
  const bool exact_requested =
      exact_enabled && ((!timed && maximum > 96) || timed_exact);
  // ALNS iterations and exact-search nodes are independent, literal controls.
  // Neither phase silently borrows a percentage of the other one's allowance.
  //
  // A wall-clock request is genuinely anytime: the explicit iteration maximum
  // remains a safety ceiling, but there is no smaller hidden timed cap. The
  // main loop searches until its phase deadline, while the bounded elite pool
  // retains diverse continuation roots for the reserved projection phase.
  const int alns_maximum = maximum;
  const int diversify_after =
      limits.stagnation_iterations > 0
          ? std::max(16, limits.stagnation_iterations / 2)
          : 128;
  int stagnation = 0;
  int since_restart = 0;
  std::vector<long double> observed_losses;
  long double initial_temperature = 1.0L;
  long double final_temperature = 0.1L;
  long double reheat_temperature = 0.0L;
  std::uniform_real_distribution<long double> unit(0.0L, 1.0L);

  // Opt-in instrumentation (HEXUDON_ALNS_TRACE): diagnoses why ALNS stalls on
  // hard instances without altering the deterministic search trajectory.
  static const bool alns_trace = std::getenv("HEXUDON_ALNS_TRACE") != nullptr;
  int trace_iterations = 0;
  int trace_invalid = 0;
  int trace_accepted = 0;
  int trace_global_improvements = 0;
  int trace_last_improvement = -1;
  int trace_restarts = 0;

  // Anytime streaming: report the incumbent whenever it strictly improves so
  // the caller can resubmit. A null sink makes this a no-op for batch callers.
  auto emit_improvement = [&] {
    if (on_improve == nullptr) return;
    const auto official = alns_official_value(best_value);
    const auto congestion = congestion_value(best_evaluation);
    const auto workload = best_evaluation.workload;
    const IncumbentRank internal_rank{
        true,
        config.players != 1
            ? "disabled"
            : (day.day + 2 == static_cast<int>(config.day_steps.size())
                   ? "rolling"
                   : "current"),
        {std::get<0>(congestion), std::get<1>(congestion),
         std::get<2>(congestion), std::get<3>(congestion),
         std::get<4>(congestion), std::get<5>(congestion)},
        {std::get<0>(workload), std::get<1>(workload),
         std::get<2>(workload)},
        std::get<3>(best_value),
        best_projection.has_value(),
        best_projection
            ? std::array<int, 3>{std::get<0>(best_projection->match),
                                 std::get<1>(best_projection->match),
                                 std::get<2>(best_projection->match)}
            : std::array<int, 3>{},
        best_projection ? best_projection->ending_fuel : 0,
        palns_enabled ? "palns" : "daily"};
    (*on_improve)(best, Score{std::get<0>(official), std::get<1>(official),
                              std::get<2>(official)},
                  internal_rank);
  };
  emit_improvement();

  for (int iteration = 0; iteration < alns_maximum; ++iteration) {
    if (expired()) break;
    if (palns_runtime) {
      if (palns_runtime->remaining_iterations <= 0) break;
      --palns_runtime->remaining_iterations;
      ++palns_diagnostics.outer_iterations;
      ++palns_diagnostics.iterations_used;
    }
    ++trace_iterations;
    const int destroy_node = select_mcts_child(0);
    const int repair_node = select_mcts_child(destroy_node);
    const int travel_node = select_mcts_child(repair_node);
    const int destroy = static_cast<int>(
        std::find(mcts[0].children.begin(), mcts[0].children.end(),
                  destroy_node) - mcts[0].children.begin());
    const int repair = static_cast<int>(
        std::find(mcts[destroy_node].children.begin(),
                  mcts[destroy_node].children.end(), repair_node) -
        mcts[destroy_node].children.begin());
    const int travel = static_cast<int>(
        std::find(mcts[repair_node].children.begin(),
                  mcts[repair_node].children.end(), travel_node) -
        mcts[repair_node].children.begin());
    const std::vector<int> mcts_path{0, destroy_node, repair_node,
                                     travel_node};

    // Periodically branch from another elite basin. High-stock multi-patrol
    // instances need coordinated route changes that cannot be reached by
    // repeatedly perturbing only the incumbent.
    AlnsSolution candidate = current.solution;
    const CandidateEvaluation* source_evaluation = &current.evaluation;
    const int elite_exploration_period = std::max(8, diversify_after / 3);
    if (elite.size() > 1 &&
        (stagnation >= elite_exploration_period || random() % 100U < 12U)) {
      const std::size_t source_limit = std::min<std::size_t>(elite.size(), 4U);
      const std::size_t source =
          static_cast<std::size_t>(random() % source_limit);
      candidate = elite[source].solution;
      source_evaluation = &elite[source].evaluation;
    }
    int visits = 0;
    for (const auto& route : candidate.skeleton.routes) visits += route.size();
    if (visits > 0) {
      const bool diversify = stagnation >= diversify_after;
      constexpr std::array<std::pair<double, double>, 6> removal_ranges{{
          {0.10, 0.35},  // random removal
          {0.18, 0.45},  // contiguous route segment
          {0.10, 0.28},  // related-neighborhood removal
          {0.14, 0.36},  // rare-brand / saving removal
          {0.24, 0.50},  // longest-route removal
          {0.14, 0.36},  // refuel-bottleneck removal
      }};
      const auto [minimum_fraction, maximum_fraction] =
          removal_ranges[static_cast<std::size_t>(destroy)];
      const double fraction =
          diversify ? 0.40 + (random() % 26) / 100.0
                    : minimum_fraction +
                          (maximum_fraction - minimum_fraction) *
                              (random() % 101) / 100.0;
      const int removed =
          std::max(1, static_cast<int>(std::ceil(visits * fraction)));
      // Mode 5 unloads the busiest refuel car's patrols so repair can re-route
      // them through a different meeting.  It is a no-op when the current plan
      // has no refuel events (e.g. all-patrol tight-fuel maps), so fall back to
      // random removal to keep the iteration productive.
      if (destroy != 5 ||
          !destroy_alns_refuel_bottleneck(candidate.skeleton,
                                          source_evaluation->trace, removed,
                                          random)) {
        destroy_lns_skeleton(config, history, graph, candidate.skeleton,
                             destroy == 5 ? 0 : destroy, removed, random);
      }
    }
    repair_lns_skeleton(config, day, history, types, graph,
                        candidate.skeleton, repair, random);
    repair_alns_travel(candidate, travel, random,
                       (features & AlnsStableTravel) != 0U);
    auto plan = decode_lns_skeleton(config, day, types, graph, meeting_cache,
                                    candidate.skeleton, &candidate.travel,
                                    /*strict_travel=*/false);
    std::optional<CandidateEvaluation> evaluation;
    if (plan) evaluation = evaluate_candidate(config, day, history, *plan);
    std::optional<CandidateValue> value =
        evaluation ? std::optional<CandidateValue>(evaluation->value)
                   : std::nullopt;
    bool accepted = false;
    const auto current_ordinal = alns_ordinal(current.value, config, types);
    const auto current_official = alns_official_value(current.value);
    if (value) {
      std::optional<PalnsProjection> candidate_projection;
      const auto candidate_official = alns_official_value(*value);
      const auto best_official_before = alns_official_value(best_value);
      if (palns_runtime && candidate_official >= best_official_before) {
        candidate_projection = project_candidate(*plan, *evaluation);
      }
      const auto candidate_ordinal = alns_ordinal(*value, config, types);
      if (candidate_ordinal < current_ordinal &&
          observed_losses.size() < 12) {
        observed_losses.push_back(
            static_cast<long double>(current_ordinal - candidate_ordinal));
      }
      if ((iteration + 1 == 12 ||
           (observed_losses.size() == 12 && initial_temperature == 1.0L)) &&
          !observed_losses.empty()) {
        std::sort(observed_losses.begin(), observed_losses.end());
        const long double median =
            observed_losses[observed_losses.size() / 2];
        initial_temperature = std::max(1.0L, -median / std::log(0.5L));
        final_temperature = std::max(0.1L, -median / std::log(0.01L));
      }
      long double progress = alns_maximum <= 1
                                 ? 1.0L
                                 : static_cast<long double>(iteration) /
                                       static_cast<long double>(alns_maximum - 1);
      if (timed && alns_budget_ms > 0) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
        progress = std::clamp(
            static_cast<long double>(elapsed) / alns_budget_ms, 0.0L, 1.0L);
      }
      const long double temperature =
          std::max({final_temperature, reheat_temperature,
                    initial_temperature *
                        std::pow(final_temperature / initial_temperature,
                                 progress)});
      if (palns_runtime && candidate_official == current_official &&
          candidate_projection && current.projection &&
          palns_projection_rank(*candidate_projection) >=
              palns_projection_rank(*current.projection)) {
        accepted = true;
      } else if (*value >= current.value) {
        accepted = true;
      } else if (observed_losses.size() >= 3) {
        const long double probability =
            std::exp((candidate_ordinal - current_ordinal) / temperature);
        accepted = unit(random) < probability;
      }

      bool global_improvement = false;
      if (!palns_runtime) {
        global_improvement = online_improves(*evaluation, best_evaluation);
      } else if (candidate_official > best_official_before) {
        global_improvement = true;
      } else if (candidate_official == best_official_before) {
        if (candidate_projection && best_projection) {
          global_improvement =
              palns_projection_rank(*candidate_projection) >
                  palns_projection_rank(*best_projection) ||
              (palns_projection_rank(*candidate_projection) ==
                   palns_projection_rank(*best_projection) &&
               online_improves(*evaluation, best_evaluation));
        } else if (candidate_projection && !best_projection) {
          global_improvement = true;
        } else if (!candidate_projection && !best_projection) {
          global_improvement = online_improves(*evaluation, best_evaluation);
        }
      }
      const auto hash = alns_solution_hash(candidate);
      auto duplicate = std::find_if(
          elite.begin(), elite.end(),
          [&](const Elite& item) { return item.hash == hash; });
      const bool new_elite = duplicate == elite.end();
      if (global_improvement) {
        best = *plan;
        best_value = *value;
        best_evaluation = *evaluation;
        best_projection = candidate_projection;
        stagnation = 0;
        ++trace_global_improvements;
        trace_last_improvement = iteration;
        emit_improvement();
      } else {
        ++stagnation;
      }
      if (accepted) ++trace_accepted;
      if (new_elite) {
        elite.push_back({candidate, *plan, *value, *evaluation,
                         candidate_projection, hash});
      } else {
        Elite replacement{candidate, *plan, *value, *evaluation,
                          candidate_projection, hash};
        if (elite_order(replacement, *duplicate)) {
          *duplicate = std::move(replacement);
        }
      }
      trim_elite();
      if (accepted) {
        current = {std::move(candidate), std::move(*plan), *value,
                   *evaluation, candidate_projection, hash};
      }
      // Keep rewards bounded so UCB remains numerically stable across maps
      // with very different score magnitudes.  Positive local improvements,
      // new elite states, and global improvements receive progressively larger
      // rewards; invalid candidates receive zero below.
      double reward = 0.05;  // valid but dominated candidates are weak signals
      if (candidate_official == current_official) {
        reward = 0.12;
      } else if (candidate_official > current_official) {
        reward = 0.55;
        if (std::get<0>(candidate_official) >
            std::get<0>(current_official)) {
          reward += 0.25;
        } else if (std::get<1>(candidate_official) >
                   std::get<1>(current_official)) {
          reward += 0.18;
        } else if (std::get<2>(candidate_official) >
                   std::get<2>(current_official)) {
          reward += 0.12;
        }
      }
      if (candidate_ordinal > current_ordinal) reward += 0.03;
      if (accepted) reward += 0.03;
      if (new_elite) reward += 0.10;
      if (global_improvement) reward += 0.60;
      backpropagate_mcts(mcts_path, std::min(1.5, reward));
    } else {
      ++stagnation;
      ++trace_invalid;
      backpropagate_mcts(mcts_path, 0.0);
    }
    ++since_restart;
    reheat_temperature *= 0.95L;
    if (stagnation >= diversify_after &&
        since_restart >= diversify_after && elite.size() > 1) {
      auto selected = elite.begin();
      int farthest = -1;
      for (auto iterator = elite.begin(); iterator != elite.end(); ++iterator) {
        const int distance =
            alns_solution_distance(iterator->solution, elite.front().solution);
        if (distance > farthest) {
          farthest = distance;
          selected = iterator;
        }
      }
      current = *selected;
      since_restart = 0;
      reheat_temperature = 0.5L * initial_temperature;
      ++trace_restarts;
    }
    if (iteration + 1 >= minimum && limits.stagnation_iterations > 0 &&
        stagnation >= limits.stagnation_iterations) {
      break;
    }
  }
  if (alns_trace) {
    auto official = alns_official_value(best_value);
    std::ostringstream line;
    line << "ALNSTRACE day=" << day.day << " maxit=" << alns_maximum
         << " run=" << trace_iterations << " invalid=" << trace_invalid
         << " accepted=" << trace_accepted
         << " global_impr=" << trace_global_improvements
         << " last_impr=" << trace_last_improvement
         << " restarts=" << trace_restarts << " elite=" << elite.size()
         << " best=(" << std::get<0>(official) << "," << std::get<1>(official)
         << "," << std::get<2>(official) << ")";
    line << std::fixed << std::setprecision(3);
    for (int destroy = 0; destroy < destroy_operator_count; ++destroy) {
      const int dnode = mcts[0].children[static_cast<std::size_t>(destroy)];
      const double dv = std::max(1, mcts[dnode].visits);
      line << " D" << destroy << "=" << mcts[dnode].visits << "v/"
           << mcts[dnode].reward / dv << "r";
    }
    for (int repair = 0; repair < repair_operator_count; ++repair) {
      int visits = 0;
      double reward = 0.0;
      for (int destroy = 0; destroy < destroy_operator_count; ++destroy) {
        const int dnode = mcts[0].children[static_cast<std::size_t>(destroy)];
        const int rnode =
            mcts[dnode].children[static_cast<std::size_t>(repair)];
        visits += mcts[rnode].visits;
        reward += mcts[rnode].reward;
      }
      line << " R" << repair << "=" << visits << "v/"
           << reward / std::max(1, visits) << "r";
    }
    std::fprintf(stderr, "%s\n", line.str().c_str());
    std::fflush(stderr);
  }

  // A refuel car is normally moved only when the current day's patrol route
  // needs a rendezvous.  That is good for the daily score, but it can leave
  // the support car in a poor position for tomorrow.  Generate a small set of
  // end-of-day staging variants and use a cheap deterministic rollout only as
  // a tie-break among plans with the same current-day official score.  This
  // preserves the competition metric while allowing a day-1 move when it
  // genuinely improves the reachable continuation.
  static const bool disable_refuel_lookahead =
      std::getenv("HEXUDON_ALNS_DISABLE_REFUEL_LOOKAHEAD") != nullptr;
  const bool severe_fuel_pressure =
      2 * config.fuel_limit <= config.day_steps[day.day];
  if (allow_continuation && !palns_enabled && !disable_refuel_lookahead &&
      severe_fuel_pressure &&
      day.day + 1 < static_cast<int>(config.day_steps.size()) &&
      !elite.empty() &&
      std::find(types.begin(), types.end(), AgentKind::Refuel) != types.end() &&
      (!timed || std::chrono::steady_clock::now() < deadline)) {
    const auto current_official = alns_official_value(best_evaluation.value);
    std::vector<std::pair<ActionPlan, CandidateEvaluation>> sources;
    sources.emplace_back(best, best_evaluation);
    for (std::size_t index = 0;
         index < elite.size() && sources.size() < 3; ++index) {
      if (alns_official_value(elite[index].value) != current_official) continue;
      sources.emplace_back(elite[index].plan, elite[index].evaluation);
    }

    const int lookahead_ms =
        timed ? std::max(1, limits.time_limit_ms / 20) : 100;
    const auto lookahead_deadline =
        timed ? std::min(deadline, std::chrono::steady_clock::now() +
                                      std::chrono::milliseconds(lookahead_ms))
              : std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(lookahead_ms);
    if (!timed || std::chrono::steady_clock::now() < lookahead_deadline) {
      // Give the incumbent and every staged candidate an equal rollout slice.
      // A single shared deadline let the incumbent consume the whole budget;
      // candidates then returned their root score without projecting even one
      // future day, making the look-ahead incapable of selecting a move.
      const int projection_ms = timed ? std::max(1, lookahead_ms / 5) : 40;
      auto projection_deadline = [&] {
        if (!timed) {
          return std::chrono::steady_clock::time_point::max();
        }
        const auto slice = std::chrono::steady_clock::now() +
                           std::chrono::milliseconds(projection_ms);
        return std::min(lookahead_deadline, slice);
      };
      auto projection = alns_match_rollout(
          config, day, history, types, best_evaluation, 1,
          projection_deadline(), features, &own_traffic_history);
      auto projected_score_rank = [](const AlnsMatchScore& score) {
        return std::tuple{score.worst, score.discounted, score.total};
      };
      const auto base_projected_score = projected_score_rank(projection);
      auto continuation_rank = [&](const CandidateEvaluation& evaluation,
                                   const AlnsMatchScore& score) {
        return std::tuple{score.worst, score.discounted, score.total,
                          std::get<3>(evaluation.value),
                          congestion_value(evaluation), evaluation.workload};
      };
      auto best_rank = continuation_rank(best_evaluation, projection);
      int evaluated_staging = 0;
      for (const auto& source : sources) {
        if (evaluated_staging >= 4) break;
        if (timed && std::chrono::steady_clock::now() >= lookahead_deadline) {
          break;
        }
        for (const auto& staged : refuel_staging_variants(
                 config, day, types, history, source.first)) {
          if (evaluated_staging >= 4) break;
          if (timed && std::chrono::steady_clock::now() >= lookahead_deadline) {
            break;
          }
          ++evaluated_staging;
          auto evaluation = evaluate_candidate(config, day, history, staged);
          if (!evaluation ||
              alns_official_value(evaluation->value) != current_official) {
            continue;
          }
          // A staging route must actually rendezvous with a patrol today.
          // Moving the support car to an empty spot (or a patrol endpoint that
          // it never reaches during this plan) only changes traffic/positions
          // and can damage a later route without providing fuel capacity.
          if (std::get<3>(evaluation->value) <=
              std::get<3>(best_evaluation.value)) {
            continue;
          }
          auto staged_projection = alns_match_rollout(
              config, day, history, types, *evaluation, 1,
              projection_deadline(), features, &own_traffic_history);
          auto rank = continuation_rank(*evaluation, staged_projection);
          // Current-day official score is already held equal above, and this
          // block runs only on genuinely fuel-starved days. Accept a strict
          // projected-score gain or a score tie that rescues patrol fuel. On
          // high-capacity maps proactive staging can perturb a saturated route
          // and lose a real final-day serving, so it is gated out above.
          const bool projected_improvement =
              projected_score_rank(staged_projection) > base_projected_score;
          const bool severe_fuel_rescue =
              projected_score_rank(staged_projection) == base_projected_score &&
              std::get<3>(evaluation->value) >
                  std::get<3>(best_evaluation.value);
          if ((projected_improvement || severe_fuel_rescue) &&
              rank > best_rank) {
            best = staged;
            best_value = evaluation->value;
            best_evaluation = std::move(*evaluation);
            best_rank = std::move(rank);
            emit_improvement();
          }
        }
      }
    }
  }

  // Tier 2 continuation look-ahead: on every non-final day, re-rank the top
  // elites by an accurate projection of the *remaining match* (each future day
  // is planned with a small nested ALNS on the real simulator), and adopt the
  // elite whose whole-match projection is best.  Unlike the cheap positional
  // potential, the projected match distinct is correctly bounded because it is
  // produced by actually playing the days, so it can safely drive selection.
  // The previous version only fired on the penultimate day; generalizing it to
  // all days is what makes the search grade today's plan by its real future.
  const int continuation_horizon =
      static_cast<int>(config.day_steps.size()) - day.day - 1;
  if (allow_continuation && !palns_enabled && continuation_horizon >= 1 &&
      (!timed || std::chrono::steady_clock::now() < deadline) &&
      elite.size() > 1) {
    SearchLimits projection_limits = limits;
    // Fixed-iteration benchmarks use a deterministic projection depth. Timed
    // online search instead gives every nested planner a wall-clock slice below;
    // it must not inherit another small iteration/stagnation stop that makes a
    // 30-second request behave like a 5-second request.
    const int projection_iterations =
        continuation_horizon <= 1 ? 96
                                  : std::max(32, 96 / continuation_horizon);
    if (timed) {
      projection_limits.min_iterations = 0;
      projection_limits.max_iterations = limits.max_iterations;
      projection_limits.stagnation_iterations = 0;
      projection_limits.seed_iterations =
          std::min(limits.seed_iterations, projection_iterations);
    } else {
      projection_limits.min_iterations = projection_iterations;
      projection_limits.max_iterations = projection_iterations;
      projection_limits.stagnation_iterations = projection_iterations;
      projection_limits.seed_iterations = projection_iterations;
    }
    projection_limits.final_alns_iterations = -1;
    projection_limits.exact_nodes = 0;
    projection_limits.final_exact_nodes = -1;
    struct ContinuationProjection {
      std::tuple<int, int, int> match;
      int ending_fuel{};
    };
    const auto& prior_traffic = own_traffic_history;
    auto project_remaining_match = [&](const ActionPlan& current_plan,
                                       const CandidateEvaluation& current_eval,
                                       const std::chrono::steady_clock::time_point
                                           projection_deadline)
        -> std::optional<ContinuationProjection> {
      auto traffic_history = prior_traffic;
      traffic_history.push_back(current_eval.road_traffic);
      PolicyHistory next_history = history;
      next_history.submitted_actions.push_back(current_plan);
      for (const auto& acquisition : current_eval.trace.acquisitions) {
        if (const Spot* spot = spot_at(config, acquisition.spot_pos)) {
          next_history.distinct_brands.insert(spot->brand);
        }
      }
      std::vector<int> positions = current_eval.ending_positions;
      std::vector<int> fuel = current_eval.ending_fuel;
      int cumulative_daily = std::get<1>(current_eval.value);
      int servings = std::get<2>(current_eval.value);
      int distinct = std::get<0>(current_eval.value);
      for (int next_day = day.day + 1;
           next_day < static_cast<int>(config.day_steps.size()); ++next_day) {
        if (timed &&
            std::chrono::steady_clock::now() >= projection_deadline) {
          return std::nullopt;
        }
        DayInfo next;
        next.day = next_day;
        for (std::size_t agent = 0; agent < types.size(); ++agent) {
          next.agents.push_back(
              {types[agent], positions[agent], fuel[agent]});
        }
        // === Future-traffic seam ===
        // Predicts each projected day's road status. Dividing our own traffic by
        // 1 is the symmetric-opponent surrogate: if every team moves like us,
        // (our + opponents)/players ~= our own contribution, so single-team
        // traffic doubles as the multi-team estimate. Replace this call with a
        // learned predictor (e.g. a traffic GNN) to model real opponents.
        next.traffics = road_status_for_day(config, traffic_history, 1);
        SearchLimits next_projection_limits = projection_limits;
        if (timed) {
          // Split this root candidate's wall-clock slice across its remaining
          // simulated days.  The old code made these nested planners untimed;
          // on an online `solve` request they could run beyond the outer
          // deadline, so the selected continuation was never streamed in time
          // for submission.
          const int remaining_days =
              static_cast<int>(config.day_steps.size()) - next_day;
          const auto remaining =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  projection_deadline - std::chrono::steady_clock::now())
                  .count();
          if (remaining <= 0) return std::nullopt;
          next_projection_limits.time_limit_ms = static_cast<int>(
              std::max<std::int64_t>(1, remaining / remaining_days));
          // A projection is deliberately bounded by time, not by a mandatory
          // iteration floor inherited from the live optimizer.
          next_projection_limits.min_iterations = 0;
        } else {
          next_projection_limits.time_limit_ms = -1;
        }
        auto projected_plan = build_alns_plan(
            config, next, next_history, types, next_projection_limits, features,
            false, restart_salt);
        auto evaluation =
            evaluate_candidate(config, next, next_history, projected_plan);
        if (!evaluation) return std::nullopt;
        if (timed && next_day + 1 < static_cast<int>(config.day_steps.size()) &&
            std::chrono::steady_clock::now() >= projection_deadline) {
          return std::nullopt;
        }
        distinct = std::get<0>(evaluation->value);
        cumulative_daily += std::get<1>(evaluation->value);
        servings += std::get<2>(evaluation->value);
        positions = evaluation->ending_positions;
        fuel = evaluation->ending_fuel;
        traffic_history.push_back(evaluation->road_traffic);
        next_history.submitted_actions.push_back(projected_plan);
        for (const auto& acquisition : evaluation->trace.acquisitions) {
          if (const Spot* spot = spot_at(config, acquisition.spot_pos)) {
            next_history.distinct_brands.insert(spot->brand);
          }
        }
      }
      int ending_fuel = 0;
      for (std::size_t agent = 0; agent < types.size(); ++agent) {
        if (types[agent] == AgentKind::Patrol) ending_fuel += fuel[agent];
      }
      return ContinuationProjection{
          {distinct, cumulative_daily, servings}, ending_fuel};
    };
    auto continuation_rank = [&](const CandidateEvaluation& current_eval,
                                 const ContinuationProjection& projection) {
      return std::tuple{projection.match, projection.ending_fuel,
                        alns_official_value(current_eval.value),
                        congestion_value(current_eval), current_eval.workload,
                        std::get<3>(current_eval.value)};
    };
    // Build the comparison set before starting any simulation.  Every root
    // receives an equal share of the remaining online window; previously the
    // incumbent could consume it all, leaving every alternative unprojected.
    const int realized_distinct_floor =
        std::get<0>(alns_official_value(best_value));
    auto meets_floor = [&](const CandidateEvaluation& evaluation) {
      return std::get<0>(alns_official_value(evaluation.value)) >=
             realized_distinct_floor;
    };
    std::vector<const Elite*> projection_elites;
    for (const auto& candidate : elite) {
      if (projection_elites.size() >= 4U) break;
      if (candidate.plan == best || !meets_floor(candidate.evaluation)) {
        continue;
      }
      projection_elites.push_back(&candidate);
    }
    std::size_t remaining_projection_roots = 1U + projection_elites.size();
    auto next_projection_deadline = [&] {
      if (!timed) return std::chrono::steady_clock::time_point::max();
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) return now;
      const auto fair_slice = (deadline - now) /
                              static_cast<std::int64_t>(
                                  std::max<std::size_t>(
                                      1U, remaining_projection_roots));
      return std::min(deadline, now + fair_slice);
    };
    auto best_projection = project_remaining_match(
        best, best_evaluation, next_projection_deadline());
    if (remaining_projection_roots > 0U) --remaining_projection_roots;
    if (best_projection) {
      auto best_rank =
          continuation_rank(best_evaluation, *best_projection);
      // Realized banked brands are certain; the projection is a noisy estimate
      // (a weak nested planner). Never adopt a continuation whose *realized*
      // distinct is below the myopic incumbent's — that would trade a banked
      // brand for a projection that can be wrong (observed: hard/0021 7->5).
      // Continuation gains are still free among plans that tie the best realized
      // distinct (better position/fuel for future days).
      for (const Elite* candidate : projection_elites) {
        if (timed && std::chrono::steady_clock::now() >= deadline) break;
        auto projection = project_remaining_match(
            candidate->plan, candidate->evaluation,
            next_projection_deadline());
        if (remaining_projection_roots > 0U) --remaining_projection_roots;
        if (!projection) continue;
        const auto rank =
            continuation_rank(candidate->evaluation, *projection);
        if (rank > best_rank) {
          best = candidate->plan;
          best_value = candidate->value;
          best_evaluation = candidate->evaluation;
          best_projection = std::move(projection);
          best_rank = rank;
          emit_improvement();
        }
      }
      auto consider_continuation = [&](const ActionPlan& candidate_plan) {
        if (timed && std::chrono::steady_clock::now() >= deadline) return;
        auto evaluation =
            evaluate_candidate(config, day, history, candidate_plan);
        if (!evaluation) return;
        if (!meets_floor(*evaluation)) return;
        auto projection = project_remaining_match(
            candidate_plan, *evaluation, deadline);
        if (!projection) return;
        const auto rank = continuation_rank(*evaluation, *projection);
        if (rank > best_rank) {
          best = candidate_plan;
          best_value = evaluation->value;
          best_evaluation = std::move(*evaluation);
          best_projection = std::move(projection);
          best_rank = rank;
          emit_improvement();
        }
      };
      const std::optional<std::chrono::steady_clock::time_point>
          continuation_deadline =
              timed ? std::optional<std::chrono::steady_clock::time_point>(
                          deadline)
                    : std::nullopt;
      auto exact_tie = exact_route_search(
          config, day, history, types, graph, meeting_cache, best,
          best_evaluation, 32, continuation_deadline, true);
      consider_continuation(exact_tie.plan);
    }
  }
  if (exact_requested &&
      (!timed || std::chrono::steady_clock::now() < deadline)) {
    std::int64_t exact_budget = configured_exact_nodes;
    const std::optional<std::chrono::steady_clock::time_point> exact_deadline =
        timed ? std::optional<std::chrono::steady_clock::time_point>(deadline)
              : std::nullopt;
    if (day.day + 1 == static_cast<int>(config.day_steps.size()) &&
        exact_budget > 0) {
      auto route_exact = exact_route_search(
          config, day, history, types, graph, meeting_cache, best,
          best_evaluation, exact_budget, exact_deadline);
      exact_budget = route_exact.complete
                         ? 0
                         : std::max<std::int64_t>(
                               0, exact_budget - route_exact.explored_nodes);
      if (!validate_action_plan(config, day, route_exact.plan)) {
        auto route_evaluation =
            evaluate_candidate(config, day, history, route_exact.plan);
        if (route_evaluation &&
            online_improves(*route_evaluation, best_evaluation)) {
          best = std::move(route_exact.plan);
          best_value = route_exact.value;
          best_evaluation = std::move(*route_evaluation);
          emit_improvement();
        }
      }
    }
    auto exact = exact_day_search(config, day, history, types, best, best_value,
                                  exact_budget, exact_deadline, features);
    if (!validate_action_plan(config, day, exact.plan)) {
      auto exact_evaluation =
          evaluate_candidate(config, day, history, exact.plan);
      if (exact_evaluation &&
          online_improves(*exact_evaluation, best_evaluation)) {
        best = std::move(exact.plan);
        best_value = exact.value;
        best_evaluation = std::move(*exact_evaluation);
        emit_improvement();
      }
    }
  }
  if (palns_enabled && final_day) {
    if (auto final_projection = project_candidate(best, best_evaluation)) {
      best_projection = std::move(final_projection);
    }
  }
  if (palns_enabled && best_projection) {
    palns_returned_projections[best] =
        palns_projection_rank(*best_projection);
  }
  if (elite_plans != nullptr) {
    elite_plans->clear();
    elite_plans->push_back(best);
    std::set<std::vector<int>> ending_positions;
    ending_positions.insert(best_evaluation.ending_positions);
    const auto best_official = alns_official_value(best_evaluation.value);
    for (const auto& candidate : elite) {
      if (elite_plans->size() >= 8U) break;
      if (alns_official_value(candidate.value) != best_official ||
          !ending_positions.insert(candidate.evaluation.ending_positions)
               .second ||
          candidate.plan == best) {
        continue;
      }
      elite_plans->push_back(candidate.plan);
    }
  }
  return best;
}

ActionPlan build_alns_multirestart_plan(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& types,
    const SearchLimits& limits, unsigned features) {
  const int restart_count = std::clamp(limits.alns_restarts, 1, 3);
  if (restart_count == 1) {
    return build_alns_plan(config, day, history, types, limits, features);
  }

  std::vector<SearchLimits> restart_limits(
      static_cast<std::size_t>(restart_count), limits);
  const bool palns_enabled = (features & AlnsProjectedObjective) != 0U;
  if (palns_enabled) {
    palns_returned_projections.clear();
    const int total = limits.total_iterations >= 0
                          ? limits.total_iterations
                          : std::max(1, limits.max_iterations);
    for (int index = 0; index < restart_count; ++index) {
      const int share = total / restart_count + (index < total % restart_count);
      restart_limits[static_cast<std::size_t>(index)].total_iterations = share;
      restart_limits[static_cast<std::size_t>(index)].max_iterations = share;
    }
  }
  if (restart_count >= 3) {
    auto& reduced = restart_limits[2];
    reduced.aco_ants = 4;
    reduced.aco_iterations = 4;
    reduced.use_local_search_seed = false;
    reduced.alns_restarts = 1;
  }
  for (auto& restart : restart_limits) restart.alns_restarts = 1;

  constexpr std::array<std::uint64_t, 3> restart_salts{
      0ULL, 0x9e3779b97f4a7c15ULL, 0xbf58476d1ce4e5b9ULL};
  auto build_restart = [&](std::size_t index) {
        const unsigned restart_features =
            index == 2 ? features | AlnsSisrRecreate : features;
        return build_alns_plan(config, day, history, types,
                               restart_limits[index], restart_features, true,
                               restart_salts[index]);
  };
  std::vector<ActionPlan> plans;
  if (palns_enabled) {
    plans.reserve(restart_limits.size());
    for (std::size_t index = 0; index < restart_limits.size(); ++index) {
      plans.push_back(build_restart(index));
    }
  } else {
    plans = parallel_alns_restarts(restart_limits.size(), build_restart);
  }

  ActionPlan best = plans.front();
  auto best_evaluation = evaluate_candidate(config, day, history, best);
  const bool final_day =
      day.day + 1 >= static_cast<int>(config.day_steps.size());
  auto continuation_equivalent = [&](const CandidateEvaluation& left,
                                     const CandidateEvaluation& right) {
    if (left.ending_positions != right.ending_positions ||
        left.ending_fuel != right.ending_fuel ||
        left.road_traffic != right.road_traffic) {
      return false;
    }
    auto resulting_brands = [&](const CandidateEvaluation& evaluation) {
      std::set<int> brands = history.distinct_brands;
      for (const auto& acquisition : evaluation.trace.acquisitions) {
        if (const Spot* spot = spot_at(config, acquisition.spot_pos)) {
          brands.insert(spot->brand);
        }
      }
      return brands;
    };
    return resulting_brands(left) == resulting_brands(right);
  };
  for (std::size_t index = 1; index < plans.size(); ++index) {
    auto evaluation = evaluate_candidate(config, day, history, plans[index]);
    if (!evaluation) continue;
    bool improves = !best_evaluation;
    if (best_evaluation && palns_enabled) {
      const auto candidate_official = alns_official_value(evaluation->value);
      const auto incumbent_official =
          alns_official_value(best_evaluation->value);
      if (candidate_official > incumbent_official) {
        improves = true;
      } else if (candidate_official == incumbent_official) {
        const auto candidate_projection =
            palns_returned_projections.find(plans[index]);
        const auto incumbent_projection = palns_returned_projections.find(best);
        if (candidate_projection != palns_returned_projections.end() &&
            incumbent_projection != palns_returned_projections.end()) {
          improves = candidate_projection->second > incumbent_projection->second;
        } else if (candidate_projection != palns_returned_projections.end()) {
          improves = true;
        } else if (incumbent_projection == palns_returned_projections.end()) {
          improves = evaluation->value > best_evaluation->value;
        }
      }
    } else if (best_evaluation) {
      improves =
          alns_official_value(evaluation->value) >
              alns_official_value(best_evaluation->value) &&
          (final_day || continuation_equivalent(*evaluation, *best_evaluation));
    }
    if (improves) {
      best = std::move(plans[index]);
      best_evaluation = std::move(evaluation);
    }
  }
  return best;
}

namespace {

using MlnsWide = boost::multiprecision::uint128_t;

struct MlnsRank {
  std::array<MlnsWide, 3> weighted{};
  // Daily types and servings realized on the committed current day. Only the
  // current day is submitted; every spot restocks each morning and later days
  // are re-planned from the revealed state. The whole-match suffix forecast is
  // a weak decoder that systematically under-counts an aggressive current-day
  // plan's future coverage/servings, so ranking banks these realized values
  // before trusting the forecast (see mlns_rank_better). Distinct types stay a
  // discounted whole-match quantity because reaching every brand over the match
  // genuinely needs multi-day look-ahead.
  int current_daily{};
  int current_servings{};
  // Total patrol fuel carried out of the committed current day. Among plans that
  // realize the identical current-day official triple, a larger reserve enters
  // the next day fuller, so it needs fewer/later refuel rendezvous and wastes
  // fewer collection steps on detours. It is a non-forecast continuation signal
  // used only to break exact commit ties (see mlns_commit_better).
  int current_ending_fuel{};
  std::tuple<int, int, int> projected{};
  int ending_fuel{};
  std::size_t hash{};
};

struct MlnsDaySolution {
  int day{};
  LnsSkeleton skeleton;
  AlnsTravelChoices travel;
  ActionPlan plan;
  CandidateEvaluation evaluation;
  DayInfo day_info;
  PolicyHistory history_before;
  PolicyHistory history_after;
  std::vector<std::map<int, int>> traffic_history_after;
  std::array<int, 3> reward{};
};

struct MlnsSolution {
  std::vector<MlnsDaySolution> days;
  MlnsRank rank;
  bool exact_replayed_root{};
};

struct MlnsReplaySignature {
  std::vector<AgentView> agents;
  std::map<int, int> traffics;
};

struct MlnsGenome {
  std::vector<LnsSkeleton> skeletons;
  std::vector<AlnsTravelChoices> travel;
  std::vector<std::optional<ActionPlan>> replay_plans;
  std::vector<std::optional<MlnsReplaySignature>> replay_signatures;
};

struct MlnsProfileSnapshot {
  std::tuple<int, int, int> current{};
  std::tuple<int, int, int> projected{};
  int ending_fuel{};
  std::size_t hash{};
  ActionPlan committed_plan;
};

std::map<int, int> mlns_forecast_traffic(
    const MapConfig& config,
    const std::vector<std::map<int, int>>& traffic_history,
    int day_index) {
  if (day_index >= 0 &&
      static_cast<std::size_t>(day_index) < mlns_predicted_traffic.size()) {
    return mlns_predicted_traffic[static_cast<std::size_t>(day_index)];
  }
  return road_status_for_day(config, traffic_history, 1);
}

MlnsProfileSnapshot mlns_profile_snapshot(const MlnsSolution& solution) {
  const auto current = solution.days.empty()
                           ? std::tuple{0, 0, 0}
                           : alns_official_value(
                                 solution.days.front().evaluation.value);
  return {current, solution.rank.projected, solution.rank.ending_fuel,
          solution.rank.hash,
          solution.days.empty() ? ActionPlan{} : solution.days.front().plan};
}

MlnsComponentDiagnostics& mlns_profile_component(std::string_view name) {
  auto found = std::find_if(
      mlns_diagnostics.components.begin(), mlns_diagnostics.components.end(),
      [&](const MlnsComponentDiagnostics& component) {
        return component.component == name;
      });
  if (found != mlns_diagnostics.components.end()) return *found;
  mlns_diagnostics.components.push_back(
      MlnsComponentDiagnostics{std::string(name)});
  return mlns_diagnostics.components.back();
}

std::int64_t mlns_elapsed_microseconds(
    std::chrono::steady_clock::time_point started) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now() - started)
      .count();
}

bool mlns_profile_changed(const MlnsProfileSnapshot& before,
                          const MlnsProfileSnapshot& after) {
  return before.current != after.current ||
         before.projected != after.projected ||
         before.ending_fuel != after.ending_fuel || before.hash != after.hash ||
         before.committed_plan != after.committed_plan;
}

bool mlns_profile_committed_plan_changed(const MlnsProfileSnapshot& before,
                                         const MlnsProfileSnapshot& after) {
  return before.committed_plan != after.committed_plan;
}

void mlns_profile_phase(std::string_view name,
                        std::chrono::steady_clock::time_point started,
                        const std::optional<MlnsProfileSnapshot>& before,
                        const MlnsSolution& after) {
  auto& component = mlns_profile_component(name);
  ++component.calls;
  component.elapsed_microseconds += mlns_elapsed_microseconds(started);
  if (!before) return;
  const auto current = mlns_profile_snapshot(after);
  if (!mlns_profile_changed(*before, current)) return;
  ++component.incumbent_updates;
  const auto add_tuple_delta = [](std::array<std::int64_t, 3>& destination,
                                  const std::tuple<int, int, int>& left,
                                  const std::tuple<int, int, int>& right) {
    destination[0] += std::get<0>(left) - std::get<0>(right);
    destination[1] += std::get<1>(left) - std::get<1>(right);
    destination[2] += std::get<2>(left) - std::get<2>(right);
  };
  add_tuple_delta(component.current_score_gain, current.current,
                  before->current);
  add_tuple_delta(component.projected_score_gain, current.projected,
                  before->projected);
  component.ending_patrol_fuel_gain +=
      current.ending_fuel - before->ending_fuel;
}

MlnsWide mlns_power(int base, int exponent) {
  MlnsWide result = 1;
  for (int index = 0; index < exponent; ++index) {
    result *= static_cast<unsigned>(base);
  }
  return result;
}

std::vector<MlnsWide> mlns_weights(int count, int discount_percent) {
  std::vector<MlnsWide> result;
  result.reserve(static_cast<std::size_t>(count));
  for (int horizon = 0; horizon < count; ++horizon) {
    result.push_back(mlns_power(discount_percent, horizon) *
                     mlns_power(100, count - horizon - 1));
  }
  return result;
}

std::string mlns_wide_string(MlnsWide value) {
  if (value == 0) return "0";
  std::string result;
  while (value > 0) {
    result.push_back(static_cast<char>(
        '0' + (value % 10).convert_to<unsigned>()));
    value /= 10;
  }
  std::reverse(result.begin(), result.end());
  return result;
}

// MLNS commits only the current day and re-plans every later day from the
// revealed state, but it ranks whole-match candidates. Its suffix decoder is a
// cheap forecast that systematically under-counts an aggressive current-day
// plan's future daily coverage and servings (on Q06 it predicted 87/127 for a
// plan that actually realizes 91/223). Ranking therefore trusts the forecast
// only where multi-day planning is essential and irreversible — reaching every
// distinct brand over the match — and otherwise banks the current day's
// realized, submitted values before consulting the forecast.
bool mlns_rank_better(const MlnsRank& left, const MlnsRank& right) {
  // 1. Predicted final distinct types (top official objective). Distinct is
  //    permanent and match-wide, so it keeps full look-ahead: giving up a
  //    current-day brand to reach more brands overall stays available.
  if (std::get<0>(left.projected) != std::get<0>(right.projected)) {
    return std::get<0>(left.projected) > std::get<0>(right.projected);
  }
  // 2. Among equal predicted final distinct, prefer collecting brands earlier
  //    (weighted[0] discounts later days), which is robust to later-day traffic
  //    and fuel uncertainty.
  if (left.weighted[0] != right.weighted[0]) {
    return left.weighted[0] > right.weighted[0];
  }
  // 3-4. Bank the committed day's realized daily types, then servings. These
  //    reset every morning and are re-planned next day, so the reliable
  //    submitted value outranks the weak whole-match forecast below.
  if (left.current_daily != right.current_daily) {
    return left.current_daily > right.current_daily;
  }
  if (left.current_servings != right.current_servings) {
    return left.current_servings > right.current_servings;
  }
  // 5-6. Discounted whole-match daily / servings forecasts break remaining ties.
  if (left.weighted[1] != right.weighted[1]) {
    return left.weighted[1] > right.weighted[1];
  }
  if (left.weighted[2] != right.weighted[2]) {
    return left.weighted[2] > right.weighted[2];
  }
  if (left.projected != right.projected) {
    return left.projected > right.projected;
  }
  if (left.ending_fuel != right.ending_fuel) {
    return left.ending_fuel > right.ending_fuel;
  }
  return left.hash < right.hash;
}

bool mlns_rank_equal(const MlnsRank& left, const MlnsRank& right) {
  return left.weighted == right.weighted &&
         left.current_daily == right.current_daily &&
         left.current_servings == right.current_servings &&
         left.projected == right.projected &&
         left.ending_fuel == right.ending_fuel && left.hash == right.hash;
}

bool mlns_reward_better(const MlnsRank& left, const MlnsRank& right) {
  if (std::get<0>(left.projected) != std::get<0>(right.projected)) {
    return std::get<0>(left.projected) > std::get<0>(right.projected);
  }
  if (left.weighted[0] != right.weighted[0]) {
    return left.weighted[0] > right.weighted[0];
  }
  if (left.current_daily != right.current_daily) {
    return left.current_daily > right.current_daily;
  }
  if (left.current_servings != right.current_servings) {
    return left.current_servings > right.current_servings;
  }
  if (left.weighted[1] != right.weighted[1]) {
    return left.weighted[1] > right.weighted[1];
  }
  if (left.weighted[2] != right.weighted[2]) {
    return left.weighted[2] > right.weighted[2];
  }
  return left.projected > right.projected;
}

// Servings tolerance (in realized current-day servings) within which the plan
// that is COMMITTED for the current day is chosen by the whole-match
// continuation forecast rather than by raw realized current-day servings.
//
// mlns_rank_better banks the realized current-day servings before the forecast
// (weighted[1]/weighted[2]) because that forecast systematically under-counts an
// aggressive plan's future. That protects the submitted day but makes the
// committed choice greedy: on coupled days it maximizes today at the cost of
// stranding agents/fuel for tomorrow (Q06 committed day2=36 -> re-planned
// day3=24, versus day2=35 -> day3=28, a net whole-match loss). weighted[2]
// already contains the realized current day at full weight plus the discounted
// suffix, so inside a small current-day gap it directly answers "does the better
// continuation outweigh the current-day deficit?". Outside the band the realized
// advantage is protected from the biased forecast. Tuned on the
// brutal/steady/easy + online suite; override with HEXUDON_MLNS_COMMIT_TOLERANCE.
constexpr int kMlnsCommitToleranceDefault = 0;
int mlns_commit_tolerance() {
  static const int value = [] {
    const char* raw = std::getenv("HEXUDON_MLNS_COMMIT_TOLERANCE");
    if (raw == nullptr) return kMlnsCommitToleranceDefault;
    return std::max(0, std::atoi(raw));
  }();
  return value;
}

// Tie-break used among commit candidates whose current-day official triple is
// within tolerance. "fuel" prefers the plan that carries more patrol fuel out
// of the committed day before consulting the biased whole-match forecast;
// "forecast" keeps the forecast-first order. Default set by the suite sweep;
// override with HEXUDON_MLNS_COMMIT_MODE.
constexpr bool kMlnsCommitPrefersFuelDefault = false;
bool mlns_commit_prefers_fuel() {
  static const bool value = [] {
    const char* raw = std::getenv("HEXUDON_MLNS_COMMIT_MODE");
    if (raw == nullptr) return kMlnsCommitPrefersFuelDefault;
    if (std::string_view(raw) == "fuel") return true;
    if (std::string_view(raw) == "forecast") return false;
    throw std::invalid_argument(
        "HEXUDON_MLNS_COMMIT_MODE must be forecast or fuel");
  }();
  return value;
}

// Pairwise preference for the plan to commit this day. Only ever used to update
// the returned incumbent `best` (never inside a std::sort), so the bounded
// current-day tolerance need not be transitive. Distinct/daily coverage remains
// fully protected: the primary objectives are compared before servings.
bool mlns_commit_better(const MlnsRank& left, const MlnsRank& right) {
  if (std::get<0>(left.projected) != std::get<0>(right.projected)) {
    return std::get<0>(left.projected) > std::get<0>(right.projected);
  }
  if (left.weighted[0] != right.weighted[0]) {
    return left.weighted[0] > right.weighted[0];
  }
  if (left.current_daily != right.current_daily) {
    return left.current_daily > right.current_daily;
  }
  const int tolerance = mlns_commit_tolerance();
  const long long gap = static_cast<long long>(left.current_servings) -
                        static_cast<long long>(right.current_servings);
  if (gap > tolerance) return true;
  if (gap < -tolerance) return false;
  // Current-day servings are within tolerance. Optionally prefer the larger
  // carried-out fuel reserve (a non-forecast continuation signal) before the
  // biased whole-match forecast.
  if (mlns_commit_prefers_fuel() &&
      left.current_ending_fuel != right.current_ending_fuel) {
    return left.current_ending_fuel > right.current_ending_fuel;
  }
  // Otherwise let the discounted whole-match forecast (which includes today at
  // full weight) pick the better continuation.
  if (left.weighted[1] != right.weighted[1]) {
    return left.weighted[1] > right.weighted[1];
  }
  if (left.weighted[2] != right.weighted[2]) {
    return left.weighted[2] > right.weighted[2];
  }
  if (left.current_servings != right.current_servings) {
    return left.current_servings > right.current_servings;
  }
  if (left.projected != right.projected) {
    return left.projected > right.projected;
  }
  if (left.ending_fuel != right.ending_fuel) {
    return left.ending_fuel > right.ending_fuel;
  }
  return left.hash < right.hash;
}

std::string mlns_config_fingerprint(const MapConfig& config) {
  std::uint64_t value = 0x4d4c4e535f535441ULL;
  auto add = [&](std::uint64_t item) { value = lns_mix(value ^ item); };
  add(static_cast<std::uint64_t>(config.width));
  add(static_cast<std::uint64_t>(config.height));
  add(static_cast<std::uint64_t>(config.fuel_limit));
  add(static_cast<std::uint64_t>(config.players));
  add(static_cast<std::uint64_t>(config.busy_threshold));
  add(static_cast<std::uint64_t>(config.jammed_threshold));
  for (int steps : config.day_steps) add(static_cast<std::uint64_t>(steps));
  for (Terrain terrain : config.cells) {
    add(static_cast<std::uint64_t>(terrain));
  }
  for (const auto& spot : config.spots) {
    add(static_cast<std::uint64_t>(spot.brand + 1));
    add(static_cast<std::uint64_t>(spot.pos + 1));
    add(static_cast<std::uint64_t>(spot.stocks));
  }
  for (int pos : config.agents) add(static_cast<std::uint64_t>(pos + 1));
  std::ostringstream stream;
  stream << std::hex << value;
  return stream.str();
}

ActionPlan mlns_parse_actions(const json::value& value) {
  ActionPlan result;
  for (const auto& row : value.as_array()) {
    std::vector<int> actions;
    for (const auto& item : row.as_array()) {
      actions.push_back(item.to_number<int>());
    }
    result.push_back(std::move(actions));
  }
  return result;
}

bool mlns_replay_matches(const DayInfo& day,
                         const MlnsReplaySignature& signature) {
  if (day.agents.size() != signature.agents.size() ||
      day.traffics != signature.traffics) {
    return false;
  }
  for (std::size_t index = 0; index < day.agents.size(); ++index) {
    const auto& actual = day.agents[index];
    const auto& expected = signature.agents[index];
    if (actual.kind != expected.kind || actual.pos != expected.pos ||
        actual.fuel != expected.fuel) {
      return false;
    }
  }
  return true;
}

std::optional<MlnsGenome> mlns_parse_state(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& types,
    const json::value* planner_state) {
  if (planner_state == nullptr || !planner_state->is_object() || day.day == 0 ||
      history.submitted_actions.empty()) {
    return std::nullopt;
  }
  try {
    const auto& root = planner_state->as_object();
    if (root.at("schema_version").to_number<int>() != 1 ||
        root.at("policy").as_string() != "mlns" ||
        root.at("config_fingerprint").as_string() !=
            mlns_config_fingerprint(config) ||
        root.at("source_day").to_number<int>() != day.day - 1 ||
        mlns_parse_actions(root.at("committed_actions")) !=
            history.submitted_actions.back()) {
      return std::nullopt;
    }
    const auto& state_types = root.at("types").as_array();
    if (state_types.size() != types.size()) return std::nullopt;
    for (std::size_t index = 0; index < types.size(); ++index) {
      if (state_types[index].to_number<int>() !=
          static_cast<int>(types[index])) {
        return std::nullopt;
      }
    }
    const auto& suffix = root.at("suffix").as_array();
    const std::size_t expected = config.day_steps.size() -
                                 static_cast<std::size_t>(day.day);
    if (suffix.size() != expected) return std::nullopt;
    MlnsGenome result;
    result.skeletons.reserve(suffix.size());
    result.travel.reserve(suffix.size());
    result.replay_plans.reserve(suffix.size());
    result.replay_signatures.reserve(suffix.size());
    for (std::size_t offset = 0; offset < suffix.size(); ++offset) {
      const auto& entry = suffix[offset].as_object();
      if (entry.at("day").to_number<int>() !=
          day.day + static_cast<int>(offset)) {
        return std::nullopt;
      }
      LnsSkeleton skeleton;
      for (const auto& route_value : entry.at("routes").as_array()) {
        std::vector<int> route;
        for (const auto& item : route_value.as_array()) {
          const int spot = item.to_number<int>();
          if (spot < 0 || spot >= static_cast<int>(config.spots.size())) {
            return std::nullopt;
          }
          route.push_back(spot);
        }
        skeleton.routes.push_back(std::move(route));
      }
      if (skeleton.routes.size() != types.size()) return std::nullopt;
      AlnsTravelChoices travel(skeleton.routes.size());
      if (const auto* encoded = entry.if_contains("travel")) {
        if (encoded->as_array().size() != skeleton.routes.size()) {
          return std::nullopt;
        }
        for (std::size_t agent = 0; agent < travel.size(); ++agent) {
          for (const auto& item : encoded->as_array()[agent].as_array()) {
            const auto choice = item.to_number<std::uint64_t>();
            if (choice > std::numeric_limits<std::uint32_t>::max()) {
              return std::nullopt;
            }
            travel[agent].push_back(static_cast<std::uint32_t>(choice));
          }
          if (travel[agent].size() != skeleton.routes[agent].size()) {
            return std::nullopt;
          }
        }
      } else {
        for (std::size_t agent = 0; agent < travel.size(); ++agent) {
          travel[agent].resize(skeleton.routes[agent].size(), 0U);
        }
      }
      result.skeletons.push_back(std::move(skeleton));
      result.travel.push_back(std::move(travel));
      const auto* encoded_plan = entry.if_contains("plan");
      const auto* encoded_agents = entry.if_contains("expected_agents");
      const auto* encoded_traffics = entry.if_contains("expected_traffics");
      if (encoded_plan != nullptr && encoded_agents != nullptr &&
          encoded_traffics != nullptr) {
        auto plan = mlns_parse_actions(*encoded_plan);
        if (plan.size() != types.size()) return std::nullopt;
        MlnsReplaySignature signature;
        for (const auto& value : encoded_agents->as_array()) {
          const auto& encoded_agent = value.as_object();
          const int kind = encoded_agent.at("kind").to_number<int>();
          if (kind < static_cast<int>(AgentKind::Patrol) ||
              kind > static_cast<int>(AgentKind::Refuel)) {
            return std::nullopt;
          }
          signature.agents.push_back(
              {static_cast<AgentKind>(kind),
               encoded_agent.at("pos").to_number<int>(),
               encoded_agent.at("fuel").to_number<int>()});
        }
        if (signature.agents.size() != types.size()) return std::nullopt;
        for (const auto& value : encoded_traffics->as_array()) {
          const auto& encoded_traffic = value.as_object();
          signature.traffics[encoded_traffic.at("pos").to_number<int>()] =
              encoded_traffic.at("status").to_number<int>();
        }
        result.replay_plans.push_back(std::move(plan));
        result.replay_signatures.push_back(std::move(signature));
      } else if (encoded_plan == nullptr && encoded_agents == nullptr &&
                 encoded_traffics == nullptr) {
        // Version-one states produced before exact replay hints remain valid.
        result.replay_plans.push_back(std::nullopt);
        result.replay_signatures.push_back(std::nullopt);
      } else {
        return std::nullopt;
      }
    }
    return result;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

json::object mlns_serialize_state(const MapConfig& config, const DayInfo& day,
                                  const AgentTypes& types,
                                  const MlnsSolution& solution) {
  json::array state_types;
  for (auto type : types) state_types.push_back(static_cast<int>(type));
  json::array suffix;
  for (std::size_t offset = 1; offset < solution.days.size(); ++offset) {
    json::array routes;
    for (const auto& route : solution.days[offset].skeleton.routes) {
      json::array values;
      for (int spot : route) values.push_back(spot);
      routes.push_back(std::move(values));
    }
    json::array travel;
    for (const auto& choices : solution.days[offset].travel) {
      json::array values;
      for (std::uint32_t choice : choices) values.push_back(choice);
      travel.push_back(std::move(values));
    }
    json::array expected_agents;
    for (const auto& agent : solution.days[offset].day_info.agents) {
      expected_agents.push_back(
          json::object{{"kind", static_cast<int>(agent.kind)},
                       {"pos", agent.pos},
                       {"fuel", agent.fuel}});
    }
    json::array expected_traffics;
    for (const auto& [pos, status] : solution.days[offset].day_info.traffics) {
      expected_traffics.push_back(
          json::object{{"pos", pos}, {"status", status}});
    }
    suffix.push_back(json::object{{"day", solution.days[offset].day},
                                  {"routes", std::move(routes)},
                                  {"travel", std::move(travel)},
                                  {"plan", to_json(solution.days[offset].plan)},
                                  {"expected_agents",
                                   std::move(expected_agents)},
                                  {"expected_traffics",
                                   std::move(expected_traffics)}});
  }
  return json::object{
      {"schema_version", 1},
      {"policy", "mlns"},
      {"config_fingerprint", mlns_config_fingerprint(config)},
      {"source_day", day.day},
      {"types", std::move(state_types)},
      {"committed_actions", to_json(solution.days.front().plan)},
      {"suffix", std::move(suffix)}};
}

std::size_t mlns_solution_hash(const std::vector<MlnsDaySolution>& days) {
  std::size_t seed = 0;
  for (const auto& day : days) {
    seed ^= lns_skeleton_hash(day.skeleton) + 0x9e3779b9U + (seed << 6U) +
            (seed >> 2U);
  }
  return seed;
}

MlnsRank mlns_build_rank(const MapConfig& config,
                         const PolicyHistory& root_history,
                         const AgentTypes& types,
                         const std::vector<MlnsDaySolution>& days,
                         int discount_percent) {
  MlnsRank result;
  const auto weights =
      mlns_weights(static_cast<int>(days.size()), discount_percent);
  for (std::size_t index = 0; index < days.size(); ++index) {
    for (std::size_t objective = 0; objective < result.weighted.size();
         ++objective) {
      result.weighted[objective] +=
          static_cast<unsigned>(days[index].reward[objective]) * weights[index];
    }
  }
  if (!days.empty()) {
    result.current_daily = days.front().reward[1];
    result.current_servings = days.front().reward[2];
    for (std::size_t agent = 0; agent < types.size(); ++agent) {
      if (types[agent] == AgentKind::Patrol) {
        result.current_ending_fuel +=
            days.front().evaluation.ending_fuel[agent];
      }
    }
    const auto& tail = days.back();
    result.projected = {
        static_cast<int>(tail.history_after.distinct_brands.size()),
        tail.history_after.cumulative_daily_types,
        tail.history_after.total_servings};
    for (std::size_t agent = 0; agent < types.size(); ++agent) {
      if (types[agent] == AgentKind::Patrol) {
        result.ending_fuel += tail.evaluation.ending_fuel[agent];
      }
    }
  } else {
    result.projected = {
        static_cast<int>(root_history.distinct_brands.size()),
        root_history.cumulative_daily_types, root_history.total_servings};
  }
  result.hash = mlns_solution_hash(days);
  (void)config;
  return result;
}

std::optional<MlnsSolution> mlns_evaluate(
    const MapConfig& config, const DayInfo& root_day,
    const PolicyHistory& root_history, const AgentTypes& types,
    std::vector<LnsSkeleton> skeletons, int discount_percent,
    const MlnsSolution* prefix_source = nullptr, std::size_t pivot = 0,
    bool coordinated_seed = false,
    const ActionPlan* root_seed = nullptr,
    const std::vector<AlnsTravelChoices>* travel_choices = nullptr,
    const MlnsGenome* replay_hints = nullptr,
    bool strong_suffix_seed = false,
    int strong_suffix_iterations = 0,
    bool reuse_downstream_plans = false,
    const std::optional<std::chrono::steady_clock::time_point>& deadline =
        std::nullopt) {
  const std::size_t count = config.day_steps.size() -
                            static_cast<std::size_t>(root_day.day);
  if (skeletons.size() != count || pivot > count) return std::nullopt;
  MlnsSolution result;
  PolicyHistory history = root_history;
  std::vector<std::map<int, int>> traffic_history =
      reconstruct_own_traffic(config, types, root_history);
  std::vector<int> positions;
  std::vector<int> fuel;
  if (prefix_source != nullptr && pivot > 0) {
    if (prefix_source->days.size() < pivot) return std::nullopt;
    result.days.insert(result.days.end(), prefix_source->days.begin(),
                       prefix_source->days.begin() +
                           static_cast<std::ptrdiff_t>(pivot));
    const auto& prefix = result.days.back();
    history = prefix.history_after;
    traffic_history = prefix.traffic_history_after;
    positions = prefix.evaluation.ending_positions;
    fuel = prefix.evaluation.ending_fuel;
  }

  for (std::size_t offset = pivot; offset < count; ++offset) {
    if (deadline && std::chrono::steady_clock::now() >= *deadline) {
      return std::nullopt;
    }
    DayInfo info;
    if (offset == 0) {
      info = root_day;
    } else {
      if (positions.empty()) {
        const auto& previous = result.days.back().evaluation;
        positions = previous.ending_positions;
        fuel = previous.ending_fuel;
      }
      info.day = root_day.day + static_cast<int>(offset);
      for (std::size_t agent = 0; agent < types.size(); ++agent) {
        info.agents.push_back({types[agent], positions[agent], fuel[agent]});
      }
      // Future opponent actions are unavailable. Treat our candidate traffic
      // as representative of every team, matching the existing symmetric
      // online surrogate while keeping today's server traffic authoritative.
      info.traffics = mlns_forecast_traffic(config, traffic_history, info.day);
    }

    ActionPlan plan;
    std::optional<CandidateEvaluation> evaluation;
    bool replayed_exactly = false;
    if (!coordinated_seed && replay_hints != nullptr &&
        offset < replay_hints->replay_plans.size() &&
        offset < replay_hints->replay_signatures.size() &&
        replay_hints->replay_plans[offset] &&
        replay_hints->replay_signatures[offset] &&
        mlns_replay_matches(info, *replay_hints->replay_signatures[offset])) {
      plan = *replay_hints->replay_plans[offset];
      evaluation = evaluate_candidate(config, info, history, plan);
      replayed_exactly = evaluation.has_value();
    }
    if (!evaluation && coordinated_seed && offset == 0 && root_seed != nullptr) {
      plan = *root_seed;
      evaluation = evaluate_candidate(config, info, history, plan);
    } else if (!evaluation && coordinated_seed && offset > 0 &&
               strong_suffix_seed) {
      // Construct the rolling horizon once with the same strong daily roots
      // that are available after traffic is revealed. Without this, MLNS
      // starts from a good day zero followed by near-idle coordinated plans,
      // and its small neighborhoods cannot discover a credible whole match.
      auto consider = [&](ActionPlan candidate) {
        if (candidate.empty()) return;
        if (auto candidate_evaluation =
                evaluate_candidate(config, info, history, candidate);
            candidate_evaluation &&
            (!evaluation ||
             alns_official_value(candidate_evaluation->value) >
                 alns_official_value(evaluation->value))) {
          plan = std::move(candidate);
          evaluation = std::move(candidate_evaluation);
        }
      };
      const auto forced = coordinated_first_targets(config, info, history, types);
      consider(build_routing_plan("coordinated", config, info, history, types,
                                  forced, {}));
      if (config.players == 1) {
        consider(build_escort_plan(config, info, history, types));
      }
      SearchLimits suffix_limits;
      if (deadline) {
        suffix_limits.time_limit_ms = std::max(
            1, static_cast<int>(std::chrono::duration_cast<
                                    std::chrono::milliseconds>(
                                    *deadline -
                                    std::chrono::steady_clock::now())
                                    .count()));
      }
      auto aco = build_aco_plan(config, info, history, types, true,
                                suffix_limits);
      consider(aco);
      if (strong_suffix_iterations > 0 &&
          (!deadline || std::chrono::steady_clock::now() < *deadline)) {
        SearchLimits alns_limits;
        alns_limits.min_iterations = deadline ? 0 : strong_suffix_iterations;
        alns_limits.max_iterations = strong_suffix_iterations;
        alns_limits.stagnation_iterations = 0;
        alns_limits.seed_iterations = std::min(32, strong_suffix_iterations);
        alns_limits.exact_nodes = 0;
        alns_limits.final_exact_nodes = 0;
        alns_limits.alns_restarts = 1;
        alns_limits.use_legacy_seed = false;
        alns_limits.use_local_search_seed = true;
        if (deadline) {
          const int remaining_ms = std::max(
              1, static_cast<int>(std::chrono::duration_cast<
                                      std::chrono::milliseconds>(
                                      *deadline -
                                      std::chrono::steady_clock::now())
                                      .count()));
          const int remaining_offsets =
              std::max(1, static_cast<int>(count - offset));
          alns_limits.time_limit_ms =
              std::max(1, remaining_ms / remaining_offsets);
        }
        consider(build_alns_plan(config, info, history, types, alns_limits,
                                 kProductionAlnsFeatures,
                                 /*allow_continuation=*/false));
      }
      if (deadline && std::chrono::steady_clock::now() >= *deadline) {
        return std::nullopt;
      }
      for (auto staged :
           refuel_staging_variants(config, info, types, history, aco)) {
        consider(std::move(staged));
      }
    } else if (!evaluation && !coordinated_seed) {
      // Every day at and after a mutated pivot is a coupled part of the
      // neighbor. Re-decode the genome against the positions, fuel, history,
      // and predicted roads produced by the preceding day. Replaying the old
      // ActionPlan here is cheaper, but it turns an upstream route/refuel
      // mutation into a stale suffix and prevents whole-match optimization.
      if (reuse_downstream_plans && prefix_source != nullptr &&
          offset > pivot && offset < prefix_source->days.size()) {
        plan = prefix_source->days[offset].plan;
        evaluation = evaluate_candidate(config, info, history, plan);
      }
      if (!evaluation) {
        auto graph = build_aco_graph(config, info);
        auto meetings = build_aco_meeting_cache(graph, std::size_t{6});
        const AlnsTravelChoices* day_travel =
            travel_choices != nullptr && offset < travel_choices->size()
                ? &(*travel_choices)[offset]
                : nullptr;
        if (auto decoded = decode_lns_skeleton(
                config, info, types, graph, meetings, skeletons[offset],
                day_travel, /*strict_travel=*/false)) {
          plan = std::move(*decoded);
          evaluation = evaluate_candidate(config, info, history, plan);
        }
      }
    }
    if (!evaluation) {
      const auto forced = coordinated_first_targets(config, info, history, types);
      plan = build_routing_plan("coordinated", config, info, history, types,
                                forced, {});
      evaluation = evaluate_candidate(config, info, history, plan);
    }
    if (!evaluation) return std::nullopt;
    if (offset == 0 && replayed_exactly) result.exact_replayed_root = true;

    const std::size_t before_brands = history.distinct_brands.size();
    PolicyHistory after = history;
    after.submitted_actions.push_back(plan);
    for (const auto& acquisition : evaluation->trace.acquisitions) {
      if (const Spot* spot = spot_at(config, acquisition.spot_pos)) {
        after.distinct_brands.insert(spot->brand);
      }
    }
    const int daily = std::get<1>(evaluation->value);
    const int servings = std::get<2>(evaluation->value);
    after.cumulative_daily_types += daily;
    after.total_servings += servings;
    traffic_history.push_back(evaluation->road_traffic);
    positions = evaluation->ending_positions;
    fuel = evaluation->ending_fuel;
    auto canonical = lns_skeleton_from_trace(
        config, info.agents.size(), evaluation->trace);
    AlnsTravelChoices canonical_travel(canonical.routes.size());
    if (travel_choices != nullptr && offset < travel_choices->size()) {
      canonical_travel = (*travel_choices)[offset];
    }
    for (std::size_t agent = 0; agent < canonical_travel.size(); ++agent) {
      canonical_travel[agent].resize(canonical.routes[agent].size(), 0U);
    }
    result.days.push_back(
        {info.day,
         std::move(canonical),
         std::move(canonical_travel),
         std::move(plan),
         *evaluation,
         std::move(info),
         history,
         after,
         traffic_history,
         {static_cast<int>(after.distinct_brands.size() - before_brands),
          daily, servings}});
    history = std::move(after);
  }
  result.rank = mlns_build_rank(config, root_history, types, result.days,
                                discount_percent);
  return result;
}

std::optional<MlnsSolution> mlns_beam_seed(
    const MapConfig& config, const DayInfo& root_day,
    const PolicyHistory& root_history, const AgentTypes& types,
    int discount_percent, int alns_iterations, bool use_elite_pool,
    const std::optional<std::chrono::steady_clock::time_point>& deadline =
        std::nullopt) {
  const std::size_t count = config.day_steps.size() -
                            static_cast<std::size_t>(root_day.day);
  if (count == 0) return std::nullopt;
  const std::size_t beam_width =
      use_elite_pool && count <= 3
          ? 8U
          : (alns_iterations > 0 ? 2U : (count <= 3 ? 8U : 4U));
  std::vector<MlnsSolution> beam(1);

  for (std::size_t offset = 0; offset < count; ++offset) {
    if (deadline && std::chrono::steady_clock::now() >= *deadline) {
      return std::nullopt;
    }
    auto batches = parallel_indexed(beam.size(), [&](std::size_t beam_index) {
      std::vector<MlnsSolution> local_expanded;
      if (deadline && std::chrono::steady_clock::now() >= *deadline) {
        return local_expanded;
      }
      const auto& prefix = beam[beam_index];
      PolicyHistory history = root_history;
      std::vector<std::map<int, int>> traffic_history =
          reconstruct_own_traffic(config, types, root_history);
      DayInfo info;
      if (prefix.days.empty()) {
        info = root_day;
      } else {
        const auto& previous = prefix.days.back();
        history = previous.history_after;
        traffic_history = previous.traffic_history_after;
        info.day = root_day.day + static_cast<int>(offset);
        for (std::size_t agent = 0; agent < types.size(); ++agent) {
          info.agents.push_back(
              {types[agent], previous.evaluation.ending_positions[agent],
               previous.evaluation.ending_fuel[agent]});
        }
        info.traffics = mlns_forecast_traffic(config, traffic_history, info.day);
      }

      std::vector<ActionPlan> plans;
      auto add_plan = [&](ActionPlan plan) {
        if (plan.empty() ||
            std::find(plans.begin(), plans.end(), plan) != plans.end()) {
          return;
        }
        plans.push_back(std::move(plan));
      };
      const auto forced = coordinated_first_targets(config, info, history, types);
      add_plan(build_routing_plan("coordinated", config, info, history, types,
                                  forced, {}));
      if (config.players == 1) {
        add_plan(build_escort_plan(config, info, history, types));
      }
      SearchLimits seed_limits;
      if (deadline) {
        seed_limits.time_limit_ms = std::max(
            1, static_cast<int>(std::chrono::duration_cast<
                                    std::chrono::milliseconds>(
                                    *deadline -
                                    std::chrono::steady_clock::now())
                                    .count()) /
                   std::max(1, static_cast<int>(count - offset)));
      }
      std::vector<ActionPlan> alns_elite_plans;
      auto proposals = parallel_indexed(
          alns_iterations > 0 ? std::size_t{3} : std::size_t{2},
          [&](std::size_t proposal) -> ActionPlan {
            if (proposal == 0) {
              return build_aco_plan(config, info, history, types, true,
                                    seed_limits);
            }
            if (proposal == 1) {
              return build_local_search_plan(config, info, history, types,
                                             seed_limits);
            }
            SearchLimits alns_limits;
            alns_limits.min_iterations = alns_iterations;
            alns_limits.max_iterations = alns_iterations;
            alns_limits.stagnation_iterations = 0;
            alns_limits.seed_iterations = std::min(32, alns_iterations);
            alns_limits.exact_nodes = 0;
            alns_limits.final_exact_nodes = 0;
            alns_limits.alns_restarts = 1;
            alns_limits.use_legacy_seed = false;
            alns_limits.use_local_search_seed = true;
            if (deadline) {
              alns_limits.time_limit_ms = seed_limits.time_limit_ms;
              alns_limits.min_iterations = 0;
            }
            return build_alns_plan(config, info, history, types, alns_limits,
                                   kProductionAlnsFeatures,
                                   /*allow_continuation=*/true,
                                   /*restart_salt=*/0,
                                   /*on_improve=*/nullptr,
                                   use_elite_pool && count - offset <= 3
                                       ? &alns_elite_plans
                                       : nullptr);
          });
      auto aco = std::move(proposals[0]);
      add_plan(aco);
      add_plan(std::move(proposals[1]));
      for (auto staged :
           refuel_staging_variants(config, info, types, history, aco)) {
        add_plan(std::move(staged));
      }
      if (alns_iterations > 0) {
        add_plan(std::move(proposals[2]));
        for (auto& elite_plan : alns_elite_plans) {
          add_plan(std::move(elite_plan));
        }
      }
      auto evaluations = parallel_indexed(
          plans.size(), [&](std::size_t plan_index) {
            return evaluate_candidate(config, info, history,
                                      plans[plan_index]);
          });
      for (std::size_t plan_index = 0; plan_index < plans.size(); ++plan_index) {
        auto& plan = plans[plan_index];
        auto& evaluation = evaluations[plan_index];
        if (!evaluation) continue;
        const std::size_t before_brands = history.distinct_brands.size();
        PolicyHistory after = history;
        after.submitted_actions.push_back(plan);
        for (const auto& acquisition : evaluation->trace.acquisitions) {
          if (const Spot* spot = spot_at(config, acquisition.spot_pos)) {
            after.distinct_brands.insert(spot->brand);
          }
        }
        const int daily = std::get<1>(evaluation->value);
        const int servings = std::get<2>(evaluation->value);
        after.cumulative_daily_types += daily;
        after.total_servings += servings;
        auto next_traffic = traffic_history;
        next_traffic.push_back(evaluation->road_traffic);
        auto skeleton = lns_skeleton_from_trace(
            config, info.agents.size(), evaluation->trace);
        AlnsTravelChoices travel(skeleton.routes.size());
        for (std::size_t agent = 0; agent < travel.size(); ++agent) {
          travel[agent].resize(skeleton.routes[agent].size(), 0U);
        }
        MlnsSolution candidate = prefix;
        candidate.days.push_back(
            {info.day,
             std::move(skeleton),
             std::move(travel),
             std::move(plan),
             *evaluation,
             info,
             history,
             after,
             std::move(next_traffic),
             {static_cast<int>(after.distinct_brands.size() - before_brands),
              daily, servings}});
        candidate.rank = mlns_build_rank(config, root_history, types,
                                         candidate.days, discount_percent);
        local_expanded.push_back(std::move(candidate));
      }
      return local_expanded;
    });
    std::vector<MlnsSolution> expanded;
    for (auto& batch : batches) {
      expanded.insert(expanded.end(),
                      std::make_move_iterator(batch.begin()),
                      std::make_move_iterator(batch.end()));
    }
    if (deadline && std::chrono::steady_clock::now() >= *deadline) {
      return std::nullopt;
    }
    if (expanded.empty()) return std::nullopt;
    std::sort(expanded.begin(), expanded.end(),
              [](const MlnsSolution& left, const MlnsSolution& right) {
                return mlns_rank_better(left.rank, right.rank);
              });
    // Preserve distinct continuation states even when their partial official
    // reward ties. A daily-only collapse here is precisely what loses useful
    // future refuel/position choices.
    std::vector<MlnsSolution> diverse;
    for (auto& candidate : expanded) {
      const auto& tail = candidate.days.back();
      const bool duplicate = std::any_of(
          diverse.begin(), diverse.end(), [&](const MlnsSolution& retained) {
            const auto& other = retained.days.back();
            if (use_elite_pool) {
              return tail.evaluation.ending_positions ==
                     other.evaluation.ending_positions;
            }
            return tail.evaluation.ending_positions ==
                       other.evaluation.ending_positions &&
                   tail.evaluation.ending_fuel == other.evaluation.ending_fuel &&
                   tail.evaluation.road_traffic == other.evaluation.road_traffic;
          });
      if (!duplicate) diverse.push_back(std::move(candidate));
      if (diverse.size() >= beam_width) break;
    }
    if (diverse.empty()) diverse.push_back(std::move(expanded.front()));
    beam = std::move(diverse);
  }
  return *std::max_element(
      beam.begin(), beam.end(),
      [](const MlnsSolution& left, const MlnsSolution& right) {
        return mlns_rank_better(right.rank, left.rank);
      });
}

long double mlns_rank_scalar(const MapConfig& config,
                             const MlnsRank& rank,
                             const std::vector<MlnsWide>& weights) {
  MlnsWide total_weight = 0;
  for (MlnsWide weight : weights) total_weight += weight;
  std::set<int> brands;
  int daily_servings = 0;
  for (const auto& spot : config.spots) {
    brands.insert(spot.brand);
    daily_servings += spot.stocks;
  }
  const long double daily_base =
      total_weight.convert_to<long double>() * brands.size() + 1.0L;
  const long double serving_base =
      total_weight.convert_to<long double>() * daily_servings + 1.0L;
  return ((rank.weighted[0].convert_to<long double>() * daily_base +
           rank.weighted[1].convert_to<long double>()) *
              serving_base +
          rank.weighted[2].convert_to<long double>());
}

}  // namespace

PlannerResult build_mlns_plan(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& types,
    const SearchLimits& limits, const json::value* planner_state,
    const ImprovementSink* on_improve) {
  const auto started = std::chrono::steady_clock::now();
  ++mlns_diagnostics.planner_calls;
  mlns_predicted_traffic = limits.use_traffic_gnn
                               ? limits.predicted_traffic
                               : std::vector<std::map<int, int>>{};
  const bool timed = limits.time_limit_ms >= 0;
  // A small explicit iteration cap is a reproducible search mode, not an
  // anytime request. Keep its richer whole-match neighborhoods while still
  // enforcing the wall deadline as a safety backstop.
  const bool anytime_search = timed && limits.max_iterations > 256;
  enum class NeighborhoodMode { Current, Refined, Disabled };
  const auto neighborhood_mode = [] {
    const char* value = std::getenv("HEXUDON_MLNS_NEIGHBORHOOD_MODE");
    if (value == nullptr || std::string_view(value) == "current") {
      return NeighborhoodMode::Current;
    }
    if (std::string_view(value) == "refined") {
      return NeighborhoodMode::Refined;
    }
    if (std::string_view(value) == "disabled") {
      return NeighborhoodMode::Disabled;
    }
    throw std::invalid_argument(
        "HEXUDON_MLNS_NEIGHBORHOOD_MODE must be current, refined, or disabled");
  }();
  const auto deadline =
      started + std::chrono::milliseconds(std::max(0, limits.time_limit_ms));
  auto expired = [&] {
    return timed && std::chrono::steady_clock::now() >= deadline;
  };
  const std::optional<std::chrono::steady_clock::time_point>
      evaluation_deadline = timed ? std::optional(deadline) : std::nullopt;
  const std::size_t remaining_days =
      config.day_steps.size() - static_cast<std::size_t>(day.day);
  std::vector<LnsSkeleton> empty(remaining_days);
  for (auto& skeleton : empty) skeleton.routes.resize(types.size());
  const auto cold_started = std::chrono::steady_clock::now();
  auto cold = mlns_evaluate(config, day, history, types, std::move(empty),
                            limits.future_discount_percent, nullptr, 0,
                            /*coordinated_seed=*/true);
  if (!cold) {
    throw std::logic_error("MLNS failed to construct a valid cold seed");
  }
  MlnsSolution best = *cold;
  mlns_profile_phase("cold_seed", cold_started, std::nullopt, best);
  std::string winning_component = "cold_seed";
  auto record_phase = [&](std::string_view name,
                          const MlnsProfileSnapshot& before,
                          std::chrono::steady_clock::time_point phase_started) {
    const auto after = mlns_profile_snapshot(best);
    mlns_profile_phase(name, phase_started, before, best);
    if (mlns_profile_committed_plan_changed(before, after)) {
      winning_component = name;
    }
  };
  // A timed Web search still needs a credible multi-day root.  Disabling the
  // strong suffix completely made the production path start from an ACO-only
  // continuation even when tens of seconds were available; the New Question
  // fixture then predicted its eventual 30/25 tail on day zero and never
  // escaped it.  Keep this seed deliberately small because it runs once per
  // remaining day before the deadline-governed neighborhood loop.
  const int strong_suffix_iterations =
      limits.max_iterations <= 0
          ? 0
          : (timed ? (limits.time_limit_ms < 1000
                          ? 0
                          : std::min(limits.time_limit_ms < 5000 ? 64 : 256,
                                     limits.max_iterations))
                   : std::min(96, limits.max_iterations));
  auto consider_root_seed = [&](const ActionPlan& root_seed,
                                bool strong_suffix = false,
                                std::optional<
                                    std::chrono::steady_clock::time_point>
                                    seed_deadline = std::nullopt) {
    if (root_seed.empty()) return;
    if (!seed_deadline) seed_deadline = evaluation_deadline;
    std::vector<LnsSkeleton> seed_skeletons(remaining_days);
    for (auto& skeleton : seed_skeletons) {
      skeleton.routes.resize(types.size());
    }
    if (auto candidate = mlns_evaluate(
            config, day, history, types, std::move(seed_skeletons),
            limits.future_discount_percent, nullptr, 0,
            /*coordinated_seed=*/true, &root_seed, nullptr, nullptr,
            strong_suffix, strong_suffix_iterations,
            /*reuse_downstream_plans=*/false, seed_deadline);
        candidate) {
      const auto candidate_current =
          alns_official_value(candidate->days.front().evaluation.value);
      const auto best_current =
          alns_official_value(best.days.front().evaluation.value);
      const auto candidate_coverage =
          std::tuple{std::get<0>(candidate_current),
                     std::get<1>(candidate_current)};
      const auto best_coverage =
          std::tuple{std::get<0>(best_current), std::get<1>(best_current)};
      if (candidate_coverage > best_coverage ||
          (candidate_coverage == best_coverage &&
           mlns_commit_better(candidate->rank, best.rank))) {
        best = std::move(*candidate);
      }
    }
  };
  // Test one transition from the actual incumbent before longer root searches
  // can consume the request. The variant generator is a no-op when staging is
  // irrelevant, and the ordinary whole-match comparator still protects the
  // current-day coverage floor.
  int maximum_patrol_fuel = 0;
  int patrol_count = 0;
  for (std::size_t agent = 0; agent < types.size(); ++agent) {
    if (types[agent] != AgentKind::Patrol) continue;
    ++patrol_count;
    maximum_patrol_fuel = std::max(
        maximum_patrol_fuel, best.days.front().evaluation.ending_fuel[agent]);
  }
  const bool fuel_starved_transition =
      patrol_count > 0 &&
      maximum_patrol_fuel <= std::max(1, config.fuel_limit / 5);
  if (!expired() && remaining_days > 1 && fuel_starved_transition) {
    const auto transition_started = std::chrono::steady_clock::now();
    const auto transition_deadline =
        timed ? std::min(deadline,
                         transition_started +
                             std::chrono::milliseconds(std::clamp(
                                 limits.time_limit_ms / 10, 250, 1000)))
              : std::chrono::steady_clock::time_point::max();
    for (const auto& staged : refuel_staging_variants(
             config, day, types, history, best.days.front().plan)) {
      consider_root_seed(staged, /*strong_suffix=*/false,
                         transition_deadline);
      break;
    }
  }
  const bool use_dp_proposals = lns_dp_proposals_enabled(limits);
  // Preserve a bounded production-ALNS current-day floor while leaving most of
  // the request available to the whole-match search.
  if (use_dp_proposals && anytime_search &&
      remaining_days > 2 &&
      limits.time_limit_ms >= 1000 && !expired()) {
    const auto profile_before = mlns_profile_snapshot(best);
    const auto profile_started = std::chrono::steady_clock::now();
    const auto floor_deadline =
        started + std::chrono::milliseconds(
                      std::max(1, limits.time_limit_ms * 60 / 100));
    SearchLimits current_limits = limits;
    current_limits.time_limit_ms =
        std::max(250, limits.time_limit_ms * 50 / 100);
    current_limits.min_iterations = 0;
    current_limits.max_iterations = std::max(1, limits.max_iterations);
    current_limits.stagnation_iterations = 0;
    current_limits.alns_restarts = 3;
    // This is the protected baseline chain. Its budget must remain comparable
    // to ordinary ALNS; DP proposals are evaluated separately by MLNS below.
    current_limits.use_lns_dp_proposals = false;
    // Do not spend the floor's slice on its own continuation look-ahead: MLNS
    // already searches the whole match around this seed, so an internal
    // continuation phase is redundant and (via alns_anytime_budget's 50/50
    // split) halves the floor's current-day search. On coverage-saturated maps
    // that lost servings the whole-match machinery cannot recover, because the
    // floor is the only component that optimizes the committed day's servings.
    // Running it current-day-only lifts its effective budget from ~25% to ~50%
    // of the request, matching standalone ALNS's current-day effort.
    auto current_seed = build_alns_plan(
        config, day, history, types, current_limits,
        kProductionAlnsFeatures, /*allow_continuation=*/false);
    consider_root_seed(current_seed,
        /*strong_suffix=*/false, floor_deadline);
    record_phase("protected_alns_floor", profile_before, profile_started);
  }
  // Near the end of the match there is no value in spending the complete
  // deadline rediscovering mature single-day routing moves through the generic
  // whole-match neighborhood.  Reserve one ALNS chain as a root proposal, then
  // rank it with the same MLNS suffix objective.  The terminal day receives a
  // larger share because its suffix is empty; on the penultimate day the
  // remainder stays available for coupled state search.
  if (!expired() && timed && remaining_days <= 2 &&
      limits.time_limit_ms >= 1000) {
    const auto profile_before = mlns_profile_snapshot(best);
    const auto profile_started = std::chrono::steady_clock::now();
    SearchLimits route_limits = limits;
    const int route_percent = remaining_days == 1 ? 90 : 50;
    route_limits.time_limit_ms =
        std::max(250, limits.time_limit_ms * route_percent / 100);
    route_limits.min_iterations = 1;
    route_limits.max_iterations =
        anytime_search ? std::max(1, limits.max_iterations)
                       : std::max(2048, limits.max_iterations);
    route_limits.stagnation_iterations = 0;
    route_limits.alns_restarts = 1;
    auto route_seed = build_alns_plan(
        config, day, history, types, route_limits, kProductionAlnsFeatures,
        /*allow_continuation=*/remaining_days > 1);
    consider_root_seed(route_seed, /*strong_suffix=*/remaining_days > 1);
    record_phase("route_alns_seed", profile_before, profile_started);
  }
  // Tight-fuel maps need an explicit mobile-refuel construction. Evaluate it
  // as another complete match root rather than embedding refuel-specific
  // surrogate rewards in the MLNS objective.
  if (!expired() && config.players == 1) {
    const auto profile_before = mlns_profile_snapshot(best);
    const auto profile_started = std::chrono::steady_clock::now();
    consider_root_seed(build_escort_plan(config, day, history, types));
    record_phase("escort_seed", profile_before, profile_started);
  }
  // STOP branch-and-price is valid on the paper-compatible regime where each
  // unit-stock spot is its own brand cluster. Use it only as a bounded root
  // proposal: MLNS remains responsible for replaying the complete suffix and
  // may reject a locally stronger route when its continuation is worse.
  if (stop_bp_proposals_enabled() && !expired() &&
      (!timed || limits.time_limit_ms >= 1000)) {
    std::set<int> stop_brands;
    for (const auto& spot : config.spots) stop_brands.insert(spot.brand);
    const bool stop_compatible =
        !config.spots.empty() && config.spots.size() <= 20 &&
        stop_brands.size() == config.spots.size() &&
        std::all_of(config.spots.begin(), config.spots.end(),
                    [](const Spot& spot) { return spot.stocks == 1; });
    if (stop_compatible) {
      const auto profile_before = mlns_profile_snapshot(best);
      const auto profile_started = std::chrono::steady_clock::now();
      SearchLimits bp_limits = limits;
      if (timed) {
        const int remaining_ms = std::max(
            0, static_cast<int>(std::chrono::duration_cast<
                                    std::chrono::milliseconds>(
                                    deadline -
                                    std::chrono::steady_clock::now())
                                    .count()));
        bp_limits.time_limit_ms =
            std::min(remaining_ms,
                     std::max(100, limits.time_limit_ms / 10));
      }
      bp_limits.exact_nodes =
          limits.exact_nodes > 0 ? limits.exact_nodes : 50;
      if ((!timed || bp_limits.time_limit_ms >= 100) &&
          !best.days.empty()) {
        if (auto proposal = build_stop_bp_proposal(
                config, day, history, types, best.days.front().plan,
                bp_limits, /*allow_official_tie=*/true)) {
          consider_root_seed(*proposal,
                             /*strong_suffix=*/remaining_days <= 2);
        }
      }
      record_phase("stop_bp_seed", profile_before, profile_started);
    }
  }
  // Add one DP request-bank route as another whole-match root. MLNS still
  // replays it through its suffix decoder, so it cannot bypass continuation
  // state validation.
  if (use_dp_proposals && !expired() &&
      (!timed || limits.time_limit_ms >= 1000)) {
    const auto profile_before = mlns_profile_snapshot(best);
    const auto profile_started = std::chrono::steady_clock::now();
    SearchLimits dp_limits = limits;
    if (anytime_search) {
      const int remaining_ms = std::max(
          1, static_cast<int>(std::chrono::duration_cast<
                                  std::chrono::milliseconds>(
                                  deadline - std::chrono::steady_clock::now())
                                  .count()));
      // Calibrate the proposal inside one third of MLNS's remaining window;
      // the rest is retained for suffix evaluation and neighborhood search.
      dp_limits.time_limit_ms = std::max(1, remaining_ms / 3);
    }
    dp_limits.min_iterations = 0;
    // Keep the proposal bank compact: on long-horizon maps an oversized DP
    // root can win the current-day comparison while producing a weaker match
    // continuation. The three construction modes still run concurrently.
    dp_limits.max_iterations = std::min(8, std::max(1, limits.max_iterations));
    dp_limits.stagnation_iterations = 0;
    dp_limits.random_seed ^= 0x4d4c4e535f4450ULL;
    for (const auto& proposal : build_lns_dp_route_proposals(
             config, day, history, types, dp_limits, 1)) {
      if (expired()) break;
      consider_root_seed(proposal.plan, /*strong_suffix=*/remaining_days <= 2);
    }
    record_phase("dp_seed", profile_before, profile_started);
  }
  // One ACO+LS construction gives the rolling match search a competitive root
  // and initializes its suffix without nesting another optimizer inside every
  // neighborhood candidate. Keep it out of sub-second requests, where its
  // all-pairs preprocessing would consume the entire response window.
  if (!expired() && limits.use_aco_seed &&
      (!timed || limits.time_limit_ms >= 1000)) {
    const auto profile_before = mlns_profile_snapshot(best);
    const auto profile_started = std::chrono::steady_clock::now();
    SearchLimits seed_limits;
    seed_limits.aco_evaporation = limits.aco_evaporation;
    if (anytime_search) {
      seed_limits.time_limit_ms = std::max(
          1, static_cast<int>(std::chrono::duration_cast<
                                  std::chrono::milliseconds>(
                                  deadline - std::chrono::steady_clock::now())
                                  .count()));
    }
    auto root_seed =
        build_aco_plan(config, day, history, types, true, seed_limits);
    consider_root_seed(root_seed, /*strong_suffix=*/true);
    if (auto evaluation = evaluate_candidate(
            config, day, history, best.days.front().plan)) {
      for (const auto& staged : refuel_staging_variants(
               config, day, types, history, best.days.front().plan)) {
        consider_root_seed(staged);
      }
    }
    if (strong_suffix_iterations > 0 && !expired()) {
      SearchLimits alns_limits;
      alns_limits.min_iterations =
          anytime_search ? 0 : strong_suffix_iterations;
      alns_limits.max_iterations = strong_suffix_iterations;
      alns_limits.stagnation_iterations = 0;
      alns_limits.seed_iterations = std::min(32, strong_suffix_iterations);
      alns_limits.exact_nodes = 0;
      alns_limits.final_exact_nodes = 0;
      alns_limits.alns_restarts = 1;
      alns_limits.use_legacy_seed = false;
      alns_limits.use_local_search_seed = true;
      if (anytime_search) {
        alns_limits.time_limit_ms = std::max(
            1, static_cast<int>(std::chrono::duration_cast<
                                    std::chrono::milliseconds>(
                                    deadline -
                                    std::chrono::steady_clock::now())
                                    .count()));
      }
      consider_root_seed(
          build_alns_plan(config, day, history, types, alns_limits,
                          kProductionAlnsFeatures,
                          /*allow_continuation=*/false),
          /*strong_suffix=*/true);
    }
    record_phase("aco_seed", profile_before, profile_started);
  }
  // Untimed evaluation keeps its proven ALNS-backed beam.  Timed Web search
  // uses inexpensive constructive proposals over a long horizon, then enables
  // the stronger ALNS-backed expansion for the tightly coupled final days.
  if (!expired() && ((!timed && strong_suffix_iterations > 0) ||
                     (timed && limits.time_limit_ms >= 5000))) {
    const auto profile_before = mlns_profile_snapshot(best);
    const auto profile_started = std::chrono::steady_clock::now();
    const int beam_alns_iterations =
        anytime_search && remaining_days > 3
            ? 0
            : strong_suffix_iterations;
    if (auto beam_seed = mlns_beam_seed(
            config, day, history, types, limits.future_discount_percent,
            beam_alns_iterations,
            /*use_elite_pool=*/anytime_search,
            anytime_search ? evaluation_deadline : std::nullopt);
        beam_seed && mlns_commit_better(beam_seed->rank, best.rank) &&
        (!anytime_search || mlns_reward_better(beam_seed->rank, best.rank))) {
      best = std::move(*beam_seed);
    }
    record_phase("beam_seed", profile_before, profile_started);
  }
  if (!expired()) {
    const auto profile_before = mlns_profile_snapshot(best);
    const auto profile_started = std::chrono::steady_clock::now();
    if (auto warm_skeletons =
            mlns_parse_state(config, day, history, types, planner_state)) {
      if (auto warm = mlns_evaluate(
            config, day, history, types,
            std::move(warm_skeletons->skeletons),
            limits.future_discount_percent, nullptr, 0, false, nullptr,
            &warm_skeletons->travel, &*warm_skeletons, false, 0, false,
            evaluation_deadline);
          warm) {
        const auto warm_current =
            alns_official_value(warm->days.front().evaluation.value);
        const auto fresh_current =
            alns_official_value(best.days.front().evaluation.value);
        const auto warm_coverage =
            std::tuple{std::get<0>(warm_current), std::get<1>(warm_current)};
        const auto fresh_coverage =
            std::tuple{std::get<0>(fresh_current), std::get<1>(fresh_current)};
        const bool trusted_warm =
            config.players == 1 || warm_coverage >= fresh_coverage;
        if (trusted_warm &&
            mlns_commit_better(warm->rank, best.rank)) {
          best = std::move(*warm);
        }
      }
    }
    record_phase("warm_start", profile_before, profile_started);
  }
  MlnsSolution current = best;
  auto emit = [&] {
    if (on_improve == nullptr || best.days.empty()) return;
    const auto official = alns_official_value(best.days.front().evaluation.value);
    IncumbentRank rank;
    rank.available = true;
    rank.objective_mode = "mlns";
    rank.patrol_fuel = std::get<3>(best.days.front().evaluation.value);
    rank.predicted_final_available = true;
    rank.predicted_final = {
        std::get<0>(best.rank.projected), std::get<1>(best.rank.projected),
        std::get<2>(best.rank.projected)};
    rank.predicted_ending_patrol_fuel = best.rank.ending_fuel;
    rank.future_discount_percent = limits.future_discount_percent;
    for (std::size_t index = 0; index < rank.weighted_match.size(); ++index) {
      rank.weighted_match[index] = mlns_wide_string(best.rank.weighted[index]);
    }
    (*on_improve)(best.days.front().plan,
                  Score{std::get<0>(official), std::get<1>(official),
                        std::get<2>(official)},
                  rank);
  };
  emit();

  std::mt19937_64 random(lns_seed(config, day, history) ^
                         limits.random_seed ^ 0x4d4c4e535f4c4e53ULL);
  std::uniform_real_distribution<long double> unit(0.0L, 1.0L);
  const auto weights = mlns_weights(static_cast<int>(remaining_days),
                                    limits.future_discount_percent);
  int maximum_daily_servings = 0;
  for (const auto& spot : config.spots) maximum_daily_servings += spot.stocks;
  std::vector<std::uint64_t> pivot_weights;
  std::uint64_t pivot_total = 0;
  for (MlnsWide weight : weights) {
    const auto narrowed = weight.convert_to<std::uint64_t>();
    pivot_weights.push_back(narrowed);
    pivot_total += narrowed;
  }
  std::vector<long double> observed_losses;
  int stagnation = 0;
  const int maximum = std::max(0, limits.max_iterations);
  const int minimum = std::min(limits.min_iterations, maximum);
  const auto neighborhood_before = mlns_profile_snapshot(best);
  const auto neighborhood_started = std::chrono::steady_clock::now();
  // The production profiler found that the generic neighborhood consumed over
  // one third of MLNS wall time while changing only one incumbent in ten. The
  // refined mode keeps a short exploitation tail for the useful coupled moves
  // but no longer lets it consume every millisecond left by stronger seeds.
  const auto refined_neighborhood_deadline =
      timed ? std::min(deadline,
                       neighborhood_started + std::chrono::milliseconds(
                           std::max(250, limits.time_limit_ms * 15 / 100)))
            : std::chrono::steady_clock::time_point::max();
  auto neighborhood_expired = [&] {
    if (neighborhood_mode == NeighborhoodMode::Disabled) return true;
    if (expired()) return true;
    return neighborhood_mode == NeighborhoodMode::Refined && timed &&
           std::chrono::steady_clock::now() >= refined_neighborhood_deadline;
  };
  for (int iteration = 0; iteration < maximum; ++iteration) {
    if (neighborhood_expired()) break;
    std::size_t pivot = 0;
    if (iteration > 0) {
      std::uint64_t draw = pivot_total == 0 ? 0 : random() % pivot_total;
      while (pivot + 1 < pivot_weights.size() &&
             draw >= pivot_weights[pivot]) {
        draw -= pivot_weights[pivot++];
      }
      // Discounted sampling intentionally emphasizes the revealed day, but it
      // used to spend less than three percent of a seven-day run on the final
      // two days even when the incumbent explicitly forecast missing stock
      // there.  Half of the moves now target the first deficient day or its
      // predecessor, where a changed ending position can still repair that
      // deficit.  A path that already serves all stock keeps the ordinary
      // discounted distribution.
      const unsigned deficiency_focus =
          neighborhood_mode == NeighborhoodMode::Refined ? 80U : 50U;
      if (anytime_search && random() % 100U < deficiency_focus) {
        auto deficient = std::find_if(
            current.days.begin(), current.days.end(),
            [&](const MlnsDaySolution& item) {
              return item.reward[2] < maximum_daily_servings;
            });
        if (deficient != current.days.end()) {
          pivot = static_cast<std::size_t>(
              std::distance(current.days.begin(), deficient));
          if (pivot > 0 && random() % 100U < 65U) --pivot;
        }
      }
    }
    std::vector<LnsSkeleton> skeletons;
    std::vector<AlnsTravelChoices> travel;
    skeletons.reserve(current.days.size());
    travel.reserve(current.days.size());
    for (const auto& item : current.days) {
      skeletons.push_back(item.skeleton);
      travel.push_back(item.travel);
    }
    // Most moves remain one-day LNS moves, but periodically mutate a
    // contiguous two/three-day block. Refuel rendezvous and patrol routes are
    // state-coupled across day boundaries, so a single-day mutation cannot
    // discover many useful whole-match transitions.
    const std::size_t available_span = current.days.size() - pivot;
    const unsigned coupled_percent =
        neighborhood_mode == NeighborhoodMode::Refined
            ? 30U
            : (anytime_search ? 20U : 35U);
    const std::size_t span =
        available_span <= 1 || random() % 100U >= coupled_percent
            ? 1U
            : std::min<std::size_t>(
                  available_span,
                  2U +
                      static_cast<std::size_t>(!anytime_search && random() % 2U));
    for (std::size_t changed = pivot; changed < pivot + span; ++changed) {
      auto& candidate_skeleton = skeletons[changed];
      const auto& changed_day = current.days[changed];
      auto graph = build_aco_graph(config, changed_day.day_info);
      int visits = 0;
      for (const auto& route : candidate_skeleton.routes) {
        visits += static_cast<int>(route.size());
      }
      if (visits > 0) {
        const int fraction = 15 + static_cast<int>(random() % 31U);
        const int removed = std::max(1, visits * fraction / 100);
        destroy_lns_skeleton(config, changed_day.history_before, graph,
                             candidate_skeleton,
                             static_cast<int>(random() % 3U), removed, random);
      }
      repair_lns_skeleton(config, changed_day.day_info,
                          changed_day.history_before, types, graph,
                          candidate_skeleton,
                          static_cast<int>(random() % 2U), random);
      AlnsSolution travel_candidate{candidate_skeleton, travel[changed]};
      repair_alns_travel(travel_candidate,
                         1 + static_cast<int>(random() % 3U), random,
                         /*stable=*/true);
      candidate_skeleton = std::move(travel_candidate.skeleton);
      travel[changed] = std::move(travel_candidate.travel);
    }
    auto replay_skeletons = skeletons;
    auto replay_travel = travel;
    std::optional<MlnsSolution> candidate;
    if (!anytime_search) {
      auto variants = parallel_indexed(
          std::size_t{2}, [&](std::size_t variant) {
            if (variant == 0) {
              return mlns_evaluate(
                  config, day, history, types, std::move(skeletons),
                  limits.future_discount_percent, &current, pivot, false,
                  nullptr, &travel, nullptr, false, 0,
                  /*reuse_downstream_plans=*/false, evaluation_deadline);
            }
            return mlns_evaluate(
                config, day, history, types, std::move(replay_skeletons),
                limits.future_discount_percent, &current, pivot, false,
                nullptr, &replay_travel, nullptr, false, 0,
                /*reuse_downstream_plans=*/true, evaluation_deadline);
          });
      candidate = std::move(variants[0]);
      if (variants[1] &&
          (!candidate ||
           mlns_rank_better(variants[1]->rank, candidate->rank))) {
        candidate = std::move(variants[1]);
      }
    } else {
      // Most timed moves retain the cheap exact downstream replay.  Coupled
      // moves, and a periodic one-day probe, must instead re-decode the whole
      // suffix against the newly produced positions, fuel, and traffic; using
      // the old ActionPlans here turns a whole-match neighbor back into a
      // single-day search with stale future routes.
      const bool rebuild_suffix =
          span > 1 || (neighborhood_mode == NeighborhoodMode::Current &&
                       iteration % 16 == 0);
      candidate = mlns_evaluate(
          config, day, history, types, std::move(replay_skeletons),
          limits.future_discount_percent, &current, pivot, false, nullptr,
          &replay_travel, nullptr, false, 0,
          /*reuse_downstream_plans=*/!rebuild_suffix,
          evaluation_deadline);
    }
    if (!candidate) {
      ++stagnation;
      continue;
    }
    const auto candidate_current =
        alns_official_value(candidate->days.front().evaluation.value);
    const auto incumbent_current =
        alns_official_value(current.days.front().evaluation.value);
    const auto candidate_coverage =
        std::tuple{std::get<0>(candidate_current),
                   std::get<1>(candidate_current)};
    const auto incumbent_coverage =
        std::tuple{std::get<0>(incumbent_current),
                   std::get<1>(incumbent_current)};
    // Never trade away banked distinct or daily-type coverage. Servings are a
    // match-wide sum, however, so MLNS may move one serving to a later day
    // when the complete projected rank improves.
    const bool trusted_transition = config.players == 1 || pivot > 0 ||
                                    candidate_coverage >= incumbent_coverage;
    const bool improves_current =
        trusted_transition &&
        (mlns_rank_better(candidate->rank, current.rank) ||
         mlns_rank_equal(candidate->rank, current.rank));
    // In refined mode, forecast/fuel-only replacements are not reliable enough
    // to justify changing the live trajectory when the mutation touches today.
    // A future-only pivot preserves today's plan exactly, however, so it may
    // still improve the serialized continuation without risking the submitted
    // action. Requiring a current-day gain for pivot > 0 rejects every such
    // candidate by construction and silently turns MLNS into daily-only LNS.
    const bool refined_transition_gain =
        pivot > 0 ? mlns_rank_better(candidate->rank, current.rank)
                  : candidate_current > incumbent_current;
    bool accepted =
        neighborhood_mode == NeighborhoodMode::Refined
            ? trusted_transition && refined_transition_gain
            : improves_current;
    const long double current_scalar =
        mlns_rank_scalar(config, current.rank, weights);
    const long double candidate_scalar =
        mlns_rank_scalar(config, candidate->rank, weights);
    if (neighborhood_mode == NeighborhoodMode::Current && !accepted &&
        candidate_scalar < current_scalar) {
      const long double loss = current_scalar - candidate_scalar;
      if (observed_losses.size() < 12U) observed_losses.push_back(loss);
      long double probability = 0.05L;
      if (observed_losses.size() >= 3U) {
        auto sorted = observed_losses;
        std::sort(sorted.begin(), sorted.end());
        const long double median = sorted[sorted.size() / 2U];
        const long double initial = std::max(1.0L, -median / std::log(0.5L));
        const long double final = std::max(0.1L, -median / std::log(0.01L));
        long double progress = maximum <= 1
                                   ? 1.0L
                                   : static_cast<long double>(iteration) /
                                         static_cast<long double>(maximum - 1);
        if (anytime_search && limits.time_limit_ms > 0) {
          const auto elapsed =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - started)
                  .count();
          progress = std::clamp(
              static_cast<long double>(elapsed) / limits.time_limit_ms, 0.0L,
              1.0L);
        }
        const long double temperature =
            initial * std::pow(final / initial, progress);
        probability = std::exp(-loss / temperature);
      }
      accepted = trusted_transition && unit(random) < probability;
    }
    const auto best_current =
        alns_official_value(best.days.front().evaluation.value);
    const auto best_coverage =
        std::tuple{std::get<0>(best_current), std::get<1>(best_current)};
    const bool trusted_best = config.players == 1 || pivot > 0 ||
                              candidate_coverage >= best_coverage;
    const bool preserves_protected_current =
        candidate_coverage >= best_coverage;
    const bool refined_best_gain =
        pivot > 0 ? mlns_rank_better(candidate->rank, best.rank)
                  : candidate_current > best_current;
    if (trusted_best && preserves_protected_current &&
        mlns_commit_better(candidate->rank, best.rank) &&
        (neighborhood_mode != NeighborhoodMode::Refined ||
         refined_best_gain)) {
      best = *candidate;
      stagnation = 0;
      emit();
    } else {
      ++stagnation;
    }
    if (accepted) current = std::move(*candidate);
    if (neighborhood_mode == NeighborhoodMode::Refined && stagnation >= 64 &&
        iteration + 1 >= std::min(32, minimum)) {
      break;
    }
    if (iteration + 1 >= minimum && limits.stagnation_iterations > 0 &&
        stagnation >= limits.stagnation_iterations) {
      break;
    }
  }
  record_phase("neighborhood_search", neighborhood_before,
               neighborhood_started);
  const auto state_started = std::chrono::steady_clock::now();
  auto serialized_state = mlns_serialize_state(config, day, types, best);
  mlns_profile_phase("state_serialization", state_started,
                     mlns_profile_snapshot(best), best);
  ++mlns_profile_component(winning_component).final_selections;
  mlns_diagnostics.elapsed_microseconds += mlns_elapsed_microseconds(started);
  return {best.days.front().plan, std::move(serialized_state)};
}

LnsSkeleton build_rollout_skeleton(const MapConfig& config,
                                   const DayInfo& day,
                                   const PolicyHistory& history,
                                   const AgentTypes& types,
                                   const AcoGraph& graph) {
  LnsSkeleton skeleton;
  skeleton.routes.resize(types.size());
  std::vector<int> assigned(config.spots.size());
  std::set<int> planned_brands;
  const int horizon = config.day_steps[day.day];
  while (true) {
    struct Choice {
      std::size_t agent{};
      int spot{};
      int tier{};
      int remaining_stock{};
      int added{};
    };
    std::optional<Choice> choice;
    for (std::size_t agent = 0; agent < types.size(); ++agent) {
      if (types[agent] != AgentKind::Patrol) continue;
      const auto& route = skeleton.routes[agent];
      const int current_time =
          lns_route_time(config, day, graph, agent, route);
      const int from = route.empty()
                           ? graph.node_for_pos.at(day.agents[agent].pos)
                           : graph.node_for_pos.at(
                                 config.spots[route.back()].pos);
      for (std::size_t spot = 0; spot < config.spots.size(); ++spot) {
        if (assigned[spot] >= config.spots[spot].stocks ||
            std::find(route.begin(), route.end(), static_cast<int>(spot)) !=
                route.end()) {
          continue;
        }
        const int target = graph.node_for_pos.at(config.spots[spot].pos);
        int added = lns_path_time(graph, from, target);
        if (from == target) added = 1;
        if (current_time + added > horizon) continue;
        const int brand = config.spots[spot].brand;
        const int tier = !history.distinct_brands.contains(brand) &&
                                 !planned_brands.contains(brand)
                             ? 3
                             : (!planned_brands.contains(brand) ? 2 : 1);
        Choice candidate{agent, static_cast<int>(spot), tier,
                         config.spots[spot].stocks - assigned[spot], added};
        const bool better =
            !choice || candidate.tier > choice->tier ||
            (candidate.tier == choice->tier &&
             (candidate.remaining_stock > choice->remaining_stock ||
              (candidate.remaining_stock == choice->remaining_stock &&
               std::tie(candidate.added, candidate.agent, candidate.spot) <
                   std::tie(choice->added, choice->agent, choice->spot))));
        if (better) {
          choice = candidate;
        }
      }
    }
    if (!choice) break;
    skeleton.routes[choice->agent].push_back(choice->spot);
    ++assigned[choice->spot];
    planned_brands.insert(config.spots[choice->spot].brand);
  }
  return skeleton;
}

struct TypeRolloutResult {
  std::tuple<int, int, int> score;
  int residual_fuel{};
};

TypeRolloutResult run_type_rollout(
    const MapConfig& config, const AgentTypes& types,
    const AcoGraph& smooth_graph,
    const std::vector<AcoMeetingList>& smooth_meetings,
    const AcoGraph& jam_graph,
    const std::vector<AcoMeetingList>& jam_meetings, bool conservative) {
  TeamState team;
  team.id = "type-rollout";
  team.visited_today.resize(types.size());
  for (std::size_t index = 0; index < types.size(); ++index) {
    team.agents.push_back(
        {types[index], config.agents[index], config.fuel_limit});
  }
  int cumulative_daily = 0;
  for (int day_index = 0;
       day_index < static_cast<int>(config.day_steps.size()); ++day_index) {
    DayInfo day;
    day.day = day_index;
    for (const auto& agent : team.agents) {
      day.agents.push_back({agent.kind, agent.pos, agent.fuel});
    }
    const bool jammed = conservative && day_index > 0;
    for (int pos = 0; pos < config.width * config.height; ++pos) {
      if (config.cells[pos] == Terrain::Road) {
        day.traffics[pos] = jammed ? 2 : 0;
      }
    }
    team.stock.clear();
    team.daily_types.clear();
    for (auto& visited : team.visited_today) visited.clear();
    for (const auto& spot : config.spots) team.stock[spot.pos] = spot.stocks;

    const auto& graph = jammed ? jam_graph : smooth_graph;
    const auto& meetings = jammed ? jam_meetings : smooth_meetings;
    auto skeleton =
        build_rollout_skeleton(config, day, team.history, types, graph);
    auto plan = decode_lns_skeleton(config, day, types, graph, meetings,
                                    skeleton);
    ActionPlan actions =
        plan ? std::move(*plan)
             : wait_plan(types.size(), config.day_steps[day_index]);
    std::map<int, int> traffic;
    if (auto error =
            simulate_team_day(config, team, actions, day.traffics, traffic)) {
      throw std::logic_error("type rollout produced invalid plan: " + *error);
    }
    team.history.submitted_actions.push_back(actions);
    team.history.distinct_brands = team.distinct_types;
    cumulative_daily += static_cast<int>(team.daily_types.size());
  }
  int residual_fuel = 0;
  for (const auto& agent : team.agents) {
    if (agent.kind == AgentKind::Patrol) residual_fuel += agent.fuel;
  }
  return {{static_cast<int>(team.distinct_types.size()), cumulative_daily,
           team.total_servings},
          residual_fuel};
}

AgentTypes select_remote_refuel_agent_types(const MapConfig& config,
                                            int refuelers) {
  AgentTypes types(config.agents.size(), AgentKind::Patrol);
  std::map<int, int> smooth;
  std::vector<std::pair<int, int>> ranked;
  for (std::size_t index = 0; index < config.agents.size(); ++index) {
    int nearest = std::numeric_limits<int>::max() / 4;
    for (const auto& spot : config.spots) {
      nearest =
          std::min(nearest,
                   shortest_path(config, config.agents[index], spot.pos, smooth)
                       .cost);
    }
    ranked.emplace_back(nearest, static_cast<int>(index));
  }
  std::sort(ranked.begin(), ranked.end(), std::greater<>());
  for (int index = 0;
       index < std::min<int>(refuelers, static_cast<int>(ranked.size()));
       ++index) {
    types[static_cast<std::size_t>(ranked[index].second)] = AgentKind::Refuel;
  }
  return types;
}

AgentTypes select_one_refuel_agent_types(const MapConfig& config) {
  return select_remote_refuel_agent_types(config, 1);
}

AgentTypes select_lns_agent_types(const MapConfig& config) {
  const std::size_t count = config.agents.size();
  constexpr int rollout_cycles = 7;
  // Keep the synthetic 1x/2x/4x stress horizons for robustness, but also
  // evaluate the schedule that the server actually supplies in the initial
  // map configuration. The latter captures asymmetric multi-day fuel
  // pressure that the synthetic cycles can miss.
  std::array<MapConfig, 4> rollout_configs{config, config, config, config};
  const int map_span = std::max(1, config.width + config.height);
  for (std::size_t index = 0; index < 3; ++index) {
    const int horizon = map_span * (1 << index);
    rollout_configs[index].day_steps.assign(rollout_cycles, horizon);
    rollout_configs[index].day_seconds.assign(rollout_cycles, 60.0);
  }
  DayInfo smooth_day;
  smooth_day.day = 0;
  for (int pos : config.agents) {
    smooth_day.agents.push_back({AgentKind::Patrol, pos, config.fuel_limit});
  }
  DayInfo jam_day = smooth_day;
  for (int pos = 0; pos < config.width * config.height; ++pos) {
    if (config.cells[pos] == Terrain::Road) {
      smooth_day.traffics[pos] = 0;
      jam_day.traffics[pos] = 2;
    }
  }
  const auto smooth_graph = build_aco_graph(config, smooth_day);
  const auto smooth_meetings = build_aco_meeting_cache(smooth_graph);
  const auto jam_graph = build_aco_graph(config, jam_day);
  const auto jam_meetings = build_aco_meeting_cache(jam_graph);

  std::vector<unsigned> masks;
  const unsigned limit = 1U << count;
  const int minimum_refuelers =
      count <= 1
          ? 0
          : (config.players > 1 && count <= 6 &&
                     config.width * config.height > 300 &&
                     config.fuel_limit < map_span
                 ? 2
                 : (config.fuel_limit * 2 >= map_span * 21 ? 0 : 1));
  for (unsigned mask = 0; mask + 1 < limit; ++mask) {
    const int refuelers = std::popcount(mask);
    if (refuelers >= minimum_refuelers && refuelers <= 3) {
      masks.push_back(mask);
    }
  }
  struct RankedTypes {
    AgentTypes types;
    std::tuple<int, int, int> worst;
    std::tuple<int, int, int> combined;
    std::tuple<int, int, int> scheduled;
    int residual{};
    int patrols{};
    unsigned mask{};
  };
  const auto ranked = parallel_indexed(masks.size(), [&](std::size_t index) {
    const unsigned mask = masks[index];
    AgentTypes types(count, AgentKind::Patrol);
    for (std::size_t agent = 0; agent < count; ++agent) {
      if ((mask & (1U << agent)) != 0) types[agent] = AgentKind::Refuel;
    }
    std::tuple<int, int, int> worst{
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max()};
    std::tuple<int, int, int> combined{};
    std::tuple<int, int, int> scheduled{
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max()};
    int residual = 0;
    for (std::size_t rollout_index = 0;
         rollout_index < rollout_configs.size(); ++rollout_index) {
      const auto& rollout_config = rollout_configs[rollout_index];
      for (bool conservative : {false, true}) {
        const auto outcome = run_type_rollout(
            rollout_config, types, smooth_graph, smooth_meetings, jam_graph,
            jam_meetings, conservative);
        if (rollout_index < 3) {
          worst = std::min(worst, outcome.score);
          std::get<0>(combined) += std::get<0>(outcome.score);
          std::get<1>(combined) += std::get<1>(outcome.score);
          std::get<2>(combined) += std::get<2>(outcome.score);
        } else {
          scheduled = std::min(scheduled, outcome.score);
        }
        residual += outcome.residual_fuel;
      }
    }
    return RankedTypes{std::move(types), worst, combined, scheduled, residual,
                       static_cast<int>(count - std::popcount(mask)), mask};
  });
  auto better = [](const RankedTypes& left, const RankedTypes& right) {
    if (left.worst != right.worst) return left.worst < right.worst;
    if (left.combined != right.combined) {
      return left.combined < right.combined;
    }
    if (left.residual != right.residual) {
      return left.residual < right.residual;
    }
    if (left.patrols != right.patrols) {
      return left.patrols < right.patrols;
    }
    if (left.scheduled != right.scheduled) {
      return left.scheduled < right.scheduled;
    }
    return left.mask > right.mask;
  };
  const auto selected = std::max_element(ranked.begin(), ranked.end(), better);
  const int selected_refuelers =
      static_cast<int>(count) - selected->patrols;
  if (selected_refuelers == 1 ||
      (count <= 3 && selected_refuelers > 1)) {
    return select_one_refuel_agent_types(config);
  }
  int total_stock = 0;
  for (const auto& spot : config.spots) total_stock += spot.stocks;
  const bool multi_refuel_pressure =
      total_stock > 2 * config.fuel_limit ||
      (config.spots.size() >= 15 && config.fuel_limit < 50) ||
      (config.width * config.height > 500 &&
       total_stock > config.fuel_limit);
  if (minimum_refuelers <= 1 && selected_refuelers > 1 &&
      !multi_refuel_pressure) {
    return select_one_refuel_agent_types(config);
  }
  if (config.players > 1 && count <= 6 &&
      config.width * config.height <= 300 && selected_refuelers == 1) {
    return select_one_refuel_agent_types(config);
  }
  return selected->types;
}
}  // namespace hexudon
