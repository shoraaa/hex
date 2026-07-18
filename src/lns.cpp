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
#include <iomanip>
#include <iterator>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace hexudon {
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

[[maybe_unused]] int lns_path_congestion(const MapConfig& config,
                                         const DayInfo& day,
                                         const AcoGraph& graph, int from,
                                         int to) {
  if (from == to) return 0;
  const auto& paths = graph.paths[from][to];
  if (paths.empty()) return std::numeric_limits<int>::max() / 4;
  int pos = graph.nodes[from];
  int result = 0;
  for (int direction : paths.front().directions) {
    if (config.cells[pos] == Terrain::Road) {
      auto iterator = day.traffics.find(pos);
      result += 1 + (iterator == day.traffics.end() ? 0 : iterator->second);
    }
    pos = *neighbor(config, pos, direction);
  }
  return result;
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
    auto density_distance = [&](int spot, std::size_t neighbours) {
      std::vector<int> distances;
      distances.reserve(config.spots.size());
      const int node = graph.node_for_pos.at(config.spots[spot].pos);
      for (std::size_t other = 0; other < config.spots.size(); ++other) {
        if (other == static_cast<std::size_t>(spot)) continue;
        const int other_node =
            graph.node_for_pos.at(config.spots[other].pos);
        distances.push_back(lns_path_time(graph, node, other_node));
      }
      std::sort(distances.begin(), distances.end());
      int total = 0;
      for (std::size_t index = 0;
           index < std::min(neighbours, distances.size()); ++index) {
        total += distances[index];
      }
      return total;
    };
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
        const std::size_t neighbours = sisr_sort == 3 ? 2U : 5U;
        const int left_density = density_distance(left.spot, neighbours);
        const int right_density = density_distance(right.spot, neighbours);
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

struct CoverageState {
  int time{std::numeric_limits<int>::max() / 4};
  int previous{-1};
};

[[maybe_unused]] std::vector<LnsSkeleton> build_coverage_skeletons(
    const MapConfig& config, const DayInfo& day, const AgentTypes& types,
    const AcoGraph& graph) {
  std::vector<std::size_t> patrols;
  for (std::size_t agent = 0; agent < types.size(); ++agent) {
    if (types[agent] == AgentKind::Patrol) patrols.push_back(agent);
  }
  if (patrols.empty() || config.spots.size() > 14) return {};
  const int horizon = config.day_steps[day.day];
  const std::size_t masks = std::size_t{1} << config.spots.size();
  const int infinity = std::numeric_limits<int>::max() / 4;
  std::vector<LnsSkeleton> result;
  if (patrols.size() == 3) {
    std::vector<std::vector<std::vector<CoverageState>>> all_dp;
    std::vector<std::vector<int>> mask_time;
    std::vector<std::vector<int>> mask_last;
    for (std::size_t agent : patrols) {
      std::vector<std::vector<CoverageState>> dp(
          masks, std::vector<CoverageState>(config.spots.size()));
      const int start = graph.node_for_pos.at(day.agents[agent].pos);
      for (std::size_t spot = 0; spot < config.spots.size(); ++spot) {
        const int target = graph.node_for_pos.at(config.spots[spot].pos);
        const int added =
            start == target ? 1 : lns_path_time(graph, start, target);
        if (added <= horizon) dp[std::size_t{1} << spot][spot] = {added, -1};
      }
      for (std::size_t mask = 1; mask < masks; ++mask) {
        for (std::size_t last = 0; last < config.spots.size(); ++last) {
          if (dp[mask][last].time == infinity) continue;
          const int from =
              graph.node_for_pos.at(config.spots[last].pos);
          for (std::size_t spot = 0; spot < config.spots.size(); ++spot) {
            const std::size_t bit = std::size_t{1} << spot;
            if ((mask & bit) != 0) continue;
            const int target =
                graph.node_for_pos.at(config.spots[spot].pos);
            const int candidate =
                dp[mask][last].time + lns_path_time(graph, from, target);
            auto& next = dp[mask | bit][spot];
            if (candidate <= horizon && candidate < next.time) {
              next = {candidate, static_cast<int>(last)};
            }
          }
        }
      }
      std::vector<int> times(masks, infinity);
      std::vector<int> lasts(masks, -1);
      times[0] = 0;
      for (std::size_t mask = 1; mask < masks; ++mask) {
        for (std::size_t last = 0; last < config.spots.size(); ++last) {
          if (dp[mask][last].time < times[mask]) {
            times[mask] = dp[mask][last].time;
            lasts[mask] = static_cast<int>(last);
          }
        }
      }
      all_dp.push_back(std::move(dp));
      mask_time.push_back(std::move(times));
      mask_last.push_back(std::move(lasts));
    }
    std::vector<std::size_t> third_mask(masks);
    for (std::size_t wanted = 0; wanted < masks; ++wanted) {
      int best_gain = -1;
      int best_time = infinity;
      for (std::size_t mask = 0; mask < masks; ++mask) {
        if (mask_time[2][mask] == infinity) continue;
        const int gain = std::popcount(mask & wanted);
        if (std::tuple{gain, -mask_time[2][mask]} >
            std::tuple{best_gain, -best_time}) {
          best_gain = gain;
          best_time = mask_time[2][mask];
          third_mask[wanted] = mask;
        }
      }
    }
    std::array<std::size_t, 3> selected{};
    int best_servings = -1;
    int best_time = infinity;
    for (std::size_t first = 0; first < masks; ++first) {
      if (mask_time[0][first] == infinity) continue;
      for (std::size_t second = 0; second < masks; ++second) {
        if (mask_time[1][second] == infinity) continue;
        int base = 0;
        std::size_t wanted = 0;
        for (std::size_t spot = 0; spot < config.spots.size(); ++spot) {
          const int assigned = static_cast<int>((first >> spot) & 1U) +
                               static_cast<int>((second >> spot) & 1U);
          base += std::min(config.spots[spot].stocks, assigned);
          if (assigned < config.spots[spot].stocks) {
            wanted |= std::size_t{1} << spot;
          }
        }
        const std::size_t third = third_mask[wanted];
        const int servings = base + std::popcount(third & wanted);
        const int time = mask_time[0][first] + mask_time[1][second] +
                         mask_time[2][third];
        if (std::tuple{servings, -time} >
            std::tuple{best_servings, -best_time}) {
          selected = {first, second, third};
          best_servings = servings;
          best_time = time;
        }
      }
    }
    LnsSkeleton joint;
    joint.routes.resize(types.size());
    for (std::size_t index = 0; index < patrols.size(); ++index) {
      std::size_t mask = selected[index];
      int last = mask_last[index][mask];
      std::vector<int> reverse_route;
      while (last >= 0) {
        reverse_route.push_back(last);
        const int previous =
            all_dp[index][mask][static_cast<std::size_t>(last)].previous;
        mask &= ~(std::size_t{1} << static_cast<std::size_t>(last));
        last = previous;
      }
      joint.routes[patrols[index]].assign(reverse_route.rbegin(),
                                          reverse_route.rend());
    }
    result.push_back(std::move(joint));
  }
  std::sort(patrols.begin(), patrols.end());
  do {
    LnsSkeleton skeleton;
    skeleton.routes.resize(types.size());
    std::vector<int> remaining;
    for (const auto& spot : config.spots) remaining.push_back(spot.stocks);
    for (std::size_t agent : patrols) {
      std::vector<std::vector<CoverageState>> dp(
          masks, std::vector<CoverageState>(config.spots.size()));
      const int start = graph.node_for_pos.at(day.agents[agent].pos);
      for (std::size_t spot = 0; spot < config.spots.size(); ++spot) {
        if (remaining[spot] <= 0) continue;
        const int target = graph.node_for_pos.at(config.spots[spot].pos);
        const int added = start == target ? 1 : lns_path_time(graph, start, target);
        if (added <= horizon) dp[std::size_t{1} << spot][spot] = {added, -1};
      }
      for (std::size_t mask = 1; mask < masks; ++mask) {
        for (std::size_t last = 0; last < config.spots.size(); ++last) {
          if (dp[mask][last].time == infinity) continue;
          const int from =
              graph.node_for_pos.at(config.spots[last].pos);
          for (std::size_t spot = 0; spot < config.spots.size(); ++spot) {
            const std::size_t bit = std::size_t{1} << spot;
            if ((mask & bit) != 0 || remaining[spot] <= 0) continue;
            const int target =
                graph.node_for_pos.at(config.spots[spot].pos);
            const int added = lns_path_time(graph, from, target);
            const int candidate = dp[mask][last].time + added;
            auto& next = dp[mask | bit][spot];
            if (candidate <= horizon && candidate < next.time) {
              next = {candidate, static_cast<int>(last)};
            }
          }
        }
      }
      std::size_t best_mask = 0;
      int best_last = -1;
      int best_count = -1;
      int best_time = infinity;
      for (std::size_t mask = 1; mask < masks; ++mask) {
        const int count = std::popcount(mask);
        for (std::size_t last = 0; last < config.spots.size(); ++last) {
          const int time = dp[mask][last].time;
          if (time == infinity) continue;
          if (std::tuple{count, -time} >
              std::tuple{best_count, -best_time}) {
            best_mask = mask;
            best_last = static_cast<int>(last);
            best_count = count;
            best_time = time;
          }
        }
      }
      std::vector<int> reverse_route;
      std::size_t mask = best_mask;
      int last = best_last;
      while (last >= 0) {
        reverse_route.push_back(last);
        const int previous = dp[mask][static_cast<std::size_t>(last)].previous;
        mask &= ~(std::size_t{1} << static_cast<std::size_t>(last));
        last = previous;
      }
      auto& route = skeleton.routes[agent];
      route.assign(reverse_route.rbegin(), reverse_route.rend());
      for (int spot : route) --remaining[static_cast<std::size_t>(spot)];
    }
    result.push_back(std::move(skeleton));
  } while (std::next_permutation(patrols.begin(), patrols.end()));
  return result;
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

std::optional<LnsSkeleton> lns_skeleton_from_plan(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const ActionPlan& plan) {
  auto evaluated = evaluate_candidate(config, day, history, plan);
  if (!evaluated) return std::nullopt;
  std::map<int, int> spot_index;
  for (std::size_t index = 0; index < config.spots.size(); ++index) {
    spot_index[config.spots[index].pos] = static_cast<int>(index);
  }
  LnsSkeleton skeleton;
  skeleton.routes.resize(day.agents.size());
  for (const auto& event : evaluated->trace.acquisitions) {
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
    auto value = candidate_value(config, day, history, seed);
    auto skeleton = lns_skeleton_from_plan(config, day, history, seed);
    if (!value || !skeleton) continue;
    if (*value > best_value) {
      best = seed;
      best_value = *value;
    }
    const auto hash = lns_skeleton_hash(*skeleton);
    if (std::none_of(elite.begin(), elite.end(),
                     [&](const Elite& item) { return item.hash == hash; })) {
      elite.push_back({std::move(*skeleton), seed, *value, hash});
    }
  }
  std::sort(elite.begin(), elite.end(),
            [](const Elite& left, const Elite& right) {
              if (left.value != right.value) return left.value > right.value;
              return left.hash < right.hash;
            });
  if (elite.size() > 8) elite.resize(8);
  if (elite.empty()) return best;

  std::mt19937_64 random(lns_seed(config, day, history));
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

[[maybe_unused]] void exact_repair_alns_solution(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& types,
    const AcoGraph& graph, const std::vector<AcoMeetingList>& meeting_cache,
    AlnsSolution& solution, std::mt19937_64& random, bool stable_travel) {
  auto best_plan = decode_lns_skeleton(config, day, types, graph,
                                       meeting_cache, solution.skeleton,
                                       &solution.travel);
  auto best_value = best_plan
                        ? candidate_value(config, day, history, *best_plan)
                        : std::optional<CandidateValue>{};
  AlnsSolution best = solution;
  for (int attempt = 0; attempt < 3; ++attempt) {
    std::vector<std::pair<std::size_t, std::size_t>> visits;
    std::vector<std::size_t> patrols;
    for (std::size_t agent = 0; agent < solution.skeleton.routes.size();
         ++agent) {
      if (types[agent] != AgentKind::Patrol) continue;
      patrols.push_back(agent);
      for (std::size_t position = 0;
           position < solution.skeleton.routes[agent].size(); ++position) {
        visits.emplace_back(agent, position);
      }
    }
    if (visits.empty() || patrols.empty()) break;
    AlnsSolution trial = solution;
    const auto [from_agent, from_position] =
        visits[static_cast<std::size_t>(random() % visits.size())];
    const int spot = trial.skeleton.routes[from_agent][from_position];
    const std::size_t to_agent =
        patrols[static_cast<std::size_t>(random() % patrols.size())];
    if (to_agent != from_agent &&
        std::find(trial.skeleton.routes[to_agent].begin(),
                  trial.skeleton.routes[to_agent].end(), spot) !=
            trial.skeleton.routes[to_agent].end()) {
      continue;
    }
    trial.skeleton.routes[from_agent].erase(
        trial.skeleton.routes[from_agent].begin() +
        static_cast<std::ptrdiff_t>(from_position));
    auto& destination = trial.skeleton.routes[to_agent];
    const std::size_t position =
        static_cast<std::size_t>(random() % (destination.size() + 1));
    destination.insert(destination.begin() +
                           static_cast<std::ptrdiff_t>(position),
                       spot);
    repair_alns_travel(trial, 0, random, stable_travel);
    auto plan = decode_lns_skeleton(config, day, types, graph, meeting_cache,
                                    trial.skeleton, &trial.travel);
    auto value = plan ? candidate_value(config, day, history, *plan)
                      : std::optional<CandidateValue>{};
    if (value && (!best_value || *value > *best_value)) {
      best = std::move(trial);
      best_value = value;
    }
  }
  solution = std::move(best);
}

[[maybe_unused]] std::tuple<int, int, int> alns_terminal_value(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& types,
    const CandidateEvaluation& current) {
  const auto current_official = alns_official_value(current.value);
  if (day.day + 1 >= static_cast<int>(config.day_steps.size())) {
    return current_official;
  }
  PolicyHistory next_history = history;
  for (const auto& acquisition : current.trace.acquisitions) {
    if (const Spot* spot = spot_at(config, acquisition.spot_pos)) {
      next_history.distinct_brands.insert(spot->brand);
    }
  }
  auto rollout = [&](int road_status) -> std::optional<CandidateValue> {
    DayInfo next;
    next.day = day.day + 1;
    for (std::size_t agent = 0; agent < types.size(); ++agent) {
      next.agents.push_back({types[agent], current.ending_positions[agent],
                             current.ending_fuel[agent]});
    }
    for (int pos = 0; pos < config.width * config.height; ++pos) {
      if (config.cells[pos] == Terrain::Road) next.traffics[pos] = road_status;
    }
    const auto forced =
        coordinated_first_targets(config, next, next_history, types);
    const ActionPlan plan = build_routing_plan(
        "coordinated", config, next, next_history, types, forced, {});
    return candidate_value(config, next, next_history, plan);
  };
  auto smooth = rollout(0);
  auto jammed = rollout(2);
  if (!smooth || !jammed) return current_official;
  const CandidateValue conservative = std::min(*smooth, *jammed);
  return {std::get<0>(conservative),
          std::get<1>(current.value) + std::get<1>(conservative),
          std::get<2>(current.value) + std::get<2>(conservative)};
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

struct AlnsMatchScore {
  std::tuple<int, int, int> worst{};
  std::tuple<int, int, int> total{};
};

[[maybe_unused]] AlnsMatchScore alns_match_rollout(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& types,
    const CandidateEvaluation& root, int beam_width,
    const std::chrono::steady_clock::time_point& deadline) {
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
    initial.traffic_history = reconstruct_own_traffic(config, types, history);
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
        for (const auto& plan : plans) {
          if (std::chrono::steady_clock::now() >= deadline) break;
          auto evaluation =
              evaluate_candidate(config, next, next_history, plan);
          if (!evaluation) continue;
          AlnsRolloutState child = state;
          child.positions = evaluation->ending_positions;
          child.fuel = evaluation->ending_fuel;
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
        return std::tuple{left.distinct_brands.size(), left.cumulative_daily,
                          left.servings, left_fuel} >
               std::tuple{right.distinct_brands.size(), right.cumulative_daily,
                          right.servings, right_fuel};
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
          return std::tuple{left.distinct_brands.size(), left.cumulative_daily,
                            left.servings} <
                 std::tuple{right.distinct_brands.size(), right.cumulative_daily,
                            right.servings};
        });
    return std::tuple{static_cast<int>(best.distinct_brands.size()),
                      best.cumulative_daily, best.servings};
  };

  std::vector<std::tuple<int, int, int>> scores;
  if (config.players == 1) {
    scores.push_back(rollout_scenario(0));
  } else {
    for (int scenario = 0; scenario < 3; ++scenario) {
      scores.push_back(rollout_scenario(scenario));
    }
  }
  auto worst = *std::min_element(scores.begin(), scores.end());
  std::tuple<int, int, int> total{};
  for (const auto& score : scores) {
    std::get<0>(total) += std::get<0>(score);
    std::get<1>(total) += std::get<1>(score);
    std::get<2>(total) += std::get<2>(score);
  }
  return {worst, total};
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

ActionPlan build_alns_plan(const MapConfig& config, const DayInfo& day,
                           const PolicyHistory& history,
                           const AgentTypes& types,
                            const SearchLimits& limits,
                            unsigned features,
                            bool allow_continuation,
                            std::uint64_t restart_salt,
                            const ImprovementSink* on_improve) {
  const auto started = std::chrono::steady_clock::now();
  const bool timed = limits.time_limit_ms >= 0;
  const auto deadline =
      started + std::chrono::milliseconds(std::max(0, limits.time_limit_ms));
  const bool final_day =
      day.day + 1 >= static_cast<int>(config.day_steps.size());
  const bool timed_exact =
      timed && limits.time_limit_ms >= 5000 && final_day &&
      (features & AlnsExactCompletion) != 0U;
  const auto alns_deadline =
      timed_exact
          ? started + std::chrono::milliseconds(
                          std::max(0, limits.time_limit_ms * 7 / 10))
          : deadline;
  auto expired = [&] {
    return timed && std::chrono::steady_clock::now() >= alns_deadline;
  };

  const auto forced = coordinated_first_targets(config, day, history, types);
  std::vector<ActionPlan> seeds;
  seeds.push_back(build_routing_plan("coordinated", config, day, history,
                                     types, forced, limits));
  for (const std::string policy : {"greedy", "utility_greedy", "fuel_aware",
                                   "stock_maximiser"}) {
    if (expired()) break;
    seeds.push_back(
        build_routing_plan(policy, config, day, history, types, {}, limits));
  }
  if (!expired() && limits.use_aco_seed &&
      (features & AlnsAcoSeed) != 0U &&
      (!timed || limits.time_limit_ms >= 50)) {
    SearchLimits aco_limits = limits;
    if (timed && limits.time_limit_ms < 1000) {
      if (aco_limits.aco_ants <= 0) {
        aco_limits.aco_ants = limits.time_limit_ms < 250 ? 4 : 8;
      }
      if (aco_limits.aco_iterations <= 0) {
        aco_limits.aco_iterations = limits.time_limit_ms < 250 ? 4 : 8;
      }
    }
    seeds.insert(
        seeds.begin(),
        build_aco_plan(config, day, history, types, true, aco_limits));
  }
  if (!expired() && (!timed || limits.time_limit_ms >= 250) &&
      (features & AlnsSharedPreprocessing) == 0U) {
    SearchLimits legacy_limits = limits;
    if (timed) {
      legacy_limits.time_limit_ms =
          std::max(25, std::min(1000, limits.time_limit_ms / 5));
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
  if (!expired() && limits.use_local_search_seed &&
      (!timed || limits.time_limit_ms >= 500)) {
    seeds.push_back(build_local_search_plan(config, day, history, types, limits));
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
  const auto own_traffic_history =
      config.players == 1
          ? reconstruct_own_traffic(config, types, history)
          : std::vector<std::map<int, int>>{};
  auto congestion_value = [&](const CandidateEvaluation& evaluation) {
    // The live single-team practice game derives future traffic entirely from
    // our own actions. Multi-team benchmark matches include external traffic,
    // so self-congestion is not a reliable tie-break there.
    if (config.players != 1) return std::tuple{0, 0, 0, 0, 0, 0};
    const bool exact_rolling_day =
        day.day + 2 == static_cast<int>(config.day_steps.size());
    if (!exact_rolling_day) {
      int jammed_pressure = 0;
      int busy_pressure = 0;
      int squared_traffic = 0;
      int total_traffic = 0;
      for (const auto& [pos, traffic] : evaluation.road_traffic) {
        (void)pos;
        jammed_pressure +=
            std::max(0, traffic - config.jammed_threshold + 1);
        busy_pressure +=
            std::max(0, traffic - config.busy_threshold + 1);
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
      const int current = evaluation.road_traffic.contains(pos)
                              ? evaluation.road_traffic.at(pos)
                              : 0;
      const int prior = previous && previous->contains(pos)
                            ? previous->at(pos)
                            : 0;
      const int traffic = current + prior;
      jammed_roads += traffic >= config.jammed_threshold;
      busy_roads += traffic >= config.busy_threshold;
      jammed_pressure +=
          std::max(0, traffic - config.jammed_threshold + 1);
      busy_pressure +=
          std::max(0, traffic - config.busy_threshold + 1);
      squared_traffic += traffic * traffic;
      total_traffic += traffic;
    }
    return std::tuple{-jammed_roads, -busy_roads, -jammed_pressure,
                      -busy_pressure, -squared_traffic, -total_traffic};
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
  if (expired()) return best;

  AcoGraph graph = build_aco_graph(config, day);
  if (expired()) return best;
  std::optional<std::vector<AcoMeetingList>> shared_meeting_cache;
  if (limits.use_legacy_seed && limits.seed_iterations > 0 &&
      (features & AlnsSharedPreprocessing) != 0U &&
      (!timed || limits.time_limit_ms >= 250)) {
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
      legacy_limits.time_limit_ms =
          std::max(25, std::min(1000, limits.time_limit_ms / 5));
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

  struct Elite {
    AlnsSolution solution;
    ActionPlan plan;
    CandidateValue value;
    CandidateEvaluation evaluation;
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
    auto skeleton = lns_skeleton_from_plan(config, day, history, seed);
    if (!evaluation || !skeleton) continue;
    AlnsSolution solution{std::move(*skeleton), {}};
    repair_alns_travel(solution, 0, zero_random);
    const auto hash = alns_solution_hash(solution);
    if (std::none_of(elite.begin(), elite.end(),
                     [&](const Elite& item) { return item.hash == hash; })) {
      elite.push_back({std::move(solution), seed, evaluation->value,
                       *evaluation, hash});
    }
  }
  if (elite.empty()) return best;
  auto elite_order = [&](const Elite& left, const Elite& right) {
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
  std::sort(elite.begin(), elite.end(), elite_order);
  const std::size_t elite_limit = 12U;
  if (elite.size() > elite_limit) elite.resize(elite_limit);
  std::mt19937_64 random(lns_seed(config, day, history) ^ 0x414c4e53ULL ^
                         restart_salt);
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
  const int configured_exact_nodes =
      final_day && limits.final_exact_nodes >= 0
          ? limits.final_exact_nodes
          : limits.exact_nodes;
  const bool exact_requested =
      (features & AlnsExactCompletion) != 0U &&
      configured_exact_nodes > 0 && ((!timed && maximum > 96) || timed_exact);
  // ALNS iterations and exact-search nodes are independent, literal controls.
  // Neither phase silently borrows a percentage of the other one's allowance.
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
    (*on_improve)(best, Score{std::get<0>(official), std::get<1>(official),
                              std::get<2>(official)});
  };
  emit_improvement();

  for (int iteration = 0; iteration < alns_maximum; ++iteration) {
    if (expired()) break;
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
    const auto current_ordinal =
        alns_ordinal(current.value, config, types);
    const auto current_official = alns_official_value(current.value);
    if (value) {
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
      if (timed && limits.time_limit_ms > 0) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
        progress = std::clamp(
            static_cast<long double>(elapsed) / limits.time_limit_ms, 0.0L,
            1.0L);
      }
      const long double temperature =
          std::max({final_temperature, reheat_temperature,
                    initial_temperature *
                        std::pow(final_temperature / initial_temperature,
                                 progress)});
      if (*value >= current.value) {
        accepted = true;
      } else if (observed_losses.size() >= 3) {
        const long double probability = std::exp(
            static_cast<long double>(candidate_ordinal - current_ordinal) /
            temperature);
        accepted = unit(random) < probability;
      }

      const bool global_improvement =
          online_improves(*evaluation, best_evaluation);
      const auto hash = alns_solution_hash(candidate);
      auto duplicate = std::find_if(
          elite.begin(), elite.end(),
          [&](const Elite& item) { return item.hash == hash; });
      const bool new_elite = duplicate == elite.end();
      if (global_improvement) {
        best = *plan;
        best_value = *value;
        best_evaluation = *evaluation;
        stagnation = 0;
        ++trace_global_improvements;
        trace_last_improvement = iteration;
        emit_improvement();
      } else {
        ++stagnation;
      }
      if (accepted) ++trace_accepted;
      if (new_elite) {
        elite.push_back(
            {candidate, *plan, *value, *evaluation, hash});
      } else if (*value > duplicate->value) {
        *duplicate = {candidate, *plan, *value, *evaluation, hash};
      }
      std::sort(elite.begin(), elite.end(), elite_order);
      if (elite.size() > elite_limit) elite.resize(elite_limit);
      if (accepted) {
        current = {std::move(candidate), std::move(*plan), *value,
                   *evaluation, hash};
      }
      // Keep rewards bounded so UCB remains numerically stable across maps
      // with very different score magnitudes.  Positive local improvements,
      // new elite states, and global improvements receive progressively larger
      // rewards; invalid candidates receive zero below.
      const auto candidate_official = alns_official_value(*value);
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
  if (allow_continuation && config.players == 1 &&
      day.day + 2 == static_cast<int>(config.day_steps.size()) &&
      (!timed || std::chrono::steady_clock::now() < deadline) &&
      elite.size() > 1) {
    SearchLimits projection_limits = limits;
    projection_limits.time_limit_ms = -1;
    projection_limits.min_iterations = 64;
    projection_limits.max_iterations = 64;
    projection_limits.stagnation_iterations = 64;
    projection_limits.seed_iterations = 64;
    projection_limits.final_alns_iterations = -1;
    projection_limits.exact_nodes = 0;
    projection_limits.final_exact_nodes = -1;
    struct ContinuationProjection {
      std::tuple<int, int, int> match;
      int ending_fuel{};
    };
    const auto& prior_traffic = own_traffic_history;
    auto project_remaining_match = [&](const ActionPlan& current_plan,
                                       const CandidateEvaluation& current_eval)
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
        DayInfo next;
        next.day = next_day;
        for (std::size_t agent = 0; agent < types.size(); ++agent) {
          next.agents.push_back(
              {types[agent], positions[agent], fuel[agent]});
        }
        next.traffics = road_status_for_day(config, traffic_history, 1);
        auto projected_plan = build_alns_plan(
            config, next, next_history, types, projection_limits, features,
            false, restart_salt);
        auto evaluation =
            evaluate_candidate(config, next, next_history, projected_plan);
        if (!evaluation) return std::nullopt;
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
    auto best_projection = project_remaining_match(best, best_evaluation);
    if (best_projection) {
      auto best_rank =
          continuation_rank(best_evaluation, *best_projection);
      for (const auto& candidate : elite) {
        if (timed && std::chrono::steady_clock::now() >= deadline) break;
        auto projection =
            project_remaining_match(candidate.plan, candidate.evaluation);
        if (!projection) continue;
        const auto rank =
            continuation_rank(candidate.evaluation, *projection);
        if (rank > best_rank) {
          best = candidate.plan;
          best_value = candidate.value;
          best_evaluation = candidate.evaluation;
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
        auto projection =
            project_remaining_match(candidate_plan, *evaluation);
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
  auto plans = parallel_alns_restarts(
      restart_limits.size(), [&](std::size_t index) {
        const unsigned restart_features =
            index == 2 ? features | AlnsSisrRecreate : features;
        return build_alns_plan(config, day, history, types,
                               restart_limits[index], restart_features, true,
                               restart_salts[index]);
      });

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
    if (!best_evaluation ||
        (alns_official_value(evaluation->value) >
             alns_official_value(best_evaluation->value) &&
         (final_day ||
          continuation_equivalent(*evaluation, *best_evaluation)))) {
      best = std::move(plans[index]);
      best_evaluation = std::move(evaluation);
    }
  }
  return best;
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
