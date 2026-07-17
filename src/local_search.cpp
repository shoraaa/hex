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
ActionPlan improve_with_route_substitutions(
    const MapConfig& config, const DayInfo& day, const PolicyHistory& history,
    ActionPlan best, const std::vector<ActionPlan>& alternatives,
    std::size_t passes,
    bool parallel_evaluation) {
  auto best_value = candidate_value(config, day, history, best);
  // Best-improvement coordinate descent. Each pass evaluates every independent
  // one-route substitution, then applies exactly one deterministic improvement.
  for (std::size_t pass = 0; pass < passes; ++pass) {
    std::vector<ActionPlan> mutations;
    for (const auto& alternative : alternatives) {
      for (std::size_t agent = 0; agent < best.size(); ++agent) {
        if (best[agent] == alternative[agent]) continue;
        ActionPlan mutation = best;
        mutation[agent] = alternative[agent];
        mutations.push_back(std::move(mutation));
      }
    }
    if (mutations.empty()) break;
    std::vector<std::optional<std::tuple<int, int, int, int>>> values;
    if (parallel_evaluation) {
      values = parallel_indexed(mutations.size(), [&](std::size_t index) {
        return candidate_value(config, day, history, mutations[index]);
      });
    } else {
      values.reserve(mutations.size());
      for (const auto& mutation : mutations) {
        values.push_back(candidate_value(config, day, history, mutation));
      }
    }
    std::optional<std::size_t> improved;
    for (std::size_t index = 0; index < mutations.size(); ++index) {
      if (!values[index] || (best_value && *values[index] <= *best_value)) continue;
      if (!improved || *values[index] > *values[*improved]) improved = index;
    }
    if (!improved) break;
    best = std::move(mutations[*improved]);
    best_value = values[*improved];
  }
  return best;
}

ActionPlan build_local_search_plan(const MapConfig& config, const DayInfo& day,
                                   const PolicyHistory& history,
                                    const AgentTypes& types,
                                    const SearchLimits& limits) {
  const auto forced = coordinated_first_targets(config, day, history, types);
  const ActionPlan seed =
      build_routing_plan("coordinated", config, day, history, types, forced,
                         limits);
  const std::array<std::string, 4> alternative_policies{
      "greedy", "utility_greedy", "fuel_aware", "stock_maximiser"};
  const auto alternatives = parallel_indexed(
      alternative_policies.size(), [&](std::size_t index) {
        return build_routing_plan(alternative_policies[index], config, day,
                                  history, types, {}, limits);
      });
  std::vector<ActionPlan> initial{seed};
  initial.insert(initial.end(), alternatives.begin(), alternatives.end());
  const auto initial_values = parallel_indexed(initial.size(), [&](std::size_t index) {
    return candidate_value(config, day, history, initial[index]);
  });
  std::size_t initial_best = 0;
  for (std::size_t index = 1; index < initial.size(); ++index) {
    if (initial_values[index] &&
        (!initial_values[initial_best] ||
         *initial_values[index] > *initial_values[initial_best])) {
      initial_best = index;
    }
  }
  return improve_with_route_substitutions(
      config, day, history, initial[initial_best], alternatives,
      limits.local_search_passes > 0 ? static_cast<std::size_t>(limits.local_search_passes)
                                     : types.size(),
      true);
}
}  // namespace hexudon
