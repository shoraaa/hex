#include "hexudon/internal.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hexudon {
namespace {

constexpr int kScale = 1000;
constexpr int kMaxFinalists = 5;
constexpr int kScenarioCount = 3;
constexpr int kRiskHalfStd = 500; // 0.5 in fixed-point arithmetic.

struct DpRequest {
  int spot{};
  int copy{};
  bool operator==(const DpRequest &) const = default;
};

struct DpRoute {
  std::vector<DpRequest> requests;
};

struct DpSolution {
  std::vector<DpRoute> routes;
  std::vector<DpRequest> bank;
};

struct DpEvent {
  std::size_t patrol{};
  std::size_t boundary{};
  int cell{};
  int lower{};
  int upper{};
  int nominal{};
  int service{};
  std::size_t refuel{};
  bool optional{};
  int priority{};
  std::size_t next_boundary{};
};

struct DpRouteTrace {
  std::vector<int> cells;
  std::vector<int> edge_time;
  std::vector<int> edge_fuel;
  std::vector<int> edge_start;
  std::vector<int> waits;
  std::vector<DpEvent> events;
  int end_time{};
  int end_fuel{};
};

struct DpDecode {
  ActionPlan plan;
  std::vector<DpRouteTrace> patrol;
  std::vector<DpEvent> events;
  std::optional<CandidateEvaluation> evaluation;
  int critical_slack{kScale};
  int repeated_road_dwell{};
  bool valid{};
};

struct DpRank {
  // The first two fields are intentionally separated from today's daily score:
  // a chain collected today is never traded for a future option, while future
  // Tier-1 reachability may trade against today's Tier-2/Tier-3 score.
  int banked_match{};
  int projected_match{};
  int daily{};
  int future_daily{};
  int servings{};
  int slack{};
  int fuel{};
  int hygiene{};
  std::size_t hash{};
};

struct RoadFilter {
  std::map<int, double> mean;
  std::map<int, double> uncertainty;
  std::vector<std::map<int, int>> observed_status;
  std::vector<std::map<int, int>> own_dwell;
};

struct RoadScenario {
  std::map<int, int> roads;
  int weight{};
  std::string name;
};

DpDecode decode_solution(const MapConfig &config, const DayInfo &day,
                         const PolicyHistory &history, const DpSolution &input);
DpDecode normalize_and_decode(const MapConfig &config, const DayInfo &day,
                              const PolicyHistory &history,
                              const AgentTypes &types, DpSolution &solution);
std::vector<RoadScenario> build_scenarios(const MapConfig &config,
                                          const DayInfo &day,
                                          const RoadFilter &filter,
                                          const std::map<int, int> &own_today,
                                          int next_day);
DpRank rank_decoded(const MapConfig &config, const DayInfo &day,
                    const PolicyHistory &history, const DpDecode &decoded,
                    const RoadFilter &filter, int discount_percent,
                    const std::map<int, int> &previous_dwell);
int remaining_days(const MapConfig &config, int day);
bool same_route_shape(const DpSolution &left, const DpSolution &right);

auto rank_quality(const DpRank &rank) {
  // Preserve banked Tier-1 coverage first; projected future reachability may
  // trade against today's lower tiers when the realized match objective is
  // otherwise tied.
  return std::tie(rank.banked_match, rank.projected_match, rank.daily,
                  rank.future_daily, rank.servings, rank.slack, rank.fuel,
                  rank.hygiene);
}

bool better_rank_quality(const DpRank &left, const DpRank &right) {
  return rank_quality(left) > rank_quality(right);
}

bool better_rank(const DpRank &left, const DpRank &right) {
  if (rank_quality(left) != rank_quality(right))
    return better_rank_quality(left, right);
  // Hashes provide reproducible ordering only; they are not an optimization
  // objective and must not create fake ALNS improvements.
  return left.hash < right.hash;
}

std::uint64_t dp_mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::uint64_t dp_seed(const MapConfig &config, const DayInfo &day,
                      const PolicyHistory &history, std::uint64_t salt = 0) {
  std::uint64_t seed = 0x4c4e535f4450ULL ^ salt;
  auto add = [&](std::uint64_t value) { seed = dp_mix(seed ^ value); };
  add(config.width);
  add(config.height);
  add(config.fuel_limit);
  add(day.day);
  for (const auto &agent : day.agents) {
    add(static_cast<std::uint64_t>(agent.pos + 1));
    add(static_cast<std::uint64_t>(agent.fuel + 1));
    add(static_cast<std::uint64_t>(agent.kind));
  }
  for (int brand : history.distinct_brands)
    add(brand + 1);
  return seed;
}

int cached_path_cost(const MapConfig &config, int source, int target,
                     const std::map<int, int> &roads) {
  thread_local std::unordered_map<std::uint64_t, int> cache;
  std::uint64_t key =
      dp_mix(static_cast<std::uint64_t>(config.width * 4099 + config.height));
  // This cache is process/thread local and may see more than one generated
  // map (notably in regression tests).  Geometry is part of shortest-path
  // identity; dimensions and road statuses alone are insufficient.
  key = dp_mix(key ^ static_cast<std::uint64_t>(
                         reinterpret_cast<std::uintptr_t>(config.cells.data())));
  key = dp_mix(key ^ static_cast<std::uint64_t>(config.busy_threshold + 1));
  key = dp_mix(key ^
               static_cast<std::uint64_t>((config.jammed_threshold + 1) * 17));
  key = dp_mix(key ^ static_cast<std::uint64_t>(source + 1));
  key = dp_mix(key ^ static_cast<std::uint64_t>(target + 1));
  for (const auto &[pos, status] : roads) {
    key = dp_mix(key ^ static_cast<std::uint64_t>((pos + 1) * 3 + status));
  }
  if (auto iterator = cache.find(key); iterator != cache.end()) {
    return iterator->second;
  }
  const int result = shortest_path(config, source, target, roads).cost;
  if (cache.size() > 250'000)
    cache.clear();
  cache.emplace(key, result);
  return result;
}

std::size_t solution_hash(const DpSolution &solution) {
  std::size_t hash = 0xcbf29ce484222325ULL;
  auto add = [&](std::size_t value) {
    hash ^= value + 1U;
    hash *= 0x100000001b3ULL;
  };
  for (std::size_t agent = 0; agent < solution.routes.size(); ++agent) {
    add(agent + 1);
    for (const auto &request : solution.routes[agent].requests) {
      add(static_cast<std::size_t>(request.spot + 1));
      add(static_cast<std::size_t>(request.copy + 1));
    }
    add(0xfeedU);
  }
  for (const auto &request : solution.bank) {
    add(static_cast<std::size_t>(request.spot + 1));
    add(static_cast<std::size_t>(request.copy + 1));
  }
  return hash;
}

std::string config_fingerprint_dp(const MapConfig &config) {
  std::uint64_t value = 0x4c4e534450434647ULL;
  auto add = [&](std::uint64_t item) { value = dp_mix(value ^ item); };
  add(config.width);
  add(config.height);
  add(config.players);
  add(config.fuel_limit);
  add(config.busy_threshold);
  add(config.jammed_threshold);
  for (int steps : config.day_steps)
    add(steps);
  for (Terrain terrain : config.cells)
    add(static_cast<int>(terrain));
  for (const auto &spot : config.spots) {
    add(spot.brand + 1);
    add(spot.pos + 1);
    add(spot.stocks);
  }
  for (int pos : config.agents)
    add(pos + 1);
  std::ostringstream stream;
  stream << std::hex << value;
  return stream.str();
}

int direction_between_dp(const MapConfig &config, int from, int to) {
  for (int direction = 0; direction < 6; ++direction) {
    if (neighbor(config, from, direction) == to)
      return direction;
  }
  return -1;
}

[[maybe_unused]] int path_fuel_dp(const MapConfig &config, int source,
                                  const std::vector<int> &directions) {
  int fuel = 0;
  int cursor = source;
  for (int direction : directions) {
    fuel += terrain_fuel(config, cursor);
    const auto next = neighbor(config, cursor, direction);
    if (!next)
      return std::numeric_limits<int>::max() / 4;
    cursor = *next;
  }
  return fuel;
}

[[maybe_unused]] int path_road_dwell_dp(const MapConfig &config, int source,
                                        const std::vector<int> &directions) {
  int result = 0;
  int cursor = source;
  for (int direction : directions) {
    if (config.cells[cursor] == Terrain::Road)
      ++result;
    const auto next = neighbor(config, cursor, direction);
    if (!next)
      break;
    cursor = *next;
  }
  return result;
}

[[maybe_unused]] std::vector<int>
action_path_from_cells(const MapConfig &config, const std::vector<int> &cells) {
  std::vector<int> directions;
  if (cells.empty())
    return directions;
  directions.reserve(cells.size() > 0 ? cells.size() - 1 : 0);
  for (std::size_t index = 1; index < cells.size(); ++index) {
    const int direction =
        direction_between_dp(config, cells[index - 1], cells[index]);
    if (direction < 0)
      return {};
    directions.push_back(direction);
  }
  return directions;
}

std::vector<DpRequest> all_requests(const MapConfig &config,
                                    std::size_t patrol_count) {
  std::vector<DpRequest> result;
  for (std::size_t spot = 0; spot < config.spots.size(); ++spot) {
    const int copies = std::min<int>(config.spots[spot].stocks,
                                     static_cast<int>(patrol_count));
    for (int copy = 0; copy < copies; ++copy) {
      result.push_back({static_cast<int>(spot), copy});
    }
  }
  return result;
}

bool request_in_routes(const DpSolution &solution, const DpRequest &request) {
  return std::any_of(solution.routes.begin(), solution.routes.end(),
                     [&](const DpRoute &route) {
                       return std::find(route.requests.begin(),
                                        route.requests.end(),
                                        request) != route.requests.end();
                     });
}

bool same_spot_in_route(const DpRoute &route, int spot) {
  return std::any_of(
      route.requests.begin(), route.requests.end(),
      [&](const DpRequest &request) { return request.spot == spot; });
}

void canonicalize_bank(DpSolution &solution, const MapConfig &config,
                       std::size_t patrol_count) {
  std::vector<DpRequest> used;
  std::vector<int> counts(config.spots.size(), 0);
  for (auto &route : solution.routes) {
    std::vector<DpRequest> filtered;
    for (const auto &request : route.requests) {
      if (request.spot < 0 ||
          request.spot >= static_cast<int>(config.spots.size()) ||
          same_spot_in_route(DpRoute{filtered}, request.spot) ||
          counts[request.spot] >= config.spots[request.spot].stocks) {
        continue;
      }
      filtered.push_back(request);
      ++counts[request.spot];
      used.push_back(request);
    }
    route.requests = std::move(filtered);
  }
  solution.bank.clear();
  for (const auto &request : all_requests(config, patrol_count)) {
    if (!request_in_routes(solution, request))
      solution.bank.push_back(request);
  }
}

DpSolution solution_from_trace(const MapConfig &config,
                               std::size_t route_count,
                               std::size_t patrol_count,
                               const SimulationTrace &trace) {
  DpSolution solution;
  solution.routes.resize(route_count);
  std::vector<int> next_copy(config.spots.size(), 0);
  std::map<int, int> spot_for_pos;
  for (std::size_t spot = 0; spot < config.spots.size(); ++spot)
    spot_for_pos[config.spots[spot].pos] = static_cast<int>(spot);
  for (const auto &acquisition : trace.acquisitions) {
    const auto found = spot_for_pos.find(acquisition.spot_pos);
    if (found == spot_for_pos.end() || acquisition.agent >= route_count)
      continue;
    const int spot = found->second;
    if (same_spot_in_route(solution.routes[acquisition.agent], spot))
      continue;
    solution.routes[acquisition.agent].requests.push_back(
        {spot, next_copy[spot]++});
  }
  canonicalize_bank(solution, config, patrol_count);
  return solution;
}

LnsSkeleton skeleton_from_trace_dp(const MapConfig &config,
                                   const SimulationTrace &trace,
                                   std::size_t agent_count) {
  LnsSkeleton skeleton;
  skeleton.routes.resize(agent_count);
  std::map<int, int> spot_index;
  for (std::size_t spot = 0; spot < config.spots.size(); ++spot)
    spot_index[config.spots[spot].pos] = static_cast<int>(spot);
  for (const auto &acquisition : trace.acquisitions) {
    if (acquisition.agent >= skeleton.routes.size())
      continue;
    const auto found = spot_index.find(acquisition.spot_pos);
    if (found == spot_index.end())
      continue;
    auto &route = skeleton.routes[acquisition.agent];
    if (std::find(route.begin(), route.end(), found->second) == route.end())
      route.push_back(found->second);
  }
  return skeleton;
}

void remove_request(DpSolution &solution, std::size_t agent,
                    std::size_t position) {
  if (agent >= solution.routes.size() ||
      position >= solution.routes[agent].requests.size()) {
    return;
  }
  solution.routes[agent].requests.erase(
      solution.routes[agent].requests.begin() +
      static_cast<std::ptrdiff_t>(position));
}

std::vector<std::pair<std::size_t, std::size_t>>
route_visits(const DpSolution &solution) {
  std::vector<std::pair<std::size_t, std::size_t>> result;
  for (std::size_t agent = 0; agent < solution.routes.size(); ++agent) {
    for (std::size_t position = 0;
         position < solution.routes[agent].requests.size(); ++position) {
      result.emplace_back(agent, position);
    }
  }
  return result;
}

DpDecode normalize_and_decode(const MapConfig &config, const DayInfo &day,
                              const PolicyHistory &history,
                              const AgentTypes &types, DpSolution &solution) {
  const std::size_t patrols = static_cast<std::size_t>(
      std::count(types.begin(), types.end(), AgentKind::Patrol));
  canonicalize_bank(solution, config, patrols);
  while (true) {
    DpDecode decoded = decode_solution(config, day, history, solution);
    if (decoded.valid) {
      // The grid decoder may collect a spot incidentally on a shortest path,
      // and it may truncate the tail of an overlong request sequence.  Keep
      // the phenotype and request bank aligned with what the simulator
      // actually served; otherwise unreachable tail requests disappear from
      // the bank and poison every later destroy/repair move.
      if (decoded.evaluation) {
        solution = solution_from_trace(config, solution.routes.size(), patrols,
                                       decoded.evaluation->trace);
      }
      return decoded;
    }
    std::optional<std::size_t> target;
    for (std::size_t agent = 0; agent < solution.routes.size(); ++agent) {
      if (types[agent] != AgentKind::Patrol ||
          solution.routes[agent].requests.empty())
        continue;
      if (!target || solution.routes[agent].requests.size() >
                         solution.routes[*target].requests.size()) {
        target = agent;
      }
    }
    if (!target) {
      DpDecode fallback;
      fallback.plan = wait_plan(types.size(), config.day_steps[day.day]);
      fallback.evaluation =
          evaluate_candidate(config, day, history, fallback.plan);
      fallback.valid = fallback.evaluation.has_value();
      return fallback;
    }
    solution.routes[*target].requests.pop_back();
    canonicalize_bank(solution, config, patrols);
  }
}

} // namespace

namespace {

struct DpSearchContext {
  const MapConfig &config;
  const DayInfo &day;
  const PolicyHistory &history;
  const AgentTypes &types;
  const RoadFilter &filter;
  int discount_percent{};
  std::map<int, int> previous_dwell;
  std::mt19937_64 &random;
};

[[maybe_unused]] int approximate_route_time(const MapConfig &config,
                                            const DayInfo &day,
                                            const DpRoute &route,
                                            std::size_t agent) {
  int cursor = day.agents[agent].pos;
  int total = 0;
  for (const auto &request : route.requests) {
    if (request.spot < 0 ||
        request.spot >= static_cast<int>(config.spots.size()))
      return std::numeric_limits<int>::max() / 4;
    const auto path = shortest_path(
        config, cursor, config.spots[request.spot].pos, day.traffics);
    if (path.cost >= std::numeric_limits<int>::max() / 8)
      return path.cost;
    total += path.cost == 0 ? 1 : path.cost;
    cursor = config.spots[request.spot].pos;
  }
  return total;
}

int insertion_delta(const MapConfig &config, const DayInfo &day,
                    const DpRoute &route, std::size_t agent,
                    std::size_t position, int spot) {
  const int previous =
      position == 0 ? day.agents[agent].pos
                    : config.spots[route.requests[position - 1].spot].pos;
  const int next = position == route.requests.size()
                       ? -1
                       : config.spots[route.requests[position].spot].pos;
  const int first =
      cached_path_cost(config, previous, config.spots[spot].pos, day.traffics);
  const int second = next < 0 ? 0
                              : cached_path_cost(config, config.spots[spot].pos,
                                                 next, day.traffics);
  const int before =
      next < 0 ? 0 : cached_path_cost(config, previous, next, day.traffics);
  if (first >= std::numeric_limits<int>::max() / 8 ||
      second >= std::numeric_limits<int>::max() / 8) {
    return std::numeric_limits<int>::max() / 4;
  }
  return std::max(1, first + second - before);
}

int request_tier(const MapConfig &config, const PolicyHistory &history,
                 const DpSolution &solution, int spot) {
  const int brand = config.spots[spot].brand;
  bool planned = false;
  for (const auto &route : solution.routes) {
    for (const auto &request : route.requests) {
      if (config.spots[request.spot].brand == brand)
        planned = true;
    }
  }
  if (!history.distinct_brands.contains(brand) && !planned)
    return 3;
  if (!planned)
    return 2;
  return 1;
}

std::optional<std::pair<DpSolution, DpDecode>>
insert_and_decode(const DpSearchContext &context, const DpSolution &source,
                  const DpRequest &request, std::size_t agent,
                  std::size_t position) {
  DpSolution candidate = source;
  if (agent >= candidate.routes.size() ||
      same_spot_in_route(candidate.routes[agent], request.spot))
    return std::nullopt;
  candidate.routes[agent].requests.insert(
      candidate.routes[agent].requests.begin() +
          static_cast<std::ptrdiff_t>(position),
      request);
  canonicalize_bank(candidate, context.config,
                    std::count(context.types.begin(), context.types.end(),
                               AgentKind::Patrol));
  DpDecode decoded = normalize_and_decode(
      context.config, context.day, context.history, context.types, candidate);
  if (!decoded.valid || same_route_shape(candidate, source))
    return std::nullopt;
  return std::make_pair(std::move(candidate), std::move(decoded));
}

DpRank rank_candidate(const DpSearchContext &context,
                      const DpSolution &solution, const DpDecode &decoded,
                      bool full = false) {
  if (!full && decoded.valid && decoded.evaluation) {
    DpRank rank;
    rank.banked_match = std::get<0>(decoded.evaluation->value);
    rank.daily = std::get<1>(decoded.evaluation->value);
    rank.servings = std::get<2>(decoded.evaluation->value);
    rank.slack = decoded.critical_slack;
    rank.fuel = std::accumulate(decoded.evaluation->ending_fuel.begin(),
                                decoded.evaluation->ending_fuel.end(), 0);
    rank.hygiene = -decoded.repeated_road_dwell;
    rank.hash = solution_hash(solution);
    return rank;
  }
  DpRank rank = rank_decoded(context.config, context.day, context.history,
                             decoded, context.filter, context.discount_percent,
                             context.previous_dwell);
  if (!full)
    rank.future_daily = 0;
  rank.hash = solution_hash(solution);
  return rank;
}

DpSolution construct_solution(const DpSearchContext &context, int mode,
                              int iteration_cap = -1,
                              const std::optional<
                                  std::chrono::steady_clock::time_point>
                                  &deadline = std::nullopt) {
  DpSolution solution;
  solution.routes.resize(context.types.size());
  canonicalize_bank(solution, context.config,
                    std::count(context.types.begin(), context.types.end(),
                               AgentKind::Patrol));
  int moves = 0;
  const std::size_t patrol_count = static_cast<std::size_t>(std::count(
      context.types.begin(), context.types.end(), AgentKind::Patrol));
  const int effective_cap =
      iteration_cap < 0
          ? static_cast<int>(all_requests(context.config, patrol_count).size())
          : iteration_cap;
  while (!solution.bank.empty() && moves < effective_cap) {
    if (deadline && std::chrono::steady_clock::now() >= *deadline)
      break;
    struct Option {
      DpRequest request;
      std::size_t agent{};
      std::size_t position{};
      int tier{};
      int delta{};
      int regret{};
    };
    std::vector<Option> options;
    for (const auto &request : solution.bank) {
      int best = std::numeric_limits<int>::max() / 4;
      int second = std::numeric_limits<int>::max() / 4;
      std::optional<Option> best_option;
      for (std::size_t agent = 0; agent < context.types.size(); ++agent) {
        if (context.types[agent] != AgentKind::Patrol ||
            same_spot_in_route(solution.routes[agent], request.spot))
          continue;
        for (std::size_t position = 0;
             position <= solution.routes[agent].requests.size(); ++position) {
          const int delta = insertion_delta(context.config, context.day,
                                            solution.routes[agent], agent,
                                            position, request.spot);
          if (delta >= std::numeric_limits<int>::max() / 8)
            continue;
          if (delta < best) {
            second = best;
            best = delta;
            best_option = Option{
                request,
                agent,
                position,
                request_tier(context.config, context.history, solution, request.spot),
                delta,
                0};
          } else if (delta < second) {
            second = delta;
          }
        }
      }
      if (best_option) {
        best_option->regret = second >= std::numeric_limits<int>::max() / 8
                                  ? context.config.day_steps[context.day.day]
                                  : second - best;
        options.push_back(*best_option);
      }
    }
    if (options.empty())
      break;
    std::sort(options.begin(), options.end(),
              [&](const Option &left, const Option &right) {
                if (mode != 2 && left.tier != right.tier)
                  return left.tier > right.tier;
                if ((mode == 1 || mode == 3) && left.regret != right.regret) {
                  return left.regret > right.regret;
                }
                if (mode == 2)
                  return left.delta < right.delta;
                return std::tie(left.delta, left.tier, left.request.spot,
                                left.request.copy) <
                       std::tie(right.delta, right.tier, right.request.spot,
                                right.request.copy);
              });
    const std::size_t evaluation_limit =
        std::min<std::size_t>(options.size(), 6);
    std::size_t selected = 0;
    bool randomized = false;
    if (mode == 2) {
      selected = static_cast<std::size_t>(context.random() % options.size());
      randomized = true;
    } else if (mode == 3 && options.size() > 1 && context.random() % 5 == 0) {
      selected = static_cast<std::size_t>(context.random() % options.size());
      randomized = true;
    }
    DpRank best_rank;
    std::optional<DpSolution> best_solution;
    std::size_t evaluated = 0;
    for (std::size_t offset = 0; offset < options.size(); ++offset) {
      if (deadline && evaluated > 0 &&
          std::chrono::steady_clock::now() >= *deadline) {
        break;
      }
      const std::size_t index = randomized ? (selected + offset) % options.size()
                                           : offset;
      const auto &option = options[index];
      auto candidate = insert_and_decode(context, solution, option.request,
                                         option.agent, option.position);
      if (!candidate)
        continue;
      const DpRank rank =
          rank_candidate(context, candidate->first, candidate->second);
      if (!best_solution || better_rank(rank, best_rank)) {
        best_rank = rank;
        best_solution = std::move(candidate->first);
      }
      ++evaluated;
      if (randomized)
        break;
      if (evaluated >= evaluation_limit)
        break;
    }
    if (!best_solution)
      break;
    solution = std::move(*best_solution);
    ++moves;
  }
  canonicalize_bank(solution, context.config,
                    patrol_count);
  return solution;
}

} // namespace

namespace {

void destroy_dp(const DpSearchContext &context, DpSolution &solution, int mode,
                int remove_count, const DpDecode &decoded) {
  auto visits = route_visits(solution);
  if (visits.empty())
    return;
  remove_count = std::clamp(remove_count, 1, static_cast<int>(visits.size()));
  std::vector<std::pair<std::size_t, std::size_t>> removed;
  if (mode == 0) {
    std::shuffle(visits.begin(), visits.end(), context.random);
    removed.assign(visits.begin(), visits.begin() + remove_count);
  } else if (mode == 1) {
    std::vector<std::tuple<int, int, std::size_t, std::size_t>> ranked;
    std::map<int, int> brand_count;
    for (const auto &route : solution.routes) {
      for (const auto &request : route.requests) {
        ++brand_count[context.config.spots[request.spot].brand];
      }
    }
    for (const auto [agent, position] : visits) {
      const auto &request = solution.routes[agent].requests[position];
      const int brand = context.config.spots[request.spot].brand;
      const int prize =
          brand_count[brand] > 1
              ? 1
              : (!context.history.distinct_brands.contains(brand) ? 1000 : 100);
      const int saving =
          insertion_delta(context.config, context.day, solution.routes[agent],
                          agent, position, request.spot);
      ranked.emplace_back(prize, std::max(1, saving), agent, position);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto &left, const auto &right) {
                // Low marginal prize and high saved resource are the worst
                // requests.
                return std::tuple{std::get<0>(left), -std::get<1>(left)} <
                       std::tuple{std::get<0>(right), -std::get<1>(right)};
              });
    for (int index = 0; index < remove_count; ++index) {
      removed.emplace_back(std::get<2>(ranked[index]),
                           std::get<3>(ranked[index]));
    }
  } else if (mode == 2) {
    const auto seed = visits[context.random() % visits.size()];
    const int seed_spot =
        solution.routes[seed.first].requests[seed.second].spot;
    std::sort(
        visits.begin(), visits.end(), [&](const auto &left, const auto &right) {
          const auto &l = solution.routes[left.first].requests[left.second];
          const auto &r = solution.routes[right.first].requests[right.second];
          const int l_same = context.config.spots[l.spot].brand ==
                             context.config.spots[seed_spot].brand;
          const int r_same = context.config.spots[r.spot].brand ==
                             context.config.spots[seed_spot].brand;
          const int l_distance =
              shortest_path(context.config, context.config.spots[seed_spot].pos,
                            context.config.spots[l.spot].pos,
                            context.day.traffics)
                  .cost;
          const int r_distance =
              shortest_path(context.config, context.config.spots[seed_spot].pos,
                            context.config.spots[r.spot].pos,
                            context.day.traffics)
                  .cost;
          return std::tuple{-l_same, l_distance, left.first, left.second} <
                 std::tuple{-r_same, r_distance, right.first, right.second};
        });
    removed.assign(visits.begin(), visits.begin() + remove_count);
  } else if (mode == 3) {
    const auto seed = visits[context.random() % visits.size()];
    const int brand =
        context.config
            .spots[solution.routes[seed.first].requests[seed.second].spot]
            .brand;
    for (const auto [agent, position] : visits) {
      const int candidate_brand =
          context.config.spots[solution.routes[agent].requests[position].spot]
              .brand;
      if (candidate_brand == brand)
        removed.emplace_back(agent, position);
    }
  } else if (mode == 4) {
    std::size_t target_agent = visits.front().first;
    std::size_t segment_begin = 0;
    std::size_t segment_end = 0;
    for (std::size_t agent = 0; agent < decoded.patrol.size(); ++agent) {
      if (agent >= solution.routes.size() ||
          solution.routes[agent].requests.empty())
        continue;
      std::vector<std::size_t> boundaries{0};
      for (const auto &event : decoded.events) {
        if (event.patrol == agent)
          boundaries.push_back(event.boundary);
      }
      boundaries.push_back(decoded.patrol[agent].edge_time.size());
      std::sort(boundaries.begin(), boundaries.end());
      for (std::size_t index = 1; index < boundaries.size(); ++index) {
        if (boundaries[index] - boundaries[index - 1] >
            segment_end - segment_begin) {
          target_agent = agent;
          segment_begin = boundaries[index - 1];
          segment_end = boundaries[index];
        }
      }
    }
    std::size_t cursor = 0;
    const auto &cells = decoded.patrol[target_agent].cells;
    for (std::size_t position = 0;
         position < solution.routes[target_agent].requests.size(); ++position) {
      const int target =
          context.config
              .spots[solution.routes[target_agent].requests[position].spot]
              .pos;
      while (cursor + 1 < cells.size() && cells[cursor] != target)
        ++cursor;
      if (cursor >= segment_begin && cursor <= segment_end) {
        removed.emplace_back(target_agent, position);
      }
    }
    if (removed.empty())
      removed.push_back(visits.front());
    if (removed.size() > static_cast<std::size_t>(remove_count)) {
      removed.resize(remove_count);
    }
  } else if (mode == 5) {
    std::map<std::pair<std::size_t, int>, int> acquisition_step;
    if (decoded.evaluation) {
      for (const auto &acquisition : decoded.evaluation->trace.acquisitions) {
        acquisition_step[{acquisition.agent, acquisition.spot_pos}] =
            acquisition.step;
      }
    }
    const int horizon = context.config.day_steps[context.day.day];
    const int center = static_cast<int>(
        context.random() % static_cast<std::uint64_t>(horizon + 1));
    std::sort(
        visits.begin(), visits.end(), [&](const auto &left, const auto &right) {
          auto acquisition_time = [&](const auto &item) {
            const int spot =
                solution.routes[item.first].requests[item.second].spot;
            const auto key =
                std::pair{item.first, context.config.spots[spot].pos};
            return acquisition_step.contains(key) ? acquisition_step.at(key)
                                                  : horizon;
          };
          return std::abs(acquisition_time(left) - center) <
                 std::abs(acquisition_time(right) - center);
        });
    removed.assign(visits.begin(), visits.begin() + remove_count);
  } else {
    // Whole-car removal.
    std::vector<std::size_t> patrols;
    for (std::size_t agent = 0; agent < solution.routes.size(); ++agent) {
      if (!solution.routes[agent].requests.empty())
        patrols.push_back(agent);
    }
    if (patrols.empty())
      return;
    const std::size_t agent = patrols[context.random() % patrols.size()];
    for (std::size_t position = 0;
         position < solution.routes[agent].requests.size(); ++position) {
      removed.emplace_back(agent, position);
    }
  }
  std::sort(removed.begin(), removed.end(),
            [](const auto &left, const auto &right) {
              return left.first != right.first ? left.first > right.first
                                               : left.second > right.second;
            });
  for (const auto [agent, position] : removed)
    remove_request(solution, agent, position);
  canonicalize_bank(solution, context.config,
                    std::count(context.types.begin(), context.types.end(),
                               AgentKind::Patrol));
}

DpSolution repair_dp(
    const DpSearchContext &context, DpSolution solution, int mode,
    int iteration_cap = -1,
    const std::optional<std::chrono::steady_clock::time_point> &deadline =
        std::nullopt) {
  canonicalize_bank(solution, context.config,
                    std::count(context.types.begin(), context.types.end(),
                               AgentKind::Patrol));
  const auto expired = [&] {
    return deadline && std::chrono::steady_clock::now() >= *deadline;
  };
  int moves = 0;
  const int effective_cap =
      iteration_cap < 0
          ? std::min<int>(32, static_cast<int>(solution.bank.size()))
          : iteration_cap;
  while (!solution.bank.empty() && moves < effective_cap && !expired()) {
    struct Option {
      DpRequest request;
      std::size_t agent{};
      std::size_t position{};
      int tier{};
      int delta{};
      int regret2{};
      int regret3{};
      std::uint64_t tie{};
    };
    struct Candidate {
      DpSolution solution;
      DpDecode decoded;
      DpRank rank;
    };
    std::vector<Option> options;
    for (const auto &request : solution.bank) {
      if (expired())
        break;
      std::vector<Option> placements;
      for (std::size_t agent = 0; agent < context.types.size(); ++agent) {
        if (expired())
          break;
        if (context.types[agent] != AgentKind::Patrol ||
            same_spot_in_route(solution.routes[agent], request.spot))
          continue;
        for (std::size_t position = 0;
             position <= solution.routes[agent].requests.size(); ++position) {
          if (expired())
            break;
          const int delta = insertion_delta(context.config, context.day,
                                            solution.routes[agent], agent,
                                            position, request.spot);
          if (delta >= context.config.day_steps[context.day.day] * 2)
            continue;
          placements.push_back({request, agent, position,
                                request_tier(context.config, context.history,
                                             solution, request.spot),
                                delta, 0, 0, context.random()});
        }
      }
      if (placements.empty())
        continue;
      std::sort(placements.begin(), placements.end(),
                [](const Option &left, const Option &right) {
                  return std::tie(left.delta, left.tie, left.agent,
                                  left.position) <
                         std::tie(right.delta, right.tie, right.agent,
                                  right.position);
                });
      Option best = placements.front();
      best.regret2 = placements.size() > 1
                         ? placements[1].delta - placements[0].delta
                         : context.config.day_steps[context.day.day];
      best.regret3 = placements.size() > 2
                         ? placements[2].delta - placements[0].delta
                         : best.regret2;
      options.push_back(best);
    }
    if (expired() || options.empty())
      break;
    std::sort(options.begin(), options.end(),
              [&](const Option &left, const Option &right) {
                if (mode == 3 && left.tier != right.tier)
                  return left.tier > right.tier;
                if (mode == 1 && left.regret2 != right.regret2) {
                  return left.regret2 > right.regret2;
                }
                if (mode == 2 && left.regret3 != right.regret3) {
                  return left.regret3 > right.regret3;
                }
                return std::tuple{left.tier, -left.delta, left.tie} >
                       std::tuple{right.tier, -right.delta, right.tie};
              });
    std::vector<Candidate> candidates;
    const std::size_t exact_limit = std::min<std::size_t>(options.size(), 8);
    for (std::size_t index = 0;
         index < options.size() && candidates.size() < exact_limit; ++index) {
      if (expired())
        break;
      const auto &option = options[index];
      auto inserted = insert_and_decode(context, solution, option.request,
                                        option.agent, option.position);
      if (!inserted)
        continue;
      auto candidate_solution = std::move(inserted->first);
      auto candidate_decode = std::move(inserted->second);
      auto rank = rank_candidate(context, candidate_solution, candidate_decode);
      rank.hash = solution_hash(candidate_solution);
      candidates.push_back(
          {std::move(candidate_solution), std::move(candidate_decode), rank});
    }
    if (candidates.empty())
      break;
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &left, const Candidate &right) {
                return better_rank(left.rank, right.rank);
              });
    solution = std::move(candidates.front().solution);
    ++moves;
  }
  canonicalize_bank(solution, context.config,
                    std::count(context.types.begin(), context.types.end(),
                               AgentKind::Patrol));
  return solution;
}

} // namespace

namespace {

struct SearchItem {
  DpSolution solution;
  DpDecode decoded;
  DpRank rank;
};

struct SearchOutcome {
  SearchItem best;
  std::vector<SearchItem> elite;
};

int weighted_choice(const std::vector<double> &weights,
                    std::mt19937_64 &random) {
  const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (total <= 0.0)
    return 0;
  std::uniform_real_distribution<double> distribution(0.0, total);
  double draw = distribution(random);
  for (std::size_t index = 0; index < weights.size(); ++index) {
    draw -= weights[index];
    if (draw <= 0.0)
      return static_cast<int>(index);
  }
  return static_cast<int>(weights.size() - 1);
}

std::set<int> collected_brands(const MapConfig &config,
                               const SearchItem &item) {
  std::set<int> result;
  if (!item.decoded.evaluation)
    return result;
  for (const auto &acquisition : item.decoded.evaluation->trace.acquisitions) {
    if (const Spot *spot = spot_at(config, acquisition.spot_pos))
      result.insert(spot->brand);
  }
  return result;
}

bool same_elite_state(const MapConfig &config, const SearchItem &left,
                      const SearchItem &right) {
  if (same_route_shape(left.solution, right.solution))
    return true;
  if (!left.decoded.evaluation || !right.decoded.evaluation)
    return false;
  const auto &left_evaluation = *left.decoded.evaluation;
  const auto &right_evaluation = *right.decoded.evaluation;
  return left_evaluation.ending_positions == right_evaluation.ending_positions &&
         left_evaluation.ending_fuel == right_evaluation.ending_fuel &&
         left_evaluation.road_traffic == right_evaluation.road_traffic &&
         collected_brands(config, left) == collected_brands(config, right) &&
         left.rank.daily == right.rank.daily &&
         left.rank.servings == right.rank.servings;
}

void insert_elite(const MapConfig &config, SearchItem item,
                  std::vector<SearchItem> &elite,
                  std::size_t limit = 12) {
  auto duplicate = std::find_if(
      elite.begin(), elite.end(),
      [&](const SearchItem &existing) {
        return same_elite_state(config, item, existing);
      });
  if (duplicate != elite.end()) {
    if (better_rank(item.rank, duplicate->rank))
      *duplicate = std::move(item);
    std::sort(elite.begin(), elite.end(),
              [](const SearchItem &left, const SearchItem &right) {
                return better_rank(left.rank, right.rank);
              });
    return;
  }
  elite.push_back(std::move(item));
  std::sort(elite.begin(), elite.end(),
            [](const SearchItem &left, const SearchItem &right) {
              return better_rank(left.rank, right.rank);
            });
  if (elite.size() > limit)
    elite.resize(limit);
}

SearchOutcome search_current_day(
    const DpSearchContext &context, int maximum_iterations,
    int minimum_iterations, int stagnation_limit,
    const std::optional<std::chrono::steady_clock::time_point> &deadline,
    const ImprovementSink *on_improve) {
  std::vector<SearchItem> seeds;
  // Warm-start the direct route phenotype from the established lightweight LNS
  // constructors (zero neighborhood iterations), then re-decode it entirely
  // through LNS-DP.  This imports only a visit ordering, not legacy refuel
  // tokens, paths, timings, or feasibility decisions.
  SearchLimits warm_limits;
  warm_limits.min_iterations = 0;
  warm_limits.max_iterations = 0;
  warm_limits.stagnation_iterations = 0;
  warm_limits.time_limit_ms = -1;
  warm_limits.random_seed = context.random();
  const ActionPlan warm_plan =
      build_lns_plan(context.config, context.day, context.history,
                     context.types, warm_limits);
  if (const auto warm_evaluation = evaluate_candidate(
          context.config, context.day, context.history, warm_plan)) {
    DpSolution solution = solution_from_trace(
        context.config, context.types.size(),
        static_cast<std::size_t>(std::count(context.types.begin(),
                                            context.types.end(),
                                            AgentKind::Patrol)),
        warm_evaluation->trace);
    DpDecode decoded = normalize_and_decode(
        context.config, context.day, context.history, context.types, solution);
    if (decoded.valid) {
      DpRank rank = rank_candidate(context, solution, decoded, true);
      seeds.push_back({std::move(solution), std::move(decoded), rank});
    }
  }
  for (int mode = 0; mode < 4; ++mode) {
    if (deadline && !seeds.empty() &&
        std::chrono::steady_clock::now() >= *deadline) {
      break;
    }
    DpSolution solution = construct_solution(context, mode, -1, deadline);
    DpDecode decoded = normalize_and_decode(
        context.config, context.day, context.history, context.types, solution);
    if (!decoded.valid)
      continue;
    DpRank rank = rank_candidate(context, solution, decoded, true);
    seeds.push_back({std::move(solution), std::move(decoded), rank});
  }
  if (seeds.empty()) {
    DpSolution empty;
    empty.routes.resize(context.types.size());
    canonicalize_bank(empty, context.config,
                      std::count(context.types.begin(), context.types.end(),
                                 AgentKind::Patrol));
    auto decoded = normalize_and_decode(context.config, context.day,
                                        context.history, context.types, empty);
    auto rank = rank_candidate(context, empty, decoded, true);
    seeds.push_back({std::move(empty), std::move(decoded), rank});
  }
  std::sort(seeds.begin(), seeds.end(),
            [](const SearchItem &left, const SearchItem &right) {
              return better_rank(left.rank, right.rank);
            });
  SearchOutcome outcome;
  outcome.best = seeds.front();
  SearchItem current = seeds.front();
  for (auto &seed : seeds)
    insert_elite(context.config, seed, outcome.elite);

  auto emit = [&](const SearchItem &item) {
    if (!on_improve || !item.decoded.evaluation)
      return;
    const auto official = alns_official_value(item.decoded.evaluation->value);
    IncumbentRank internal;
    internal.available = true;
    internal.objective_mode = "lns_dp";
    internal.future_discount_percent = context.discount_percent;
    internal.weighted_match = {
        std::to_string(item.rank.banked_match),
        std::to_string(item.rank.projected_match),
        std::to_string(item.rank.daily * kScale + item.rank.future_daily)};
    internal.predicted_final_available =
        remaining_days(context.config, context.day.day) > 0;
    internal.predicted_final = {item.rank.banked_match, item.rank.future_daily,
                                item.rank.servings};
    internal.predicted_ending_patrol_fuel = item.rank.fuel;
    (*on_improve)(item.decoded.plan,
                  Score{std::get<0>(official), std::get<1>(official),
                        std::get<2>(official)},
                  internal);
  };
  emit(outcome.best);

  std::vector<double> destroy_weights(7, 1.0);
  std::vector<double> repair_weights(4, 1.0);
  int stagnation = 0;
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
    if (deadline && std::chrono::steady_clock::now() >= *deadline)
      break;
    int destroy = iteration < 28
                      ? iteration / 4
                      : weighted_choice(destroy_weights, context.random);
    int repair = iteration < 28
                     ? iteration % 4
                     : weighted_choice(repair_weights, context.random);
    DpSolution candidate_solution = current.solution;
    const auto visits = route_visits(candidate_solution);
    if (!visits.empty()) {
      const int minimum =
          std::max(1, static_cast<int>(std::ceil(visits.size() * 0.05)));
      const int maximum =
          std::max(minimum, static_cast<int>(std::ceil(visits.size() * 0.30)));
      const int remove =
          minimum +
          static_cast<int>(context.random() %
                           static_cast<std::uint64_t>(maximum - minimum + 1));
      destroy_dp(context, candidate_solution, destroy, remove, current.decoded);
    }
    candidate_solution = repair_dp(context, std::move(candidate_solution),
                                   repair, -1, deadline);
    DpDecode decoded =
        normalize_and_decode(context.config, context.day, context.history,
                             context.types, candidate_solution);
    if (!decoded.valid) {
      ++stagnation;
      continue;
    }
    DpRank rank = rank_candidate(context, candidate_solution, decoded, true);
    SearchItem candidate{std::move(candidate_solution), std::move(decoded),
                         rank};
    const bool improves_current =
        better_rank_quality(candidate.rank, current.rank);
    bool accepted = improves_current;
    // Annealing is deliberately confined to Tier 3 and shaping: no accepted
    // move may lose banked/current projected Tier 1 or combined Tier 2.
    if (!accepted && candidate.rank.banked_match == current.rank.banked_match &&
        candidate.rank.projected_match == current.rank.projected_match &&
        candidate.rank.daily == current.rank.daily &&
        candidate.rank.future_daily == current.rank.future_daily) {
      const int loss =
          std::max(0, current.rank.servings - candidate.rank.servings);
      const double progress =
          maximum_iterations <= 1
              ? 1.0
              : static_cast<double>(iteration) / (maximum_iterations - 1);
      const double temperature = std::max(0.1, 2.0 * (1.0 - progress));
      accepted = unit(context.random) < std::exp(-loss / temperature);
    }
    const bool global = better_rank_quality(candidate.rank, outcome.best.rank);
    if (global) {
      outcome.best = candidate;
      stagnation = 0;
      destroy_weights[destroy] += 8.0;
      repair_weights[repair] += 8.0;
      emit(outcome.best);
    } else {
      ++stagnation;
      if (accepted) {
        destroy_weights[destroy] += 2.0;
        repair_weights[repair] += 2.0;
      }
    }
    insert_elite(context.config, candidate, outcome.elite);
    if (accepted)
      current = std::move(candidate);
    if ((iteration + 1) % 32 == 0) {
      for (double &value : destroy_weights)
        value = 0.8 * value + 0.2;
      for (double &value : repair_weights)
        value = 0.8 * value + 0.2;
    }
    if (iteration + 1 >= minimum_iterations && stagnation_limit > 0 &&
        stagnation >= stagnation_limit)
      break;
  }
  return outcome;
}

struct RecourseValue {
  int match{};
  int daily{};
  int servings{};
  int fuel{};
  int residual_match_potential{};
};

PolicyHistory history_after_today(const MapConfig &config,
                                  const PolicyHistory &history,
                                  const DpDecode &decoded) {
  PolicyHistory next = history;
  if (!decoded.evaluation)
    return next;
  for (const auto &event : decoded.evaluation->trace.acquisitions) {
    if (const Spot *spot = spot_at(config, event.spot_pos)) {
      next.distinct_brands.insert(spot->brand);
    }
  }
  next.submitted_actions.push_back(decoded.plan);
  next.cumulative_daily_types += std::get<1>(decoded.evaluation->value);
  next.total_servings += std::get<2>(decoded.evaluation->value);
  return next;
}

RecourseValue solve_recourse(
    const MapConfig &config, const DayInfo &day, const PolicyHistory &history,
    const AgentTypes &types, const CandidateEvaluation &today,
    const RoadScenario &scenario, const RoadFilter &filter,
    int discount_percent, std::mt19937_64 &random, int local_iterations,
    const std::optional<std::chrono::steady_clock::time_point> &deadline) {
  if (day.day + 1 >= static_cast<int>(config.day_steps.size())) {
    return {std::get<0>(today.value), 0, 0, 0, 0};
  }
  DayInfo future;
  future.day = day.day + 1;
  future.traffics = scenario.roads;
  for (std::size_t index = 0; index < types.size(); ++index) {
    future.agents.push_back(
        {types[index], today.ending_positions[index],
         types[index] == AgentKind::Patrol ? today.ending_fuel[index] : 0});
  }
  DpSearchContext context{config, future,           history, types,
                          filter, discount_percent, {},      random};
  DpSolution solution = construct_solution(context, 3, 8, deadline);
  DpDecode decoded =
      normalize_and_decode(config, future, history, types, solution);
  if (!decoded.valid || !decoded.evaluation) {
    return {std::get<0>(today.value), 0, 0, 0, 0};
  }
  SearchItem best{solution, decoded,
                  rank_candidate(context, solution, decoded, false)};
  for (int iteration = 0; iteration < local_iterations; ++iteration) {
    if (deadline && std::chrono::steady_clock::now() >= *deadline)
      break;
    DpSolution candidate = best.solution;
    const auto visits = route_visits(candidate);
    if (!visits.empty()) {
      destroy_dp(context, candidate, iteration % 7,
                 std::max(1, static_cast<int>(visits.size() / 5)),
                 best.decoded);
    }
    candidate = repair_dp(context, std::move(candidate), iteration % 4, 4,
                          deadline);
    auto evaluated =
        normalize_and_decode(config, future, history, types, candidate);
    if (!evaluated.valid)
      continue;
    auto rank = rank_candidate(context, candidate, evaluated, false);
    if (better_rank(rank, best.rank)) {
      best = {std::move(candidate), std::move(evaluated), rank};
    }
  }
  const auto &value = best.decoded.evaluation->value;
  int fuel = 0;
  for (int amount : best.decoded.evaluation->ending_fuel)
    fuel += amount;
  const DpRank projected =
      rank_candidate(context, best.solution, best.decoded, true);
  return {std::get<0>(value), std::get<1>(value), std::get<2>(value), fuel,
          projected.projected_match};
}

struct JointRank {
  int banked_today{};
  int risk_match{};
  int combined_daily{};
  int combined_servings{};
  int fuel{};
  std::size_t hash{};
};

bool better_joint(const JointRank &left, const JointRank &right) {
  const auto left_quality =
      std::tie(left.banked_today, left.risk_match, left.combined_daily,
               left.combined_servings, left.fuel);
  const auto right_quality =
      std::tie(right.banked_today, right.risk_match, right.combined_daily,
               right.combined_servings, right.fuel);
  if (left_quality != right_quality)
    return left_quality > right_quality;
  return left.hash < right.hash;
}

JointRank joint_rank(
    const DpSearchContext &context, const SearchItem &item,
    int recourse_iterations,
    const std::optional<std::chrono::steady_clock::time_point> &deadline) {
  JointRank result;
  if (!item.decoded.evaluation)
    return result;
  const auto &today = *item.decoded.evaluation;
  result.banked_today = std::get<0>(today.value);
  const auto scenarios =
      build_scenarios(context.config, context.day, context.filter,
                      today.road_traffic, context.day.day + 1);
  const PolicyHistory future_history =
      history_after_today(context.config, context.history, item.decoded);
  std::array<RecourseValue, kScenarioCount> values{};
  long long mean_match = 0;
  long long mean_daily = 0;
  long long mean_servings = 0;
  long long mean_fuel = 0;
  for (std::size_t index = 0; index < scenarios.size(); ++index) {
    values[index] = solve_recourse(
        context.config, context.day, future_history, context.types, today,
        scenarios[index], context.filter, context.discount_percent,
        context.random, recourse_iterations, deadline);
    mean_match +=
        static_cast<long long>(scenarios[index].weight) *
        (values[index].match * kScale + values[index].residual_match_potential);
    mean_daily +=
        static_cast<long long>(scenarios[index].weight) * values[index].daily;
    mean_servings += static_cast<long long>(scenarios[index].weight) *
                     values[index].servings;
    mean_fuel +=
        static_cast<long long>(scenarios[index].weight) * values[index].fuel;
  }
  mean_match /= kScale;
  long long variance = 0;
  for (std::size_t index = 0; index < scenarios.size(); ++index) {
    const long long scaled =
        static_cast<long long>(values[index].match) * kScale +
        values[index].residual_match_potential;
    const long long difference = scaled - mean_match;
    variance += static_cast<long long>(scenarios[index].weight) * difference *
                difference / kScale;
  }
  const int standard_deviation =
      static_cast<int>(std::sqrt(std::max<long long>(0, variance)));
  result.risk_match =
      static_cast<int>(mean_match) - standard_deviation * kRiskHalfStd / kScale;
  result.combined_daily =
      std::get<1>(today.value) * kScale + static_cast<int>(mean_daily);
  result.combined_servings =
      std::get<2>(today.value) * kScale + static_cast<int>(mean_servings);
  result.fuel = static_cast<int>(mean_fuel);
  result.hash = solution_hash(item.solution);
  return result;
}

} // namespace

namespace {

int remaining_days(const MapConfig &config, int day) {
  return std::max(0, static_cast<int>(config.day_steps.size()) - day - 1);
}

std::map<int, int> roads_for_scenario(const MapConfig &config,
                                      const RoadScenario &scenario) {
  std::map<int, int> roads;
  for (int pos = 0; pos < config.width * config.height; ++pos) {
    if (config.cells[pos] == Terrain::Road) {
      auto iterator = scenario.roads.find(pos);
      roads[pos] = iterator == scenario.roads.end() ? 0 : iterator->second;
    }
  }
  return roads;
}

int scenario_reachability(const MapConfig &config, const std::vector<int> &ends,
                          int brand, const RoadScenario &scenario,
                          int horizon) {
  int best = std::numeric_limits<int>::max() / 4;
  std::map<int, int> roads = roads_for_scenario(config, scenario);
  for (std::size_t index = 0; index < config.spots.size(); ++index) {
    if (config.spots[index].brand != brand)
      continue;
    for (int end : ends) {
      best = std::min(
          best, cached_path_cost(config, end, config.spots[index].pos, roads));
    }
  }
  if (best >= std::numeric_limits<int>::max() / 8)
    return 0;
  const int slack_target = std::max(1, static_cast<int>(0.30 * horizon));
  const int slack = horizon - best;
  if (slack <= 0)
    return 0;
  return std::clamp(slack * kScale / slack_target, 0, kScale);
}

int potential_match(const MapConfig &config, const DayInfo &day,
                    const PolicyHistory &history,
                    const CandidateEvaluation &evaluation,
                    const std::vector<RoadScenario> &scenarios,
                    int discount_percent) {
  std::set<int> known = history.distinct_brands;
  for (const auto &acquisition : evaluation.trace.acquisitions) {
    if (const Spot *spot = spot_at(config, acquisition.spot_pos)) {
      known.insert(spot->brand);
    }
  }
  std::set<int> all;
  for (const auto &spot : config.spots)
    all.insert(spot.brand);
  const int future = remaining_days(config, day.day);
  if (future <= 0)
    return 0;
  std::vector<int> ends;
  for (std::size_t index = 0; index < evaluation.ending_positions.size();
       ++index) {
    if (index < day.agents.size() &&
        day.agents[index].kind == AgentKind::Patrol) {
      ends.push_back(evaluation.ending_positions[index]);
    }
  }
  long long total = 0;
  for (int brand : all) {
    if (known.contains(brand))
      continue;
    long long weighted = 0;
    for (const auto &scenario : scenarios) {
      weighted +=
          static_cast<long long>(scenario.weight) *
          scenario_reachability(
              config, ends, brand, scenario,
              config.day_steps[std::min<int>(
                  day.day + 1, static_cast<int>(config.day_steps.size()) - 1)]);
    }
    const int probability = static_cast<int>(weighted / 1000);
    int reach = probability;
    // Compound a one-day option over the remaining days.  This is deliberately
    // a shaping estimate, not an admissible bound.
    for (int index = 1; index < future; ++index) {
      reach = kScale - ((kScale - reach) * (kScale - probability)) / kScale;
    }
    int defer = future + 1;
    for (int offset = 1; offset <= future; ++offset) {
      const int horizon = config.day_steps[static_cast<std::size_t>(
          std::min<int>(day.day + offset,
                        static_cast<int>(config.day_steps.size()) - 1))];
      if (scenario_reachability(config, ends, brand, scenarios[1], horizon) >=
          kScale) {
        defer = offset;
        break;
      }
    }
    int discount = kScale;
    for (int index = 1; index < defer; ++index) {
      discount = discount * std::clamp(discount_percent, 1, 100) / 100;
    }
    total += static_cast<long long>(reach) * discount / kScale;
  }
  return static_cast<int>(std::clamp<long long>(total, 0, 2'000'000));
}

int future_daily_capacity(const MapConfig &config, const DayInfo &day,
                          const CandidateEvaluation &evaluation,
                          const std::vector<RoadScenario> &scenarios) {
  const int future = remaining_days(config, day.day);
  if (future <= 0)
    return 0;
  const int horizon = config.day_steps[std::min<int>(
      day.day + 1, static_cast<int>(config.day_steps.size()) - 1)];
  std::set<int> reachable;
  const std::map<int, int> roads =
      roads_for_scenario(config, scenarios.front());
  for (std::size_t agent = 0; agent < evaluation.ending_positions.size();
       ++agent) {
    if (agent >= day.agents.size() ||
        day.agents[agent].kind != AgentKind::Patrol)
      continue;
    for (const auto &spot : config.spots) {
      if (cached_path_cost(config, evaluation.ending_positions[agent], spot.pos,
                           roads) <= horizon / 2) {
        reachable.insert(spot.brand);
      }
    }
  }
  return static_cast<int>(reachable.size());
}

DpRank rank_decoded(const MapConfig &config, const DayInfo &day,
                    const PolicyHistory &history, const DpDecode &decoded,
                    const RoadFilter &filter, int discount_percent,
                    const std::map<int, int> &previous_dwell) {
  DpRank rank;
  if (!decoded.valid || !decoded.evaluation)
    return rank;
  const auto &evaluation = *decoded.evaluation;
  const auto scenarios = build_scenarios(config, day, filter,
                                         evaluation.road_traffic, day.day + 1);
  rank.banked_match = std::get<0>(evaluation.value);
  rank.projected_match = potential_match(config, day, history, evaluation,
                                         scenarios, discount_percent);
  rank.daily = std::get<1>(evaluation.value);
  rank.future_daily = future_daily_capacity(config, day, evaluation, scenarios);
  rank.servings = std::get<2>(evaluation.value);
  rank.slack = decoded.critical_slack;
  for (int fuel : evaluation.ending_fuel)
    rank.fuel += fuel;
  int repeated = 0;
  for (const auto &[pos, dwell] : evaluation.road_traffic) {
    const auto previous =
        previous_dwell.contains(pos) ? previous_dwell.at(pos) : 0;
    repeated += dwell * (1 + previous);
  }
  rank.hygiene = -repeated;
  rank.hash = solution_hash(DpSolution{});
  return rank;
}

bool same_route_shape(const DpSolution &left, const DpSolution &right) {
  return solution_hash(left) == solution_hash(right);
}

} // namespace

namespace {

std::vector<int> shortest_cells_dp(const MapConfig &config, int source,
                                   int target,
                                   const std::map<int, int> &roads) {
  const auto path = shortest_path(config, source, target, roads);
  if (path.cost >= std::numeric_limits<int>::max() / 8)
    return {};
  std::vector<int> cells{source};
  int cursor = source;
  for (int direction : path.directions) {
    const auto next = neighbor(config, cursor, direction);
    if (!next)
      return {};
    cursor = *next;
    cells.push_back(cursor);
  }
  return cells;
}

[[maybe_unused]] std::vector<int> route_targets(const MapConfig &config,
                                                const DpRoute &route) {
  std::vector<int> result;
  result.reserve(route.requests.size());
  for (const auto &request : route.requests) {
    if (request.spot >= 0 &&
        request.spot < static_cast<int>(config.spots.size())) {
      result.push_back(config.spots[request.spot].pos);
    }
  }
  return result;
}

[[maybe_unused]] void expand_route(const MapConfig &config, const DayInfo &day,
                                   const DpRoute &route, DpRouteTrace &result) {
  (void)config;
  result = {};
  int cursor = day.agents.empty() ? -1 : day.agents[0].pos;
  (void)cursor;
  for (std::size_t index = 0; index < route.requests.size(); ++index) {
    (void)index;
  }
}

void expand_route_from_agent(const MapConfig &config, const DayInfo &day,
                             std::size_t agent, const DpRoute &route,
                             const std::map<int, int> &roads,
                             DpRouteTrace &result) {
  result = {};
  int cursor = day.agents[agent].pos;
  int elapsed = 0;
  const int horizon = config.day_steps.at(static_cast<std::size_t>(day.day));
  result.cells.push_back(cursor);
  for (const auto &request : route.requests) {
    if (request.spot < 0 ||
        request.spot >= static_cast<int>(config.spots.size())) {
      continue;
    }
    const int target = config.spots[request.spot].pos;
    const auto cells = shortest_cells_dp(config, cursor, target, roads);
    if (cells.empty())
      break;
    for (std::size_t index = 1; index < cells.size(); ++index) {
      const int from = cells[index - 1];
      const int duration = terrain_time(config, from, roads);
      if (elapsed + duration > horizon)
        break;
      result.edge_time.push_back(duration);
      result.edge_fuel.push_back(terrain_fuel(config, from));
      result.edge_start.push_back(0);
      result.cells.push_back(cells[index]);
      elapsed += duration;
    }
    cursor = result.cells.back();
    if (cursor != target || elapsed >= horizon)
      break;
  }
  result.waits.assign(result.edge_time.size() + 1, 0);
}

struct FuelChoice {
  std::vector<std::size_t> boundaries;
  int events{};
  int nominal_wait{};
  int end_fuel{};
};

FuelChoice choose_fuel_breakpoints(const MapConfig &config, const DayInfo &day,
                                   std::size_t agent, DpRouteTrace &route) {
  const int edges = static_cast<int>(route.edge_time.size());
  const int infinity = std::numeric_limits<int>::max() / 4;
  const int initial_fuel = day.agents[agent].fuel;
  std::vector<int> prefix(edges + 1, 0);
  for (int index = 0; index < edges; ++index) {
    prefix[index + 1] = prefix[index] + route.edge_fuel[index];
  }
  if (prefix[edges] <= initial_fuel) {
    route.events.clear();
    return {{}, 0, 0, initial_fuel - prefix[edges]};
  }
  // best[b] is the minimum number of services needed to arrive at boundary b
  // and refuel there, ready to traverse edge b.  The old implementation used
  // an end sentinel in the same array and reconstructed `cursor` rather than
  // its parent; that dropped the first (and commonly only) recharge event.
  std::vector<int> best(edges, infinity);
  std::vector<int> parent(edges, -1);
  for (int boundary = 1; boundary < edges; ++boundary) {
    if (prefix[boundary] <= initial_fuel)
      best[boundary] = 1;
  }
  // If even the first edge exceeds the carried fuel, an initial service can
  // still make the route feasible.  route_events_from_breakpoints inserts the
  // mandatory first-step wait so that this service happens before movement.
  if (edges > 0 && route.edge_fuel[0] > initial_fuel)
    best[0] = 1;
  for (int from = 0; from < edges; ++from) {
    if (best[from] == infinity)
      continue;
    for (int to = from + 1; to < edges; ++to) {
      if (prefix[to] - prefix[from] > config.fuel_limit)
        break;
      if (best[to] >= best[from] + 1) {
        best[to] = best[from] + 1;
        parent[to] = from;
      }
    }
  }
  int last = -1;
  int event_count = infinity;
  for (int boundary = 0; boundary < edges; ++boundary) {
    if (best[boundary] == infinity ||
        prefix[edges] - prefix[boundary] > config.fuel_limit) {
      continue;
    }
    // Prefer the latest equally short partition: it preserves initial fuel and
    // gives a mobile refuel car the widest useful arrival deadline.
    if (best[boundary] <= event_count) {
      event_count = best[boundary];
      last = boundary;
    }
  }
  if (last < 0)
    return {{}, infinity, infinity, 0};
  std::vector<std::size_t> reverse;
  for (int cursor = last; cursor >= 0; cursor = parent[cursor]) {
    reverse.push_back(static_cast<std::size_t>(cursor));
    if (parent[cursor] < 0)
      break;
  }
  std::reverse(reverse.begin(), reverse.end());
  // The DP above counts a service per selected segment boundary.  Build event
  // windows later from the exact patrol schedule; at this stage choose the
  // minimum-event partition and retain broad windows as a tie-break.
  const int end_fuel =
      config.fuel_limit - (prefix[edges] - prefix[static_cast<std::size_t>(last)]);
  return {std::move(reverse), event_count, 0, end_fuel};
}

void assign_nominal_times(DpRouteTrace &route) {
  int time = 0;
  for (std::size_t index = 0; index < route.edge_time.size(); ++index) {
    route.edge_start[index] = time;
    time += route.edge_time[index];
  }
  route.end_time = time;
}

std::vector<DpEvent> route_events_from_breakpoints(const MapConfig &config,
                                                   std::size_t patrol,
                                                   DpRouteTrace &route,
                                                   const FuelChoice &fuel) {
  std::vector<DpEvent> events;
  if (!fuel.boundaries.empty() && fuel.boundaries.front() == 0) {
    route.waits[0] += 1;
    for (int &start : route.edge_start)
      ++start;
    ++route.end_time;
  }
  for (std::size_t event_index = 0; event_index < fuel.boundaries.size();
       ++event_index) {
    const std::size_t boundary = fuel.boundaries[event_index];
    if (boundary >= route.edge_time.size())
      continue;
    const int start = route.edge_start[boundary];
    const int duration = route.edge_time[boundary];
    const int lower = std::max(1, start);
    const int upper = std::max(lower, start + duration - 1);
    const int cell = route.cells[boundary];
    events.push_back(
        {patrol, boundary, cell, lower, upper, lower, lower, 0, false, 0});
    events.back().next_boundary =
        event_index + 1 < fuel.boundaries.size()
            ? fuel.boundaries[event_index + 1]
            : route.edge_time.size();
  }
  (void)config;
  return events;
}

struct RefuelScheduleState {
  int pos{};
  int time{};
  std::vector<int> actions;
};

[[maybe_unused]] bool append_path_actions(const MapConfig &config,
                                          const std::map<int, int> &roads,
                                          RefuelScheduleState &state,
                                          int target) {
  const auto path = shortest_path(config, state.pos, target, roads);
  if (path.cost >= std::numeric_limits<int>::max() / 8)
    return false;
  int cursor = state.pos;
  for (int direction : path.directions) {
    const int duration = terrain_time(config, cursor, roads);
    state.actions.push_back(direction);
    state.time += duration;
    const auto next = neighbor(config, cursor, direction);
    if (!next)
      return false;
    cursor = *next;
  }
  state.pos = target;
  return true;
}

bool schedule_events(const MapConfig &config, const DayInfo &day,
                     const std::map<int, int> &roads,
                     std::vector<DpRouteTrace> &patrol,
                     std::vector<DpEvent> &events, ActionPlan &refuel_actions,
                     int order_mode = 0) {
  const std::size_t count = day.agents.size();
  std::vector<std::size_t> refuels;
  for (std::size_t index = 0; index < count; ++index) {
    if (day.agents[index].kind == AgentKind::Refuel)
      refuels.push_back(index);
  }
  if (events.empty()) {
    refuel_actions.resize(count);
    return true;
  }
  if (refuels.empty())
    return false;
  std::sort(
      events.begin(), events.end(),
      [order_mode](const DpEvent &left, const DpEvent &right) {
        if (order_mode == 1) {
          return std::tie(left.optional, left.patrol, left.boundary,
                          left.upper, left.lower) <
                 std::tie(right.optional, right.patrol, right.boundary,
                          right.upper, right.lower);
        }
        return std::tie(left.upper, left.optional, left.priority, left.lower,
                        left.patrol, left.boundary) <
               std::tie(right.upper, right.optional, right.priority,
                        right.lower, right.patrol, right.boundary);
      });
  std::vector<RefuelScheduleState> states(count);
  std::vector<std::optional<std::size_t>> last_boundary(patrol.size());
  for (std::size_t refuel : refuels) {
    states[refuel].pos = day.agents[refuel].pos;
  }
  for (auto &event : events) {
    struct EventWindow {
      std::size_t boundary{};
      int cell{};
      int lower{};
      int upper{};
    };
    std::vector<EventWindow> windows;
    if (event.optional) {
      if (event.patrol < patrol.size() &&
          !patrol[event.patrol].cells.empty()) {
        event.lower = std::max(1, patrol[event.patrol].end_time);
        windows.push_back({event.boundary, event.cell, event.lower, event.upper});
      }
    } else if (event.patrol < patrol.size()) {
      const auto &route = patrol[event.patrol];
      const std::size_t edges = route.edge_fuel.size();
      std::vector<int> prefix(edges + 1, 0);
      for (std::size_t edge = 0; edge < edges; ++edge)
        prefix[edge + 1] = prefix[edge] + route.edge_fuel[edge];
      const std::size_t next = std::min(event.next_boundary, edges);
      const std::size_t begin = last_boundary[event.patrol]
                                    ? *last_boundary[event.patrol] + 1
                                    : 0;
      for (std::size_t boundary = begin; boundary < next && boundary < edges;
           ++boundary) {
        if (boundary == 0 && event.boundary != 0)
          continue;
        const bool reachable = last_boundary[event.patrol]
                                   ? prefix[boundary] -
                                             prefix[*last_boundary[event.patrol]] <=
                                         config.fuel_limit
                                   : prefix[boundary] <=
                                         day.agents[event.patrol].fuel;
        if (!reachable ||
            prefix[next] - prefix[boundary] > config.fuel_limit) {
          continue;
        }
        const int start = route.edge_start[boundary];
        const int lower = std::max(1, start);
        const int upper =
            std::max(lower, start + route.edge_time[boundary] - 1);
        windows.push_back(
            {boundary, route.cells[boundary], lower, upper});
      }
    }
    if (windows.empty()) {
      windows.push_back({event.boundary, event.cell, event.lower, event.upper});
    }
    bool chosen = false;
    std::size_t best_refuel = refuels.front();
    int best_service = std::numeric_limits<int>::max();
    int best_delay = std::numeric_limits<int>::max();
    EventWindow best_window = windows.front();
    std::vector<int> best_path;
    for (const auto &window : windows) {
      for (std::size_t refuel : refuels) {
        const auto path =
            shortest_path(config, states[refuel].pos, window.cell, roads);
        if (path.cost >= std::numeric_limits<int>::max() / 8)
          continue;
        const int arrival = states[refuel].time + path.cost;
        const int service = std::max(window.lower, arrival);
        const int delay = std::max(0, service - window.upper);
        if (!chosen ||
            std::tie(delay, service, refuel, window.boundary) <
                std::tie(best_delay, best_service, best_refuel,
                         best_window.boundary)) {
          chosen = true;
          best_refuel = refuel;
          best_service = service;
          best_delay = delay;
          best_window = window;
          best_path = path.directions;
        }
      }
    }
    if (!chosen)
      return false;
    if (event.optional && best_service > best_window.upper) {
      event.service = -1;
      continue;
    }
    event.boundary = best_window.boundary;
    event.cell = best_window.cell;
    event.lower = best_window.lower;
    event.upper = best_window.upper;
    if (!event.optional)
      last_boundary[event.patrol] = event.boundary;
    event.refuel = best_refuel;
    event.service = best_service;
    auto &state = states[best_refuel];
    int cursor = state.pos;
    for (int direction : best_path) {
      state.actions.push_back(direction);
      state.time += terrain_time(config, cursor, roads);
      const auto next = neighbor(config, cursor, direction);
      if (!next)
        return false;
      cursor = *next;
    }
    state.pos = event.cell;
    if (state.time < event.service) {
      state.actions.push_back(-(event.service - state.time));
      state.time = event.service;
    }
    const int wait = std::max(0, event.service - event.upper);
    if (wait > 0 && event.patrol < patrol.size() &&
        event.boundary < patrol[event.patrol].waits.size()) {
      patrol[event.patrol].waits[event.boundary] += wait;
      for (std::size_t index = event.boundary;
           index < patrol[event.patrol].edge_start.size(); ++index) {
        patrol[event.patrol].edge_start[index] += wait;
      }
      event.lower += wait;
      event.upper += wait;
      event.nominal += wait;
      patrol[event.patrol].end_time += wait;
      for (auto &downstream : events) {
        if (&downstream == &event || downstream.patrol != event.patrol ||
            downstream.boundary <= event.boundary) {
          continue;
        }
        downstream.lower += wait;
        if (!downstream.optional)
          downstream.upper += wait;
        downstream.nominal += wait;
      }
    }
  }
  refuel_actions.assign(count, {});
  for (std::size_t refuel : refuels)
    refuel_actions[refuel] = states[refuel].actions;
  events.erase(std::remove_if(events.begin(), events.end(),
                              [](const DpEvent &event) {
                                return event.optional && event.service < 0;
                              }),
               events.end());
  return true;
}

void append_patrol_actions(const MapConfig &config, const DayInfo &day,
                           const std::map<int, int> &roads, std::size_t agent,
                           DpRouteTrace &route, std::vector<int> &actions) {
  int cursor = day.agents[agent].pos;
  for (std::size_t edge = 0; edge < route.edge_time.size(); ++edge) {
    if (edge < route.waits.size() && route.waits[edge] > 0) {
      actions.push_back(-route.waits[edge]);
    }
    const auto next =
        neighbor(config, cursor,
                 direction_between_dp(config, cursor, route.cells[edge + 1]));
    if (!next)
      break;
    actions.push_back(
        direction_between_dp(config, cursor, route.cells[edge + 1]));
    cursor = *next;
    (void)roads;
  }
  if (!route.waits.empty() && route.waits.back() > 0) {
    actions.push_back(-route.waits.back());
  }
}

DpDecode decode_solution(const MapConfig &config, const DayInfo &day,
                         const PolicyHistory &history,
                         const DpSolution &input) {
  DpDecode result;
  DpSolution solution = input;
  std::size_t patrol_count = 0;
  for (auto type : day.agents)
    patrol_count += type.kind == AgentKind::Patrol;
  canonicalize_bank(solution, config, patrol_count);
  result.patrol.resize(day.agents.size());
  result.plan.assign(day.agents.size(), {});
  std::vector<DpEvent> events;
  for (std::size_t agent = 0; agent < day.agents.size(); ++agent) {
    if (day.agents[agent].kind != AgentKind::Patrol)
      continue;
    expand_route_from_agent(config, day, agent, solution.routes[agent],
                            day.traffics, result.patrol[agent]);
    assign_nominal_times(result.patrol[agent]);
    const auto fuel =
        choose_fuel_breakpoints(config, day, agent, result.patrol[agent]);
    if (fuel.events >= std::numeric_limits<int>::max() / 8) {
      result.valid = false;
      return result;
    }
    auto route_events = route_events_from_breakpoints(
        config, agent, result.patrol[agent], fuel);
    result.patrol[agent].end_fuel = fuel.end_fuel;
    if (day.day + 1 < static_cast<int>(config.day_steps.size()) &&
        fuel.end_fuel < config.fuel_limit &&
        !result.patrol[agent].cells.empty()) {
      const std::size_t boundary = result.patrol[agent].edge_time.size();
      const int lower = std::max(1, result.patrol[agent].end_time);
      const int upper = config.day_steps[static_cast<std::size_t>(day.day)];
      route_events.push_back(
          {agent, boundary, result.patrol[agent].cells.back(), lower, upper,
           lower, lower, 0, true, fuel.end_fuel});
    }
    events.insert(events.end(), route_events.begin(), route_events.end());
  }
  const int horizon = config.day_steps.at(static_cast<std::size_t>(day.day));
  const auto base_patrol = result.patrol;
  const auto base_events = events;
  std::optional<DpDecode> best_decoded;
  const int schedule_modes = base_events.size() > 4 ? 2 : 1;
  for (int order_mode = 0; order_mode < schedule_modes; ++order_mode) {
    DpDecode candidate;
    candidate.patrol = base_patrol;
    candidate.plan.assign(day.agents.size(), {});
    candidate.events = base_events;
    ActionPlan refuel_actions;
    if (!schedule_events(config, day, day.traffics, candidate.patrol,
                         candidate.events, refuel_actions, order_mode)) {
      continue;
    }
    for (std::size_t agent = 0; agent < day.agents.size(); ++agent) {
      if (day.agents[agent].kind == AgentKind::Patrol) {
        append_patrol_actions(config, day, day.traffics, agent,
                              candidate.patrol[agent], candidate.plan[agent]);
      } else if (agent < refuel_actions.size()) {
        candidate.plan[agent] = refuel_actions[agent];
      }
    }
    // Feasible-by-construction truncation: retain the longest action prefix
    // that fits the day, then wait.  A long tail is unserved prize, not an
    // invalid candidate.
    for (std::size_t agent = 0; agent < candidate.plan.size(); ++agent) {
      int elapsed = 0;
      int cursor = day.agents[agent].pos;
      std::vector<int> truncated;
      for (int action : candidate.plan[agent]) {
        if (action < 0) {
          const int duration = std::min(-action, horizon - elapsed);
          if (duration > 0) {
            truncated.push_back(-duration);
            elapsed += duration;
          }
        } else {
          const int duration = terrain_time(config, cursor, day.traffics);
          if (elapsed + duration > horizon)
            break;
          truncated.push_back(action);
          elapsed += duration;
          const auto next = neighbor(config, cursor, action);
          if (!next)
            break;
          cursor = *next;
        }
        if (elapsed >= horizon)
          break;
      }
      if (elapsed < horizon)
        truncated.push_back(-(horizon - elapsed));
      candidate.plan[agent] = std::move(truncated);
    }
    candidate.evaluation =
        evaluate_candidate(config, day, history, candidate.plan);
    if (!candidate.evaluation)
      continue;
    candidate.valid = true;
    if (!best_decoded ||
        candidate.evaluation->value > best_decoded->evaluation->value) {
      best_decoded = std::move(candidate);
    }
  }
  if (!best_decoded) {
    result.valid = false;
    return result;
  }
  result = std::move(*best_decoded);
  result.critical_slack = horizon;
  std::set<int> covered_today;
  for (const auto &acquisition : result.evaluation->trace.acquisitions) {
    const Spot *spot = spot_at(config, acquisition.spot_pos);
    if (!spot)
      continue;
    const bool critical = !covered_today.contains(spot->brand) ||
                          !history.distinct_brands.contains(spot->brand);
    covered_today.insert(spot->brand);
    const bool sync_dependent = std::any_of(
        result.events.begin(), result.events.end(), [&](const DpEvent &event) {
          return event.patrol == acquisition.agent && !event.optional;
        });
    if (critical && sync_dependent &&
        acquisition.step > static_cast<int>(0.95 * horizon)) {
      result.critical_slack =
          std::min(result.critical_slack, horizon - acquisition.step);
    }
  }
  for (const auto &[pos, traffic] : result.evaluation->road_traffic) {
    if (config.cells[pos] == Terrain::Road) {
      result.repeated_road_dwell += traffic * traffic;
    }
  }
  return result;
}

} // namespace

namespace {

void update_filter(RoadFilter &filter, const MapConfig &config,
                   const DayInfo &day, int source_day) {
  if (source_day < 0)
    return;
  const int window = source_day == 0 ? 1 : 2;
  std::map<int, int> own;
  const int begin =
      std::max(0, static_cast<int>(filter.own_dwell.size()) - window);
  for (int index = begin; index < static_cast<int>(filter.own_dwell.size());
       ++index) {
    for (const auto &[pos, dwell] :
         filter.own_dwell[static_cast<std::size_t>(index)]) {
      own[pos] += dwell;
    }
  }
  const int players = std::max(1, config.players);
  const int patrols = static_cast<int>(config.agents.size());
  int horizon_sum = 0;
  const int horizon_begin = std::max(0, source_day - window + 1);
  for (int index = horizon_begin;
       index <= source_day && index < static_cast<int>(config.day_steps.size());
       ++index) {
    horizon_sum += config.day_steps[static_cast<std::size_t>(index)];
  }
  for (int pos = 0; pos < config.width * config.height; ++pos) {
    if (config.cells[pos] != Terrain::Road)
      continue;
    const auto status_it = day.traffics.find(pos);
    const int status = status_it == day.traffics.end() ? 0 : status_it->second;
    long long lower = 0;
    long long upper =
        static_cast<long long>(players) * patrols * std::max(1, horizon_sum);
    if (status == 0) {
      upper = static_cast<long long>(players) * config.busy_threshold - 1;
    } else if (status == 1) {
      lower = static_cast<long long>(players) * config.busy_threshold;
      upper = static_cast<long long>(players) * config.jammed_threshold - 1;
    } else {
      lower = static_cast<long long>(players) * config.jammed_threshold;
    }
    const int own_dwell = own[pos];
    const double opponent_count = static_cast<double>(std::max(1, players - 1));
    double low_per_day = 0.0;
    double high_per_day = 0.0;
    if (players > 1) {
      low_per_day =
          std::max<long long>(0, lower - own_dwell) / (opponent_count * window);
      high_per_day =
          std::max<long long>(0, upper - own_dwell) / (opponent_count * window);
    }
    const double observation = 0.5 * (low_per_day + high_per_day);
    const double spread = 0.5 * std::max(0.0, high_per_day - low_per_day);
    auto &mean = filter.mean[pos];
    auto &uncertainty = filter.uncertainty[pos];
    if (!filter.observed_status.empty()) {
      mean = 0.5 * mean + 0.5 * observation;
      uncertainty = 0.5 * uncertainty + 0.5 * spread;
    } else {
      mean = observation;
      uncertainty = spread;
    }
  }
  filter.observed_status.push_back(day.traffics);
  if (filter.observed_status.size() > 2)
    filter.observed_status.erase(filter.observed_status.begin());
  if (filter.own_dwell.size() > 2)
    filter.own_dwell.erase(filter.own_dwell.begin());
}

RoadFilter parse_filter_state(const MapConfig &config, const DayInfo &day,
                              const PolicyHistory &history,
                              const AgentTypes &types,
                              const json::value *planner_state) {
  RoadFilter filter;
  if (planner_state && planner_state->is_object() && day.day > 0) {
    try {
      const auto &root = planner_state->as_object();
      if (root.at("schema_version").to_number<int>() == 1 &&
          root.at("policy").as_string() == "lns_dp" &&
          root.at("config_fingerprint").as_string() ==
              config_fingerprint_dp(config) &&
          root.at("source_day").to_number<int>() == day.day - 1 &&
          root.at("committed_actions").is_array() &&
          !history.submitted_actions.empty() &&
          root.at("committed_actions") ==
              to_json(history.submitted_actions.back())) {
        const auto &encoded_types = root.at("types").as_array();
        bool types_match = encoded_types.size() == types.size();
        for (std::size_t index = 0; types_match && index < types.size();
             ++index) {
          types_match = encoded_types[index].to_number<int>() ==
                        static_cast<int>(types[index]);
        }
        if (types_match) {
          const auto &means = root.at("means").as_array();
          const auto &uncertainty = root.at("uncertainty").as_array();
          for (const auto &item : means) {
            const auto &object = item.as_object();
            filter.mean[object.at("pos").to_number<int>()] =
                object.at("value").to_number<double>();
          }
          for (const auto &item : uncertainty) {
            const auto &object = item.as_object();
            filter.uncertainty[object.at("pos").to_number<int>()] =
                object.at("value").to_number<double>();
          }
          auto decode_maps = [](const json::value &value) {
            std::vector<std::map<int, int>> result;
            for (const auto &row : value.as_array()) {
              std::map<int, int> map;
              for (const auto &item : row.as_array()) {
                const auto &object = item.as_object();
                map[object.at("pos").to_number<int>()] =
                    object.at("value").to_number<int>();
              }
              result.push_back(std::move(map));
            }
            return result;
          };
          filter.observed_status = decode_maps(root.at("observed_status"));
          filter.own_dwell = decode_maps(root.at("own_dwell"));
          return filter;
        }
      }
    } catch (const std::exception &) {
      // A stale state is deliberately ignored.  The next valid plan will
      // serialize a fresh filter rather than failing the competition day.
    }
  }
  (void)history;
  (void)types;
  // The Day-1 prior is deliberately mild.  It supplies uncertainty without
  // turning every road into a pessimistic jam before any observation exists.
  const double prior = std::max(0.0, config.busy_threshold * 0.25);
  for (int pos = 0; pos < config.width * config.height; ++pos) {
    if (config.cells[pos] == Terrain::Road) {
      filter.mean[pos] = prior;
      filter.uncertainty[pos] = prior;
    }
  }
  return filter;
}

json::object serialize_filter_state(const MapConfig &config, const DayInfo &day,
                                    const AgentTypes &types,
                                    const PolicyHistory &history,
                                    const ActionPlan &actions,
                                    const RoadFilter &filter) {
  json::array encoded_types;
  for (auto type : types)
    encoded_types.push_back(static_cast<int>(type));
  auto encode_maps = [](const std::vector<std::map<int, int>> &maps) {
    json::array result;
    for (const auto &map : maps) {
      json::array row;
      for (const auto &[pos, value] : map) {
        row.push_back(json::object{{"pos", pos}, {"value", value}});
      }
      result.push_back(std::move(row));
    }
    return result;
  };
  json::array means;
  for (const auto &[pos, value] : filter.mean) {
    means.push_back(json::object{{"pos", pos}, {"value", value}});
  }
  json::array uncertainty;
  for (const auto &[pos, value] : filter.uncertainty) {
    uncertainty.push_back(json::object{{"pos", pos}, {"value", value}});
  }
  (void)history;
  return json::object{{"schema_version", 1},
                      {"policy", "lns_dp"},
                      {"config_fingerprint", config_fingerprint_dp(config)},
                      {"source_day", day.day},
                      {"types", std::move(encoded_types)},
                      {"committed_actions", to_json(actions)},
                      {"means", std::move(means)},
                      {"uncertainty", std::move(uncertainty)},
                      {"observed_status", encode_maps(filter.observed_status)},
                      {"own_dwell", encode_maps(filter.own_dwell)}};
}

std::map<int, int> action_road_dwell(const MapConfig &config,
                                     const DayInfo &day,
                                     const ActionPlan &actions) {
  TeamState team;
  for (const auto &agent : day.agents) {
    team.agents.push_back({agent.kind, agent.pos, agent.fuel});
  }
  team.visited_today.resize(team.agents.size());
  for (const auto &spot : config.spots)
    team.stock[spot.pos] = spot.stocks;
  std::map<int, int> traffic;
  if (simulate_team_day(config, team, actions, day.traffics, traffic))
    return {};
  return traffic;
}

std::vector<RoadScenario> build_scenarios(const MapConfig &config,
                                          const DayInfo &day,
                                          const RoadFilter &filter,
                                          const std::map<int, int> &own_today,
                                          int next_day) {
  std::vector<RoadScenario> scenarios;
  const int window = next_day == 1 ? 1 : 2;
  std::map<int, int> forecast;
  std::map<int, int> pessimistic;
  std::map<int, int> smooth;
  std::map<int, int> corridor;
  std::map<int, bool> pressure;
  for (int pos = 0; pos < config.width * config.height; ++pos) {
    if (config.cells[pos] != Terrain::Road)
      continue;
    long long own = 0;
    if (window == 2 && !filter.own_dwell.empty()) {
      const auto &previous = filter.own_dwell.back();
      if (previous.contains(pos))
        own += previous.at(pos);
    }
    for (const auto &[cell, dwell] : own_today) {
      if (cell == pos)
        own += dwell;
    }
    const double opponents =
        filter.mean.contains(pos)
            ? filter.mean.at(pos) * std::max(0, config.players - 1) * window
            : 0.0;
    const double total = (own + opponents) / std::max(1, config.players);
    const int status = total >= config.jammed_threshold
                           ? 2
                           : (total >= config.busy_threshold ? 1 : 0);
    forecast[pos] = status;
    smooth[pos] = 0;
    const double upper =
        (filter.mean.contains(pos) ? filter.mean.at(pos) : 0.0) +
        (filter.uncertainty.contains(pos) ? filter.uncertainty.at(pos) : 0.0);
    pressure[pos] =
        status > 0 || upper * std::max(1, config.players - 1) * window >=
                          config.busy_threshold * 0.75;
  }
  for (const auto &agent : day.agents) {
    if (agent.kind != AgentKind::Patrol)
      continue;
    for (const auto &spot : config.spots) {
      const auto path = shortest_path(config, agent.pos, spot.pos, forecast);
      int cursor = agent.pos;
      for (int direction : path.directions) {
        if (config.cells[cursor] == Terrain::Road)
          ++corridor[cursor];
        const auto next = neighbor(config, cursor, direction);
        if (!next)
          break;
        cursor = *next;
      }
    }
  }
  std::vector<int> corridor_values;
  for (const auto &[pos, count] : corridor) {
    (void)pos;
    if (count > 0)
      corridor_values.push_back(count);
  }
  std::sort(corridor_values.begin(), corridor_values.end());
  const int corridor_cut =
      corridor_values.empty() ? std::numeric_limits<int>::max()
                              : corridor_values[corridor_values.size() * 3 / 4];
  for (const auto &[pos, status] : forecast) {
    const bool hot_corridor =
        corridor.contains(pos) && corridor.at(pos) >= corridor_cut;
    const bool bump = status > 0 || (pressure[pos] && hot_corridor);
    pessimistic[pos] = bump ? std::min(2, status + 1) : status;
  }
  scenarios.push_back({forecast, 500, "forecast"});
  scenarios.push_back({pessimistic, 350, "pessimistic"});
  scenarios.push_back({smooth, 150, "smooth"});
  return scenarios;
}

} // namespace

AgentTypes select_lns_dp_agent_types(const MapConfig &config,
                                     const SearchLimits &limits,
                                     const AgentTypeImprovementSink *on_improve) {
  // Agent roles are fixed before the match, outside the per-day route
  // genotype.  Reuse the production multi-horizon rollout selector here so a
  // two-request myopic screen cannot discard every refuel mask on a low-fuel
  // map.  LNS-DP remains independent at the solution/decoder layer.
  return select_lns_agent_types(config, limits, on_improve);
}

PlannerResult build_lns_dp_plan(const MapConfig &config, const DayInfo &day,
                                const PolicyHistory &history,
                                const AgentTypes &types,
                                const SearchLimits &limits,
                                const json::value *planner_state,
                                const ImprovementSink *on_improve) {
  RoadFilter filter =
      parse_filter_state(config, day, history, types, planner_state);
  if (day.day > 0)
    update_filter(filter, config, day, day.day - 1);
  const std::map<int, int> previous_dwell =
      filter.own_dwell.empty() ? std::map<int, int>{} : filter.own_dwell.back();
  std::mt19937_64 random(dp_seed(config, day, history, limits.random_seed));
  DpSearchContext context{config,         day,
                          history,        types,
                          filter,         limits.future_discount_percent,
                          previous_dwell, random};
  if (on_improve) {
    const ActionPlan initial =
        wait_plan(types.size(), config.day_steps[day.day]);
    const auto evaluated = evaluate_candidate(config, day, history, initial);
    IncumbentRank rank;
    rank.available = true;
    rank.objective_mode = "lns_dp";
    rank.future_discount_percent = limits.future_discount_percent;
    if (evaluated) {
      const auto official = alns_official_value(evaluated->value);
      (*on_improve)(initial,
                    Score{std::get<0>(official), std::get<1>(official),
                          std::get<2>(official)},
                    rank);
    }
  }
  const bool timed = limits.time_limit_ms >= 0;
  const bool final_day =
      day.day + 1 >= static_cast<int>(config.day_steps.size());
  const auto started = std::chrono::steady_clock::now();
  const auto total_deadline =
      timed ? std::optional(started + std::chrono::milliseconds(
                                          std::max(0, limits.time_limit_ms)))
            : std::nullopt;
  const auto main_deadline =
      timed && !final_day
          ? std::optional(started + std::chrono::milliseconds(std::max(
                                        1, limits.time_limit_ms * 60 / 100)))
          : total_deadline;
  const int main_iterations =
      final_day ? limits.max_iterations
                : std::max(1, limits.max_iterations * 60 / 100);
  SearchOutcome outcome = search_current_day(
      context, main_iterations,
      std::min(main_iterations, limits.min_iterations),
      limits.stagnation_iterations, main_deadline, on_improve);
  SearchItem selected = outcome.best;
  JointRank selected_joint;
  bool joint_available = false;

  if (!final_day && selected.decoded.evaluation) {
    std::vector<SearchItem> finalists = outcome.elite;
    std::sort(finalists.begin(), finalists.end(),
              [](const SearchItem &left, const SearchItem &right) {
                return better_rank(left.rank, right.rank);
              });
    if (finalists.size() > kMaxFinalists)
      finalists.resize(kMaxFinalists);
    for (auto &finalist : finalists) {
      if (total_deadline && std::chrono::steady_clock::now() >= *total_deadline)
        break;
      const JointRank rank = joint_rank(context, finalist, 0, total_deadline);
      if (!joint_available || better_joint(rank, selected_joint)) {
        joint_available = true;
        selected_joint = rank;
        selected = finalist;
      }
    }
    SearchItem current = selected;
    const int joint_iterations = std::max(1, limits.max_iterations * 30 / 100);
    for (int iteration = 0; iteration < joint_iterations; ++iteration) {
      if (total_deadline && std::chrono::steady_clock::now() >= *total_deadline)
        break;
      DpSolution candidate_solution = current.solution;
      const auto visits = route_visits(candidate_solution);
      if (!visits.empty()) {
        destroy_dp(context, candidate_solution, iteration % 7,
                   std::max(1, static_cast<int>(visits.size() / 5)),
                   current.decoded);
      }
      candidate_solution = repair_dp(context, std::move(candidate_solution),
                                     iteration % 4, 4, total_deadline);
      DpDecode decoded =
          normalize_and_decode(config, day, history, types, candidate_solution);
      if (!decoded.valid)
        continue;
      DpRank rank = rank_candidate(context, candidate_solution, decoded, true);
      SearchItem candidate{std::move(candidate_solution), std::move(decoded),
                           rank};
      const JointRank projected =
          joint_rank(context, candidate, 2, total_deadline);
      if (!joint_available || better_joint(projected, selected_joint)) {
        joint_available = true;
        selected_joint = projected;
        selected = candidate;
      }
      current = std::move(candidate);
    }
  }

  if (!selected.decoded.valid || !selected.decoded.evaluation) {
    selected.decoded.plan = wait_plan(types.size(), config.day_steps[day.day]);
    selected.decoded.evaluation =
        evaluate_candidate(config, day, history, selected.decoded.plan);
    selected.decoded.valid = selected.decoded.evaluation.has_value();
  }
  if (on_improve && selected.decoded.evaluation) {
    const auto official =
        alns_official_value(selected.decoded.evaluation->value);
    IncumbentRank rank;
    rank.available = true;
    rank.objective_mode = "lns_dp";
    rank.future_discount_percent = limits.future_discount_percent;
    rank.weighted_match = std::array<std::string, 3>{
        std::to_string(selected.rank.banked_match),
        std::to_string(joint_available ? selected_joint.risk_match
                                       : selected.rank.projected_match),
        std::to_string(joint_available ? selected_joint.combined_daily
                                       : selected.rank.daily * kScale +
                                             selected.rank.future_daily)};
    rank.predicted_final_available = joint_available;
    rank.predicted_final = std::array<int, 3>{
        selected.rank.banked_match,
        joint_available ? selected_joint.combined_daily / kScale
                        : selected.rank.future_daily,
        joint_available ? selected_joint.combined_servings / kScale
                        : selected.rank.servings};
    rank.predicted_ending_patrol_fuel = selected.rank.fuel;
    (*on_improve)(selected.decoded.plan,
                  Score{std::get<0>(official), std::get<1>(official),
                        std::get<2>(official)},
                  rank);
  }
  filter.own_dwell.push_back(
      action_road_dwell(config, day, selected.decoded.plan));
  if (filter.own_dwell.size() > 2)
    filter.own_dwell.erase(filter.own_dwell.begin());
  return {selected.decoded.plan,
          serialize_filter_state(config, day, types, history,
                                 selected.decoded.plan, filter)};
}

std::vector<LnsDpProposal> build_lns_dp_route_proposals(
    const MapConfig &config, const DayInfo &day,
    const PolicyHistory &history, const AgentTypes &types,
    const SearchLimits &limits, int max_proposals) {
  std::vector<LnsDpProposal> proposals;
  if (max_proposals <= 0 || types.empty())
    return proposals;

  RoadFilter filter;
  for (const auto &[pos, status] : day.traffics) {
    (void)status;
    filter.mean[pos] = config.busy_threshold * 0.25;
    filter.uncertainty[pos] = config.busy_threshold * 0.25;
  }
  const std::uint64_t base_seed = dp_seed(
      config, day, history, limits.random_seed ^ 0x50524f504f53414cULL);
  const int maximum_request_cap = std::max(
      4, std::min(16, limits.max_iterations > 0 ? limits.max_iterations : 8));
  const std::array<int, 3> modes{3, 1, 0};
  const bool timed = limits.time_limit_ms >= 0;
  const auto started = std::chrono::steady_clock::now();
  const std::optional<std::chrono::steady_clock::time_point> deadline =
      timed ? std::optional(started + std::chrono::milliseconds(
                                          std::max(0, limits.time_limit_ms)))
            : std::nullopt;
  // The three construction modes are independent. Run every mode at the full
  // request cap against the same wall-clock deadline; parallelism buys quality
  // inside the official window instead of shrinking later proposal genomes.
  auto candidates = parallel_indexed(
      modes.size(), [&](std::size_t mode_index)
                        -> std::optional<LnsDpProposal> {
    std::uint64_t seed = base_seed;
    if (mode_index > 0) {
      seed = dp_mix(seed ^ (0x9e3779b97f4a7c15ULL * (mode_index + 1)));
    }
    std::mt19937_64 random(seed);
    DpSearchContext context{config,
                            day,
                            history,
                            types,
                            filter,
                            limits.future_discount_percent,
                            {},
                            random};
    DpSolution solution = construct_solution(
        context, modes[mode_index], maximum_request_cap, deadline);
    DpDecode decoded =
        normalize_and_decode(config, day, history, types, solution);
    if (!decoded.valid || !decoded.evaluation)
      return std::nullopt;
    LnsDpProposal proposal;
    proposal.skeleton = skeleton_from_trace_dp(
        config, decoded.evaluation->trace, types.size());
    proposal.plan = decoded.plan;
    proposal.value = decoded.evaluation->value;
    proposal.ending_fuel = std::accumulate(
        decoded.evaluation->ending_fuel.begin(),
        decoded.evaluation->ending_fuel.end(), 0);
    proposal.fuel_pressure = std::numeric_limits<int>::max();
    for (std::size_t agent = 0; agent < decoded.evaluation->ending_fuel.size();
         ++agent) {
      if (agent < day.agents.size() &&
          day.agents[agent].kind == AgentKind::Patrol) {
        proposal.fuel_pressure = std::min(
            proposal.fuel_pressure, decoded.evaluation->ending_fuel[agent]);
      }
    }
    return proposal;
  });
  for (auto &candidate : candidates) {
    if (!candidate ||
        std::any_of(proposals.begin(), proposals.end(),
                    [&](const LnsDpProposal &existing) {
                      return existing.skeleton == candidate->skeleton;
                    })) {
      continue;
    }
    proposals.push_back(std::move(*candidate));
  }
  std::sort(proposals.begin(), proposals.end(),
            [](const LnsDpProposal &left, const LnsDpProposal &right) {
              if (left.value != right.value)
                return left.value > right.value;
              if (left.fuel_pressure != right.fuel_pressure)
                return left.fuel_pressure > right.fuel_pressure;
              return left.ending_fuel > right.ending_fuel;
            });
  if (proposals.size() > static_cast<std::size_t>(max_proposals))
    proposals.resize(static_cast<std::size_t>(max_proposals));
  return proposals;
}

} // namespace hexudon
