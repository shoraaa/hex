#include "hexudon/core.hpp"

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
namespace {

const json::object& object_at(const json::object& obj, std::string_view key) {
  return obj.at(key).as_object();
}

int int_at(const json::object& obj, std::string_view key) {
  return static_cast<int>(obj.at(key).to_number<std::int64_t>());
}

double number_at(const json::object& obj, std::string_view key) {
  return obj.at(key).to_number<double>();
}

std::vector<int> int_array(const json::value& value) {
  std::vector<int> result;
  for (const auto& entry : value.as_array()) {
    result.push_back(static_cast<int>(entry.to_number<std::int64_t>()));
  }
  return result;
}

struct PathResult {
  int cost{std::numeric_limits<int>::max()};
  std::vector<int> directions;
};

PathResult shortest_path(const MapConfig& config, int source, int target,
                         const std::map<int, int>& roads) {
  const int count = config.width * config.height;
  const int infinity = std::numeric_limits<int>::max() / 4;
  std::vector<int> distance(count, infinity);
  std::vector<int> previous(count, -1);
  std::vector<int> previous_direction(count, -1);
  using QueueItem = std::pair<int, int>;
  std::priority_queue<QueueItem, std::vector<QueueItem>,
                      std::greater<QueueItem>>
      queue;
  distance[source] = 0;
  queue.emplace(0, source);
  while (!queue.empty()) {
    auto [current_distance, pos] = queue.top();
    queue.pop();
    if (current_distance != distance[pos]) continue;
    if (pos == target) break;
    for (int direction = 0; direction < 6; ++direction) {
      auto next = neighbor(config, pos, direction);
      if (!next || config.cells[*next] == Terrain::Pond) continue;
      const int candidate = current_distance + terrain_time(config, pos, roads);
      if (candidate < distance[*next] ||
          (candidate == distance[*next] &&
           direction < previous_direction[*next])) {
        distance[*next] = candidate;
        previous[*next] = pos;
        previous_direction[*next] = direction;
        queue.emplace(candidate, *next);
      }
    }
  }
  if (distance[target] == infinity) return {};
  std::vector<int> reverse;
  for (int pos = target; pos != source; pos = previous[pos]) {
    if (pos < 0 || previous[pos] < 0) return {};
    reverse.push_back(previous_direction[pos]);
  }
  std::reverse(reverse.begin(), reverse.end());
  return {distance[target], std::move(reverse)};
}

const Spot* spot_at(const MapConfig& config, int pos) {
  auto iterator = std::find_if(config.spots.begin(), config.spots.end(),
                               [pos](const Spot& spot) {
                                 return spot.pos == pos;
                               });
  return iterator == config.spots.end() ? nullptr : &*iterator;
}

struct MutableAgent {
  AgentKind kind{AgentKind::Patrol};
  int pos{};
  int fuel{};
};

struct TeamState {
  std::string id;
  std::string policy;
  std::vector<MutableAgent> agents;
  std::map<int, int> stock;
  std::vector<std::set<int>> visited_today;
  std::set<int> distinct_types;
  std::set<int> daily_types;
  int cumulative_daily_types{};
  int total_servings{};
  int refuel_events{};
  int invalid_days{};
  int valid_days{};
  std::vector<Score> daily_scores;
  std::vector<std::string> errors;
  PolicyHistory history;
};

struct PendingAction {
  bool active{};
  bool move{};
  int remaining{};
  int destination{};
  int fuel_cost{};
};

struct AcquisitionEvent {
  int step{};
  std::size_t agent{};
  int spot_pos{};
};

struct RefuelEvent {
  int step{};
  std::size_t patrol{};
  std::size_t refuel{};
  int pos{};
};

struct SimulationTrace {
  std::vector<AcquisitionEvent> acquisitions;
  std::vector<RefuelEvent> refuels;
};

std::optional<std::string> simulate_team_day(
    const MapConfig& config, TeamState& team, const ActionPlan& plan,
    const std::map<int, int>& roads, std::map<int, int>& traffic,
    SimulationTrace* trace = nullptr) {
  const int steps = config.day_steps.at(team.history.submitted_actions.size());
  const std::size_t agent_count = team.agents.size();
  if (plan.size() != agent_count) return "action row count does not match agents";

  std::vector<std::size_t> cursor(agent_count, 0);
  std::vector<PendingAction> pending(agent_count);

  auto schedule = [&](std::size_t index, int step) -> std::optional<std::string> {
    if (cursor[index] >= plan[index].size()) {
      return "agent " + std::to_string(index) + " plan ends at step " +
             std::to_string(step);
    }
    const int action = plan[index][cursor[index]++];
    PendingAction next;
    next.active = true;
    if (action <= -1) {
      next.remaining = -action;
      next.destination = team.agents[index].pos;
    } else if (action <= 5) {
      auto destination = neighbor(config, team.agents[index].pos, action);
      if (!destination) {
        return "agent " + std::to_string(index) + " leaves the map";
      }
      if (config.cells[*destination] == Terrain::Pond) {
        return "agent " + std::to_string(index) + " enters a pond";
      }
      next.move = true;
      next.destination = *destination;
      next.remaining = terrain_time(config, team.agents[index].pos, roads);
      next.fuel_cost = terrain_fuel(config, team.agents[index].pos);
    } else {
      return "agent " + std::to_string(index) + " has invalid action";
    }
    if (next.remaining <= 0 || step + next.remaining > steps) {
      return "agent " + std::to_string(index) + " exceeds day horizon";
    }
    pending[index] = next;
    return std::nullopt;
  };

  for (std::size_t index = 0; index < agent_count; ++index) {
    if (auto error = schedule(index, 0)) return error;
  }

  for (int step = 1; step <= steps; ++step) {
    for (std::size_t index = 0; index < agent_count; ++index) {
      auto& action = pending[index];
      if (!action.active) {
        return "agent " + std::to_string(index) + " is idle unexpectedly";
      }
      --action.remaining;
      if (action.remaining == 0) {
        if (action.move) {
          auto& agent = team.agents[index];
          if (agent.kind == AgentKind::Patrol) {
            if (agent.fuel < action.fuel_cost) {
              return "agent " + std::to_string(index) + " lacks fuel";
            }
            agent.fuel -= action.fuel_cost;
          }
          agent.pos = action.destination;
        }
        action.active = false;
      }
    }

    for (std::size_t index = 0; index < agent_count; ++index) {
      auto& agent = team.agents[index];
      if (agent.kind != AgentKind::Patrol) continue;
      const Spot* spot = spot_at(config, agent.pos);
      if (!spot || team.visited_today[index].contains(spot->pos)) continue;
      auto stock = team.stock.find(spot->pos);
      if (stock != team.stock.end() && stock->second > 0) {
        --stock->second;
        team.visited_today[index].insert(spot->pos);
        team.distinct_types.insert(spot->brand);
        team.daily_types.insert(spot->brand);
        ++team.total_servings;
        if (trace) trace->acquisitions.push_back({step, index, spot->pos});
      }
    }

    std::map<int, std::size_t> refuel_cells;
    for (std::size_t index = 0; index < team.agents.size(); ++index) {
      const auto& agent = team.agents[index];
      if (agent.kind == AgentKind::Refuel &&
          !refuel_cells.contains(agent.pos)) {
        refuel_cells[agent.pos] = index;
      }
    }
    for (std::size_t index = 0; index < team.agents.size(); ++index) {
      auto& agent = team.agents[index];
      if (agent.kind == AgentKind::Patrol &&
          refuel_cells.contains(agent.pos)) {
        if (trace && agent.fuel < config.fuel_limit) {
          trace->refuels.push_back(
              {step, index, refuel_cells.at(agent.pos), agent.pos});
        }
        if (agent.fuel < config.fuel_limit) ++team.refuel_events;
        agent.fuel = config.fuel_limit;
      }
    }

    for (const auto& agent : team.agents) {
      if (config.cells[agent.pos] == Terrain::Road) ++traffic[agent.pos];
    }

    if (step < steps) {
      for (std::size_t index = 0; index < agent_count; ++index) {
        if (!pending[index].active) {
          if (auto error = schedule(index, step)) return error;
        }
      }
    }
  }

  for (std::size_t index = 0; index < agent_count; ++index) {
    if (pending[index].active || cursor[index] != plan[index].size()) {
      return "agent " + std::to_string(index) +
             " plan does not end exactly at day horizon";
    }
  }
  return std::nullopt;
}

ActionPlan wait_plan(std::size_t agents, int steps) {
  return ActionPlan(agents, std::vector<int>{-steps});
}

DayInfo make_day_info(const MapConfig& config,
                      const std::vector<TeamState>& teams,
                      std::size_t own_index, int day,
                      const std::map<int, int>& roads) {
  DayInfo info;
  info.day = day;
  info.traffics = roads;
  for (const auto& agent : teams[own_index].agents) {
    info.agents.push_back({agent.kind, agent.pos, agent.fuel});
  }
  for (std::size_t index = 0; index < teams.size(); ++index) {
    if (index == own_index) continue;
    OtherTeamView other;
    other.id = teams[index].id;
    for (const auto& agent : teams[index].agents) {
      other.agents.push_back({agent.kind, agent.pos, agent.fuel});
    }
    info.others.push_back(std::move(other));
  }
  for (int pos = 0; pos < config.width * config.height; ++pos) {
    if (config.cells[pos] == Terrain::Road && !info.traffics.contains(pos)) {
      info.traffics[pos] = 0;
    }
  }
  return info;
}

std::map<int, int> road_status_for_day(
    const MapConfig& config, const std::vector<std::map<int, int>>& history,
    int players) {
  std::map<int, int> result;
  for (int pos = 0; pos < config.width * config.height; ++pos) {
    if (config.cells[pos] != Terrain::Road) continue;
    long total = 0;
    const int begin = std::max(0, static_cast<int>(history.size()) - 2);
    for (int day = begin; day < static_cast<int>(history.size()); ++day) {
      if (auto iterator = history[day].find(pos); iterator != history[day].end()) {
        total += iterator->second;
      }
    }
    const double volume = history.empty() ? 0.0 : total / double(players);
    result[pos] = volume >= config.jammed_threshold
                      ? 2
                      : (volume >= config.busy_threshold ? 1 : 0);
  }
  return result;
}

}  // namespace

MapConfig parse_map_config(const json::value& value) {
  const auto& root = value.as_object();
  MapConfig config;
  config.starts_at = number_at(root, "startsAt");
  for (const auto& item : root.at("daySeconds").as_array()) {
    config.day_seconds.push_back(item.to_number<double>());
  }
  config.day_steps = int_array(root.at("daySteps"));
  const auto& map = object_at(root, "map");
  config.height = int_at(map, "height");
  config.width = int_at(map, "width");
  for (const auto& row : map.at("cells").as_array()) {
    for (const auto& cell : row.as_array()) {
      config.cells.push_back(
          static_cast<Terrain>(cell.to_number<std::int64_t>()));
    }
  }
  for (const auto& item : root.at("spots").as_array()) {
    const auto& spot = item.as_object();
    config.spots.push_back(
        {int_at(spot, "brand"), int_at(spot, "pos"), int_at(spot, "stocks")});
  }
  config.agents = int_array(root.at("agents"));
  config.fuel_limit = int_at(root, "fuelLimits");
  config.players = int_at(root, "players");
  config.busy_threshold = int_at(root, "busyThreshold");
  config.jammed_threshold = int_at(root, "jammedThreshold");
  validate_config(config);
  return config;
}

DayInfo parse_day_info(const json::value& value) {
  const auto& root = value.as_object();
  DayInfo result;
  if (const auto* ends = root.if_contains("endsAt"); ends && !ends->is_null()) {
    result.ends_at = ends->to_number<double>();
  }
  result.day = int_at(root, "day");
  auto parse_agent = [](const json::value& item) {
    const auto& agent = item.as_object();
    return AgentView{static_cast<AgentKind>(int_at(agent, "kind")),
                     int_at(agent, "pos"), int_at(agent, "fuel")};
  };
  for (const auto& item : root.at("agents").as_array()) {
    result.agents.push_back(parse_agent(item));
  }
  if (const auto* others = root.if_contains("others")) {
    for (const auto& item : others->as_array()) {
      const auto& object = item.as_object();
      OtherTeamView team;
      const auto& id = object.at("id");
      team.id = id.is_string() ? std::string(id.as_string())
                               : std::to_string(id.to_number<std::int64_t>());
      for (const auto& agent : object.at("agents").as_array()) {
        team.agents.push_back(parse_agent(agent));
      }
      result.others.push_back(std::move(team));
    }
  }
  for (const auto& item : root.at("traffics").as_array()) {
    const auto& traffic = item.as_object();
    result.traffics[int_at(traffic, "pos")] = int_at(traffic, "status");
  }
  return result;
}

PolicyHistory parse_history(const json::value& value) {
  PolicyHistory history;
  if (value.is_null()) return history;
  const auto& root = value.as_object();
  if (const auto* brands = root.if_contains("distinct_brands")) {
    for (int brand : int_array(*brands)) history.distinct_brands.insert(brand);
  }
  if (const auto* submitted = root.if_contains("submitted_actions")) {
    for (const auto& day : submitted->as_array()) {
      ActionPlan plan;
      for (const auto& row : day.as_array()) {
        plan.push_back(int_array(row));
      }
      history.submitted_actions.push_back(std::move(plan));
    }
  }
  return history;
}

json::value to_json(const AgentTypes& types) {
  json::array result;
  for (auto type : types) result.push_back(static_cast<int>(type));
  return result;
}

json::value to_json(const ActionPlan& actions) {
  json::array result;
  for (const auto& row : actions) {
    json::array values;
    for (int action : row) values.push_back(action);
    result.push_back(std::move(values));
  }
  return result;
}

json::value to_json(const EvaluationResult& result) {
  json::array errors;
  for (const auto& error : result.errors) errors.push_back(json::value(error));
  json::array daily_scores;
  for (const auto& score : result.daily_scores) {
    daily_scores.push_back(
        json::object{{"distinct_types", score.distinct_types},
                     {"daily_types", score.cumulative_daily_types},
                     {"servings", score.total_servings}});
  }
  return json::object{
      {"score",
       json::object{{"distinct_types", result.score.distinct_types},
                    {"cumulative_daily_types",
                     result.score.cumulative_daily_types},
                    {"total_servings", result.score.total_servings}}},
      {"valid_days", result.valid_days},
      {"invalid_days", result.invalid_days},
      {"patrol_agents", result.patrol_agents},
      {"refuel_agents", result.refuel_agents},
      {"refuel_events", result.refuel_events},
      {"ending_patrol_fuel", result.ending_patrol_fuel},
      {"daily_scores", std::move(daily_scores)},
      {"errors", std::move(errors)}};
}

std::optional<int> neighbor(const MapConfig& config, int pos, int direction) {
  if (pos < 0 || pos >= config.width * config.height || direction < 0 ||
      direction > 5) {
    return std::nullopt;
  }
  const int row = pos / config.width;
  const int column = pos % config.width;
  static constexpr int even_offsets[6][2] = {
      {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {0, -1}};
  static constexpr int odd_offsets[6][2] = {
      {-1, -1}, {-1, 0}, {0, 1}, {1, 0}, {1, -1}, {0, -1}};
  const auto& offset = row % 2 == 0 ? even_offsets[direction]
                                    : odd_offsets[direction];
  const int next_row = row + offset[0];
  const int next_column = column + offset[1];
  if (next_row < 0 || next_row >= config.height || next_column < 0 ||
      next_column >= config.width) {
    return std::nullopt;
  }
  return next_row * config.width + next_column;
}

int terrain_time(const MapConfig& config, int pos,
                 const std::map<int, int>& road_status) {
  switch (config.cells.at(pos)) {
    case Terrain::Plain:
      return 2;
    case Terrain::Mountain:
      return 3;
    case Terrain::Road: {
      int status = 0;
      if (auto iterator = road_status.find(pos); iterator != road_status.end()) {
        status = iterator->second;
      }
      return status == 0 ? 1 : (status == 1 ? 2 : 4);
    }
    case Terrain::Pond:
      throw std::invalid_argument("pond has no travel time");
  }
  throw std::invalid_argument("unknown terrain");
}

int terrain_fuel(const MapConfig& config, int pos) {
  switch (config.cells.at(pos)) {
    case Terrain::Plain:
      return 1;
    case Terrain::Mountain:
    case Terrain::Road:
      return 2;
    case Terrain::Pond:
      throw std::invalid_argument("pond has no fuel cost");
  }
  throw std::invalid_argument("unknown terrain");
}

void validate_config(const MapConfig& config) {
  if (config.width < 1 || config.height < 1 ||
      config.cells.size() !=
          static_cast<std::size_t>(config.width * config.height)) {
    throw std::invalid_argument("invalid map dimensions");
  }
  if (config.day_steps.empty() ||
      config.day_steps.size() != config.day_seconds.size()) {
    throw std::invalid_argument("daySteps/daySeconds mismatch");
  }
  if (config.agents.empty() || config.fuel_limit <= 0 || config.players <= 0) {
    throw std::invalid_argument("invalid agent/fuel/player configuration");
  }
  std::set<int> spot_positions;
  for (const auto& spot : config.spots) {
    if (spot.pos < 0 || spot.pos >= config.width * config.height ||
        config.cells[spot.pos] != Terrain::Plain || spot.stocks <= 0 ||
        !spot_positions.insert(spot.pos).second) {
      throw std::invalid_argument("invalid spot placement");
    }
  }
  for (int pos : config.agents) {
    if (pos < 0 || pos >= config.width * config.height ||
        config.cells[pos] != Terrain::Plain || spot_positions.contains(pos)) {
      throw std::invalid_argument("invalid agent placement");
    }
  }
  if (config.busy_threshold >= config.jammed_threshold) {
    throw std::invalid_argument("traffic thresholds must be ordered");
  }
}

namespace {

bool is_routing_policy(const std::string& policy) {
  return policy == "greedy" || policy == "utility_greedy" ||
         policy == "fuel_aware" || policy == "stock_maximiser" ||
         policy == "coordinated" || policy == "local_search" ||
         policy == "lns" || policy == "alns" ||
         policy.starts_with("alns_") || policy == "aco" ||
         policy == "aco_ls";
}

std::size_t configured_workers(std::size_t tasks) {
  if (tasks <= 1) return tasks;
  std::size_t requested = 0;
  if (const char* value = std::getenv("HEXUDON_THREADS")) {
    try {
      requested = static_cast<std::size_t>(std::max(1, std::stoi(value)));
    } catch (const std::exception&) {
      requested = 1;
    }
  } else {
    requested = std::max(1u, std::thread::hardware_concurrency());
  }
  return std::min({tasks, requested, std::size_t{8}});
}

template <class Function>
auto parallel_indexed(std::size_t count, Function function) {
  using Result = std::invoke_result_t<Function, std::size_t>;
  std::vector<std::optional<Result>> slots(count);
  std::vector<std::exception_ptr> errors(count);
  const std::size_t workers = configured_workers(count);
  if (workers <= 1) {
    for (std::size_t index = 0; index < count; ++index) {
      slots[index] = function(index);
    }
  } else {
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
      threads.emplace_back([&] {
        while (true) {
          const std::size_t index = next.fetch_add(1);
          if (index >= count) return;
          try {
            slots[index] = function(index);
          } catch (...) {
            errors[index] = std::current_exception();
          }
        }
      });
    }
    for (auto& thread : threads) thread.join();
  }
  std::vector<Result> result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    if (errors[index]) std::rethrow_exception(errors[index]);
    result.push_back(std::move(*slots[index]));
  }
  return result;
}

int path_fuel_cost(const MapConfig& config, int source,
                   const std::vector<int>& directions) {
  int fuel = 0;
  int current = source;
  for (int direction : directions) {
    fuel += terrain_fuel(config, current);
    current = *neighbor(config, current, direction);
  }
  return fuel;
}

struct ResourcePath {
  int time{std::numeric_limits<int>::max()};
  int fuel{std::numeric_limits<int>::max()};
  std::vector<int> directions;
};

std::map<int, ResourcePath> resource_paths_to_spots(
    const MapConfig& config, int source, int fuel_limit, int time_limit,
    const std::map<int, int>& roads) {
  const int cells = config.width * config.height;
  const int stride = fuel_limit + 1;
  const int states = cells * stride;
  const int infinity = std::numeric_limits<int>::max() / 4;
  std::vector<int> best(states, infinity);
  std::vector<int> previous(states, -1);
  std::vector<int> previous_direction(states, -1);
  auto state_index = [stride](int pos, int fuel) { return pos * stride + fuel; };
  using QueueItem = std::tuple<int, int, int>;
  std::priority_queue<QueueItem, std::vector<QueueItem>,
                      std::greater<QueueItem>>
      queue;
  best[state_index(source, 0)] = 0;
  queue.emplace(0, 0, source);
  while (!queue.empty()) {
    auto [elapsed, used, pos] = queue.top();
    queue.pop();
    const int current_state = state_index(pos, used);
    if (best[current_state] != elapsed) continue;
    const int next_time = elapsed + terrain_time(config, pos, roads);
    const int next_fuel = used + terrain_fuel(config, pos);
    if (next_time > time_limit || next_fuel > fuel_limit) continue;
    for (int direction = 0; direction < 6; ++direction) {
      auto next = neighbor(config, pos, direction);
      if (!next || config.cells[*next] == Terrain::Pond) continue;
      const int next_state = state_index(*next, next_fuel);
      if (next_time < best[next_state]) {
        best[next_state] = next_time;
        previous[next_state] = current_state;
        previous_direction[next_state] = direction;
        queue.emplace(next_time, next_fuel, *next);
      }
    }
  }

  std::map<int, ResourcePath> result;
  for (const auto& spot : config.spots) {
    int chosen_state = -1;
    std::pair<int, int> chosen{infinity, infinity};
    for (int fuel = 0; fuel <= fuel_limit; ++fuel) {
      const int index = state_index(spot.pos, fuel);
      const auto candidate = std::pair{best[index], fuel};
      if (candidate < chosen) {
        chosen = candidate;
        chosen_state = index;
      }
    }
    if (chosen_state < 0 || chosen.first == infinity) continue;
    std::vector<int> reverse;
    for (int state = chosen_state; previous[state] >= 0;
         state = previous[state]) {
      reverse.push_back(previous_direction[state]);
    }
    std::reverse(reverse.begin(), reverse.end());
    result[spot.pos] = {chosen.first, chosen.second, std::move(reverse)};
  }
  return result;
}

std::vector<int> hungarian(const std::vector<std::vector<long long>>& cost) {
  if (cost.empty()) return {};
  const int rows = static_cast<int>(cost.size());
  const int columns = static_cast<int>(cost.front().size());
  if (rows > columns) throw std::invalid_argument("hungarian rows exceed columns");
  std::vector<long long> u(rows + 1), v(columns + 1);
  std::vector<int> p(columns + 1), way(columns + 1);
  for (int row = 1; row <= rows; ++row) {
    p[0] = row;
    std::vector<long long> minimum(columns + 1,
                                   std::numeric_limits<long long>::max() / 4);
    std::vector<bool> used(columns + 1);
    int column0 = 0;
    do {
      used[column0] = true;
      const int row0 = p[column0];
      long long delta = std::numeric_limits<long long>::max() / 4;
      int column1 = 0;
      for (int column = 1; column <= columns; ++column) {
        if (used[column]) continue;
        const long long current =
            cost[row0 - 1][column - 1] - u[row0] - v[column];
        if (current < minimum[column]) {
          minimum[column] = current;
          way[column] = column0;
        }
        if (minimum[column] < delta) {
          delta = minimum[column];
          column1 = column;
        }
      }
      for (int column = 0; column <= columns; ++column) {
        if (used[column]) {
          u[p[column]] += delta;
          v[column] -= delta;
        } else {
          minimum[column] -= delta;
        }
      }
      column0 = column1;
    } while (p[column0] != 0);
    do {
      const int column1 = way[column0];
      p[column0] = p[column1];
      column0 = column1;
    } while (column0 != 0);
  }
  std::vector<int> assignment(rows, -1);
  for (int column = 1; column <= columns; ++column) {
    if (p[column] != 0) assignment[p[column] - 1] = column - 1;
  }
  return assignment;
}

using ForcedPaths = std::map<std::size_t, std::pair<int, PathResult>>;

ForcedPaths coordinated_first_targets(const MapConfig& config,
                                      const DayInfo& day,
                                      const PolicyHistory& history,
                                      const AgentTypes& types) {
  std::vector<std::size_t> patrols;
  std::map<std::size_t, std::map<int, ResourcePath>> paths;
  for (std::size_t index = 0; index < day.agents.size(); ++index) {
    if (types[index] != AgentKind::Patrol || day.agents[index].fuel <= 2) continue;
    patrols.push_back(index);
  }
  const auto path_results = parallel_indexed(
      patrols.size(), [&](std::size_t position) {
        const auto index = patrols[position];
        return resource_paths_to_spots(
            config, day.agents[index].pos, day.agents[index].fuel,
            config.day_steps[day.day], day.traffics);
      });
  for (std::size_t position = 0; position < patrols.size(); ++position) {
    paths[patrols[position]] = path_results[position];
  }
  ForcedPaths forced;
  std::set<std::size_t> assigned;
  std::map<int, int> capacity;
  for (const auto& spot : config.spots) capacity[spot.pos] = spot.stocks;

  auto assign_types = [&](std::vector<int> brands) {
    std::vector<std::size_t> cars;
    for (auto patrol : patrols) {
      if (!assigned.contains(patrol)) cars.push_back(patrol);
    }
    if (cars.empty() || brands.empty()) return;
    constexpr long long unassigned = 1'000'000;
    constexpr long long unreachable = 2'000'000;
    const int real_columns = static_cast<int>(brands.size());
    std::vector<std::vector<long long>> matrix(
        cars.size(), std::vector<long long>(brands.size() + cars.size(), unassigned));
    std::map<std::pair<std::size_t, int>, std::pair<int, ResourcePath>> candidates;
    for (std::size_t row = 0; row < cars.size(); ++row) {
      for (int column = 0; column < real_columns; ++column) {
        const int brand = brands[column];
        std::optional<std::pair<int, ResourcePath>> best;
        for (const auto& spot : config.spots) {
          if (spot.brand != brand || capacity[spot.pos] <= 0) continue;
          auto iterator = paths[cars[row]].find(spot.pos);
          if (iterator == paths[cars[row]].end()) continue;
          auto candidate = std::pair{spot.pos, iterator->second};
          if (!best || std::tie(candidate.second.time, candidate.second.fuel,
                                candidate.first) <
                           std::tie(best->second.time, best->second.fuel,
                                    best->first)) {
            best = candidate;
          }
        }
        if (best) {
          candidates[{row, column}] = *best;
          matrix[row][column] = best->second.time * 100LL + best->second.fuel;
        } else {
          matrix[row][column] = unreachable;
        }
      }
    }
    const auto assignment = hungarian(matrix);
    for (std::size_t row = 0; row < assignment.size(); ++row) {
      const int column = assignment[row];
      if (column < 0 || column >= real_columns) continue;
      auto iterator = candidates.find({row, column});
      if (iterator == candidates.end()) continue;
      const auto& [spot_pos, path] = iterator->second;
      forced[cars[row]] =
          {spot_pos, PathResult{path.time, path.directions}};
      assigned.insert(cars[row]);
      --capacity[spot_pos];
    }
  };

  std::set<int> available;
  for (const auto& spot : config.spots) available.insert(spot.brand);
  std::vector<int> new_brands;
  std::vector<int> repeat_brands;
  for (int brand : available) {
    (history.distinct_brands.contains(brand) ? repeat_brands : new_brands)
        .push_back(brand);
  }
  assign_types(new_brands);
  assign_types(repeat_brands);

  for (auto patrol : patrols) {
    if (assigned.contains(patrol)) continue;
    std::optional<std::pair<int, ResourcePath>> best;
    for (const auto& spot : config.spots) {
      if (capacity[spot.pos] <= 0) continue;
      auto iterator = paths[patrol].find(spot.pos);
      if (iterator == paths[patrol].end()) continue;
      auto candidate = std::pair{spot.pos, iterator->second};
      if (!best || std::tie(candidate.second.time, candidate.second.fuel,
                            candidate.first) <
                       std::tie(best->second.time, best->second.fuel,
                                best->first)) {
        best = candidate;
      }
    }
    if (best) {
      forced[patrol] = {best->first,
                        PathResult{best->second.time,
                                   best->second.directions}};
      --capacity[best->first];
    }
  }
  return forced;
}

ActionPlan build_routing_plan(const std::string& policy,
                              const MapConfig& config, const DayInfo& day,
                              const PolicyHistory& history,
                              const AgentTypes& fixed_types,
                              const ForcedPaths& forced = {},
                              const SearchLimits& limits = {}) {
  const int horizon = config.day_steps[day.day];
  ActionPlan result(day.agents.size());
  std::set<int> planned_brands = history.distinct_brands;
  std::set<int> daily_brands;
  std::map<int, int> reservations;
  for (std::size_t index = 0; index < day.agents.size(); ++index) {
    int remaining = horizon;
    int current = day.agents[index].pos;
    int fuel = day.agents[index].fuel;
    if (fixed_types[index] == AgentKind::Refuel) {
      int target = current;
      int lowest_fuel = std::numeric_limits<int>::max();
      for (std::size_t patrol = 0; patrol < day.agents.size(); ++patrol) {
        if (fixed_types[patrol] != AgentKind::Patrol) continue;
        if (day.agents[patrol].fuel < lowest_fuel) {
          lowest_fuel = day.agents[patrol].fuel;
          target = day.agents[patrol].pos;
        }
      }
      if (policy == "coordinated" &&
          lowest_fuel >= std::max(2, config.fuel_limit / 4) &&
          !config.spots.empty()) {
        target = std::min_element(
                     config.spots.begin(), config.spots.end(),
                     [&](const Spot& left, const Spot& right) {
                       auto total = [&](int pos) {
                         long sum = 0;
                         for (const auto& other : config.spots) {
                           sum += shortest_path(config, pos, other.pos,
                                                day.traffics)
                                      .cost;
                         }
                         return sum;
                       };
                       return std::pair{total(left.pos), left.pos} <
                              std::pair{total(right.pos), right.pos};
                     })
                     ->pos;
      }
      auto path = shortest_path(config, current, target, day.traffics);
      for (int direction : path.directions) {
        const int duration = terrain_time(config, current, day.traffics);
        if (duration > remaining) break;
        result[index].push_back(direction);
        remaining -= duration;
        current = *neighbor(config, current, direction);
      }
      if (remaining > 0) result[index].push_back(-remaining);
      continue;
    }

    std::set<int> visited_targets;
    bool blocked_by_fuel = false;
    int planned_targets = 0;
    if (fuel < 2) {
      result[index].push_back(-remaining);
      continue;
    }
    bool use_forced = forced.contains(index);
    while (remaining > 0 && !blocked_by_fuel &&
           (limits.max_targets <= 0 || planned_targets < limits.max_targets)) {
      const Spot* chosen = nullptr;
      PathResult chosen_path;
      if (use_forced) {
        const auto& [pos, path] = forced.at(index);
        chosen = spot_at(config, pos);
        chosen_path = path;
        use_forced = false;
      } else {
        std::array<long long, 6> chosen_rank{
            std::numeric_limits<long long>::min(), 0, 0, 0, 0, 0};
        for (const auto& spot : config.spots) {
          const int remaining_stock = spot.stocks - reservations[spot.pos];
          if (visited_targets.contains(spot.pos) || remaining_stock <= 0) continue;
          auto path = shortest_path(config, current, spot.pos, day.traffics);
          if (path.cost == std::numeric_limits<int>::max()) continue;
          const int path_fuel = path_fuel_cost(config, current, path.directions);
          const int reserve = limits.fuel_reserve > 0 ? limits.fuel_reserve : 1;
          if (policy == "fuel_aware" && fuel < 2 * path_fuel + reserve) continue;
          const int is_new = planned_brands.contains(spot.brand) ? 0 : 1;
          const int is_daily = daily_brands.contains(spot.brand) ? 0 : 1;
          std::array<long long, 6> rank;
          if (policy == "utility_greedy") {
            const long long utility =
                (is_new ? 1000 : (is_daily ? 100 : 0)) + 10 - path.cost;
            rank = {utility, is_new, is_daily, remaining_stock, -path.cost,
                    -spot.pos};
          } else if (policy == "stock_maximiser") {
            rank = {remaining_stock, is_new, is_daily, -path.cost, -spot.pos, 0};
          } else {
            rank = {is_new, is_daily, remaining_stock, -path.cost, -path_fuel,
                    -spot.pos};
          }
          if (!chosen || rank > chosen_rank) {
            chosen = &spot;
            chosen_path = std::move(path);
            chosen_rank = rank;
          }
        }
      }
      if (!chosen) break;
      ++reservations[chosen->pos];
      visited_targets.insert(chosen->pos);
      ++planned_targets;
      if (chosen_path.directions.empty()) {
        result[index].push_back(-1);
        --remaining;
        planned_brands.insert(chosen->brand);
        daily_brands.insert(chosen->brand);
        continue;
      }
      bool reached = true;
      for (int direction : chosen_path.directions) {
        const int duration = terrain_time(config, current, day.traffics);
        const int fuel_cost = terrain_fuel(config, current);
        if (duration > remaining) {
          reached = false;
          break;
        }
        if (fuel_cost > fuel) {
          reached = false;
          blocked_by_fuel = true;
          break;
        }
        result[index].push_back(direction);
        remaining -= duration;
        fuel -= fuel_cost;
        current = *neighbor(config, current, direction);
        if (const Spot* incidental = spot_at(config, current)) {
          planned_brands.insert(incidental->brand);
          daily_brands.insert(incidental->brand);
        }
      }
      if (reached) {
        planned_brands.insert(chosen->brand);
        daily_brands.insert(chosen->brand);
      }
    }

    if (policy == "fuel_aware" && remaining > 0) {
      PathResult retreat;
      for (std::size_t refuel = 0; refuel < day.agents.size(); ++refuel) {
        if (fixed_types[refuel] != AgentKind::Refuel) continue;
        auto path = shortest_path(config, current, day.agents[refuel].pos,
                                  day.traffics);
        const int required = path_fuel_cost(config, current, path.directions);
        if (required <= fuel && path.cost < retreat.cost) retreat = std::move(path);
      }
      for (int direction : retreat.directions) {
        const int duration = terrain_time(config, current, day.traffics);
        const int cost = terrain_fuel(config, current);
        if (duration > remaining || cost > fuel) break;
        result[index].push_back(direction);
        remaining -= duration;
        fuel -= cost;
        current = *neighbor(config, current, direction);
      }
    }
    if (remaining > 0) result[index].push_back(-remaining);
    if (result[index].empty()) result[index].push_back(-horizon);
  }
  return result;
}

using CandidateValue = std::tuple<int, int, int, int>;

struct CandidateEvaluation {
  CandidateValue value;
  SimulationTrace trace;
  std::vector<int> ending_positions;
  std::vector<int> ending_fuel;
  std::map<int, int> road_traffic;
};

std::optional<CandidateEvaluation> evaluate_candidate(
    const MapConfig& config, const DayInfo& day, const PolicyHistory& history,
    const ActionPlan& plan) {
  TeamState team;
  team.id = "local-search";
  for (const auto& agent : day.agents) {
    team.agents.push_back({agent.kind, agent.pos, agent.fuel});
  }
  team.distinct_types = history.distinct_brands;
  team.visited_today.resize(team.agents.size());
  for (const auto& spot : config.spots) team.stock[spot.pos] = spot.stocks;
  team.history.submitted_actions.resize(day.day);
  std::map<int, int> traffic;
  SimulationTrace trace;
  if (simulate_team_day(config, team, plan, day.traffics, traffic, &trace)) {
    return std::nullopt;
  }
  int patrol_fuel = 0;
  std::vector<int> ending_positions;
  std::vector<int> ending_fuel;
  for (const auto& agent : team.agents) {
    if (agent.kind == AgentKind::Patrol) patrol_fuel += agent.fuel;
    ending_positions.push_back(agent.pos);
    ending_fuel.push_back(agent.fuel);
  }
  return CandidateEvaluation{
      CandidateValue{static_cast<int>(team.distinct_types.size()),
                     static_cast<int>(team.daily_types.size()),
                     team.total_servings, patrol_fuel},
      std::move(trace), std::move(ending_positions), std::move(ending_fuel),
      std::move(traffic)};
}

std::optional<CandidateValue> candidate_value(
    const MapConfig& config, const DayInfo& day, const PolicyHistory& history,
    const ActionPlan& plan) {
  auto evaluated = evaluate_candidate(config, day, history, plan);
  if (!evaluated) return std::nullopt;
  return evaluated->value;
}

std::vector<ActionPlan> refuel_staging_variants(
    const MapConfig& config, const DayInfo& day, const AgentTypes& types,
    const PolicyHistory& history, const ActionPlan& plan) {
  std::vector<ActionPlan> result;
  auto evaluation = evaluate_candidate(config, day, history, plan);
  if (!evaluation) return result;
  for (std::size_t agent = 0; agent < types.size(); ++agent) {
    if (types[agent] != AgentKind::Refuel || plan[agent].empty() ||
        plan[agent].back() >= 0) {
      continue;
    }
    const int available = -plan[agent].back();
    const int start = evaluation->ending_positions[agent];
    for (const auto& spot : config.spots) {
      auto path = shortest_path(config, start, spot.pos, day.traffics);
      if (path.cost <= 0 || path.cost > available) continue;
      ActionPlan staged = plan;
      staged[agent].pop_back();
      staged[agent].insert(staged[agent].end(), path.directions.begin(),
                           path.directions.end());
      if (available > path.cost) {
        staged[agent].push_back(-(available - path.cost));
      }
      if (!validate_action_plan(config, day, staged)) {
        result.push_back(std::move(staged));
        if (result.size() >= 4) return result;
      }
    }
  }
  return result;
}

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
                                   const SearchLimits& limits = {}) {
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

struct AcoPath {
  int time{};
  int fuel{};
  int variant{};
  std::vector<int> directions;
};

struct AcoGraph {
  std::vector<int> nodes;
  std::unordered_map<int, int> node_for_pos;
  std::vector<std::vector<std::vector<AcoPath>>> paths;
};

AcoGraph build_aco_graph(const MapConfig& config, const DayInfo& day,
                         const std::vector<int>& extra_nodes = {}) {
  AcoGraph graph;
  auto add_node = [&](int pos) {
    if (graph.node_for_pos.contains(pos)) return;
    graph.node_for_pos[pos] = static_cast<int>(graph.nodes.size());
    graph.nodes.push_back(pos);
  };
  for (const auto& agent : day.agents) add_node(agent.pos);
  for (const auto& spot : config.spots) add_node(spot.pos);
  for (int pos : extra_nodes) {
    if (pos >= 0 && pos < config.width * config.height &&
        config.cells[pos] != Terrain::Pond) {
      add_node(pos);
    }
  }

  const int node_count = static_cast<int>(graph.nodes.size());
  const int cell_count = config.width * config.height;
  const long long infinity = std::numeric_limits<long long>::max() / 4;
  // These scalarisations cheaply retain useful points between the fastest and
  // most fuel-efficient paths without materialising a width*height*fuel graph.
  static constexpr std::array<std::pair<int, int>, 7> weights{{
      {1, 0}, {4, 1}, {2, 1}, {1, 1}, {1, 2}, {1, 4}, {0, 1},
  }};

  graph.paths = parallel_indexed(node_count, [&](std::size_t source_index) {
    const int source_node = static_cast<int>(source_index);
    const int source = graph.nodes[source_node];
    std::vector<std::vector<AcoPath>> source_paths(
        static_cast<std::size_t>(node_count));
    for (const auto [time_weight, fuel_weight] : weights) {
      std::vector<long long> distance(cell_count, infinity);
      std::vector<int> elapsed(cell_count, std::numeric_limits<int>::max());
      std::vector<int> fuel(cell_count, std::numeric_limits<int>::max());
      std::vector<int> previous(cell_count, -1);
      std::vector<int> previous_direction(cell_count, -1);
      using QueueItem = std::tuple<long long, int, int, int>;
      std::priority_queue<QueueItem, std::vector<QueueItem>,
                          std::greater<QueueItem>>
          queue;
      distance[source] = 0;
      elapsed[source] = 0;
      fuel[source] = 0;
      queue.emplace(0, 0, 0, source);
      while (!queue.empty()) {
        auto [weighted, used_time, used_fuel, pos] = queue.top();
        queue.pop();
        if (std::tuple{weighted, used_time, used_fuel} !=
            std::tuple{distance[pos], elapsed[pos], fuel[pos]}) {
          continue;
        }
        const int move_time = terrain_time(config, pos, day.traffics);
        const int move_fuel = terrain_fuel(config, pos);
        for (int direction = 0; direction < 6; ++direction) {
          auto next = neighbor(config, pos, direction);
          if (!next || config.cells[*next] == Terrain::Pond) continue;
          const long long candidate_weight =
              weighted + time_weight * move_time + fuel_weight * move_fuel;
          const int candidate_time = used_time + move_time;
          const int candidate_fuel = used_fuel + move_fuel;
          const auto candidate =
              std::tuple{candidate_weight, candidate_time, candidate_fuel,
                         direction, pos};
          const auto incumbent =
              std::tuple{distance[*next], elapsed[*next], fuel[*next],
                         previous_direction[*next], previous[*next]};
          if (candidate >= incumbent) continue;
          distance[*next] = candidate_weight;
          elapsed[*next] = candidate_time;
          fuel[*next] = candidate_fuel;
          previous[*next] = pos;
          previous_direction[*next] = direction;
          queue.emplace(candidate_weight, candidate_time, candidate_fuel,
                        *next);
        }
      }

      for (int target_node = 0; target_node < node_count; ++target_node) {
        const int target = graph.nodes[target_node];
        if (distance[target] == infinity) continue;
        std::vector<int> reverse;
        for (int pos = target; pos != source; pos = previous[pos]) {
          if (pos < 0 || previous[pos] < 0) {
            reverse.clear();
            break;
          }
          reverse.push_back(previous_direction[pos]);
        }
        if (target != source && reverse.empty()) continue;
        std::reverse(reverse.begin(), reverse.end());
        auto& options = source_paths[target_node];
        const bool duplicate = std::any_of(
            options.begin(), options.end(), [&](const AcoPath& option) {
              return option.directions == reverse;
            });
        if (!duplicate) {
          options.push_back(
              {elapsed[target], fuel[target], 0, std::move(reverse)});
        }
      }
    }
    return source_paths;
  });

  for (auto& from : graph.paths) {
    for (auto& options : from) {
      std::vector<AcoPath> nondominated;
      for (std::size_t candidate = 0; candidate < options.size(); ++candidate) {
        bool dominated = false;
        for (std::size_t other = 0; other < options.size(); ++other) {
          if (candidate == other) continue;
          if (options[other].time <= options[candidate].time &&
              options[other].fuel <= options[candidate].fuel &&
              (options[other].time < options[candidate].time ||
               options[other].fuel < options[candidate].fuel)) {
            dominated = true;
            break;
          }
        }
        if (!dominated) nondominated.push_back(std::move(options[candidate]));
      }
      options = std::move(nondominated);
      std::sort(options.begin(), options.end(),
                [](const AcoPath& left, const AcoPath& right) {
                  return std::tie(left.time, left.fuel, left.directions) <
                         std::tie(right.time, right.fuel, right.directions);
                });
      for (std::size_t index = 0; index < options.size(); ++index) {
        options[index].variant = static_cast<int>(index);
      }
    }
  }
  return graph;
}

std::vector<int> alns_transit_nodes(const MapConfig& config,
                                    const AcoGraph& graph,
                                    std::size_t limit = 8,
                                    bool include_path_variants = false) {
  std::unordered_map<int, int> frequency;
  for (std::size_t from = 0; from < graph.nodes.size(); ++from) {
    for (std::size_t to = 0; to < graph.nodes.size(); ++to) {
      if (from == to || graph.paths[from][to].empty()) continue;
      const std::size_t variants = include_path_variants
                                       ? graph.paths[from][to].size()
                                       : std::size_t{1};
      for (std::size_t variant = 0; variant < variants; ++variant) {
        int pos = graph.nodes[from];
        for (int direction : graph.paths[from][to][variant].directions) {
          pos = *neighbor(config, pos, direction);
          if (!graph.node_for_pos.contains(pos)) ++frequency[pos];
        }
      }
    }
  }
  std::vector<std::pair<int, int>> ranked;
  ranked.reserve(frequency.size());
  for (const auto [pos, uses] : frequency) {
    if (uses >= 2) ranked.emplace_back(-uses, pos);
  }
  std::sort(ranked.begin(), ranked.end());
  std::vector<int> result;
  for (std::size_t index = 0; index < std::min(limit, ranked.size());
       ++index) {
    result.push_back(ranked[index].second);
  }
  return result;
}

using AcoMeetingList = std::array<int, 12>;

std::vector<AcoMeetingList> build_aco_meeting_cache(const AcoGraph& graph,
                                                    std::size_t capacity = 6) {
  capacity = std::clamp<std::size_t>(capacity, 3, AcoMeetingList{}.size());
  const std::size_t count = graph.nodes.size();
  auto blocks = parallel_indexed(count, [&](std::size_t patrol_node) {
    std::vector<AcoMeetingList> block(count * count);
    for (std::size_t refuel_node = 0; refuel_node < count; ++refuel_node) {
      for (std::size_t target_node = 0; target_node < count; ++target_node) {
        auto& meetings = block[refuel_node * count + target_node];
        meetings.fill(-1);
        std::vector<std::pair<int, int>> ranked;
        ranked.reserve(count);
        for (std::size_t node = 0; node < count; ++node) {
          const auto& patrol_paths = graph.paths[patrol_node][node];
          const auto& refuel_paths = graph.paths[refuel_node][node];
          const auto& target_paths = graph.paths[node][target_node];
          if (patrol_paths.empty() || refuel_paths.empty() ||
              target_paths.empty()) {
            continue;
          }
          ranked.emplace_back(patrol_paths.front().time +
                                  refuel_paths.front().time +
                                  target_paths.front().time,
                              static_cast<int>(node));
        }
        const std::size_t nearest =
            std::min<std::size_t>(capacity - 3, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + nearest, ranked.end());
        std::size_t used = 0;
        auto add = [&](int node) {
          if (node < 0 || used >= capacity ||
              std::find(meetings.begin(), meetings.begin() + used, node) !=
                  meetings.begin() + used) {
            return;
          }
          meetings[used++] = node;
        };
        add(static_cast<int>(patrol_node));
        add(static_cast<int>(refuel_node));
        add(static_cast<int>(target_node));
        for (std::size_t index = 0; index < nearest; ++index) {
          add(ranked[index].second);
        }
      }
    }
    return block;
  });

  std::vector<AcoMeetingList> result;
  result.reserve(count * count * count);
  for (auto& block : blocks) {
    result.insert(result.end(),
                  std::make_move_iterator(block.begin()),
                  std::make_move_iterator(block.end()));
  }
  return result;
}

struct LnsSkeleton {
  std::vector<std::vector<int>> routes;  // spot indices, keyed by agent

  bool operator==(const LnsSkeleton&) const = default;
};

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
    add(static_cast<std::uint64_t>(agent.fuel + 1));
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

enum AlnsFeature : unsigned {
  AlnsLookahead = 1U << 0U,
  AlnsExactRepair = 1U << 1U,
  AlnsStableTravel = 1U << 2U,
  AlnsRefuelDestroy = 1U << 3U,
  AlnsCorridorMeetings = 1U << 4U,
  AlnsCombinationWeights = 1U << 5U,
  AlnsSharedPreprocessing = 1U << 6U,
  AlnsExactCompletion = 1U << 7U,
  AlnsScarceBrand = 1U << 8U,
  AlnsLastChance = 1U << 9U,
  AlnsReachability = 1U << 10U,
  AlnsStockAllocation = 1U << 11U,
  AlnsRendezvousCost = 1U << 12U,
  AlnsFuelCriticality = 1U << 13U,
  AlnsOpportunityCost = 1U << 14U,
  AlnsCongestionAvoidance = 1U << 15U,
  AlnsEndPositioning = 1U << 16U,
  AlnsExactOrdering = 1U << 17U,
  AlnsExactReachableBound = 1U << 18U,
  AlnsExactServingBound = 1U << 19U,
  AlnsExactStockBound = 1U << 20U,
  AlnsExactFuelBound = 1U << 21U,
  AlnsExactDominance = 1U << 22U,
  AlnsMatchBeam = 1U << 23U,
  AlnsAcoSeed = 1U << 24U,
};

constexpr unsigned kCurrentAlnsFeatures =
    AlnsStableTravel | AlnsSharedPreprocessing | AlnsExactCompletion;
constexpr unsigned kAlnsHeuristicFeatures =
    AlnsScarceBrand | AlnsLastChance | AlnsReachability |
    AlnsStockAllocation | AlnsRendezvousCost | AlnsFuelCriticality |
    AlnsOpportunityCost | AlnsCongestionAvoidance | AlnsEndPositioning |
    AlnsExactOrdering | AlnsExactReachableBound | AlnsExactServingBound |
    AlnsExactStockBound | AlnsExactFuelBound | AlnsExactDominance;
constexpr unsigned kAlnsExactHeuristicFeatures =
    AlnsExactOrdering | AlnsExactReachableBound | AlnsExactServingBound |
    AlnsExactStockBound | AlnsExactFuelBound | AlnsExactDominance;
constexpr unsigned kAcceptedExactBoundFeatures =
    AlnsExactReachableBound | AlnsExactServingBound | AlnsExactStockBound |
    AlnsExactFuelBound;
constexpr unsigned kProductionAlnsFeatures =
    kCurrentAlnsFeatures | kAcceptedExactBoundFeatures | AlnsAcoSeed;

int lns_path_congestion(const MapConfig& config, const DayInfo& day,
                        const AcoGraph& graph, int from, int to) {
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
  int regret{};
  int score{};
};

void repair_lns_skeleton(const MapConfig& config, const DayInfo& day,
                         const PolicyHistory& history,
                         const AgentTypes& types, const AcoGraph& graph,
                         LnsSkeleton& skeleton, int mode,
                         std::mt19937_64& random,
                         unsigned features = 0U) {
  const int horizon = config.day_steps[day.day];
  int maximum_visits = 0;
  int patrol_count = 0;
  for (auto type : types) patrol_count += type == AgentKind::Patrol;
  for (const auto& spot : config.spots) {
    maximum_visits += std::min(spot.stocks, patrol_count);
  }
  std::map<int, int> brand_spots;
  std::map<int, int> brand_stock;
  for (const auto& spot : config.spots) {
    ++brand_spots[spot.brand];
    brand_stock[spot.brand] += spot.stocks;
  }

  auto total_visits = [&] {
    std::size_t total = 0;
    for (const auto& route : skeleton.routes) total += route.size();
    return static_cast<int>(total);
  };

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
    std::map<int, int> brand_opportunities;
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
          int score = 0;
          if ((features & AlnsScarceBrand) != 0U) {
            score += 64 / std::max(1, brand_spots[brand]);
            score += 64 / std::max(1, brand_stock[brand]);
          }
          if ((features & AlnsReachability) != 0U) {
            const int slack = horizon - current_time - delta;
            score += 64 * std::max(0, slack) / std::max(1, horizon);
          }
          if ((features & AlnsStockAllocation) != 0U) {
            const int remaining = config.spots[spot].stocks - assigned[spot];
            score += 16 * std::min(4, remaining);
          }
          if ((features & AlnsRendezvousCost) != 0U) {
            int rendezvous = horizon;
            for (std::size_t refuel = 0; refuel < types.size(); ++refuel) {
              if (types[refuel] != AgentKind::Refuel) continue;
              const int refuel_node =
                  graph.node_for_pos.at(day.agents[refuel].pos);
              rendezvous = std::min(rendezvous,
                                    lns_path_time(graph, refuel_node, target));
            }
            score += std::max(0, 64 - rendezvous);
          }
          if ((features & AlnsFuelCriticality) != 0U) {
            const auto& first_paths = graph.paths[previous][target];
            const auto& second_paths = next < 0 ? graph.paths[target][target]
                                                : graph.paths[target][next];
            const int first_fuel = first_paths.empty() ? config.fuel_limit
                                                        : first_paths.front().fuel;
            const int second_fuel =
                next < 0 ? 0 : (second_paths.empty() ? config.fuel_limit
                                                     : second_paths.front().fuel);
            const int margin = day.agents[agent].fuel - first_fuel - second_fuel;
            score += std::clamp(margin, -64, 64);
          }
          if ((features & AlnsOpportunityCost) != 0U) {
            const int displacement = std::max(0, second - before);
            score += std::max(0, 64 - 4 * displacement);
          }
          if ((features & AlnsCongestionAvoidance) != 0U) {
            int congestion = lns_path_congestion(config, day, graph, previous,
                                                 target);
            if (next >= 0) {
              congestion +=
                  lns_path_congestion(config, day, graph, target, next);
              congestion -=
                  lns_path_congestion(config, day, graph, previous, next);
            }
            score += std::max(-64, 64 - 8 * congestion);
          }
          if ((features & AlnsEndPositioning) != 0U &&
              position == route.size()) {
            int onward = horizon;
            for (std::size_t other = 0; other < config.spots.size(); ++other) {
              if (other == spot || assigned[other] >= config.spots[other].stocks)
                continue;
              const int other_node =
                  graph.node_for_pos.at(config.spots[other].pos);
              onward = std::min(onward,
                                lns_path_time(graph, target, other_node));
            }
            score += std::max(0, 64 - onward);
          }
          options.push_back({static_cast<int>(spot), agent, position, tier,
                             delta, 0, score});
        }
      }
      if (options.empty()) continue;
      std::sort(options.begin(), options.end(), [](const auto& left,
                                                   const auto& right) {
        return std::tie(left.delta, left.agent, left.position) <
               std::tie(right.delta, right.agent, right.position);
      });
      options.front().regret =
          options.size() > 1 ? options[1].delta - options[0].delta : horizon;
      candidates.push_back(options.front());
      ++brand_opportunities[brand];
    }
    if (candidates.empty()) break;
    std::sort(candidates.begin(), candidates.end(), [&](const auto& left,
                                                        const auto& right) {
      if (mode == 1 || mode == 3) {
        if (left.tier != right.tier) return left.tier > right.tier;
        if (left.regret != right.regret) return left.regret > right.regret;
      }
      int left_score = left.score;
      int right_score = right.score;
      if ((features & AlnsLastChance) != 0U) {
        left_score +=
            128 / std::max(1, brand_opportunities[config.spots[left.spot].brand]);
        right_score += 128 /
                       std::max(1, brand_opportunities[
                                       config.spots[right.spot].brand]);
      }
      if (left_score != right_score) return left_score > right_score;
      return std::tie(left.delta, left.agent, left.position) <
             std::tie(right.delta, right.agent, right.position);
    });
    std::size_t selected = 0;
    if (mode == 2 || mode == 3) {
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

std::vector<LnsSkeleton> build_coverage_skeletons(
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

bool destroy_alns_refuel_bottleneck(LnsSkeleton& skeleton,
                                    const SimulationTrace& trace,
                                    int remove_count,
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

using AlnsTravelChoices = std::vector<std::vector<std::uint32_t>>;

std::optional<ActionPlan> decode_lns_skeleton(
    const MapConfig& config, const DayInfo& day, const AgentTypes& types,
    const AcoGraph& graph, const std::vector<AcoMeetingList>& meeting_cache,
    const LnsSkeleton& skeleton,
    const AlnsTravelChoices* travel_choices = nullptr) {
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
    } else if (travel_choices) {
      return std::nullopt;
    }
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

ActionPlan build_coverage_plan(const MapConfig& config, const DayInfo& day,
                               const PolicyHistory& history,
                               const AgentTypes& types) {
  const auto graph = build_aco_graph(config, day);
  const auto meetings = build_aco_meeting_cache(graph);
  ActionPlan best = wait_plan(types.size(), config.day_steps[day.day]);
  auto best_value = candidate_value(config, day, history, best);
  for (const auto& skeleton :
       build_coverage_skeletons(config, day, types, graph)) {
    auto plan = decode_lns_skeleton(config, day, types, graph, meetings,
                                    skeleton);
    if (!plan) continue;
    auto value = candidate_value(config, day, history, *plan);
    if (value && (!best_value || *value > *best_value)) {
      best = std::move(*plan);
      best_value = value;
    }
  }
  return best;
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
                          const AcoGraph* shared_graph = nullptr,
                          const std::vector<AcoMeetingList>* shared_meetings =
                              nullptr) {
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

constexpr unsigned kAllAlnsFeatures =
    AlnsLookahead | AlnsExactRepair | AlnsStableTravel |
    AlnsRefuelDestroy | AlnsCorridorMeetings | AlnsCombinationWeights |
    AlnsSharedPreprocessing;

unsigned alns_features_for_policy(const std::string& policy) {
  if (policy == "lns" || policy == "alns") {
    return kProductionAlnsFeatures;
  }
  if (policy == "alns_all_patrol") return kProductionAlnsFeatures;
  if (policy == "alns_current") return kCurrentAlnsFeatures;
  if (policy == "alns_h_all") {
    return kCurrentAlnsFeatures | kAlnsHeuristicFeatures;
  }
  if (policy == "alns_h_exact_all") {
    return kCurrentAlnsFeatures | kAlnsExactHeuristicFeatures;
  }
  if (policy == "alns_h_bounds") {
    return kCurrentAlnsFeatures | kAcceptedExactBoundFeatures;
  }
  if (policy == "alns_match") {
    return kCurrentAlnsFeatures | kAcceptedExactBoundFeatures | AlnsMatchBeam;
  }
  if (policy == "alns_h_scarce")
    return kCurrentAlnsFeatures | AlnsScarceBrand;
  if (policy == "alns_h_last_chance")
    return kCurrentAlnsFeatures | AlnsLastChance;
  if (policy == "alns_h_reachability")
    return kCurrentAlnsFeatures | AlnsReachability;
  if (policy == "alns_h_stock")
    return kCurrentAlnsFeatures | AlnsStockAllocation;
  if (policy == "alns_h_rendezvous")
    return kCurrentAlnsFeatures | AlnsRendezvousCost;
  if (policy == "alns_h_fuel")
    return kCurrentAlnsFeatures | AlnsFuelCriticality;
  if (policy == "alns_h_opportunity")
    return kCurrentAlnsFeatures | AlnsOpportunityCost;
  if (policy == "alns_h_congestion")
    return kCurrentAlnsFeatures | AlnsCongestionAvoidance;
  if (policy == "alns_h_end_position")
    return kCurrentAlnsFeatures | AlnsEndPositioning;
  if (policy == "alns_h_exact_order")
    return kCurrentAlnsFeatures | AlnsExactOrdering;
  if (policy == "alns_h_reachable_bound")
    return kCurrentAlnsFeatures | AlnsExactReachableBound;
  if (policy == "alns_h_serving_bound")
    return kCurrentAlnsFeatures | AlnsExactServingBound;
  if (policy == "alns_h_stock_bound")
    return kCurrentAlnsFeatures | AlnsExactStockBound;
  if (policy == "alns_h_fuel_bound")
    return kCurrentAlnsFeatures | AlnsExactFuelBound;
  if (policy == "alns_h_dominance")
    return kCurrentAlnsFeatures | AlnsExactDominance;
  if (policy == "alns_exact") return AlnsExactCompletion;
  if (policy == "alns_survivor") {
    return AlnsStableTravel | AlnsSharedPreprocessing;
  }
  if (policy == "alns_only_lookahead") return AlnsLookahead;
  if (policy == "alns_only_exact_repair") return AlnsExactRepair;
  if (policy == "alns_only_stable_travel") return AlnsStableTravel;
  if (policy == "alns_only_refuel_destroy") return AlnsRefuelDestroy;
  if (policy == "alns_only_corridor") return AlnsCorridorMeetings;
  if (policy == "alns_only_combo_weights") return AlnsCombinationWeights;
  if (policy == "alns_only_shared_preprocess") {
    return AlnsSharedPreprocessing;
  }
  if (policy == "alns_candidate") return kAllAlnsFeatures;
  if (policy == "alns_no_lookahead") {
    return kAllAlnsFeatures & ~AlnsLookahead;
  }
  if (policy == "alns_no_exact_repair") {
    return kAllAlnsFeatures & ~AlnsExactRepair;
  }
  if (policy == "alns_no_stable_travel") {
    return kAllAlnsFeatures & ~AlnsStableTravel;
  }
  if (policy == "alns_no_refuel_destroy") {
    return kAllAlnsFeatures & ~AlnsRefuelDestroy;
  }
  if (policy == "alns_no_corridor") {
    return kAllAlnsFeatures & ~AlnsCorridorMeetings;
  }
  if (policy == "alns_no_combo_weights") {
    return kAllAlnsFeatures & ~AlnsCombinationWeights;
  }
  if (policy == "alns_no_shared_preprocess") {
    return kAllAlnsFeatures & ~AlnsSharedPreprocessing;
  }
  return 0U;
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

std::int64_t alns_ordinal(const CandidateValue& value,
                          const MapConfig& config,
                          const AgentTypes& types) {
  std::set<int> brands;
  int maximum_servings = 0;
  for (const auto& spot : config.spots) {
    brands.insert(spot.brand);
    maximum_servings += spot.stocks;
  }
  const int patrols = static_cast<int>(
      std::count(types.begin(), types.end(), AgentKind::Patrol));
  const std::int64_t brand_base = static_cast<std::int64_t>(brands.size()) + 1;
  const std::int64_t serving_base =
      static_cast<std::int64_t>(maximum_servings) + 1;
  const std::int64_t fuel_base =
      static_cast<std::int64_t>(patrols) * config.fuel_limit + 1;
  auto [distinct, daily, servings, fuel] = value;
  return (((static_cast<std::int64_t>(distinct) * brand_base + daily) *
               serving_base +
           servings) *
              fuel_base +
          fuel);
}

std::tuple<int, int, int> alns_official_value(const CandidateValue& value) {
  return {std::get<0>(value), std::get<1>(value), std::get<2>(value)};
}

void exact_repair_alns_solution(
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

std::tuple<int, int, int> alns_terminal_value(
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

struct ExactDayResult {
  ActionPlan plan;
  CandidateValue value;
  std::int64_t explored_nodes{};
  bool complete{};
};

ExactDayResult exact_day_search(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& types,
    const ActionPlan& incumbent_plan, const CandidateValue& incumbent_value,
    const CandidateEvaluation& incumbent_evaluation,
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
      (features & (AlnsExactOrdering | AlnsExactReachableBound |
                   AlnsExactStockBound | AlnsExactFuelBound)) != 0U;
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
  std::uint64_t incumbent_new_brands = 0;
  for (const auto& acquisition : incumbent_evaluation.trace.acquisitions) {
    auto spot_iterator = spot_for_pos.find(acquisition.spot_pos);
    if (spot_iterator == spot_for_pos.end()) continue;
    const auto bit =
        std::uint64_t{1} << spot_brand[spot_iterator->second];
    if ((initially_known & bit) == 0) incumbent_new_brands |= bit;
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
  bool optimum_reached = result.value == absolute_upper;
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
    return upper > result.value;
  };

  std::set<std::vector<int>> exact_states;
  auto first_visit = [&](const ExactDayState& state) {
    if ((features & AlnsExactDominance) == 0U) return true;
    std::vector<int> key;
    key.reserve(8 + state.positions.size() * 8 + state.stock.size() +
                state.traffic.size());
    key.push_back(state.step);
    for (std::size_t agent = 0; agent < agent_count; ++agent) {
      key.push_back(state.positions[agent]);
      key.push_back(state.fuel[agent]);
      key.push_back(static_cast<int>(state.visited[agent]));
      key.push_back(static_cast<int>(state.visited[agent] >> 32U));
      const auto& pending = state.pending[agent];
      key.push_back(pending.active);
      key.push_back(pending.move);
      key.push_back(pending.remaining);
      key.push_back(pending.destination);
      key.push_back(pending.fuel_cost);
    }
    key.insert(key.end(), state.stock.begin(), state.stock.end());
    key.push_back(static_cast<int>(state.daily_brands));
    key.push_back(static_cast<int>(state.daily_brands >> 32U));
    key.push_back(static_cast<int>(state.new_brands));
    key.push_back(static_cast<int>(state.new_brands >> 32U));
    key.push_back(state.servings);
    if (day.day + 1 < static_cast<int>(config.day_steps.size())) {
      key.insert(key.end(), state.traffic.begin(), state.traffic.end());
    }
    return exact_states.insert(std::move(key)).second;
  };

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
      const bool unrestricted_continuation =
          (features & AlnsMatchBeam) != 0U ||
          day.day + 1 >= static_cast<int>(config.day_steps.size());
      bool continuation_safe = unrestricted_continuation;
      if (!unrestricted_continuation) {
        continuation_safe =
            state.positions == incumbent_evaluation.ending_positions;
        if (continuation_safe && state.new_brands != incumbent_new_brands) {
          continuation_safe = false;
        }
      }
      for (std::size_t agent = 0; !unrestricted_continuation &&
                                      continuation_safe && agent < agent_count;
           ++agent) {
        if (types[agent] == AgentKind::Patrol &&
            state.fuel[agent] != incumbent_evaluation.ending_fuel[agent]) {
          continuation_safe = false;
        }
      }
      for (int pos = 0; !unrestricted_continuation && continuation_safe &&
                        pos < config.width * config.height; ++pos) {
        const int incumbent_traffic =
            incumbent_evaluation.road_traffic.contains(pos)
                ? incumbent_evaluation.road_traffic.at(pos)
                : 0;
        if (state.traffic[pos] != incumbent_traffic) continuation_safe = false;
      }
      if (continuation_safe && value > result.value) {
        result.value = value;
        result.plan = state.actions;
        optimum_reached = value == absolute_upper;
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
    if ((features & AlnsExactOrdering) != 0U) {
      auto choice_score = [&](int choice) {
        int score = choice == preferred ? 500000 : 0;
        if (choice < 0) return score - 1000;
        const int destination = *neighbor(config, state.positions[agent], choice);
        if (types[agent] == AgentKind::Patrol) {
          auto spot_iterator = spot_for_pos.find(destination);
          if (spot_iterator != spot_for_pos.end()) {
            const int spot = spot_iterator->second;
            const auto visited_bit = std::uint64_t{1} << spot;
            if ((state.visited[agent] & visited_bit) == 0 &&
                state.stock[spot] > 0) {
              const auto brand_bit = std::uint64_t{1} << spot_brand[spot];
              if ((initially_known & brand_bit) == 0 &&
                  (state.new_brands & brand_bit) == 0) {
                score += 1000000;
              } else if ((state.daily_brands & brand_bit) == 0) {
                score += 100000;
              }
              score += 1000 * std::min(9, state.stock[spot]);
            }
          }
          for (std::size_t other = 0; other < agent_count; ++other) {
            if (types[other] == AgentKind::Refuel &&
                state.positions[other] == destination &&
                state.fuel[agent] < config.fuel_limit) {
              score += 10000;
            }
          }
        } else {
          for (std::size_t other = 0; other < agent_count; ++other) {
            if (types[other] == AgentKind::Patrol &&
                state.positions[other] == destination &&
                state.fuel[other] < config.fuel_limit) {
              score += 10000;
            }
          }
        }
        int nearest = horizon;
        for (std::size_t spot = 0; spot < config.spots.size(); ++spot) {
          if (state.stock[spot] <= 0) continue;
          nearest = std::min(nearest, spot_steps[spot][destination]);
        }
        score += std::max(0, 100 - nearest);
        if (config.cells[state.positions[agent]] == Terrain::Road) {
          auto iterator = day.traffics.find(state.positions[agent]);
          score -= 10 * (iterator == day.traffics.end() ? 0
                                                        : iterator->second);
        }
        return score;
      };
      std::stable_sort(choices.begin(), choices.end(), [&](int left, int right) {
        return choice_score(left) > choice_score(right);
      });
    } else if (preferred != -2) {
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

AlnsMatchScore alns_match_rollout(
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

ActionPlan build_alns_plan(const MapConfig& config, const DayInfo& day,
                           const PolicyHistory& history,
                           const AgentTypes& types,
                           const SearchLimits& limits,
                           unsigned features) {
  const auto started = std::chrono::steady_clock::now();
  const bool timed = limits.time_limit_ms >= 0;
  const auto deadline =
      started + std::chrono::milliseconds(std::max(0, limits.time_limit_ms));
  const bool match_search =
      (features & AlnsMatchBeam) != 0U &&
      ((!timed && limits.max_iterations > 96) ||
       (timed && limits.time_limit_ms >= 250)) &&
      day.day + 1 < static_cast<int>(config.day_steps.size());
  const auto alns_deadline =
      match_search && timed
          ? started + std::chrono::milliseconds(
                          std::max(0, limits.time_limit_ms * 5 / 10))
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
  if (!expired() && (features & AlnsAcoSeed) != 0U &&
      (!timed || limits.time_limit_ms >= 50)) {
    SearchLimits aco_limits;
    if (timed && limits.time_limit_ms < 1000) {
      aco_limits.aco_ants = limits.time_limit_ms < 250 ? 4 : 8;
      aco_limits.aco_iterations = limits.time_limit_ms < 250 ? 4 : 8;
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
  if (!expired() && (!timed || limits.time_limit_ms >= 500)) {
    seeds.push_back(build_local_search_plan(config, day, history, types, limits));
  }

  ActionPlan best = seeds.front();
  CandidateEvaluation best_evaluation =
      *evaluate_candidate(config, day, history, best);
  CandidateValue best_value = best_evaluation.value;
  auto online_improves = [&](const CandidateEvaluation& candidate,
                             const CandidateEvaluation& incumbent) {
    return alns_official_value(candidate.value) >
           alns_official_value(incumbent.value);
  };
  for (const auto& seed : seeds) {
    if (auto evaluation = evaluate_candidate(config, day, history, seed);
        evaluation && online_improves(*evaluation, best_evaluation)) {
      best = seed;
      best_value = evaluation->value;
      best_evaluation = std::move(*evaluation);
    }
  }
  std::tuple<int, int, int> best_terminal =
      alns_official_value(best_value);
  if ((features & AlnsLookahead) != 0U) {
    best_terminal =
        alns_terminal_value(config, day, history, types, best_evaluation);
    for (const auto& seed : seeds) {
      auto evaluation = evaluate_candidate(config, day, history, seed);
      if (!evaluation) continue;
      const auto terminal =
          alns_terminal_value(config, day, history, types, *evaluation);
      const auto prefix =
          std::tuple{std::get<0>(evaluation->value),
                     std::get<1>(evaluation->value)};
      const auto best_prefix =
          std::tuple{std::get<0>(best_value), std::get<1>(best_value)};
      if (terminal > best_terminal && prefix >= best_prefix) {
        best = seed;
        best_value = evaluation->value;
        best_evaluation = std::move(*evaluation);
        best_terminal = terminal;
      }
    }
  }
  if (expired()) return best;

  AcoGraph graph = build_aco_graph(config, day);
  if (expired()) return best;
  std::optional<std::vector<AcoMeetingList>> shared_meeting_cache;
  if ((features & AlnsSharedPreprocessing) != 0U &&
      (!timed || limits.time_limit_ms >= 250)) {
    shared_meeting_cache = build_aco_meeting_cache(graph);
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
    ActionPlan legacy = build_lns_plan(config, day, history, types,
                                       legacy_limits, &graph,
                                       &*shared_meeting_cache);
    seeds.insert(seeds.begin(), legacy);
    auto evaluation = evaluate_candidate(config, day, history, legacy);
    if (evaluation) {
      const auto terminal =
          (features & AlnsLookahead) != 0U
              ? alns_terminal_value(config, day, history, types, *evaluation)
              : alns_official_value(evaluation->value);
      const auto prefix =
          std::tuple{std::get<0>(evaluation->value),
                     std::get<1>(evaluation->value)};
      const auto best_prefix =
          std::tuple{std::get<0>(best_value), std::get<1>(best_value)};
      const bool choose_legacy =
          (features & AlnsLookahead) != 0U
              ? terminal > best_terminal && prefix >= best_prefix
              : online_improves(*evaluation, best_evaluation);
      if (choose_legacy) {
        best = std::move(legacy);
        best_value = evaluation->value;
        best_evaluation = std::move(*evaluation);
        best_terminal = terminal;
      }
    }
  }
  if (expired()) return best;
  const bool corridor = (features & AlnsCorridorMeetings) != 0U;
  std::vector<int> transit_nodes;
  if (corridor || config.spots.size() <= 15) {
    transit_nodes = alns_transit_nodes(
        config, graph, corridor ? std::size_t{24} : std::size_t{8}, corridor);
  }
  if (!transit_nodes.empty() && !expired()) {
    graph = build_aco_graph(config, day, transit_nodes);
    shared_meeting_cache.reset();
  }
  if (expired()) return best;
  auto meeting_cache = shared_meeting_cache
                           ? std::move(*shared_meeting_cache)
                           : build_aco_meeting_cache(
                                 graph, corridor ? std::size_t{12}
                                                 : std::size_t{6});
  if (expired()) return best;
  if (match_search) {
    int coverage_index = 0;
    for (const auto& skeleton :
         build_coverage_skeletons(config, day, types, graph)) {
      if (auto plan = decode_lns_skeleton(config, day, types, graph,
                                          meeting_cache, skeleton)) {
        seeds.push_back(std::move(*plan));
      }
      if (coverage_index++ != 0) continue;
      for (std::uint32_t code : {1U, 2U, 3U, 5U, 6U, 10U, 12U, 24U}) {
        AlnsTravelChoices choices(skeleton.routes.size());
        for (std::size_t agent = 0; agent < skeleton.routes.size(); ++agent) {
          choices[agent].assign(skeleton.routes[agent].size(), code);
        }
        if (auto plan = decode_lns_skeleton(config, day, types, graph,
                                            meeting_cache, skeleton,
                                            &choices)) {
          seeds.push_back(std::move(*plan));
        }
      }
    }
    const auto unstaged = seeds;
    for (const auto& seed : unstaged) {
      auto variants =
          refuel_staging_variants(config, day, types, history, seed);
      seeds.insert(seeds.end(), std::make_move_iterator(variants.begin()),
                   std::make_move_iterator(variants.end()));
    }
  }

  struct MatchRoot {
    ActionPlan plan;
    CandidateEvaluation evaluation;
    std::size_t hash{};
  };
  std::vector<MatchRoot> match_roots;
  if (match_search) {
    for (const auto& seed : seeds) {
      auto evaluation = evaluate_candidate(config, day, history, seed);
      if (!evaluation) continue;
      std::size_t hash = 0xcbf29ce484222325ULL;
      for (int pos : evaluation->ending_positions) {
        hash = (hash ^ static_cast<std::size_t>(pos + 1)) * 0x100000001b3ULL;
      }
      for (int fuel : evaluation->ending_fuel) {
        hash = (hash ^ static_cast<std::size_t>(fuel + 1)) * 0x100000001b3ULL;
      }
      for (const auto [pos, traffic] : evaluation->road_traffic) {
        hash = (hash ^ static_cast<std::size_t>(pos + 1)) * 0x100000001b3ULL;
        hash = (hash ^ static_cast<std::size_t>(traffic + 1)) *
               0x100000001b3ULL;
      }
      auto duplicate = std::find_if(match_roots.begin(), match_roots.end(),
                                    [&](const auto& root) {
                                      return root.hash == hash;
                                    });
      if (duplicate == match_roots.end()) {
        match_roots.push_back({seed, std::move(*evaluation), hash});
      } else if (evaluation->value > duplicate->evaluation.value) {
        *duplicate = {seed, std::move(*evaluation), hash};
      }
    }
    std::sort(match_roots.begin(), match_roots.end(),
              [](const auto& left, const auto& right) {
                if (left.evaluation.value != right.evaluation.value) {
                  return left.evaluation.value > right.evaluation.value;
                }
                return left.hash < right.hash;
              });
    if (match_roots.size() > 64) match_roots.resize(64);
  }

  struct Elite {
    AlnsSolution solution;
    ActionPlan plan;
    CandidateValue value;
    CandidateEvaluation evaluation;
    std::size_t hash{};
  };
  std::vector<Elite> elite;
  std::mt19937_64 zero_random(0);
  for (const auto& seed : seeds) {
    auto evaluation = evaluate_candidate(config, day, history, seed);
    auto skeleton = lns_skeleton_from_plan(config, day, history, seed);
    if (!evaluation || !skeleton) continue;
    AlnsSolution solution{std::move(*skeleton), {}};
    repair_alns_travel(solution, 0, zero_random);
    const auto hash = alns_solution_hash(solution);
    if (std::none_of(elite.begin(), elite.end(),
                     [&](const Elite& item) { return item.hash == hash; })) {
      elite.push_back({std::move(solution), seed, evaluation->value,
                       std::move(*evaluation), hash});
    }
  }
  if (elite.empty()) return best;
  auto elite_order = [](const Elite& left, const Elite& right) {
    if (left.value != right.value) return left.value > right.value;
    return left.hash < right.hash;
  };
  std::sort(elite.begin(), elite.end(), elite_order);
  const std::size_t elite_limit = match_search ? 32U : 12U;
  if (elite.size() > elite_limit) elite.resize(elite_limit);

  std::mt19937_64 random(lns_seed(config, day, history) ^ 0x414c4e53ULL);
  // Re-initialise seed travel choices with the real deterministic generator.
  for (auto& item : elite) repair_alns_travel(item.solution, 0, random);
  Elite current = elite.front();
  std::array<double, 5> destroy_weights{1, 1, 1, 1, 1};
  std::array<double, 4> repair_weights{1, 1, 1, 1};
  std::array<double, 4> travel_weights{1, 1, 1, 1};
  std::array<double, 5> destroy_scores{};
  std::array<double, 4> repair_scores{};
  std::array<double, 4> travel_scores{};
  std::array<int, 5> destroy_uses{};
  std::array<int, 4> repair_uses{};
  std::array<int, 4> travel_uses{};
  std::array<double, 80> combination_weights;
  combination_weights.fill(1.0);
  std::array<double, 80> combination_scores{};
  std::array<int, 80> combination_uses{};
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
  auto update_weights = [](auto& weights, auto& scores, auto& uses) {
    for (std::size_t index = 0; index < weights.size(); ++index) {
      if (uses[index] > 0) {
        const double observed = scores[index] / uses[index];
        weights[index] =
            std::max(0.1, 0.8 * weights[index] + 0.2 * observed);
      }
      scores[index] = 0;
      uses[index] = 0;
    }
  };

  const int minimum = std::max(0, limits.min_iterations);
  const int maximum = std::max(minimum, limits.max_iterations);
  const bool exact_requested =
      (features & AlnsExactCompletion) != 0U &&
      (maximum > 96 || (timed && limits.time_limit_ms >= 5000));
  const int alns_maximum =
      exact_requested
          ? std::min(maximum, std::max(96, std::min(2048, maximum / 4)))
          : maximum;
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

  int iterations_executed = 0;
  for (int iteration = 0; iteration < alns_maximum; ++iteration) {
    if (expired()) break;
    iterations_executed = iteration + 1;
    int combination = -1;
    int destroy = 0;
    int repair = 0;
    int travel = 0;
    if ((features & AlnsCombinationWeights) != 0U) {
      combination = choose(combination_weights);
      destroy = combination / 16;
      repair = (combination / 4) % 4;
      travel = combination % 4;
      ++combination_uses[combination];
    } else {
      destroy = choose(destroy_weights);
      repair = choose(repair_weights);
      travel = choose(travel_weights);
      ++destroy_uses[destroy];
      ++repair_uses[repair];
      ++travel_uses[travel];
    }

    AlnsSolution candidate = current.solution;
    int visits = 0;
    for (const auto& route : candidate.skeleton.routes) visits += route.size();
    if (visits > 0) {
      const bool diversify = stagnation >= diversify_after;
      const double fraction = diversify
                                  ? 0.40 + (random() % 26) / 100.0
                                  : 0.10 + (random() % 26) / 100.0;
      const int removed =
          std::max(1, static_cast<int>(std::ceil(visits * fraction)));
      const bool refuel_destroyed =
          destroy == 4 && (features & AlnsRefuelDestroy) != 0U &&
          destroy_alns_refuel_bottleneck(candidate.skeleton,
                                         current.evaluation.trace,
                                         removed, random);
      if (!refuel_destroyed) {
        destroy_lns_skeleton(config, history, graph, candidate.skeleton,
                             destroy, removed, random);
      }
    }
    repair_lns_skeleton(config, day, history, types, graph,
                        candidate.skeleton, repair, random, features);
    repair_alns_travel(candidate, travel, random,
                       (features & AlnsStableTravel) != 0U);
    if ((features & AlnsExactRepair) != 0U && iteration % 8 == 0) {
      exact_repair_alns_solution(config, day, history, types, graph,
                                 meeting_cache, candidate, random,
                                 (features & AlnsStableTravel) != 0U);
    }
    auto plan = decode_lns_skeleton(config, day, types, graph, meeting_cache,
                                    candidate.skeleton, &candidate.travel);
    std::optional<CandidateEvaluation> evaluation;
    if (plan) evaluation = evaluate_candidate(config, day, history, *plan);
    std::optional<CandidateValue> value =
        evaluation ? std::optional<CandidateValue>(evaluation->value)
                   : std::nullopt;
    double reward = 0.0;
    bool accepted = false;
    const auto current_ordinal =
        alns_ordinal(current.value, config, types);
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

      const auto priority_prefix =
          std::tuple{std::get<0>(*value), std::get<1>(*value)};
      const auto best_priority_prefix =
          std::tuple{std::get<0>(best_value), std::get<1>(best_value)};
      const bool lookahead_eligible =
          alns_official_value(*value) > alns_official_value(best_value);
      const auto candidate_terminal =
          (features & AlnsLookahead) != 0U && lookahead_eligible
              ? alns_terminal_value(config, day, history, types, *evaluation)
              : alns_official_value(*value);
      const bool global_improvement =
          (features & AlnsLookahead) != 0U
              ? lookahead_eligible && candidate_terminal > best_terminal &&
                    priority_prefix >= best_priority_prefix
              : online_improves(*evaluation, best_evaluation);
      const bool current_improvement = *value > current.value;
      const auto hash = alns_solution_hash(candidate);
      auto duplicate = std::find_if(
          elite.begin(), elite.end(),
          [&](const Elite& item) { return item.hash == hash; });
      const bool new_elite = duplicate == elite.end();
      if (global_improvement) {
        best = *plan;
        best_value = *value;
        best_evaluation = *evaluation;
        best_terminal = candidate_terminal;
        stagnation = 0;
        reward = 12.0;
      } else {
        ++stagnation;
        reward = current_improvement || new_elite ? 6.0
                                                  : (accepted ? 2.0 : 0.0);
      }
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
    } else {
      ++stagnation;
    }
    if (combination >= 0) {
      combination_scores[combination] += reward;
    } else {
      destroy_scores[destroy] += reward;
      repair_scores[repair] += reward;
      travel_scores[travel] += reward;
    }
    ++since_restart;

    if ((iteration + 1) % 32 == 0) {
      if ((features & AlnsCombinationWeights) != 0U) {
        update_weights(combination_weights, combination_scores,
                       combination_uses);
      } else {
        update_weights(destroy_weights, destroy_scores, destroy_uses);
        update_weights(repair_weights, repair_scores, repair_uses);
        update_weights(travel_weights, travel_scores, travel_uses);
      }
    }
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
    }
    if (iteration + 1 >= minimum && limits.stagnation_iterations > 0 &&
        stagnation >= limits.stagnation_iterations &&
        !match_search) {
      break;
    }
  }
  if (match_search && (!timed || std::chrono::steady_clock::now() < deadline)) {
    int beam_width = 2;
    std::size_t root_limit = 4;
    if (timed && limits.time_limit_ms >= 500) {
      beam_width = 4;
      root_limit = 8;
    }
    if (timed && limits.time_limit_ms >= 2000) {
      beam_width = 8;
      root_limit = 16;
    }
    if (timed && limits.time_limit_ms >= 10000) {
      beam_width = 16;
      root_limit = 24;
    }
    const auto rollout_deadline =
        timed ? started + std::chrono::milliseconds(
                              std::max(0, limits.time_limit_ms * 6 / 10))
              : std::chrono::steady_clock::time_point::max();
    auto selected_score = alns_match_rollout(
        config, day, history, types, best_evaluation, beam_width,
        rollout_deadline);
    const auto required_prefix =
        std::tuple{std::get<0>(best_value), std::get<1>(best_value)};
    std::size_t examined = 0;
    for (const auto& candidate : match_roots) {
      if (examined++ >= root_limit ||
          (timed && std::chrono::steady_clock::now() >= rollout_deadline)) {
        break;
      }
      if (alns_official_value(candidate.evaluation.value) <
          alns_official_value(best_value)) {
        continue;
      }
      const auto score = alns_match_rollout(
          config, day, history, types, candidate.evaluation, beam_width,
          rollout_deadline);
      if (std::tuple{score.worst, score.total,
                     alns_official_value(candidate.evaluation.value)} >
          std::tuple{selected_score.worst, selected_score.total,
                     alns_official_value(best_value)}) {
        best = candidate.plan;
        best_value = candidate.evaluation.value;
        best_evaluation = candidate.evaluation;
        selected_score = score;
      }
    }
    examined = 0;
    for (const auto& candidate : elite) {
      if (examined++ >= root_limit ||
          (timed && std::chrono::steady_clock::now() >= rollout_deadline)) {
        break;
      }
      const auto prefix =
          std::tuple{std::get<0>(candidate.value),
                     std::get<1>(candidate.value)};
      if (prefix < required_prefix ||
          alns_official_value(candidate.value) <
              alns_official_value(best_value)) {
        continue;
      }
      const auto score = alns_match_rollout(
          config, day, history, types, candidate.evaluation, beam_width,
          rollout_deadline);
      if (std::tuple{score.worst, score.total,
                     alns_official_value(candidate.value)} >
          std::tuple{selected_score.worst, selected_score.total,
                     alns_official_value(best_value)}) {
        best = candidate.plan;
        best_value = candidate.value;
        best_evaluation = candidate.evaluation;
        selected_score = score;
      }
    }
  }
  if (exact_requested &&
      (!timed || std::chrono::steady_clock::now() < deadline)) {
    const std::int64_t exact_budget =
        std::max<std::int64_t>(0, static_cast<std::int64_t>(maximum) -
                                     iterations_executed);
    const std::optional<std::chrono::steady_clock::time_point> exact_deadline =
        timed ? std::optional<std::chrono::steady_clock::time_point>(
                    match_search
                        ? started + std::chrono::milliseconds(
                                        std::max(0, limits.time_limit_ms * 9 / 10))
                        : deadline)
              : std::nullopt;
    auto exact = exact_day_search(config, day, history, types, best, best_value,
                                  best_evaluation, exact_budget,
                                  exact_deadline, features);
    if (exact.value > best_value &&
        !validate_action_plan(config, day, exact.plan)) {
      auto exact_evaluation =
          evaluate_candidate(config, day, history, exact.plan);
      bool choose_exact = exact_evaluation.has_value();
      if (choose_exact && match_search) {
        const auto comparison_deadline =
            timed ? deadline : std::chrono::steady_clock::time_point::max();
        const auto incumbent_score = alns_match_rollout(
            config, day, history, types, best_evaluation, 2,
            comparison_deadline);
        const auto exact_score = alns_match_rollout(
            config, day, history, types, *exact_evaluation, 2,
            comparison_deadline);
        choose_exact =
            std::tuple{exact_score.worst, exact_score.total,
                       alns_official_value(exact.value)} >
            std::tuple{incumbent_score.worst, incumbent_score.total,
                       alns_official_value(best_value)};
      }
      if (choose_exact) {
        best = std::move(exact.plan);
        best_value = exact.value;
        best_evaluation = std::move(*exact_evaluation);
      }
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
  std::array<MapConfig, 3> rollout_configs{config, config, config};
  const int map_span = std::max(1, config.width + config.height);
  for (std::size_t index = 0; index < rollout_configs.size(); ++index) {
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
    int residual = 0;
    for (const auto& rollout_config : rollout_configs) {
      for (bool conservative : {false, true}) {
        const auto outcome = run_type_rollout(
            rollout_config, types, smooth_graph, smooth_meetings, jam_graph,
            jam_meetings, conservative);
        worst = std::min(worst, outcome.score);
        std::get<0>(combined) += std::get<0>(outcome.score);
        std::get<1>(combined) += std::get<1>(outcome.score);
        std::get<2>(combined) += std::get<2>(outcome.score);
        residual += outcome.residual_fuel;
      }
    }
    return RankedTypes{std::move(types), worst, combined, residual,
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
    mix(agent.fuel);
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
                          const SearchLimits& limits = {}) {
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

}  // namespace

MapConfig online_information_config(const MapConfig& config, int current_day,
                                    bool initial_map_only) {
  MapConfig view = config;
  const int map_span = std::max(1, config.width + config.height);
  for (std::size_t day = 0; day < view.day_steps.size(); ++day) {
    if (initial_map_only || static_cast<int>(day) > current_day) {
      view.day_steps[day] = map_span;
      view.day_seconds[day] = 60.0;
    }
  }
  return view;
}

AgentTypes select_agent_types_online(const std::string& policy,
                                     const MapConfig& config) {
  AgentTypes types(config.agents.size(), AgentKind::Patrol);
  if (policy == "wait") return types;
  if (policy == "hotspot") {
    std::fill(types.begin(), types.end(), AgentKind::Refuel);
    return types;
  }
  if (!is_routing_policy(policy)) {
    throw std::invalid_argument("unknown policy: " + policy);
  }
  if (policy == "lns" || policy == "alns" ||
      policy.starts_with("alns_")) {
    if (policy == "alns_all_patrol") return types;
    return select_lns_agent_types(config);
  }
  const int refuel_count = std::max(1, static_cast<int>(types.size()) / 4);
  std::vector<std::pair<int, int>> candidates;
  std::map<int, int> smooth;
  for (std::size_t index = 0; index < config.agents.size(); ++index) {
    int nearest = std::numeric_limits<int>::max() / 4;
    for (const auto& spot : config.spots) {
      nearest = std::min(nearest, shortest_path(config, config.agents[index],
                                                spot.pos, smooth)
                                      .cost);
    }
    candidates.emplace_back(nearest, static_cast<int>(index));
  }
  std::sort(candidates.begin(), candidates.end(), [](auto left, auto right) {
    if (left.first != right.first) return left.first > right.first;
    return left.second > right.second;
  });
  for (int index = 0; index < refuel_count; ++index) {
    types[candidates[index].second] = AgentKind::Refuel;
  }
  return types;
}

AgentTypes select_agent_types(const std::string& policy,
                              const MapConfig& config) {
  return select_agent_types_online(
      policy, online_information_config(config, -1, true));
}

ActionPlan plan_day_online(const std::string& policy, const MapConfig& config,
                           const DayInfo& day,
                           const PolicyHistory& history,
                           const AgentTypes& fixed_types,
                           const SearchLimits& limits) {
  if (day.day < 0 || day.day >= static_cast<int>(config.day_steps.size())) {
    throw std::invalid_argument("day index outside config");
  }
  const int horizon = config.day_steps[day.day];
  if (policy == "wait") return wait_plan(day.agents.size(), horizon);
  if (day.agents.size() != fixed_types.size()) {
    throw std::invalid_argument("agent type count mismatch");
  }
  ActionPlan result(day.agents.size());

  if (policy == "hotspot") {
    std::vector<int> roads;
    for (int pos = 0; pos < config.width * config.height; ++pos) {
      if (config.cells[pos] == Terrain::Road) roads.push_back(pos);
    }
    for (std::size_t index = 0; index < day.agents.size(); ++index) {
      int remaining = horizon;
      int current = day.agents[index].pos;
      PathResult best;
      for (int road : roads) {
        auto path = shortest_path(config, current, road, day.traffics);
        if (path.cost < best.cost) best = std::move(path);
      }
      for (int direction : best.directions) {
        int duration = terrain_time(config, current, day.traffics);
        if (duration > remaining) break;
        result[index].push_back(direction);
        remaining -= duration;
        current = *neighbor(config, current, direction);
      }
      if (remaining > 0) result[index].push_back(-remaining);
    }
    return result;
  }
  if (!is_routing_policy(policy)) {
    throw std::invalid_argument("unknown policy: " + policy);
  }
  if (policy == "local_search") {
    return build_local_search_plan(config, day, history, fixed_types, limits);
  }
  if (policy == "alns_coverage") {
    return build_coverage_plan(config, day, history, fixed_types);
  }
  if (policy == "lns" || policy == "alns" ||
      policy.starts_with("alns_")) {
    return build_alns_plan(config, day, history, fixed_types, limits,
                           alns_features_for_policy(policy));
  }
  if (policy == "aco") {
    return build_aco_plan(config, day, history, fixed_types, false, limits);
  }
  if (policy == "aco_ls") {
    return build_aco_plan(config, day, history, fixed_types, true, limits);
  }
  if (policy == "coordinated") {
    const auto forced =
        coordinated_first_targets(config, day, history, fixed_types);
    return build_routing_plan(policy, config, day, history, fixed_types,
                              forced, limits);
  }
  return build_routing_plan(policy, config, day, history, fixed_types, {},
                            limits);
}

ActionPlan plan_day(const std::string& policy, const MapConfig& config,
                    const DayInfo& day, const PolicyHistory& history,
                    const AgentTypes& fixed_types,
                    const SearchLimits& limits) {
  if (day.day < 0 || day.day >= static_cast<int>(config.day_steps.size())) {
    throw std::invalid_argument("day index outside config");
  }
  const auto view = online_information_config(config, day.day, false);
  return plan_day_online(policy, view, day, history, fixed_types, limits);
}

std::optional<std::string> validate_action_plan(const MapConfig& config,
                                                const DayInfo& day,
                                                const ActionPlan& plan) {
  if (day.day < 0 || day.day >= static_cast<int>(config.day_steps.size())) {
    return "day index outside config";
  }
  TeamState team;
  team.id = "validation";
  for (const auto& agent : day.agents) {
    team.agents.push_back({agent.kind, agent.pos, agent.fuel});
  }
  team.visited_today.resize(team.agents.size());
  for (const auto& spot : config.spots) team.stock[spot.pos] = spot.stocks;
  team.history.submitted_actions.resize(day.day);
  std::map<int, int> ignored_traffic;
  return simulate_team_day(config, team, plan, day.traffics, ignored_traffic);
}

std::optional<Score> score_action_plan(const MapConfig& config,
                                       const DayInfo& day,
                                       const PolicyHistory& history,
                                       const ActionPlan& plan) {
  TeamState team;
  team.id = "score";
  team.distinct_types = history.distinct_brands;
  for (const auto& agent : day.agents) {
    team.agents.push_back({agent.kind, agent.pos, agent.fuel});
  }
  team.visited_today.resize(team.agents.size());
  for (const auto& spot : config.spots) team.stock[spot.pos] = spot.stocks;
  team.history.submitted_actions.resize(day.day);
  std::map<int, int> ignored_traffic;
  if (simulate_team_day(config, team, plan, day.traffics, ignored_traffic)) {
    return std::nullopt;
  }
  return Score{static_cast<int>(team.distinct_types.size()),
               static_cast<int>(team.daily_types.size()),
               team.total_servings};
}

EvaluationResult evaluate_scenario(const json::value& scenario,
                                   const std::string& policy,
                                   const SearchLimits& limits) {
  const auto& root = scenario.as_object();
  const MapConfig config = parse_map_config(root.at("config"));
  std::vector<std::string> policies{policy};
  if (const auto* opponents = root.if_contains("opponents")) {
    for (const auto& item : opponents->as_array()) {
      policies.emplace_back(item.as_string());
    }
  }
  if (static_cast<int>(policies.size()) != config.players) {
    throw std::invalid_argument("scenario opponents do not match players");
  }

  std::vector<TeamState> teams;
  for (std::size_t team_index = 0; team_index < policies.size(); ++team_index) {
    TeamState team;
    team.id = std::to_string(team_index);
    team.policy = policies[team_index];
    AgentTypes types = select_agent_types(team.policy, config);
    for (std::size_t index = 0; index < config.agents.size(); ++index) {
      team.agents.push_back(
          {types[index], config.agents[index], config.fuel_limit});
    }
    team.visited_today.resize(team.agents.size());
    teams.push_back(std::move(team));
  }

  std::vector<std::map<int, int>> traffic_history;
  for (int day_index = 0; day_index < static_cast<int>(config.day_steps.size());
       ++day_index) {
    const auto roads =
        road_status_for_day(config, traffic_history, static_cast<int>(teams.size()));
    for (auto& team : teams) {
      team.stock.clear();
      team.daily_types.clear();
      for (auto& visited : team.visited_today) visited.clear();
      for (const auto& spot : config.spots) team.stock[spot.pos] = spot.stocks;
    }

    std::vector<ActionPlan> plans;
    for (std::size_t team_index = 0; team_index < teams.size(); ++team_index) {
      auto info = make_day_info(config, teams, team_index, day_index, roads);
      AgentTypes fixed;
      for (const auto& agent : teams[team_index].agents) fixed.push_back(agent.kind);
      plans.push_back(plan_day(teams[team_index].policy, config, info,
                               teams[team_index].history, fixed,
                               team_index == 0 ? limits : SearchLimits{}));
    }

    std::map<int, int> day_traffic;
    for (std::size_t team_index = 0; team_index < teams.size(); ++team_index) {
      const int previous_servings = teams[team_index].total_servings;
      auto trial = teams[team_index];
      std::map<int, int> ignored_traffic;
      auto error = simulate_team_day(config, trial, plans[team_index], roads,
                                     ignored_traffic);
      if (error) {
        ++teams[team_index].invalid_days;
        teams[team_index].errors.push_back("day " + std::to_string(day_index) +
                                          ": " + *error);
        plans[team_index] =
            wait_plan(teams[team_index].agents.size(), config.day_steps[day_index]);
      } else {
        ++teams[team_index].valid_days;
      }
      auto actual_error = simulate_team_day(config, teams[team_index],
                                            plans[team_index], roads, day_traffic);
      if (actual_error) {
        throw std::logic_error("fallback simulation failed: " + *actual_error);
      }
      teams[team_index].history.submitted_actions.push_back(plans[team_index]);
      teams[team_index].history.distinct_brands = teams[team_index].distinct_types;
      teams[team_index].cumulative_daily_types +=
          static_cast<int>(teams[team_index].daily_types.size());
      teams[team_index].daily_scores.push_back(
          {static_cast<int>(teams[team_index].distinct_types.size()),
           static_cast<int>(teams[team_index].daily_types.size()),
           teams[team_index].total_servings - previous_servings});
    }
    traffic_history.push_back(std::move(day_traffic));
  }

  const auto& own = teams.front();
  int patrol_agents = 0;
  int refuel_agents = 0;
  int ending_patrol_fuel = 0;
  for (const auto& agent : own.agents) {
    if (agent.kind == AgentKind::Patrol) {
      ++patrol_agents;
      ending_patrol_fuel += agent.fuel;
    } else {
      ++refuel_agents;
    }
  }
  return {{static_cast<int>(own.distinct_types.size()),
           own.cumulative_daily_types, own.total_servings},
          own.valid_days, own.invalid_days, patrol_agents, refuel_agents,
          own.refuel_events, ending_patrol_fuel, own.daily_scores, own.errors};
}

}  // namespace hexudon
