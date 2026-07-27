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


std::optional<std::string> simulate_team_day(
    const MapConfig& config, TeamState& team, const ActionPlan& plan,
    const std::map<int, int>& roads,     std::map<int, int>& traffic,
    SimulationTrace* trace) {
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

  auto capture_frame = [&](int step, const std::vector<int>& collected) {
    if (!trace || !trace->capture_frames) return;
    json::array agents;
    for (const auto& agent : team.agents) {
      agents.push_back(json::object{{"cell", agent.pos},
                                    {"fuel", agent.fuel},
                                    {"type", static_cast<int>(agent.kind)}});
    }
    json::array collected_json;
    for (int position : collected) collected_json.push_back(position);
    trace->frames.push_back(json::object{
        {"step", step},
        {"agents", std::move(agents)},
        {"collected", std::move(collected_json)},
        {"servings", team.total_servings},
        {"types", static_cast<int>(team.distinct_types.size())}});
  };

  capture_frame(0, {});

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

    std::vector<int> collected;
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
        if (trace) {
          trace->acquisitions.push_back({step, index, spot->pos});
          collected.push_back(spot->pos);
        }
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

    capture_frame(step, collected);

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
  auto parse_agent = [](const json::value& item, bool require_fuel) {
    const auto& agent = item.as_object();
    int fuel = 0;
    if (agent.if_contains("fuel") || require_fuel) {
      fuel = int_at(agent, "fuel");
    }
    return AgentView{static_cast<AgentKind>(int_at(agent, "kind")),
                     int_at(agent, "pos"), fuel};
  };
  for (const auto& item : root.at("agents").as_array()) {
    result.agents.push_back(parse_agent(item, true));
  }
  if (const auto* others = root.if_contains("others")) {
    for (const auto& item : others->as_array()) {
      const auto& object = item.as_object();
      OtherTeamView team;
      const auto& id = object.at("id");
      team.id = id.is_string() ? std::string(id.as_string())
                               : std::to_string(id.to_number<std::int64_t>());
      for (const auto& agent : object.at("agents").as_array()) {
        // Real-match payloads intentionally hide opponent fuel. Other-team
        // views are informational only; planning and validation use our own
        // agents, whose fuel remains required above.
        team.agents.push_back(parse_agent(agent, false));
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
  if (const auto* daily = root.if_contains("cumulative_daily_types")) {
    history.cumulative_daily_types = daily->to_number<int>();
  }
  if (const auto* servings = root.if_contains("total_servings")) {
    history.total_servings = servings->to_number<int>();
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
  json::object output{
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
  if (result.palns_diagnostics.total_iterations > 0) {
    const auto& diagnostics = result.palns_diagnostics;
    const int fallbacks = diagnostics.projection_iteration_fallbacks +
                          diagnostics.projection_deadline_fallbacks;
    const double fallback_percentage =
        diagnostics.projection_requests == 0
            ? 0.0
            : 100.0 * static_cast<double>(fallbacks) /
                  diagnostics.projection_requests;
    output["palns_diagnostics"] = json::object{
        {"total_iterations", diagnostics.total_iterations},
        {"iterations_used", diagnostics.iterations_used},
        {"outer_iterations", diagnostics.outer_iterations},
        {"projection_iterations", diagnostics.projection_iterations},
        {"projection_requests", diagnostics.projection_requests},
        {"projection_completed", diagnostics.projection_completed},
        {"projection_cache_hits", diagnostics.projection_cache_hits},
        {"projection_iteration_fallbacks",
         diagnostics.projection_iteration_fallbacks},
        {"projection_deadline_fallbacks",
         diagnostics.projection_deadline_fallbacks},
        {"projection_fallback_percentage", fallback_percentage}};
  }
  if (result.mlns_diagnostics.planner_calls > 0) {
    const auto& diagnostics = result.mlns_diagnostics;
    json::array components;
    for (const auto& component : diagnostics.components) {
      json::array current_gain;
      json::array projected_gain;
      for (auto value : component.current_score_gain) {
        current_gain.push_back(value);
      }
      for (auto value : component.projected_score_gain) {
        projected_gain.push_back(value);
      }
      components.push_back(json::object{
          {"component", component.component},
          {"calls", component.calls},
          {"elapsed_microseconds", component.elapsed_microseconds},
          {"incumbent_updates", component.incumbent_updates},
          {"final_selections", component.final_selections},
          {"current_score_gain", std::move(current_gain)},
          {"projected_score_gain", std::move(projected_gain)},
          {"ending_patrol_fuel_gain", component.ending_patrol_fuel_gain}});
    }
    output["mlns_profile"] = json::object{
        {"planner_calls", diagnostics.planner_calls},
        {"elapsed_microseconds", diagnostics.elapsed_microseconds},
        {"components", std::move(components)}};
  }
  return output;
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


bool is_routing_policy(const std::string& policy) {
  return policy == "greedy" || policy == "utility_greedy" ||
         policy == "fuel_aware" || policy == "stock_maximiser" ||
         policy == "coordinated" || policy == "local_search" ||
         policy == "lns" || policy == "alns" || policy == "palns" ||
         policy == "mlns" || policy == "simple_lns" || policy == "lns_dp" ||
         policy == "aco" ||
         policy == "aco_ls" || policy == "stop_bp" || policy == "bp";
}

thread_local std::size_t alns_restart_worker_count = 1;
thread_local std::size_t parallel_worker_divisor = 1;

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
  const std::size_t divisor =
      std::max<std::size_t>(1, alns_restart_worker_count) *
      std::max<std::size_t>(1, parallel_worker_divisor);
  requested = std::max<std::size_t>(1, requested / divisor);
  return std::min({tasks, requested, std::size_t{8}});
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
                              const ForcedPaths& forced,
                              const SearchLimits& limits) {
  const int horizon = config.day_steps[day.day];
  ActionPlan result(day.agents.size());
  std::set<int> planned_brands = history.distinct_brands;
  std::set<int> daily_brands;
  std::map<int, int> reservations;

  struct RoutingState {
    int remaining{};
    int current{};
    int fuel{};
    int planned_targets{};
    std::set<int> visited_targets;
    bool blocked{};
    bool use_forced{};
  };
  std::vector<RoutingState> states;
  states.reserve(day.agents.size());
  for (std::size_t index = 0; index < day.agents.size(); ++index) {
    states.push_back({horizon, day.agents[index].pos, day.agents[index].fuel,
                      0, {}, false, forced.contains(index)});
    auto& state = states.back();
    if (fixed_types[index] == AgentKind::Refuel) {
      int target = state.current;
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
      auto path = shortest_path(config, state.current, target, day.traffics);
      for (int direction : path.directions) {
        const int duration =
            terrain_time(config, state.current, day.traffics);
        if (duration > state.remaining) break;
        result[index].push_back(direction);
        state.remaining -= duration;
        state.current = *neighbor(config, state.current, direction);
      }
      if (state.remaining > 0) result[index].push_back(-state.remaining);
    } else if (state.fuel < 2) {
      state.blocked = true;
    }
  }

  // Assign one destination per patrol per round. Global brand and stock state
  // is still shared, but no low-index patrol can consume the remaining day
  // before the other patrols receive their next opportunity.
  bool made_progress = true;
  while (made_progress) {
    made_progress = false;
    for (std::size_t index = 0; index < day.agents.size(); ++index) {
      if (fixed_types[index] != AgentKind::Patrol) continue;
      auto& state = states[index];
      if (state.remaining <= 0 || state.blocked ||
          (limits.max_targets > 0 &&
           state.planned_targets >= limits.max_targets)) {
        continue;
      }
      const Spot* chosen = nullptr;
      PathResult chosen_path;
      if (state.use_forced) {
        const auto& [pos, path] = forced.at(index);
        chosen = spot_at(config, pos);
        chosen_path = path;
        state.use_forced = false;
      } else {
        std::array<long long, 6> chosen_rank{
            std::numeric_limits<long long>::min(), 0, 0, 0, 0, 0};
        for (const auto& spot : config.spots) {
          const int remaining_stock = spot.stocks - reservations[spot.pos];
          if (state.visited_targets.contains(spot.pos) ||
              remaining_stock <= 0) {
            continue;
          }
          auto path =
              shortest_path(config, state.current, spot.pos, day.traffics);
          if (path.cost == std::numeric_limits<int>::max()) continue;
          const int path_fuel =
              path_fuel_cost(config, state.current, path.directions);
          const int reserve = limits.fuel_reserve > 0 ? limits.fuel_reserve : 1;
          if (policy == "fuel_aware" &&
              state.fuel < 2 * path_fuel + reserve) {
            continue;
          }
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
      if (!chosen) continue;
      made_progress = true;
      ++reservations[chosen->pos];
      state.visited_targets.insert(chosen->pos);
      ++state.planned_targets;
      if (chosen_path.directions.empty()) {
        result[index].push_back(-1);
        --state.remaining;
        planned_brands.insert(chosen->brand);
        daily_brands.insert(chosen->brand);
        continue;
      }
      bool reached = true;
      for (int direction : chosen_path.directions) {
        const int duration =
            terrain_time(config, state.current, day.traffics);
        const int fuel_cost = terrain_fuel(config, state.current);
        if (duration > state.remaining) {
          reached = false;
          state.blocked = true;
          break;
        }
        if (fuel_cost > state.fuel) {
          reached = false;
          state.blocked = true;
          break;
        }
        result[index].push_back(direction);
        state.remaining -= duration;
        state.fuel -= fuel_cost;
        state.current = *neighbor(config, state.current, direction);
        if (const Spot* incidental = spot_at(config, state.current)) {
          planned_brands.insert(incidental->brand);
          daily_brands.insert(incidental->brand);
        }
      }
      if (reached) {
        planned_brands.insert(chosen->brand);
        daily_brands.insert(chosen->brand);
      }
    }
  }

  for (std::size_t index = 0; index < day.agents.size(); ++index) {
    if (fixed_types[index] != AgentKind::Patrol) continue;
    auto& state = states[index];
    if (policy == "fuel_aware" && state.remaining > 0) {
      PathResult retreat;
      for (std::size_t refuel = 0; refuel < day.agents.size(); ++refuel) {
        if (fixed_types[refuel] != AgentKind::Refuel) continue;
        auto path = shortest_path(config, state.current, day.agents[refuel].pos,
                                  day.traffics);
        const int required =
            path_fuel_cost(config, state.current, path.directions);
        if (required <= state.fuel && path.cost < retreat.cost) {
          retreat = std::move(path);
        }
      }
      for (int direction : retreat.directions) {
        const int duration =
            terrain_time(config, state.current, day.traffics);
        const int cost = terrain_fuel(config, state.current);
        if (duration > state.remaining || cost > state.fuel) break;
        result[index].push_back(direction);
        state.remaining -= duration;
        state.fuel -= cost;
        state.current = *neighbor(config, state.current, direction);
      }
    }
    if (state.remaining > 0) result[index].push_back(-state.remaining);
    if (result[index].empty()) result[index].push_back(-horizon);
  }
  return result;
}


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
  int active_patrols = 0;
  int maximum_collections = 0;
  int squared_collections = 0;
  std::vector<int> ending_positions;
  std::vector<int> ending_fuel;
  for (std::size_t index = 0; index < team.agents.size(); ++index) {
    const auto& agent = team.agents[index];
    if (agent.kind == AgentKind::Patrol) patrol_fuel += agent.fuel;
    if (agent.kind == AgentKind::Patrol) {
      const int collections =
          static_cast<int>(team.visited_today[index].size());
      active_patrols += collections > 0;
      maximum_collections = std::max(maximum_collections, collections);
      squared_collections += collections * collections;
    }
    ending_positions.push_back(agent.pos);
    ending_fuel.push_back(agent.fuel);
  }
  return CandidateEvaluation{
      CandidateValue{static_cast<int>(team.distinct_types.size()),
                     static_cast<int>(team.daily_types.size()),
                     team.total_servings, patrol_fuel},
      WorkloadValue{active_patrols, -maximum_collections,
                    -squared_collections},
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
    struct StageTarget {
      std::tuple<int, int, int, int> rank;
      PathResult path;
    };
    std::vector<StageTarget> targets;
    std::set<int> target_positions;
    auto add_target = [&](int target, int priority, int secondary) {
      if (!target_positions.insert(target).second) return;
      auto path = shortest_path(config, start, target, day.traffics);
      if (path.cost <= 0 || path.cost > available) return;
      targets.push_back(
          {{priority, secondary, path.cost, target}, std::move(path)});
    };

    // First try to rendezvous with the uniquely most fuel-constrained patrol.
    // If several patrols tie for the minimum, choosing one is speculative: the
    // other equally urgent patrol can make the next-day route strictly worse.
    int lowest_patrol_fuel = std::numeric_limits<int>::max();
    int lowest_patrol_count = 0;
    for (std::size_t patrol = 0; patrol < types.size(); ++patrol) {
      if (types[patrol] != AgentKind::Patrol) continue;
      const int fuel = evaluation->ending_fuel[patrol];
      if (fuel < lowest_patrol_fuel) {
        lowest_patrol_fuel = fuel;
        lowest_patrol_count = 1;
      } else if (fuel == lowest_patrol_fuel) {
        ++lowest_patrol_count;
      }
    }
    if (lowest_patrol_count == 1) {
      for (std::size_t patrol = 0; patrol < types.size(); ++patrol) {
        if (types[patrol] != AgentKind::Patrol ||
            evaluation->ending_fuel[patrol] != lowest_patrol_fuel) {
          continue;
        }
        add_target(evaluation->ending_positions[patrol], 0,
                   lowest_patrol_fuel);
      }
    }
    std::sort(targets.begin(), targets.end(),
              [](const StageTarget& left, const StageTarget& right) {
                return left.rank < right.rank;
              });
    for (auto& target : targets) {
      ActionPlan staged = plan;
      staged[agent].pop_back();
      staged[agent].insert(staged[agent].end(), target.path.directions.begin(),
                           target.path.directions.end());
      if (available > target.path.cost) {
        staged[agent].push_back(-(available - target.path.cost));
      }
      if (!validate_action_plan(config, day, staged)) {
        result.push_back(std::move(staged));
        if (result.size() >= 4) return result;
      }
    }
  }
  return result;
}



AcoGraph build_aco_graph(const MapConfig& config, const DayInfo& day,
                         const std::vector<int>& extra_nodes) {
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
                                    std::size_t limit,
                                    bool include_path_variants) {
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


std::vector<AcoMeetingList> build_aco_meeting_cache(const AcoGraph& graph,
                                                    std::size_t capacity) {
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










AgentTypes select_agent_types_online(const std::string& policy,
                                     const MapConfig& config,
                                     const SearchLimits& limits) {
  AgentTypes types(config.agents.size(), AgentKind::Patrol);
  if (policy == "wait") return types;
  if (policy == "hotspot") {
    std::fill(types.begin(), types.end(), AgentKind::Refuel);
    return types;
  }
  if (!is_routing_policy(policy)) {
    throw std::invalid_argument("unknown policy: " + policy);
  }
  if (policy == "simple_lns") {
    return select_simple_lns_agent_types(config, limits);
  }
  if (policy == "lns_dp") {
    return select_lns_dp_agent_types(config, limits);
  }
  if (policy == "lns" || policy == "alns" || policy == "palns" ||
      policy == "mlns" ||
      policy == "stop_bp" ||
      policy == "bp") {
    return select_lns_agent_types(config, limits);
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
                              const MapConfig& config,
                              const SearchLimits& limits) {
  // The complete daySteps/daySeconds schedule is part of the initial map
  // configuration. Keep it available for the one-time role decision: fuel
  // pressure and useful refueling assignments depend on the actual match
  // horizons, not a synthetic placeholder schedule.
  return select_agent_types_online(policy, config, limits);
}

// Number of independent MLNS trajectories to race per day (best-of-K portfolio).
// Default 1 is exactly the historical single-trajectory search. Override with
// HEXUDON_MLNS_PORTFOLIO.
int mlns_portfolio_count() {
  static const int value = [] {
    const char* raw = std::getenv("HEXUDON_MLNS_PORTFOLIO");
    if (raw == nullptr) return 1;
    int parsed = 1;
    try {
      parsed = std::stoi(raw);
    } catch (const std::exception&) {
      throw std::invalid_argument("HEXUDON_MLNS_PORTFOLIO must be an integer");
    }
    if (parsed < 1 || parsed > 8) {
      throw std::invalid_argument("HEXUDON_MLNS_PORTFOLIO must be in [1,8]");
    }
    return parsed;
  }();
  return value;
}

// Compare non-negative decimal integers without narrowing MLNS's uint128
// discounted ranks. Values are emitted canonically, but trimming leading zeros
// keeps this helper robust to older/custom planners.
int compare_decimal_rank(std::string_view left, std::string_view right) {
  while (left.size() > 1 && left.front() == '0') left.remove_prefix(1);
  while (right.size() > 1 && right.front() == '0') right.remove_prefix(1);
  if (left.size() != right.size()) return left.size() < right.size() ? -1 : 1;
  if (left == right) return 0;
  return left < right ? -1 : 1;
}

// Run `count` MLNS trajectories in parallel and keep the one with the same
// continuation-aware rank used inside MLNS. Comparing only the committed day's
// official score is greedy: it can select one extra serving today while
// stranding the shared refuel car for tomorrow, making best-of-K worse than K=1.
// The timed search is nondeterministic run-to-run (wall-clock cutoffs) with a
// wide spread; because `configured_workers` caps each trajectory's intra-search
// parallelism at 8, machines with more cores have spare capacity that a
// best-of-K portfolio converts into a higher, lower-variance floor -- it simply
// selects the luckiest trajectory (e.g. one that routed the single refuel car
// efficiently enough to keep patrols collecting) without altering the search.
// Only trajectory 0 receives the streaming sink, so anytime `solve` output
// stays single-threaded; the chosen plan is still emitted as the final line.
PlannerResult build_mlns_portfolio(const MapConfig& config, const DayInfo& day,
                                   const PolicyHistory& history,
                                   const AgentTypes& types,
                                   const SearchLimits& limits,
                                   const json::value* planner_state,
                                   const ImprovementSink* on_improve,
                                   int count) {
  // Run each trajectory on its own raw thread so it keeps the full per-search
  // worker budget (configured_workers is capped at 8 regardless of machine
  // size). K trajectories therefore occupy up to 8K cores -- the point is to
  // spend the cores a single capped search leaves idle on >8-core hardware, not
  // to split one search's budget. Each trajectory is thus as strong as a
  // standalone run, and best-of-K can only raise the floor.
  const std::size_t k = static_cast<std::size_t>(count);
  std::vector<PlannerResult> plans(k);
  std::vector<std::optional<Score>> scores(k);
  std::vector<std::exception_ptr> errors(k);
  std::vector<std::thread> threads;
  threads.reserve(k);
  for (std::size_t index = 0; index < k; ++index) {
    threads.emplace_back([&, index] {
      try {
        SearchLimits local = limits;
        local.random_seed =
            limits.random_seed ^
            (static_cast<std::uint64_t>(index) * 0x9E3779B97F4A7C15ULL);
        // Only trajectory 0 streams incumbents; the rest run silently so the
        // anytime `solve` sink is never called from more than one thread.
        const ImprovementSink* sink = index == 0 ? on_improve : nullptr;
        plans[index] = build_mlns_plan(config, day, history, types, local,
                                       planner_state, sink);
        scores[index] =
            score_action_plan(config, day, history, plans[index].actions);
      } catch (...) {
        errors[index] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) thread.join();
  for (auto& error : errors) {
    if (error) std::rethrow_exception(error);
  }
  auto better = [&](std::size_t left, std::size_t right) {
    const auto& left_score = scores[left];
    const auto& right_score = scores[right];
    if (static_cast<bool>(left_score) != static_cast<bool>(right_score)) {
      return left_score.has_value();
    }
    if (!left_score) return false;
    const auto& left_rank = plans[left].rank;
    const auto& right_rank = plans[right].rank;
    if (static_cast<bool>(left_rank) != static_cast<bool>(right_rank)) {
      return left_rank.has_value();
    }
    if (left_rank && right_rank && left_rank->available &&
        right_rank->available && left_rank->predicted_final_available &&
        right_rank->predicted_final_available) {
      if (left_rank->predicted_final[0] != right_rank->predicted_final[0]) {
        return left_rank->predicted_final[0] > right_rank->predicted_final[0];
      }
      const int distinct_weight = compare_decimal_rank(
          left_rank->weighted_match[0], right_rank->weighted_match[0]);
      if (distinct_weight != 0) return distinct_weight > 0;
      if (left_score->cumulative_daily_types !=
          right_score->cumulative_daily_types) {
        return left_score->cumulative_daily_types >
               right_score->cumulative_daily_types;
      }
      if (left_score->total_servings != right_score->total_servings) {
        return left_score->total_servings > right_score->total_servings;
      }
      for (std::size_t objective = 1; objective < 3; ++objective) {
        const int weighted = compare_decimal_rank(
            left_rank->weighted_match[objective],
            right_rank->weighted_match[objective]);
        if (weighted != 0) return weighted > 0;
      }
      if (left_rank->predicted_final != right_rank->predicted_final) {
        return left_rank->predicted_final > right_rank->predicted_final;
      }
      if (left_rank->predicted_ending_patrol_fuel !=
          right_rank->predicted_ending_patrol_fuel) {
        return left_rank->predicted_ending_patrol_fuel >
               right_rank->predicted_ending_patrol_fuel;
      }
    }
    const auto left_official =
        std::tuple{left_score->distinct_types,
                   left_score->cumulative_daily_types,
                   left_score->total_servings};
    const auto right_official =
        std::tuple{right_score->distinct_types,
                   right_score->cumulative_daily_types,
                   right_score->total_servings};
    if (left_official != right_official) return left_official > right_official;
    return plans[left].actions < plans[right].actions;
  };
  std::size_t best = 0;
  for (std::size_t index = 1; index < k; ++index) {
    if (better(index, best)) best = index;
  }
  return std::move(plans[best]);
}

ActionPlan plan_day_online(const std::string& policy, const MapConfig& config,
                           const DayInfo& day,
                           const PolicyHistory& history,
                           const AgentTypes& fixed_types,
                           const SearchLimits& limits,
                           const ImprovementSink* on_improve = nullptr) {
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
  if (policy == "simple_lns") {
    return build_simple_lns_plan(config, day, history, fixed_types, limits,
                                 nullptr, on_improve)
        .actions;
  }
  if (policy == "lns_dp") {
    return build_lns_dp_plan(config, day, history, fixed_types, limits,
                             nullptr, on_improve)
        .actions;
  }
  if (policy == "mlns") {
    const int portfolio = mlns_portfolio_count();
    if (portfolio > 1) {
      return build_mlns_portfolio(config, day, history, fixed_types, limits,
                                  nullptr, on_improve, portfolio)
          .actions;
    }
    return build_mlns_plan(config, day, history, fixed_types, limits, nullptr,
                           on_improve)
        .actions;
  }
  if (policy == "lns" || policy == "alns" || policy == "palns") {
    SearchLimits resolved_limits = limits;
    const bool projected = policy == "palns";
    if (projected) {
      if (resolved_limits.total_iterations < 0) {
        resolved_limits.total_iterations = 1536;
      }
      resolved_limits.min_iterations = 0;
      resolved_limits.max_iterations = resolved_limits.total_iterations;
      resolved_limits.stagnation_iterations = 0;
      resolved_limits.continuation_time_percent = 0;
      resolved_limits.alns_restarts = resolved_limits.palns_restarts;
    }
    if (resolved_limits.alns_restarts == 0) {
      resolved_limits.alns_restarts = policy == "alns" ? 3 : 1;
    }
    if (on_improve != nullptr) {
      // Anytime streaming runs one sequential instance so the callback stays
      // single-threaded; the loop's own diversification uses the full budget.
      return build_alns_plan(config, day, history, fixed_types, resolved_limits,
                             alns_features_for_policy(policy),
                             /*allow_continuation=*/true, /*restart_salt=*/0,
                             on_improve);
    }
    return build_alns_multirestart_plan(
        config, day, history, fixed_types, resolved_limits,
        alns_features_for_policy(policy));
  }
  if (policy == "aco") {
    return build_aco_plan(config, day, history, fixed_types, false, limits);
  }
  if (policy == "aco_ls") {
    return build_aco_plan(config, day, history, fixed_types, true, limits);
  }
  if (policy == "stop_bp" || policy == "bp") {
    return build_stop_bp_plan(config, day, history, fixed_types, limits);
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
                    const SearchLimits& limits,
                    const ImprovementSink* on_improve) {
  if (day.day < 0 || day.day >= static_cast<int>(config.day_steps.size())) {
    throw std::invalid_argument("day index outside config");
  }
  // Future day horizons are also known from the initial map configuration.
  // Passing the full config lets continuation-aware search optimize ending
  // positions and fuel for the real schedule while still using only the
  // current day's revealed traffic and agent state.
  return plan_day_online(policy, config, day, history, fixed_types, limits,
                         on_improve);
}

PlannerResult plan_day_with_state(
    const std::string& policy, const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& fixed_types,
    const SearchLimits& limits, const json::value* planner_state,
    const ImprovementSink* on_improve) {
  if (day.day < 0 || day.day >= static_cast<int>(config.day_steps.size())) {
    throw std::invalid_argument("day index outside config");
  }
  if (policy == "mlns") {
    const int portfolio = mlns_portfolio_count();
    if (portfolio > 1) {
      return build_mlns_portfolio(config, day, history, fixed_types, limits,
                                  planner_state, on_improve, portfolio);
    }
    return build_mlns_plan(config, day, history, fixed_types, limits,
                           planner_state, on_improve);
  }
  if (policy == "simple_lns") {
    return build_simple_lns_plan(config, day, history, fixed_types, limits,
                                 planner_state, on_improve);
  }
  if (policy == "lns_dp") {
    return build_lns_dp_plan(config, day, history, fixed_types, limits,
                             planner_state, on_improve);
  }
  return {plan_day_online(policy, config, day, history, fixed_types, limits,
                          on_improve),
          std::nullopt};
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

json::value trace_action_plan(const MapConfig& config, const DayInfo& day,
                              const PolicyHistory& history,
                              const ActionPlan& plan) {
  TeamState team;
  team.id = "trace";
  team.distinct_types = history.distinct_brands;
  for (const auto& agent : day.agents) {
    team.agents.push_back({agent.kind, agent.pos, agent.fuel});
  }
  team.visited_today.resize(team.agents.size());
  for (const auto& spot : config.spots) team.stock[spot.pos] = spot.stocks;
  team.history.submitted_actions.resize(day.day);
  std::map<int, int> traffic;
  SimulationTrace trace;
  trace.capture_frames = true;
  const auto error =
      simulate_team_day(config, team, plan, day.traffics, traffic, &trace);
  json::array frames;
  for (auto& frame : trace.frames) frames.push_back(std::move(frame));
  json::array acquisitions;
  for (const auto& event : trace.acquisitions) {
    acquisitions.push_back(json::object{{"step", event.step},
                                        {"agent", event.agent},
                                        {"spot", event.spot_pos}});
  }
  json::object result{{"valid", !error.has_value()},
                      {"error", error ? json::value(*error) : json::value()},
                      {"frames", std::move(frames)},
                      {"acquisitions", std::move(acquisitions)}};
  json::array own_traffic;
  for (const auto& [pos, volume] : traffic) {
    if (volume > 0) {
      own_traffic.push_back(json::object{{"pos", pos}, {"volume", volume}});
    }
  }
  result["own_traffic"] = std::move(own_traffic);
  if (!error) {
    result["score"] = json::object{
        {"distinct_types", static_cast<int>(team.distinct_types.size())},
        {"daily_types", static_cast<int>(team.daily_types.size())},
        {"servings", team.total_servings},
        {"refuel_events", team.refuel_events}};
  }
  return result;
}

namespace {

EvaluationResult evaluate_scenario_impl(const json::value& scenario,
                                        const std::string& policy,
                                        const SearchLimits& limits,
                                        json::array* replay_days) {
  const auto& root = scenario.as_object();
  const MapConfig config = parse_map_config(root.at("config"));
  std::vector<std::string> policies{policy};
  if (const auto* opponents = root.if_contains("opponents")) {
    for (const auto& item : opponents->as_array()) {
      policies.emplace_back(item.as_string());
    }
  }
  std::vector<std::map<int, int>> fixed_opponent_traffic;
  int fixed_opponent_players = 0;
  if (const auto* fixed = root.if_contains("fixedOpponentTraffic")) {
    if (const auto* count = root.if_contains("fixedOpponentPlayers")) {
      fixed_opponent_players = count->to_number<int>();
    }
    if (fixed_opponent_players <= 0 ||
        static_cast<int>(policies.size()) + fixed_opponent_players !=
            config.players) {
      throw std::invalid_argument(
          "fixed opponent count does not match players");
    }
    for (const auto& encoded_day : fixed->as_array()) {
      std::map<int, int> traffic;
      for (const auto& encoded_entry : encoded_day.as_array()) {
        const auto& entry = encoded_entry.as_object();
        const int pos = entry.at("pos").to_number<int>();
        const int volume = entry.at("volume").to_number<int>();
        if (pos < 0 || pos >= config.width * config.height ||
            config.cells[pos] != Terrain::Road || volume < 0) {
          throw std::invalid_argument("invalid fixed opponent traffic");
        }
        if (volume > 0) traffic[pos] += volume;
      }
      fixed_opponent_traffic.push_back(std::move(traffic));
    }
    if (fixed_opponent_traffic.size() != config.day_steps.size()) {
      throw std::invalid_argument(
          "fixed opponent traffic does not match days");
    }
  } else if (static_cast<int>(policies.size()) != config.players) {
    throw std::invalid_argument("scenario opponents do not match players");
  }
  const bool search_for_all_players =
      root.if_contains("searchForAllPlayers") != nullptr &&
      root.at("searchForAllPlayers").as_bool();
  std::vector<std::uint64_t> player_seeds(policies.size(), 0);
  if (const auto* seeds = root.if_contains("playerSeeds")) {
    const auto& values = seeds->as_array();
    if (values.size() != policies.size()) {
      throw std::invalid_argument("scenario playerSeeds do not match players");
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
      const auto value = values[index].to_number<std::int64_t>();
      if (value < 0) throw std::invalid_argument("player seed must be nonnegative");
      player_seeds[index] = static_cast<std::uint64_t>(value);
    }
  }

  std::vector<TeamState> teams;
  for (std::size_t team_index = 0; team_index < policies.size(); ++team_index) {
    TeamState team;
    team.id = std::to_string(team_index);
    team.policy = policies[team_index];
    const SearchLimits type_limits =
        team_index == 0 || search_for_all_players ? limits : SearchLimits{};
    AgentTypes types = select_agent_types(team.policy, config, type_limits);
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
        road_status_for_day(config, traffic_history, config.players);
    for (auto& team : teams) {
      team.stock.clear();
      team.daily_types.clear();
      for (auto& visited : team.visited_today) visited.clear();
      for (const auto& spot : config.spots) team.stock[spot.pos] = spot.stocks;
    }

    json::array all_team_starts;
    if (replay_days != nullptr) {
      for (const auto& team : teams) {
        json::array agents;
        for (const auto& agent : team.agents) {
          agents.push_back(json::object{
              {"cell", agent.pos},
              {"fuel", agent.fuel},
              {"type", static_cast<int>(agent.kind)}});
        }
        all_team_starts.push_back(json::object{
            {"team_id", team.id}, {"agents", std::move(agents)}});
      }
    }

    std::vector<ActionPlan> plans;
    for (std::size_t team_index = 0; team_index < teams.size(); ++team_index) {
      auto info = make_day_info(config, teams, team_index, day_index, roads);
      AgentTypes fixed;
      for (const auto& agent : teams[team_index].agents) fixed.push_back(agent.kind);
      SearchLimits team_limits =
          team_index == 0 || search_for_all_players ? limits : SearchLimits{};
      team_limits.random_seed ^= player_seeds[team_index];
      auto planned = plan_day_with_state(
          teams[team_index].policy, config, info, teams[team_index].history,
          fixed, team_limits,
          teams[team_index].planner_state
              ? &*teams[team_index].planner_state
              : nullptr);
      plans.push_back(std::move(planned.actions));
      teams[team_index].planner_state =
          planned.planner_state
              ? std::optional<json::value>(json::value(*planned.planner_state))
              : std::nullopt;
    }
    json::array all_team_actions;
    if (replay_days != nullptr) {
      for (std::size_t team_index = 0; team_index < teams.size(); ++team_index) {
        all_team_actions.push_back(json::object{
            {"team_id", teams[team_index].id},
            {"actions", to_json(plans[team_index])}});
      }
    }

    std::map<int, int> opponent_day_traffic;
    if (!fixed_opponent_traffic.empty()) {
      opponent_day_traffic = fixed_opponent_traffic[day_index];
    }
    std::map<int, int> own_day_traffic;
    std::map<int, int> day_traffic = opponent_day_traffic;
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
      SimulationTrace trace;
      trace.capture_frames = replay_days != nullptr && team_index == 0;
      std::map<int, int> team_day_traffic;
      auto actual_error = simulate_team_day(
          config, teams[team_index], plans[team_index], roads, team_day_traffic,
          trace.capture_frames ? &trace : nullptr);
      if (actual_error) {
        throw std::logic_error("fallback simulation failed: " + *actual_error);
      }
      for (const auto& [pos, volume] : team_day_traffic) {
        day_traffic[pos] += volume;
        if (team_index > 0) opponent_day_traffic[pos] += volume;
      }
      if (team_index == 0) own_day_traffic = team_day_traffic;
      teams[team_index].history.submitted_actions.push_back(plans[team_index]);
      teams[team_index].history.distinct_brands = teams[team_index].distinct_types;
      teams[team_index].cumulative_daily_types +=
          static_cast<int>(teams[team_index].daily_types.size());
      teams[team_index].history.cumulative_daily_types =
          teams[team_index].cumulative_daily_types;
      teams[team_index].history.total_servings = teams[team_index].total_servings;
      teams[team_index].daily_scores.push_back(
          {static_cast<int>(teams[team_index].distinct_types.size()),
           static_cast<int>(teams[team_index].daily_types.size()),
           teams[team_index].total_servings - previous_servings});
      if (replay_days != nullptr && team_index == 0) {
        json::object road_condition;
        for (const auto& [position, status] : roads) {
          road_condition[std::to_string(position)] = status;
        }
        json::array frames;
        for (auto& frame : trace.frames) frames.push_back(std::move(frame));
        json::array acquisitions;
        for (const auto& event : trace.acquisitions) {
          acquisitions.push_back(json::object{{"step", event.step},
                                              {"agent", event.agent},
                                              {"spot", event.spot_pos}});
        }
        replay_days->push_back(json::object{
            {"day", day_index},
            {"steps", config.day_steps[day_index]},
            {"road_condition", std::move(road_condition)},
            {"teams",
             json::array{json::object{
                 {"team_id", "local"},
                 {"policy", policy},
                 {"actions", to_json(plans[team_index])},
                 {"frames", std::move(frames)},
                 {"acquisitions", std::move(acquisitions)},
                 {"types", static_cast<int>(teams[team_index].distinct_types.size())},
                 {"daily_types", static_cast<int>(teams[team_index].daily_types.size())},
                 {"servings", teams[team_index].total_servings}}}},
            {"all_team_starts", std::move(all_team_starts)},
            {"all_team_actions", std::move(all_team_actions)}});
      }
    }
    if (replay_days != nullptr) {
      json::array encoded_opponent_traffic;
      for (const auto& [pos, volume] : opponent_day_traffic) {
        if (volume > 0) {
          encoded_opponent_traffic.push_back(
              json::object{{"pos", pos}, {"volume", volume}});
        }
      }
      replay_days->back().as_object()["opponent_traffic"] =
          std::move(encoded_opponent_traffic);
      json::array encoded_own_traffic;
      for (const auto& [pos, volume] : own_day_traffic) {
        if (volume > 0) {
          encoded_own_traffic.push_back(
              json::object{{"pos", pos}, {"volume", volume}});
        }
      }
      replay_days->back().as_object()["own_traffic"] =
          std::move(encoded_own_traffic);
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
          own.refuel_events, ending_patrol_fuel, own.daily_scores, own.errors,
          current_palns_diagnostics(), current_mlns_diagnostics()};
}

}  // namespace

EvaluationResult evaluate_scenario(const json::value& scenario,
                                   const std::string& policy,
                                   const SearchLimits& limits) {
  reset_palns_diagnostics();
  reset_mlns_diagnostics();
  return evaluate_scenario_impl(scenario, policy, limits, nullptr);
}

json::value evaluate_scenario_replay(const json::value& scenario,
                                     const std::string& policy,
                                     const SearchLimits& limits) {
  reset_palns_diagnostics();
  reset_mlns_diagnostics();
  json::array days;
  auto result = evaluate_scenario_impl(scenario, policy, limits, &days);
  auto output = to_json(result).as_object();
  output["replay"] = json::object{{"days", std::move(days)}};
  return output;
}

}  // namespace hexudon
