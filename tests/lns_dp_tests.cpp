#include "hexudon/internal.hpp"

#include <algorithm>
#include <cassert>
#include <map>

namespace {

hexudon::MapConfig test_config() {
  hexudon::MapConfig config;
  config.width = 4;
  config.height = 4;
  config.day_steps = {20, 20};
  config.day_seconds = {1.0, 1.0};
  config.cells.assign(16, hexudon::Terrain::Plain);
  for (int pos : {5, 6, 9, 10})
    config.cells[pos] = hexudon::Terrain::Road;
  config.spots = {{0, 3, 2}, {1, 7, 1}, {2, 11, 2}};
  config.agents = {0, 12, 15};
  config.fuel_limit = 6;
  config.players = 2;
  config.busy_threshold = 2;
  config.jammed_threshold = 4;
  return config;
}

hexudon::DayInfo first_day(const hexudon::MapConfig &config,
                           const hexudon::AgentTypes &types) {
  hexudon::DayInfo day;
  day.day = 0;
  for (std::size_t index = 0; index < types.size(); ++index) {
    day.agents.push_back(
        {types[index], config.agents[index],
         types[index] == hexudon::AgentKind::Patrol ? config.fuel_limit : 0});
  }
  for (int pos = 0; pos < config.width * config.height; ++pos) {
    if (config.cells[pos] == hexudon::Terrain::Road)
      day.traffics[pos] = 0;
  }
  return day;
}

} // namespace

int main() {
  const auto config = test_config();
  hexudon::SearchLimits limits;
  limits.min_iterations = 0;
  limits.max_iterations = 1;
  limits.stagnation_iterations = 0;
  limits.future_discount_percent = 90;
  const auto types = hexudon::select_lns_dp_agent_types(config, limits);
  assert(types.size() == config.agents.size());
  assert(std::find(types.begin(), types.end(), hexudon::AgentKind::Patrol) !=
         types.end());
  auto timed_role_limits = limits;
  timed_role_limits.time_limit_ms = 100;
  timed_role_limits.max_iterations = 1'000'000;
  timed_role_limits.use_lns_dp_proposals = true;
  const auto timed_types =
      hexudon::select_lns_dp_agent_types(config, timed_role_limits);
  assert(timed_types.size() == config.agents.size());
  assert(std::find(timed_types.begin(), timed_types.end(),
                   hexudon::AgentKind::Patrol) != timed_types.end());
  auto long_high_fuel = config;
  long_high_fuel.day_steps.assign(10, 20);
  long_high_fuel.day_seconds.assign(10, 30.0);
  long_high_fuel.fuel_limit = 1'000;
  const auto horizon_limited_types =
      hexudon::select_lns_dp_agent_types(long_high_fuel, timed_role_limits);
  assert(horizon_limited_types.size() == long_high_fuel.agents.size());
  assert(std::find(horizon_limited_types.begin(), horizon_limited_types.end(),
                   hexudon::AgentKind::Patrol) !=
         horizon_limited_types.end());
  const auto day0 = first_day(config, types);
  hexudon::PolicyHistory history;
  int callbacks = 0;
  hexudon::ImprovementSink sink = [&](const hexudon::ActionPlan &actions,
                                      const hexudon::Score &,
                                      const hexudon::IncumbentRank &rank) {
    assert(!hexudon::validate_action_plan(config, day0, actions));
    assert(rank.available);
    assert(rank.objective_mode == "lns_dp");
    ++callbacks;
  };
  const auto first = hexudon::build_lns_dp_plan(config, day0, history, types,
                                                limits, nullptr, &sink);
  assert(!hexudon::validate_action_plan(config, day0, first.actions));
  assert(first.planner_state);
  assert(first.planner_state->at("policy").as_string() == "lns_dp");
  assert(first.planner_state->at("schema_version").to_number<int>() == 1);
  assert(callbacks > 0);

  const auto proposals = hexudon::build_lns_dp_route_proposals(
      config, day0, history, types, limits, 2);
  assert(!proposals.empty());
  for (const auto &proposal : proposals) {
    assert(!hexudon::validate_action_plan(config, day0, proposal.plan));
    assert(proposal.skeleton.routes.size() == types.size());
  }

  const auto repeated = hexudon::build_lns_dp_plan(config, day0, history, types,
                                                   limits, nullptr, nullptr);
  assert(repeated.actions == first.actions);

  hexudon::TeamState team;
  for (const auto &agent : day0.agents) {
    team.agents.push_back({agent.kind, agent.pos, agent.fuel});
  }
  team.visited_today.resize(types.size());
  for (const auto &spot : config.spots)
    team.stock[spot.pos] = spot.stocks;
  std::map<int, int> traffic;
  assert(!hexudon::simulate_team_day(config, team, first.actions, day0.traffics,
                                     traffic));
  history.submitted_actions.push_back(first.actions);
  history.distinct_brands = team.distinct_types;
  history.cumulative_daily_types = static_cast<int>(team.daily_types.size());
  history.total_servings = team.total_servings;
  for (auto &visited : team.visited_today)
    visited.clear();
  team.daily_types.clear();
  team.stock.clear();
  for (const auto &spot : config.spots)
    team.stock[spot.pos] = spot.stocks;
  const auto roads =
      hexudon::road_status_for_day(config, {traffic}, config.players);
  const auto day1 = hexudon::make_day_info(config, {team}, 0, 1, roads);
  const boost::json::value saved_state(*first.planner_state);
  const auto second = hexudon::build_lns_dp_plan(config, day1, history, types,
                                                 limits, &saved_state, nullptr);
  assert(!hexudon::validate_action_plan(config, day1, second.actions));
  assert(second.planner_state);
  assert(second.planner_state->at("source_day").to_number<int>() == 1);

  // A patrol cannot reach the only spot on its initial fuel.  LNS-DP must
  // realize the route through its own full-recharge DP and rendezvous schedule.
  auto refuel_config = test_config();
  refuel_config.spots = {{0, 3, 1}};
  refuel_config.agents = {0, 0, 15};
  refuel_config.fuel_limit = 2;
  refuel_config.day_steps = {20};
  refuel_config.day_seconds = {1.0};
  const hexudon::AgentTypes refuel_types{hexudon::AgentKind::Patrol,
                                         hexudon::AgentKind::Refuel,
                                         hexudon::AgentKind::Refuel};
  const auto refuel_day = first_day(refuel_config, refuel_types);
  const auto refuel_plan = hexudon::build_lns_dp_plan(
      refuel_config, refuel_day, {}, refuel_types, limits, nullptr, nullptr);
  assert(!hexudon::validate_action_plan(refuel_config, refuel_day,
                                        refuel_plan.actions));
  hexudon::TeamState refuel_team;
  for (const auto &agent : refuel_day.agents) {
    refuel_team.agents.push_back({agent.kind, agent.pos, agent.fuel});
  }
  refuel_team.visited_today.resize(refuel_types.size());
  refuel_team.stock[3] = 1;
  std::map<int, int> refuel_traffic;
  hexudon::SimulationTrace refuel_trace;
  assert(!hexudon::simulate_team_day(refuel_config, refuel_team,
                                     refuel_plan.actions, refuel_day.traffics,
                                     refuel_traffic, &refuel_trace));
  assert(!refuel_trace.acquisitions.empty());
  assert(!refuel_trace.refuels.empty());

  // Regression: the minimum-event fuel DP must retain its first/only
  // breakpoint.  The refuel car starts far enough away that incidental
  // start-cell co-location cannot hide a missing rendezvous.
  hexudon::MapConfig one_break = refuel_config;
  one_break.width = 4;
  one_break.height = 1;
  one_break.cells.assign(4, hexudon::Terrain::Plain);
  one_break.spots = {{0, 3, 1}};
  one_break.agents = {0, 3};
  one_break.fuel_limit = 2;
  one_break.day_steps = {12};
  one_break.day_seconds = {1.0};
  const hexudon::AgentTypes one_break_types{hexudon::AgentKind::Patrol,
                                            hexudon::AgentKind::Refuel};
  const auto selected_one_break =
      hexudon::select_lns_dp_agent_types(one_break, limits);
  assert(std::find(selected_one_break.begin(), selected_one_break.end(),
                   hexudon::AgentKind::Refuel) != selected_one_break.end());
  const auto one_break_day = first_day(one_break, one_break_types);
  const auto one_break_plan = hexudon::build_lns_dp_plan(
      one_break, one_break_day, {}, one_break_types, limits, nullptr, nullptr);
  assert(!hexudon::validate_action_plan(one_break, one_break_day,
                                        one_break_plan.actions));
  hexudon::TeamState one_break_team;
  for (const auto &agent : one_break_day.agents)
    one_break_team.agents.push_back({agent.kind, agent.pos, agent.fuel});
  one_break_team.visited_today.resize(one_break_types.size());
  one_break_team.stock[3] = 1;
  std::map<int, int> one_break_traffic;
  hexudon::SimulationTrace one_break_trace;
  assert(!hexudon::simulate_team_day(one_break, one_break_team,
                                     one_break_plan.actions,
                                     one_break_day.traffics,
                                     one_break_traffic, &one_break_trace));
  assert(!one_break_trace.acquisitions.empty());
  assert(!one_break_trace.refuels.empty());
  assert(one_break_team.agents[0].fuel == one_break.fuel_limit);

  // Regression: prize-collecting request construction must not stop at the old
  // fixed 16-request cap on stock-rich instances.
  auto stock_rich = test_config();
  stock_rich.spots = {{0, 0, 4}, {1, 1, 4}, {2, 2, 4},
                      {3, 3, 4}, {4, 4, 4}};
  stock_rich.agents = {0, 0, 0, 0};
  stock_rich.fuel_limit = 100;
  stock_rich.day_steps = {100};
  stock_rich.day_seconds = {1.0};
  const hexudon::AgentTypes stock_types(4, hexudon::AgentKind::Patrol);
  const auto stock_day = first_day(stock_rich, stock_types);
  const auto stock_plan = hexudon::build_lns_dp_plan(
      stock_rich, stock_day, {}, stock_types, limits, nullptr, nullptr);
  hexudon::TeamState stock_team;
  for (const auto &agent : stock_day.agents)
    stock_team.agents.push_back({agent.kind, agent.pos, agent.fuel});
  stock_team.visited_today.resize(stock_types.size());
  for (const auto &spot : stock_rich.spots)
    stock_team.stock[spot.pos] = spot.stocks;
  std::map<int, int> stock_traffic;
  assert(!hexudon::simulate_team_day(stock_rich, stock_team,
                                     stock_plan.actions,
                                     stock_day.traffics, stock_traffic));
  assert(stock_team.total_servings > 16);
  return 0;
}
