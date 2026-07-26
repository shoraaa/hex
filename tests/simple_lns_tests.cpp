#include "hexudon/core.hpp"

#include <cassert>
#include <iostream>
#include <vector>

namespace {

hexudon::MapConfig intercept_config() {
  hexudon::MapConfig config;
  config.starts_at = 0;
  config.day_seconds = {60, 60};
  config.day_steps = {12, 12};
  config.height = 1;
  config.width = 5;
  config.cells.assign(5, hexudon::Terrain::Plain);
  config.spots = {{0, 4, 1}, {1, 3, 1}};
  config.agents = {0, 2};
  config.fuel_limit = 2;
  config.players = 1;
  config.busy_threshold = 2;
  config.jammed_threshold = 4;
  return config;
}

void test_simple_lns_intercepts_and_serializes_suffix() {
  const auto config = intercept_config();
  const hexudon::AgentTypes types = {hexudon::AgentKind::Patrol,
                                     hexudon::AgentKind::Refuel};
  hexudon::DayInfo day;
  day.day = 0;
  day.agents = {{types[0], 0, 2}, {types[1], 2, 0}};
  hexudon::SearchLimits limits;
  limits.min_iterations = 1;
  limits.max_iterations = 4;
  limits.stagnation_iterations = 0;
  limits.future_discount_percent = 90;

  const auto planned = hexudon::plan_day_with_state(
      "simple_lns", config, day, {}, types, limits);
  assert(!hexudon::validate_action_plan(config, day, planned.actions));
  assert(planned.planner_state);
  assert(planned.planner_state->at("policy").as_string() == "simple_lns");
  assert(planned.planner_state->at("suffix").as_array().size() == 1);

  const auto trace =
      hexudon::trace_action_plan(config, day, {}, planned.actions).as_object();
  assert(trace.at("valid").as_bool());
  assert(trace.at("own_traffic").is_array());
  const auto& score = trace.at("score").as_object();
  assert(score.at("servings").as_int64() >= 1);
  assert(score.at("refuel_events").as_int64() >= 1);
  // A rendezvous is selected on the patrol's future route. The patrol must
  // not be held at its arbitrary day-one starting cell while it is chased.
  assert(!planned.actions[0].empty());
  assert(planned.actions[0].front() >= 0);
}

void test_simple_lns_can_collect_multiple_stock_copies() {
  hexudon::MapConfig config;
  config.starts_at = 0;
  config.day_seconds = {60};
  config.day_steps = {5};
  config.height = 1;
  config.width = 1;
  config.cells = {hexudon::Terrain::Plain};
  config.spots = {{0, 0, 5}};
  config.agents = {0};
  config.fuel_limit = 1;
  config.players = 1;
  config.busy_threshold = 2;
  config.jammed_threshold = 4;

  const hexudon::AgentTypes types = {hexudon::AgentKind::Patrol};
  hexudon::DayInfo day;
  day.day = 0;
  day.agents = {{types[0], 0, 1}};
  hexudon::SearchLimits limits;
  limits.min_iterations = 1;
  limits.max_iterations = 4;
  limits.stagnation_iterations = 0;
  limits.future_discount_percent = 90;

  const auto planned = hexudon::plan_day_with_state(
      "simple_lns", config, day, {}, types, limits);
  const auto trace =
      hexudon::trace_action_plan(config, day, {}, planned.actions).as_object();
  assert(trace.at("valid").as_bool());
  assert(trace.at("score").as_object().at("servings").as_int64() == 5);
}

void test_real_match_opponents_may_hide_fuel() {
  const auto payload = boost::json::parse(R"json({
    "day": 0,
    "agents": [{"kind": 0, "pos": 4, "fuel": 12}],
    "others": [{
      "id": 6,
      "agents": [{"kind": 1, "pos": 9}, {"kind": 0, "pos": 10}]
    }],
    "traffics": [{"pos": 2, "status": 0}]
  })json");

  const auto day = hexudon::parse_day_info(payload);
  assert(day.agents.size() == 1);
  assert(day.agents[0].fuel == 12);
  assert(day.others.size() == 1);
  assert(day.others[0].id == "6");
  assert(day.others[0].agents.size() == 2);
  assert(day.others[0].agents[0].fuel == 0);
  assert(day.others[0].agents[1].fuel == 0);
}

void test_mlns_streams_its_valid_cold_incumbent() {
  const auto config = intercept_config();
  const hexudon::AgentTypes types = {hexudon::AgentKind::Patrol,
                                     hexudon::AgentKind::Refuel};
  hexudon::DayInfo day;
  day.day = 0;
  day.agents = {{types[0], 0, 2}, {types[1], 2, 0}};
  hexudon::SearchLimits limits;
  limits.time_limit_ms = 50;
  limits.min_iterations = 0;
  limits.max_iterations = 0;
  limits.stagnation_iterations = 0;

  std::vector<hexudon::ActionPlan> incumbents;
  const hexudon::ImprovementSink sink =
      [&](const hexudon::ActionPlan& actions, const hexudon::Score&,
          const hexudon::IncumbentRank&) { incumbents.push_back(actions); };
  const auto planned = hexudon::plan_day_with_state(
      "mlns", config, day, {}, types, limits, nullptr, &sink);

  assert(!incumbents.empty());
  assert(!hexudon::validate_action_plan(config, day, incumbents.front()));
  for (std::size_t index = 1; index < incumbents.size(); ++index) {
    assert(incumbents[index] != incumbents[index - 1]);
  }
  assert(!hexudon::validate_action_plan(config, day, planned.actions));
}

}  // namespace

int main() {
  test_simple_lns_intercepts_and_serializes_suffix();
  test_simple_lns_can_collect_multiple_stock_copies();
  test_real_match_opponents_may_hide_fuel();
  test_mlns_streams_its_valid_cold_incumbent();
  std::cout << "simple_lns tests passed\n";
}
