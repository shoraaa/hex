#include "hexudon/core.hpp"
#include "hexudon/internal.hpp"

#include <boost/json.hpp>

#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>

namespace {

hexudon::MapConfig small_config() {
  hexudon::MapConfig config;
  config.starts_at = 1700000000;
  config.day_seconds = {60};
  config.day_steps = {8};
  config.height = 3;
  config.width = 3;
  config.cells.assign(9, hexudon::Terrain::Plain);
  config.cells[1] = hexudon::Terrain::Road;
  config.cells[7] = hexudon::Terrain::Mountain;
  config.spots = {{0, 4, 2}};
  config.agents = {0, 2, 6};
  config.fuel_limit = 8;
  config.players = 1;
  config.busy_threshold = 2;
  config.jammed_threshold = 4;
  return config;
}

void test_hex_neighbors() {
  const auto config = small_config();
  assert(hexudon::neighbor(config, 4, 0) == 0);  // odd row upper-left
  assert(hexudon::neighbor(config, 4, 1) == 1);
  assert(hexudon::neighbor(config, 1, 3) == 5);  // even row lower-right
  assert(!hexudon::neighbor(config, 0, 0));
  assert(!hexudon::neighbor(config, 2, 2));
}

void test_costs() {
  const auto config = small_config();
  assert(hexudon::terrain_time(config, 0, {}) == 2);
  assert(hexudon::terrain_fuel(config, 0) == 1);
  assert(hexudon::terrain_time(config, 7, {}) == 3);
  assert(hexudon::terrain_fuel(config, 7) == 2);
  assert(hexudon::terrain_time(config, 1, {{1, 0}}) == 1);
  assert(hexudon::terrain_time(config, 1, {{1, 1}}) == 2);
  assert(hexudon::terrain_time(config, 1, {{1, 2}}) == 4);
}

void test_wait_evaluation() {
  const auto scenario = boost::json::parse(R"json(
    {"config":{"startsAt":1700000000,"daySeconds":[60],"daySteps":[8],
    "map":{"height":3,"width":3,"cells":[[0,1,0],[0,0,0],[0,2,0]]},
    "spots":[{"brand":0,"pos":4,"stocks":2}],"agents":[0,2,6],
    "fuelLimits":8,"players":1,"busyThreshold":2,"jammedThreshold":4},
    "opponents":[]})json");
  const auto result = hexudon::evaluate_scenario(scenario, "wait");
  assert(result.invalid_days == 0);
  assert(result.valid_days == 1);
  assert(result.score.total_servings == 0);
  assert(result.patrol_agents == 3);
  assert(result.refuel_agents == 0);
  assert(result.refuel_events == 0);
}

void test_validation_respects_refuel_order_and_horizon() {
  auto config = small_config();
  config.day_steps = {4};
  hexudon::DayInfo day;
  day.day = 0;
  day.agents = {
      {hexudon::AgentKind::Patrol, 0, 0},
      {hexudon::AgentKind::Refuel, 0, 8},
  };
  hexudon::ActionPlan valid{{-1, 2, -1}, {-4}};
  assert(!hexudon::validate_action_plan(config, day, valid));

  day.agents[1].pos = 2;
  const auto fuel_error = hexudon::validate_action_plan(config, day, valid);
  assert(fuel_error && fuel_error->find("lacks fuel") != std::string::npos);

  hexudon::ActionPlan too_long{{-5}, {-4}};
  const auto horizon_error = hexudon::validate_action_plan(config, day, too_long);
  assert(horizon_error && horizon_error->find("horizon") != std::string::npos);
}

void test_trace_action_plan_emits_authoritative_frames() {
  const auto config = small_config();
  hexudon::DayInfo day;
  day.day = 0;
  for (int pos : config.agents) {
    day.agents.push_back(
        {hexudon::AgentKind::Patrol, pos, config.fuel_limit});
  }
  const hexudon::ActionPlan plan(config.agents.size(),
                                 std::vector<int>{-config.day_steps[0]});
  const auto trace =
      hexudon::trace_action_plan(config, day, {}, plan).as_object();
  assert(trace.at("valid").as_bool());
  const auto& frames = trace.at("frames").as_array();
  assert(frames.size() == static_cast<std::size_t>(config.day_steps[0] + 1));
  assert(frames.front().as_object().at("step").as_int64() == 0);
  assert(frames.back().as_object().at("step").as_int64() ==
         config.day_steps[0]);
  assert(frames.back().as_object().at("agents").as_array().size() ==
         config.agents.size());
}

void test_solve_streams_monotone_improvements() {
  // The anytime `solve` path reports every improving incumbent through an
  // ImprovementSink. Emissions must be valid and non-decreasing in official
  // score, and the last emission must be the returned plan.
  auto config = small_config();
  hexudon::DayInfo day;
  day.day = 0;
  day.traffics = {{1, 0}};
  const auto types = hexudon::select_agent_types("alns", config);
  day.agents.clear();
  for (std::size_t index = 0; index < types.size(); ++index) {
    day.agents.push_back({types[index], config.agents[index], config.fuel_limit});
  }
  hexudon::SearchLimits limits;
  limits.time_limit_ms = -1;
  limits.min_iterations = 1;
  limits.max_iterations = 200;
  limits.stagnation_iterations = 0;
  std::vector<std::tuple<int, int, int>> scores;
  std::vector<hexudon::IncumbentRank> internal_ranks;
  std::vector<hexudon::ActionPlan> plans;
  hexudon::ImprovementSink sink =
      [&](const hexudon::ActionPlan& plan, const hexudon::Score& score,
          const hexudon::IncumbentRank& internal_rank) {
        plans.push_back(plan);
        scores.emplace_back(score.distinct_types, score.cumulative_daily_types,
                            score.total_servings);
        internal_ranks.push_back(internal_rank);
      };
  const auto best =
      hexudon::plan_day("alns", config, day, {}, types, limits, &sink);
  assert(!scores.empty());
  assert(internal_ranks.size() == scores.size());
  for (const auto& rank : internal_ranks) {
    assert(rank.available);
    assert(rank.congestion_mode == "current");
  }
  for (const auto& plan : plans) {
    assert(!hexudon::validate_action_plan(config, day, plan));
  }
  for (std::size_t index = 1; index < scores.size(); ++index) {
    assert(!(scores[index] < scores[index - 1]));
  }
  assert(!hexudon::validate_action_plan(config, day, best));
}

void test_refuel_escort_reaches_distant_brands() {
  // A patrol with fuel 2 can move only two plain tiles alone, so the far brand
  // at pos 8 is out of reach. The refuel-escort seed pairs it with the refuel
  // car (which burns no fuel), refuelling it every step so it sweeps the line
  // and collects both brands.
  hexudon::MapConfig config;
  config.starts_at = 0;
  config.day_seconds = {60};
  config.day_steps = {30};
  config.height = 1;
  config.width = 9;
  config.cells.assign(9, hexudon::Terrain::Plain);
  config.spots = {{0, 3, 1}, {1, 8, 1}};
  config.agents = {0, 0};
  config.fuel_limit = 2;
  config.players = 1;
  config.busy_threshold = 2;
  config.jammed_threshold = 4;
  hexudon::DayInfo day;
  day.day = 0;
  day.agents = {{hexudon::AgentKind::Patrol, 0, 2},
                {hexudon::AgentKind::Refuel, 0, 2}};
  const hexudon::AgentTypes types = {hexudon::AgentKind::Patrol,
                                     hexudon::AgentKind::Refuel};
  const auto escort = hexudon::build_escort_plan(config, day, {}, types);
  assert(!escort.empty());
  assert(!hexudon::validate_action_plan(config, day, escort));
  const auto escort_score = hexudon::score_action_plan(config, day, {}, escort);
  assert(escort_score && escort_score->distinct_types == 2);
  // The full solver keeps the escort incumbent.
  const auto solved = hexudon::plan_day("alns", config, day, {}, types);
  const auto solved_score = hexudon::score_action_plan(config, day, {}, solved);
  assert(solved_score && solved_score->distinct_types == 2);
}

void test_refuel_staging_targets_low_fuel_patrol_endpoints() {
  hexudon::MapConfig config;
  config.starts_at = 0;
  config.day_seconds = {60, 60};
  config.day_steps = {8, 8};
  config.height = 3;
  config.width = 3;
  config.cells.assign(9, hexudon::Terrain::Plain);
  config.spots = {{0, 8, 1}};
  config.agents = {0, 4};
  config.fuel_limit = 8;
  config.players = 1;
  config.busy_threshold = 2;
  config.jammed_threshold = 4;

  hexudon::DayInfo day;
  day.day = 0;
  day.agents = {{hexudon::AgentKind::Refuel, 0, 8},
                {hexudon::AgentKind::Patrol, 4, 1}};
  const hexudon::AgentTypes types = {hexudon::AgentKind::Refuel,
                                     hexudon::AgentKind::Patrol};
  const hexudon::ActionPlan idle{{-8}, {-8}};
  const auto idle_evaluation =
      hexudon::evaluate_candidate(config, day, {}, idle);
  assert(idle_evaluation);

  const auto variants =
      hexudon::refuel_staging_variants(config, day, types, {}, idle);
  assert(std::any_of(variants.begin(), variants.end(), [&](const auto& plan) {
    const auto evaluation =
        hexudon::evaluate_candidate(config, day, {}, plan);
    return evaluation && evaluation->ending_positions[0] == 4 &&
           evaluation->ending_fuel[1] > idle_evaluation->ending_fuel[1];
  }));
}

void test_all_converted_policies_produce_valid_daily_answers() {
  auto config = small_config();
  hexudon::DayInfo day;
  day.day = 0;
  day.traffics = {{1, 0}};
  const std::vector<std::string> policies = {
      "greedy",          "utility_greedy", "fuel_aware",
      "stock_maximiser", "coordinated",    "local_search",
      "lns",             "alns",           "aco",
      "aco_ls",
  };
  for (const auto& policy : policies) {
    const auto types = hexudon::select_agent_types(policy, config);
    day.agents.clear();
    for (std::size_t index = 0; index < types.size(); ++index) {
      day.agents.push_back({types[index], config.agents[index], config.fuel_limit});
    }
    const auto plan =
        hexudon::plan_day(policy, config, day, {}, types);
    const auto error = hexudon::validate_action_plan(config, day, plan);
    assert(!error);
  }
}

void test_lns_is_deterministic_and_rescues_a_stranded_patrol() {
  auto config = small_config();
  config.day_steps = {8};
  hexudon::DayInfo day;
  day.day = 0;
  day.traffics = {{1, 0}};
  hexudon::AgentTypes types = {
      hexudon::AgentKind::Patrol,
      hexudon::AgentKind::Refuel,
  };
  day.agents = {
      {types[0], 0, 0},
      {types[1], 0, config.fuel_limit},
  };
  hexudon::SearchLimits limits;
  limits.min_iterations = 32;
  limits.max_iterations = 32;
  limits.stagnation_iterations = 32;
  const auto first =
      hexudon::plan_day("lns", config, day, {}, types, limits);
  const auto second =
      hexudon::plan_day("lns", config, day, {}, types, limits);
  assert(first == second);
  assert(!hexudon::validate_action_plan(config, day, first));
  assert(first[0] != std::vector<int>{-config.day_steps[0]});

  const auto alns_first =
      hexudon::plan_day("alns", config, day, {}, types, limits);
  const auto alns_second =
      hexudon::plan_day("alns", config, day, {}, types, limits);
  assert(alns_first == alns_second);
  assert(first == alns_first);
  assert(!hexudon::validate_action_plan(config, day, alns_first));
  assert(alns_first[0] != std::vector<int>{-config.day_steps[0]});

  limits.time_limit_ms = 0;
  limits.min_iterations = 1;
  limits.max_iterations = 1'000'000;
  limits.stagnation_iterations = 0;
  const auto deadline_plan =
      hexudon::plan_day("alns", config, day, {}, types, limits);
  assert(!hexudon::validate_action_plan(config, day, deadline_plan));
}

void test_aco_is_deterministic_and_coordinates_refueling() {
  auto config = small_config();
  hexudon::DayInfo day;
  day.day = 0;
  day.traffics = {{1, 0}};
  hexudon::AgentTypes types = {
      hexudon::AgentKind::Patrol,
      hexudon::AgentKind::Refuel,
  };
  day.agents = {
      {types[0], 0, 0},
      {types[1], 0, config.fuel_limit},
  };

  const auto first = hexudon::plan_day("aco", config, day, {}, types);
  const auto second = hexudon::plan_day("aco", config, day, {}, types);
  assert(first == second);
  assert(!hexudon::validate_action_plan(config, day, first));
  assert(first[0] != std::vector<int>{-config.day_steps[0]});

  const auto refined = hexudon::plan_day("aco_ls", config, day, {}, types);
  assert(!hexudon::validate_action_plan(config, day, refined));
}

void test_aco_handles_patrol_starting_on_spot() {
  auto config = small_config();
  hexudon::DayInfo day;
  day.day = 0;
  hexudon::AgentTypes types = {
      hexudon::AgentKind::Patrol,
      hexudon::AgentKind::Refuel,
  };
  day.agents = {
      {types[0], 4, config.fuel_limit},
      {types[1], 2, config.fuel_limit},
  };
  const auto plan = hexudon::plan_day("aco", config, day, {}, types);
  assert(!hexudon::validate_action_plan(config, day, plan));
  assert(!plan[0].empty());
}

void test_fuel_aware_holds_low_fuel_patrol_for_rescue() {
  auto config = small_config();
  hexudon::DayInfo day;
  day.day = 0;
  day.traffics = {{1, 0}};
  hexudon::AgentTypes types = {
      hexudon::AgentKind::Patrol,
      hexudon::AgentKind::Refuel,
      hexudon::AgentKind::Patrol,
  };
  day.agents = {
      {types[0], 0, 1},
      {types[1], 2, config.fuel_limit},
      {types[2], 6, config.fuel_limit},
  };
  const auto plan = hexudon::plan_day("fuel_aware", config, day, {}, types);
  assert(plan[0] == std::vector<int>{-config.day_steps[0]});
  assert(!hexudon::validate_action_plan(config, day, plan));
}

void test_high_iteration_alns_reaches_known_daily_optimum() {
  const auto scenario = boost::json::parse(R"json(
    {"config":{"startsAt":1700000000,"daySeconds":[60],"daySteps":[6],
    "map":{"height":3,"width":3,"cells":[[0,0,0],[0,0,0],[0,0,0]]},
    "spots":[{"brand":0,"pos":1,"stocks":1},
             {"brand":1,"pos":4,"stocks":1}],"agents":[0],
    "fuelLimits":10,"players":1,"busyThreshold":2,"jammedThreshold":4},
    "opponents":[]})json");
  hexudon::SearchLimits limits;
  limits.min_iterations = 1;
  limits.max_iterations = 10'000;
  limits.stagnation_iterations = 32;
  const auto result = hexudon::evaluate_scenario(scenario, "alns", limits);
  assert(result.invalid_days == 0);
  assert(result.score.distinct_types == 2);
  assert(result.score.cumulative_daily_types == 2);
  assert(result.score.total_servings == 2);
}

void test_alns_final_day_quality_is_monotone_with_iteration_budget() {
  auto config = small_config();
  config.day_steps = {14};
  config.spots = {
      {0, 1, 2},
      {1, 4, 2},
      {2, 8, 2},
  };
  hexudon::DayInfo day;
  day.day = 0;
  const hexudon::AgentTypes types(config.agents.size(),
                                  hexudon::AgentKind::Patrol);
  for (int pos : config.agents) {
    day.agents.push_back(
        {hexudon::AgentKind::Patrol, pos, config.fuel_limit});
  }

  hexudon::SearchLimits short_limits;
  short_limits.min_iterations = 32;
  short_limits.max_iterations = 32;
  short_limits.stagnation_iterations = 0;
  const auto short_plan =
      hexudon::plan_day("alns", config, day, {}, types, short_limits);
  const auto short_score =
      hexudon::score_action_plan(config, day, {}, short_plan);
  assert(short_score);

  hexudon::SearchLimits long_limits;
  long_limits.min_iterations = 32;
  long_limits.max_iterations = 6'000;
  long_limits.stagnation_iterations = 0;
  const auto long_plan =
      hexudon::plan_day("alns", config, day, {}, types, long_limits);
  const auto long_score = hexudon::score_action_plan(config, day, {}, long_plan);
  assert(long_score);
  assert((std::tuple{long_score->distinct_types,
                     long_score->cumulative_daily_types,
                     long_score->total_servings} >=
          std::tuple{short_score->distinct_types,
                     short_score->cumulative_daily_types,
                     short_score->total_servings}));
}

void test_alns_same_day_multirestart_protects_single_restart_score() {
  auto config = small_config();
  config.day_steps = {14};
  config.spots = {
      {0, 1, 2},
      {1, 4, 2},
      {2, 8, 2},
  };
  hexudon::DayInfo day;
  day.day = 0;
  const hexudon::AgentTypes types(config.agents.size(),
                                  hexudon::AgentKind::Patrol);
  for (int pos : config.agents) {
    day.agents.push_back(
        {hexudon::AgentKind::Patrol, pos, config.fuel_limit});
  }

  hexudon::SearchLimits single_limits;
  single_limits.min_iterations = 32;
  single_limits.max_iterations = 128;
  single_limits.stagnation_iterations = 0;
  single_limits.alns_restarts = 1;
  const auto single_plan =
      hexudon::plan_day("alns", config, day, {}, types, single_limits);
  const auto single_score =
      hexudon::score_action_plan(config, day, {}, single_plan);
  assert(single_score);

  auto multi_limits = single_limits;
  multi_limits.alns_restarts = 3;
  const auto multi_plan =
      hexudon::plan_day("alns", config, day, {}, types, multi_limits);
  const auto multi_score =
      hexudon::score_action_plan(config, day, {}, multi_plan);
  assert(multi_score);
  assert((std::tuple{multi_score->distinct_types,
                     multi_score->cumulative_daily_types,
                     multi_score->total_servings} >=
          std::tuple{single_score->distinct_types,
                     single_score->cumulative_daily_types,
                     single_score->total_servings}));

  auto automatic_limits = single_limits;
  automatic_limits.alns_restarts = 0;
  auto two_restart_limits = single_limits;
  two_restart_limits.alns_restarts = 2;
  const auto automatic_plan =
      hexudon::plan_day("alns", config, day, {}, types, automatic_limits);
  const auto two_restart_plan =
      hexudon::plan_day("alns", config, day, {}, types, two_restart_limits);
  assert(automatic_plan == two_restart_plan);
}

void test_online_alns_accepts_full_schedule_and_preserves_daily_quality() {
  auto config = small_config();
  config.day_steps = {14, 8, 32, 20};
  config.day_seconds = {60, 10, 120, 30};
  config.spots = {
      {0, 1, 2},
      {1, 4, 2},
      {2, 8, 2},
  };
  hexudon::DayInfo day;
  day.day = 0;
  day.traffics = {{1, 0}};
  const hexudon::AgentTypes types(config.agents.size(),
                                  hexudon::AgentKind::Patrol);
  for (int pos : config.agents) {
    day.agents.push_back(
        {hexudon::AgentKind::Patrol, pos, config.fuel_limit});
  }
  hexudon::SearchLimits limits;
  limits.min_iterations = 96;
  limits.max_iterations = 96;
  limits.stagnation_iterations = 96;

  const auto aco =
      hexudon::plan_day("aco_ls", config, day, {}, types, limits);
  const auto alns =
      hexudon::plan_day("alns", config, day, {}, types, limits);
  const auto aco_score = hexudon::score_action_plan(config, day, {}, aco);
  const auto alns_score = hexudon::score_action_plan(config, day, {}, alns);
  assert(aco_score);
  assert(alns_score);
  assert((std::tuple{alns_score->distinct_types,
                     alns_score->cumulative_daily_types,
                     alns_score->total_servings} >=
          std::tuple{aco_score->distinct_types,
                     aco_score->cumulative_daily_types,
                     aco_score->total_servings}));

  auto altered = config;
  altered.day_steps = {14, 32, 8, 32};
  altered.day_seconds = {60, 30, 60, 30};
  const auto altered_alns =
      hexudon::plan_day("alns", altered, day, {}, types, limits);
  const auto altered_score =
      hexudon::score_action_plan(altered, day, {}, altered_alns);
  assert(altered_score);
}

void test_alns_balances_collections_after_official_score() {
  hexudon::MapConfig config;
  config.starts_at = 1700000000;
  config.day_seconds = {60};
  config.day_steps = {20};
  config.height = 5;
  config.width = 5;
  config.cells.assign(25, hexudon::Terrain::Plain);
  config.spots = {
      {0, 1, 1}, {1, 3, 1}, {2, 5, 1}, {3, 9, 1},
      {4, 15, 1}, {5, 19, 1}, {6, 21, 1}, {7, 23, 1},
  };
  config.agents = {0, 4, 20, 24};
  config.fuel_limit = 40;
  config.players = 1;
  config.busy_threshold = 2;
  config.jammed_threshold = 4;

  hexudon::DayInfo day;
  day.day = 0;
  const hexudon::AgentTypes types(config.agents.size(),
                                  hexudon::AgentKind::Patrol);
  for (int pos : config.agents) {
    day.agents.push_back(
        {hexudon::AgentKind::Patrol, pos, config.fuel_limit});
  }
  hexudon::SearchLimits limits;
  limits.min_iterations = 96;
  limits.max_iterations = 96;
  limits.stagnation_iterations = 96;
  const auto plan =
      hexudon::plan_day("alns", config, day, {}, types, limits);
  const auto trace =
      hexudon::trace_action_plan(config, day, {}, plan).as_object();
  assert(trace.at("valid").as_bool());
  assert(trace.at("score").as_object().at("servings").as_int64() == 8);
  std::vector<int> collections(types.size());
  for (const auto& item : trace.at("acquisitions").as_array()) {
    const auto& event = item.as_object();
    ++collections[event.at("agent").to_number<std::size_t>()];
  }
  for (int count : collections) assert(count >= 1);
  assert(*std::max_element(collections.begin(), collections.end()) <= 3);
}

}  // namespace

int main() {
  test_hex_neighbors();
  test_costs();
  test_wait_evaluation();
  test_validation_respects_refuel_order_and_horizon();
  test_trace_action_plan_emits_authoritative_frames();
  test_all_converted_policies_produce_valid_daily_answers();
  test_fuel_aware_holds_low_fuel_patrol_for_rescue();
  test_lns_is_deterministic_and_rescues_a_stranded_patrol();
  test_aco_is_deterministic_and_coordinates_refueling();
  test_aco_handles_patrol_starting_on_spot();
  test_high_iteration_alns_reaches_known_daily_optimum();
  test_alns_final_day_quality_is_monotone_with_iteration_budget();
  test_alns_same_day_multirestart_protects_single_restart_score();
  test_online_alns_accepts_full_schedule_and_preserves_daily_quality();
  test_alns_balances_collections_after_official_score();
  test_solve_streams_monotone_improvements();
  test_refuel_escort_reaches_distant_brands();
  test_refuel_staging_targets_low_fuel_patrol_endpoints();
  std::cout << "core tests passed\n";
}
