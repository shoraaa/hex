#include "hexudon/core.hpp"

#include <boost/json.hpp>

#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

std::string read_stdin() {
  return {std::istreambuf_iterator<char>(std::cin),
          std::istreambuf_iterator<char>()};
}

hexudon::AgentTypes parse_types(const boost::json::value& value) {
  hexudon::AgentTypes result;
  for (const auto& item : value.as_array()) {
    const int kind = static_cast<int>(item.to_number<std::int64_t>());
    if (kind < 0 || kind > 1) throw std::invalid_argument("invalid agent kind");
    result.push_back(static_cast<hexudon::AgentKind>(kind));
  }
  return result;
}

hexudon::ActionPlan parse_actions(const boost::json::value& value) {
  hexudon::ActionPlan result;
  for (const auto& row : value.as_array()) {
    std::vector<int> actions;
    for (const auto& item : row.as_array()) {
      actions.push_back(static_cast<int>(item.to_number<std::int64_t>()));
    }
    result.push_back(std::move(actions));
  }
  return result;
}

boost::json::value incumbent_rank_to_json(
    const hexudon::IncumbentRank& rank) {
  if (!rank.available) return nullptr;
  boost::json::array congestion;
  for (int value : rank.congestion) congestion.push_back(value);
  boost::json::array workload;
  for (int value : rank.workload) workload.push_back(value);
  boost::json::array predicted_final;
  for (int value : rank.predicted_final) predicted_final.push_back(value);
  boost::json::object result{
      {"objective_mode", rank.objective_mode},
      {"congestion_mode", rank.congestion_mode},
      {"congestion", std::move(congestion)},
      {"workload", std::move(workload)},
      {"patrol_fuel", rank.patrol_fuel},
      {"predicted_final_available", rank.predicted_final_available},
      {"predicted_final", std::move(predicted_final)},
      {"predicted_ending_patrol_fuel",
       rank.predicted_ending_patrol_fuel},
      {"prediction_horizon_days", rank.prediction_horizon_days}};
  if (rank.objective_mode == "mlns" ||
      rank.objective_mode == "simple_lns" ||
      rank.objective_mode == "lns_dp") {
    boost::json::array weighted;
    for (const auto& value : rank.weighted_match) {
      weighted.push_back(boost::json::value(value));
    }
    result["future_discount_percent"] = rank.future_discount_percent;
    result["weighted_match"] = std::move(weighted);
  }
  return result;
}

hexudon::SearchLimits parse_search_limits(const boost::json::object& object,
                                          const std::string& policy) {
  hexudon::SearchLimits limits;
  bool explicit_time_limit = false;
  bool explicit_min_iterations = false;
  bool explicit_max_iterations = false;
  bool explicit_stagnation = false;
  bool explicit_future_discount = false;
  auto assign = [](const boost::json::object& search, std::string_view key,
                   int& target) {
    if (const auto* item = search.if_contains(key)) {
      const auto value = item->to_number<std::int64_t>();
      if (value < std::numeric_limits<int>::min() ||
          value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("integer hyperparameter is out of range");
      }
      target = static_cast<int>(value);
    }
  };
  auto assign_double = [](const boost::json::object& search,
                          std::string_view key, double& target) {
    if (const auto* item = search.if_contains(key)) target = item->to_number<double>();
  };
  auto assign_bool = [](const boost::json::object& search,
                        std::string_view key, bool& target) {
    if (const auto* item = search.if_contains(key)) target = item->as_bool();
  };
  auto assign_seed = [](const boost::json::object& search,
                        std::string_view key, std::uint64_t& target) {
    if (const auto* item = search.if_contains(key)) {
      const auto value = item->to_number<std::int64_t>();
      if (value < 0) throw std::invalid_argument("random seed must be nonnegative");
      target = static_cast<std::uint64_t>(value);
    }
  };
  if (const auto* value = object.if_contains("search")) {
    const auto& search = value->as_object();
    explicit_time_limit = search.contains("timeLimitMs");
    explicit_min_iterations = search.contains("minIterations");
    explicit_max_iterations = search.contains("maxIterations");
    explicit_stagnation = search.contains("stagnationIterations");
    explicit_future_discount = search.contains("futureDiscountPercent");
    assign(search, "timeLimitMs", limits.time_limit_ms);
    assign(search, "minIterations", limits.min_iterations);
    assign(search, "maxIterations", limits.max_iterations);
    assign(search, "stagnationIterations", limits.stagnation_iterations);
    assign(search, "seedIterations", limits.seed_iterations);
    assign(search, "finalAlnsIterations", limits.final_alns_iterations);
    assign(search, "exactNodes", limits.exact_nodes);
    assign(search, "finalExactNodes", limits.final_exact_nodes);
    assign(search, "alnsRestarts", limits.alns_restarts);
    assign(search, "continuationTimePercent",
           limits.continuation_time_percent);
    assign(search, "exactTimePercent", limits.exact_time_percent);
    assign(search, "totalIterations", limits.total_iterations);
    assign(search, "palnsProjectionIterations",
           limits.palns_projection_iterations);
    assign(search, "palnsRestarts", limits.palns_restarts);
    assign(search, "futureDiscountPercent", limits.future_discount_percent);
    assign(search, "mlnsLookaheadDays", limits.mlns_lookahead_days);
    assign(search, "mlnsCommitTolerance", limits.mlns_commit_tolerance);
    assign_bool(search, "mlnsAdaptiveCommitTolerance",
                limits.mlns_adaptive_commit_tolerance);
    assign(search, "acoAnts", limits.aco_ants);
    assign(search, "acoIterations", limits.aco_iterations);
    assign_double(search, "acoEvaporation", limits.aco_evaporation);
    assign_bool(search, "useAcoSeed", limits.use_aco_seed);
    assign_bool(search, "useLegacySeed", limits.use_legacy_seed);
    assign_bool(search, "useLocalSearchSeed", limits.use_local_search_seed);
    assign_bool(search, "useLnsDpProposals", limits.use_lns_dp_proposals);
    assign_bool(search, "useTrafficGnn", limits.use_traffic_gnn);
    assign_seed(search, "randomSeed", limits.random_seed);
  }
  if (const auto* value = object.if_contains("hyperparameters")) {
    const auto& hyperparameters = value->as_object();
    explicit_time_limit =
        explicit_time_limit || hyperparameters.contains("time_limit_ms");
    explicit_min_iterations =
        explicit_min_iterations || hyperparameters.contains("min_iterations");
    explicit_max_iterations =
        explicit_max_iterations || hyperparameters.contains("max_iterations") ||
        hyperparameters.contains("alns_iterations");
    explicit_stagnation =
        explicit_stagnation || hyperparameters.contains("stagnation_iterations");
    explicit_future_discount = explicit_future_discount ||
                               hyperparameters.contains(
                                   "future_discount_percent");
    assign(hyperparameters, "time_limit_ms", limits.time_limit_ms);
    assign(hyperparameters, "min_iterations", limits.min_iterations);
    assign(hyperparameters, "max_iterations", limits.max_iterations);
    assign(hyperparameters, "alns_iterations", limits.max_iterations);
    assign(hyperparameters, "stagnation_iterations", limits.stagnation_iterations);
    assign(hyperparameters, "seed_iterations", limits.seed_iterations);
    assign(hyperparameters, "final_alns_iterations",
           limits.final_alns_iterations);
    assign(hyperparameters, "exact_nodes", limits.exact_nodes);
    assign(hyperparameters, "final_exact_nodes", limits.final_exact_nodes);
    assign(hyperparameters, "alns_restarts", limits.alns_restarts);
    assign(hyperparameters, "continuation_time_percent",
           limits.continuation_time_percent);
    assign(hyperparameters, "exact_time_percent",
           limits.exact_time_percent);
    assign(hyperparameters, "total_iterations", limits.total_iterations);
    assign(hyperparameters, "future_discount_percent",
           limits.future_discount_percent);
    assign(hyperparameters, "lookahead_days", limits.mlns_lookahead_days);
    assign(hyperparameters, "commit_tolerance", limits.mlns_commit_tolerance);
    assign_bool(hyperparameters, "adaptive_commit_tolerance",
                limits.mlns_adaptive_commit_tolerance);
    assign(hyperparameters, "aco_ants", limits.aco_ants);
    assign(hyperparameters, "aco_iterations", limits.aco_iterations);
    assign_double(hyperparameters, "aco_evaporation", limits.aco_evaporation);
    assign_bool(hyperparameters, "use_aco_seed", limits.use_aco_seed);
    assign_bool(hyperparameters, "use_legacy_seed", limits.use_legacy_seed);
    assign_bool(hyperparameters, "use_local_search_seed",
                limits.use_local_search_seed);
    assign_bool(hyperparameters, "use_lns_dp_proposals",
                limits.use_lns_dp_proposals);
    assign_bool(hyperparameters, "use_traffic_gnn", limits.use_traffic_gnn);
    assign_seed(hyperparameters, "random_seed", limits.random_seed);
    assign(hyperparameters, "max_targets", limits.max_targets);
    assign(hyperparameters, "fuel_reserve", limits.fuel_reserve);
    assign(hyperparameters, "passes", limits.local_search_passes);
    assign(hyperparameters, "ants", limits.aco_ants);
    assign(hyperparameters, "iterations", limits.aco_iterations);
    assign_double(hyperparameters, "evaporation", limits.aco_evaporation);
  }
  if (const auto* value = object.if_contains("predictedTraffic")) {
    for (const auto& item : value->as_array()) {
      const auto& entry = item.as_object();
      const int day = static_cast<int>(entry.at("day").to_number<std::int64_t>());
      if (day < 0) throw std::invalid_argument("predicted traffic day must be nonnegative");
      if (limits.predicted_traffic.size() <= static_cast<std::size_t>(day)) {
        limits.predicted_traffic.resize(static_cast<std::size_t>(day) + 1);
      }
      auto& roads = limits.predicted_traffic[static_cast<std::size_t>(day)];
      const auto& traffic = entry.at("traffics").as_array();
      for (const auto& road : traffic) {
        const auto& point = road.as_object();
        const int pos = static_cast<int>(point.at("pos").to_number<std::int64_t>());
        const int status = static_cast<int>(point.at("status").to_number<std::int64_t>());
        if (pos < 0 || status < 0 || status > 2) {
          throw std::invalid_argument("invalid predicted traffic point");
        }
        roads[pos] = status;
      }
    }
    if (!limits.predicted_traffic.empty()) limits.use_traffic_gnn = true;
  }
  if (explicit_time_limit && !explicit_max_iterations) {
    limits.max_iterations = 10'000'000;
  }
  if (explicit_time_limit && !explicit_stagnation) {
    limits.stagnation_iterations = 0;
  }
  if (policy == "mlns") {
    if (!explicit_min_iterations) limits.min_iterations = 32;
    // A timed MLNS request is deadline-governed. The old tuned stagnation
    // default stopped most non-final days after only 16 unproductive moves.
    if (!explicit_stagnation) {
      limits.stagnation_iterations = limits.time_limit_ms >= 0 ? 0 : 16;
    }
    if (!explicit_future_discount) limits.future_discount_percent = 90;
  } else if (policy == "simple_lns") {
    // Simple LNS derives its quality from repeated destroy-and-repair. Do not
    // stop the bounded untimed run after only a handful of whole-suffix moves.
    if (!explicit_min_iterations) limits.min_iterations = 32;
    if (!explicit_max_iterations) limits.max_iterations = 128;
    if (!explicit_stagnation) limits.stagnation_iterations = 0;
    if (!explicit_future_discount) limits.future_discount_percent = 90;
  } else if (policy == "lns_dp") {
    if (!explicit_min_iterations) limits.min_iterations = 4;
    if (!explicit_max_iterations) limits.max_iterations = 16;
    if (!explicit_stagnation) limits.stagnation_iterations = 0;
    if (!explicit_future_discount) limits.future_discount_percent = 90;
  }
  if (limits.min_iterations < 0 || limits.max_iterations < 0 ||
      limits.stagnation_iterations < 0 ||
      limits.min_iterations > limits.max_iterations || limits.max_targets < 0 ||
      limits.fuel_reserve < 0 || limits.local_search_passes < 0 ||
      limits.aco_ants < 0 || limits.aco_iterations < 0 ||
      limits.seed_iterations < 0 || limits.final_alns_iterations < -1 ||
      limits.exact_nodes < 0 || limits.final_exact_nodes < -1 ||
      limits.alns_restarts < 0 ||
      limits.total_iterations < -1 ||
      limits.palns_projection_iterations < 1 || limits.palns_restarts < 1 ||
      limits.future_discount_percent < 1 ||
      limits.future_discount_percent > 100 ||
      limits.mlns_lookahead_days < 0 ||
      limits.mlns_commit_tolerance < -1 ||
      limits.continuation_time_percent < 0 ||
      limits.continuation_time_percent > 100 ||
      limits.exact_time_percent < 0 || limits.exact_time_percent > 100 ||
      limits.aco_evaporation <= 0.0 || limits.aco_evaporation >= 1.0) {
    throw std::invalid_argument("invalid search limits");
  }
  return limits;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: hexudon <types|plan|solve|check|trace|eval|visualize> <policy>\n";
      return 2;
    }
    const std::string command = argv[1];
    const std::string policy = argv[2];
    const auto input = boost::json::parse(read_stdin());
    boost::json::value output;
    if (command == "types") {
      if (input.is_object() && input.as_object().if_contains("config")) {
        const auto& object = input.as_object();
        const auto config = hexudon::parse_map_config(object.at("config"));
        output = hexudon::to_json(hexudon::select_agent_types(
            policy, config, parse_search_limits(object, policy)));
      } else {
        const auto config = hexudon::parse_map_config(input);
        output = hexudon::to_json(hexudon::select_agent_types(policy, config));
      }
    } else if (command == "select") {
      const auto& object = input.as_object();
      const auto config = hexudon::parse_map_config(object.at("config"));
      const auto limits = parse_search_limits(object, policy);
      bool emitted = false;
      boost::json::object last_line;
      const hexudon::AgentTypeImprovementSink sink =
          [&](const hexudon::AgentTypes& types, const hexudon::Score& score,
              const std::string& phase) {
            boost::json::object line{
                {"types", hexudon::to_json(types)},
                {"score", boost::json::array{score.distinct_types,
                                             score.cumulative_daily_types,
                                             score.total_servings}},
                {"phase", phase}};
            last_line = line;
            std::cout << boost::json::serialize(line) << '\n';
            std::cout.flush();
            emitted = true;
          };
      const auto types =
          hexudon::select_agent_types(policy, config, limits, &sink);
      if (!emitted || !last_line.if_contains("types") ||
          last_line.at("types") != hexudon::to_json(types)) {
        sink(types, {}, "final");
      }
      last_line["kind"] = "final";
      last_line["types"] = hexudon::to_json(types);
      std::cout << boost::json::serialize(last_line) << '\n';
      std::cout.flush();
      return 0;
    } else if (command == "plan") {
      const auto& object = input.as_object();
      const auto config = hexudon::parse_map_config(object.at("config"));
      const auto day = hexudon::parse_day_info(object.at("day_info"));
      const auto history = object.if_contains("history")
                               ? hexudon::parse_history(object.at("history"))
                               : hexudon::PolicyHistory{};
      const auto types = parse_types(object.at("types"));
      const auto* planner_state = object.if_contains("planner_state");
      auto planned = hexudon::plan_day_with_state(
          policy, config, day, history, types, parse_search_limits(object, policy),
          planner_state);
      const bool include_state =
          object.if_contains("include_planner_state") != nullptr &&
          object.at("include_planner_state").as_bool();
      if (include_state) {
        output = boost::json::object{
            {"actions", hexudon::to_json(planned.actions)},
            {"planner_state",
             planned.planner_state
                 ? boost::json::value(*planned.planner_state)
                 : boost::json::value()}};
      } else {
        output = hexudon::to_json(planned.actions);
      }
    } else if (command == "solve") {
      // Anytime streaming: run the planner to its deadline and print one NDJSON
      // line per incumbent advance with its current-day score, internal
      // tie-break rank, and actions. The last line is the final timed plan to
      // submit, including any continuation-aware look-ahead replacement.
      const auto& object = input.as_object();
      const auto config = hexudon::parse_map_config(object.at("config"));
      const auto day = hexudon::parse_day_info(object.at("day_info"));
      const auto history = object.if_contains("history")
                               ? hexudon::parse_history(object.at("history"))
                               : hexudon::PolicyHistory{};
      const auto types = parse_types(object.at("types"));
      const auto limits = parse_search_limits(object, policy);
      bool emitted = false;
      boost::json::object last_line;
      const hexudon::ImprovementSink sink =
          [&](const hexudon::ActionPlan& plan, const hexudon::Score& score,
              const hexudon::IncumbentRank& internal_rank) {
            boost::json::object line{
                {"score", boost::json::array{score.distinct_types,
                                             score.cumulative_daily_types,
                                             score.total_servings}},
                {"internal_rank", incumbent_rank_to_json(internal_rank)},
                {"actions", hexudon::to_json(plan)}};
            last_line = line;
            std::cout << boost::json::serialize(line) << '\n';
            std::cout.flush();
            emitted = true;
          };
      auto planned = hexudon::plan_day_with_state(
          policy, config, day, history, types, limits,
          object.if_contains("planner_state"), &sink);
      const auto& best = planned.actions;
      if (!emitted) {
        // Policies without streaming support (or an early deadline) still owe
        // the caller one authoritative plan.
        hexudon::Score score{};
        if (auto scored = hexudon::score_action_plan(config, day, history, best)) {
          score = *scored;
        }
        sink(best, score, {});
      }
      if (planned.planner_state) {
        last_line["kind"] = "final";
        last_line["actions"] = hexudon::to_json(best);
        if (auto scored =
                hexudon::score_action_plan(config, day, history, best)) {
          last_line["score"] =
              boost::json::array{scored->distinct_types,
                                 scored->cumulative_daily_types,
                                 scored->total_servings};
        }
        if (planned.rank) {
          last_line["internal_rank"] =
              incumbent_rank_to_json(*planned.rank);
        }
        last_line["planner_state"] = *planned.planner_state;
        std::cout << boost::json::serialize(last_line) << '\n';
        std::cout.flush();
      }
      return 0;
    } else if (command == "eval") {
      output = hexudon::to_json(
          hexudon::evaluate_scenario(input, policy,
                                     parse_search_limits(input.as_object(), policy)));
    } else if (command == "visualize") {
      output = hexudon::evaluate_scenario_replay(
          input, policy, parse_search_limits(input.as_object(), policy));
    } else if (command == "check") {
      const auto& object = input.as_object();
      const auto config = hexudon::parse_map_config(object.at("config"));
      const auto day = hexudon::parse_day_info(object.at("day_info"));
      const auto actions = parse_actions(object.at("actions"));
      const auto error = hexudon::validate_action_plan(config, day, actions);
      boost::json::object checked{
          {"valid", !error.has_value()},
          {"error", error ? boost::json::value(*error) : boost::json::value()}};
      const auto history = object.if_contains("history")
                               ? hexudon::parse_history(object.at("history"))
                               : hexudon::PolicyHistory{};
      if (auto score =
              hexudon::score_action_plan(config, day, history, actions)) {
        checked["score"] = boost::json::object{
            {"distinct_types", score->distinct_types},
            {"daily_types", score->cumulative_daily_types},
            {"servings", score->total_servings}};
      }
      output = std::move(checked);
    } else if (command == "trace") {
      const auto& object = input.as_object();
      const auto config = hexudon::parse_map_config(object.at("config"));
      const auto day = hexudon::parse_day_info(object.at("day_info"));
      const auto actions = parse_actions(object.at("actions"));
      const auto history = object.if_contains("history")
                               ? hexudon::parse_history(object.at("history"))
                               : hexudon::PolicyHistory{};
      output = hexudon::trace_action_plan(config, day, history, actions);
    } else {
      throw std::invalid_argument("unknown command: " + command);
    }
    std::cout << boost::json::serialize(output) << '\n';
    return 0;
  } catch (const std::exception& error) {
    boost::json::object output{{"error", error.what()}};
    std::cerr << boost::json::serialize(output) << '\n';
    return 1;
  }
}
