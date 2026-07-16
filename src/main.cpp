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

hexudon::SearchLimits parse_search_limits(const boost::json::object& object) {
  hexudon::SearchLimits limits;
  bool explicit_time_limit = false;
  bool explicit_max_iterations = false;
  bool explicit_stagnation = false;
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
  if (const auto* value = object.if_contains("search")) {
    const auto& search = value->as_object();
    explicit_time_limit = search.contains("timeLimitMs");
    explicit_max_iterations = search.contains("maxIterations");
    explicit_stagnation = search.contains("stagnationIterations");
    assign(search, "timeLimitMs", limits.time_limit_ms);
    assign(search, "minIterations", limits.min_iterations);
    assign(search, "maxIterations", limits.max_iterations);
    assign(search, "stagnationIterations", limits.stagnation_iterations);
  }
  if (const auto* value = object.if_contains("hyperparameters")) {
    const auto& hyperparameters = value->as_object();
    explicit_time_limit =
        explicit_time_limit || hyperparameters.contains("time_limit_ms");
    explicit_max_iterations =
        explicit_max_iterations || hyperparameters.contains("max_iterations");
    explicit_stagnation =
        explicit_stagnation || hyperparameters.contains("stagnation_iterations");
    assign(hyperparameters, "time_limit_ms", limits.time_limit_ms);
    assign(hyperparameters, "min_iterations", limits.min_iterations);
    assign(hyperparameters, "max_iterations", limits.max_iterations);
    assign(hyperparameters, "stagnation_iterations", limits.stagnation_iterations);
    assign(hyperparameters, "max_targets", limits.max_targets);
    assign(hyperparameters, "fuel_reserve", limits.fuel_reserve);
    assign(hyperparameters, "passes", limits.local_search_passes);
    assign(hyperparameters, "ants", limits.aco_ants);
    assign(hyperparameters, "iterations", limits.aco_iterations);
    assign_double(hyperparameters, "evaporation", limits.aco_evaporation);
  }
  if (explicit_time_limit && !explicit_max_iterations) {
    limits.max_iterations = 10'000'000;
  }
  if (explicit_time_limit && !explicit_stagnation) {
    limits.stagnation_iterations = 0;
  }
  if (limits.min_iterations < 0 || limits.max_iterations < 0 ||
      limits.stagnation_iterations < 0 ||
      limits.min_iterations > limits.max_iterations || limits.max_targets < 0 ||
      limits.fuel_reserve < 0 || limits.local_search_passes < 0 ||
      limits.aco_ants < 0 || limits.aco_iterations < 0 ||
      limits.aco_evaporation <= 0.0 || limits.aco_evaporation >= 1.0) {
    throw std::invalid_argument("invalid search limits");
  }
  return limits;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: hexudon <types|plan|check|eval> <policy>\n";
      return 2;
    }
    const std::string command = argv[1];
    const std::string policy = argv[2];
    const auto input = boost::json::parse(read_stdin());
    boost::json::value output;
    if (command == "types") {
      const auto config = hexudon::parse_map_config(input);
      output = hexudon::to_json(hexudon::select_agent_types(policy, config));
    } else if (command == "plan") {
      const auto& object = input.as_object();
      const auto config = hexudon::parse_map_config(object.at("config"));
      const auto day = hexudon::parse_day_info(object.at("day_info"));
      const auto history = object.if_contains("history")
                               ? hexudon::parse_history(object.at("history"))
                               : hexudon::PolicyHistory{};
      const auto types = parse_types(object.at("types"));
      output = hexudon::to_json(
          hexudon::plan_day(policy, config, day, history, types,
                            parse_search_limits(object)));
    } else if (command == "eval") {
      output = hexudon::to_json(
          hexudon::evaluate_scenario(input, policy,
                                     parse_search_limits(input.as_object())));
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
