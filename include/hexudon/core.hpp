#pragma once

#include <boost/json.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace hexudon {

namespace json = boost::json;

enum class Terrain : int { Plain = 0, Road = 1, Mountain = 2, Pond = 3 };
enum class AgentKind : int { Patrol = 0, Refuel = 1 };

struct Spot {
  int brand{};
  int pos{};
  int stocks{};
};

struct MapConfig {
  double starts_at{};
  std::vector<double> day_seconds;
  std::vector<int> day_steps;
  int height{};
  int width{};
  std::vector<Terrain> cells;
  std::vector<Spot> spots;
  std::vector<int> agents;
  int fuel_limit{};
  int players{1};
  int busy_threshold{};
  int jammed_threshold{};
};

struct AgentView {
  AgentKind kind{AgentKind::Patrol};
  int pos{};
  int fuel{};
};

struct OtherTeamView {
  std::string id;
  std::vector<AgentView> agents;
};

struct DayInfo {
  std::optional<double> ends_at;
  int day{};
  std::vector<AgentView> agents;
  std::vector<OtherTeamView> others;
  std::map<int, int> traffics;
};

struct PolicyHistory {
  std::set<int> distinct_brands;
  std::vector<std::vector<std::vector<int>>> submitted_actions;
};

using AgentTypes = std::vector<AgentKind>;
using ActionPlan = std::vector<std::vector<int>>;

struct Score {
  int distinct_types{};
  int cumulative_daily_types{};
  int total_servings{};
};

struct EvaluationResult {
  Score score;
  int valid_days{};
  int invalid_days{};
  int patrol_agents{};
  int refuel_agents{};
  int refuel_events{};
  int ending_patrol_fuel{};
  std::vector<Score> daily_scores;
  std::vector<std::string> errors;
};

struct SearchLimits {
  // A negative time limit selects deterministic iteration-only search.
  int time_limit_ms{-1};
  int min_iterations{96};
  int max_iterations{96};
  int stagnation_iterations{96};
  int max_targets{};
  int fuel_reserve{};
  int local_search_passes{};
  int aco_ants{};
  int aco_iterations{};
  double aco_evaporation{0.85};
};

MapConfig parse_map_config(const json::value& value);
DayInfo parse_day_info(const json::value& value);
PolicyHistory parse_history(const json::value& value);
json::value to_json(const AgentTypes& types);
json::value to_json(const ActionPlan& actions);
json::value to_json(const EvaluationResult& result);

std::optional<int> neighbor(const MapConfig& config, int pos, int direction);
int terrain_time(const MapConfig& config, int pos,
                 const std::map<int, int>& road_status);
int terrain_fuel(const MapConfig& config, int pos);
void validate_config(const MapConfig& config);

AgentTypes select_agent_types(const std::string& policy,
                              const MapConfig& config);
ActionPlan plan_day(const std::string& policy, const MapConfig& config,
                    const DayInfo& day, const PolicyHistory& history,
                    const AgentTypes& fixed_types,
                    const SearchLimits& limits = {});
std::optional<std::string> validate_action_plan(const MapConfig& config,
                                                const DayInfo& day,
                                                const ActionPlan& plan);
std::optional<Score> score_action_plan(const MapConfig& config,
                                       const DayInfo& day,
                                       const PolicyHistory& history,
                                       const ActionPlan& plan);
// Validate and replay one day using the authoritative simulator.  The result
// contains `valid`, `error`, `score`, and a per-step `frames` array suitable
// for the web map/replay view.
json::value trace_action_plan(const MapConfig& config, const DayInfo& day,
                              const PolicyHistory& history,
                              const ActionPlan& plan);
EvaluationResult evaluate_scenario(const json::value& scenario,
                                   const std::string& policy,
                                   const SearchLimits& limits = {});

}  // namespace hexudon
