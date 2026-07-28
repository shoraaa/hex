#pragma once

#include <boost/json.hpp>

#include <array>
#include <cstdint>
#include <functional>
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
  int cumulative_daily_types{};
  int total_servings{};
};

using AgentTypes = std::vector<AgentKind>;
using ActionPlan = std::vector<std::vector<int>>;

struct Score {
  int distinct_types{};
  int cumulative_daily_types{};
  int total_servings{};
};

using AgentTypeImprovementSink = std::function<void(
    const AgentTypes&, const Score&, const std::string&)>;

// Exact secondary rank used to distinguish plans with the same official score.
// Each array is compared lexicographically and higher is better. Congestion
// mode is "disabled", "current", or "rolling"; the latter includes prior-day
// self-traffic at the penultimate-day continuation boundary.
struct IncumbentRank {
  bool available{};
  std::string congestion_mode{"disabled"};
  std::array<int, 6> congestion{};
  std::array<int, 3> workload{};
  int patrol_fuel{};
  bool predicted_final_available{};
  std::array<int, 3> predicted_final{};
  int predicted_ending_patrol_fuel{};
  std::string objective_mode{"daily"};
  int future_discount_percent{};
  std::array<std::string, 3> weighted_match{};
  // Number of days represented by predicted_final for bounded MLNS runs.
  int prediction_horizon_days{};
};

struct PalnsDiagnostics {
  int total_iterations{};
  int iterations_used{};
  int outer_iterations{};
  int projection_iterations{};
  int projection_requests{};
  int projection_completed{};
  int projection_cache_hits{};
  int projection_iteration_fallbacks{};
  int projection_deadline_fallbacks{};
};

// Wall-clock and incumbent-attribution counters for the production MLNS
// pipeline. Components are intentionally coarse, non-overlapping top-level
// phases: nested decoder/evaluation work is charged to the phase that asked
// for it. Gain vectors are phase-boundary deltas, not independent ablations.
struct MlnsComponentDiagnostics {
  std::string component;
  int calls{};
  std::int64_t elapsed_microseconds{};
  int incumbent_updates{};
  int final_selections{};
  std::array<std::int64_t, 3> current_score_gain{};
  std::array<std::int64_t, 3> projected_score_gain{};
  std::int64_t ending_patrol_fuel_gain{};
};

struct MlnsDiagnostics {
  int planner_calls{};
  std::int64_t elapsed_microseconds{};
  std::vector<MlnsComponentDiagnostics> components;
};

// Anytime callback invoked whenever the solver advances its incumbent. The
// final callback is the plan returned at the wall-clock deadline, including a
// continuation-aware replacement selected by remaining-match look-ahead. The
// score is the authoritative current-day triple; it need not describe the
// projected continuation objective used for that final replacement.
using ImprovementSink = std::function<void(
    const ActionPlan&, const Score&, const IncumbentRank&)>;

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
  PalnsDiagnostics palns_diagnostics;
  MlnsDiagnostics mlns_diagnostics;
};

struct SearchLimits {
  // A negative time limit selects deterministic iteration-only search.
  int time_limit_ms{-1};
  int min_iterations{96};
  int max_iterations{96};
  int stagnation_iterations{96};
  int seed_iterations{32};
  int final_alns_iterations{-1};
  int exact_nodes{0};
  int final_exact_nodes{-1};
  int max_targets{};
  int fuel_reserve{};
  int local_search_passes{};
  int aco_ants{};
  int aco_iterations{};
  double aco_evaporation{0.85};
  // Optional deterministic diversification salt. Zero preserves the canonical
  // competition trajectory; local multi-player simulations assign one salt per
  // player so identical ALNS policies do not collapse to identical routes.
  std::uint64_t random_seed{};
  bool use_aco_seed{true};
  bool use_legacy_seed{true};
  bool use_local_search_seed{true};
  // Experimental additive route proposals from the independent LNS-DP
  // decoder. Disabled by default until the per-case no-regression gate passes.
  bool use_lns_dp_proposals{false};
  // Zero selects the policy default: three for ALNS and one for LNS.
  int alns_restarts{};
  // Percentage of a timed non-final-day budget reserved for continuation
  // simulation. The remainder belongs to current-day ALNS.
  int continuation_time_percent{25};
  // Percentage of a timed final-day budget reserved for exact completion when
  // a positive exact-node allowance is enabled.
  int exact_time_percent{30};
  // PALNS owns one shared iteration ledger across current-day and projected
  // future-day ALNS loops. A negative value selects the compiled default.
  int total_iterations{-1};
  // Developer-only PALNS tuning overrides. Public API/UI surfaces expose only
  // total_iterations, time_limit_ms, and exact-search controls.
  int palns_projection_iterations{8};
  int palns_restarts{1};
  // MLNS discounts each additional prediction horizon geometrically by this
  // integer percentage. 90 means 1, 0.9, 0.9^2, ... .
  int future_discount_percent{90};
  // Optional MLNS rolling horizon. Zero preserves the complete suffix;
  // positive values simulate only that many days from the revealed day.
  int mlns_lookahead_days{};
  // Maximum realized current-day serving deficit that a better continuation
  // may accept after distinct and daily coverage tie. Negative means no
  // request-level override, preserving the legacy environment fallback.
  int mlns_commit_tolerance{-1};
  // Replace MLNS's symmetric self-traffic suffix forecast with maps supplied
  // by the optional traffic-GNN predictor. The current revealed day remains
  // authoritative; entries are used only for future suffix days.
  bool use_traffic_gnn{false};
  std::vector<std::map<int, int>> predicted_traffic;
};

struct PlannerResult {
  ActionPlan actions;
  std::optional<json::object> planner_state;
  std::optional<IncumbentRank> rank;
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
                              const MapConfig& config,
                              const SearchLimits& limits = {},
                              const AgentTypeImprovementSink* on_improve = nullptr);
// When `on_improve` is non-null the ALNS policies run as a single sequential
// anytime search and invoke the sink on every strictly better incumbent; other
// policies are unaffected (the caller emits their single result).
ActionPlan plan_day(const std::string& policy, const MapConfig& config,
                    const DayInfo& day, const PolicyHistory& history,
                    const AgentTypes& fixed_types,
                    const SearchLimits& limits = {},
                    const ImprovementSink* on_improve = nullptr);
// State-aware variant used by MLNS controllers. Other policies ignore
// `planner_state` and return no state. The legacy plan_day API stays stable.
PlannerResult plan_day_with_state(
    const std::string& policy, const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& fixed_types,
    const SearchLimits& limits = {},
    const json::value* planner_state = nullptr,
    const ImprovementSink* on_improve = nullptr);
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
// Evaluate a complete local scenario and retain the authoritative per-step
// simulator frames for the primary policy.  This is intentionally separate
// from evaluate_scenario so large benchmark grades do not carry replay data.
json::value evaluate_scenario_replay(const json::value& scenario,
                                     const std::string& policy,
                                     const SearchLimits& limits = {});

}  // namespace hexudon
