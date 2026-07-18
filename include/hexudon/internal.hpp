#pragma once

#include "hexudon/core.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace hexudon {

// ---------------------------------------------------------------------------
// Shared internal infrastructure exposed to the per-algorithm translation
// units (local_search / aco / lns / bp).  These symbols used to live in
// core.cpp's anonymous namespace; they are now external linkage so each
// algorithm file can reuse the simulator, routing engine, ACO graph and
// candidate evaluation without duplicating them.
// ---------------------------------------------------------------------------

struct PathResult {
  int cost{std::numeric_limits<int>::max()};
  std::vector<int> directions;
};

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
  std::vector<json::object> frames;
  bool capture_frames{};
};

struct ResourcePath {
  int time{std::numeric_limits<int>::max()};
  int fuel{std::numeric_limits<int>::max()};
  std::vector<int> directions;
};

using ForcedPaths = std::map<std::size_t, std::pair<int, PathResult>>;

using CandidateValue = std::tuple<int, int, int, int>;
using WorkloadValue = std::tuple<int, int, int>;

struct CandidateEvaluation {
  CandidateValue value;
  WorkloadValue workload;
  SimulationTrace trace;
  std::vector<int> ending_positions;
  std::vector<int> ending_fuel;
  std::map<int, int> road_traffic;
};

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

using AcoMeetingList = std::array<int, 12>;

struct LnsSkeleton {
  std::vector<std::vector<int>> routes;

  bool operator==(const LnsSkeleton&) const = default;
};

using AlnsTravelChoices = std::vector<std::vector<std::uint32_t>>;

struct ExactDayResult {
  ActionPlan plan;
  CandidateValue value;
  std::int64_t explored_nodes{};
  bool complete{};
};

enum AlnsFeature : unsigned {
  AlnsStableTravel = 1U << 0U,
  AlnsSharedPreprocessing = 1U << 1U,
  AlnsExactCompletion = 1U << 2U,
  AlnsExactReachableBound = 1U << 3U,
  AlnsExactServingBound = 1U << 4U,
  AlnsExactStockBound = 1U << 5U,
  AlnsExactFuelBound = 1U << 6U,
  AlnsAcoSeed = 1U << 7U,
  AlnsSisrRecreate = 1U << 8U,
};

constexpr unsigned kAcceptedExactBoundFeatures =
    AlnsExactReachableBound | AlnsExactServingBound | AlnsExactStockBound |
    AlnsExactFuelBound;
constexpr unsigned kProductionAlnsFeatures =
    AlnsStableTravel | AlnsSharedPreprocessing | AlnsExactCompletion |
    kAcceptedExactBoundFeatures | AlnsAcoSeed;

// Simulation / parsing helpers.
const Spot* spot_at(const MapConfig& config, int pos);
PathResult shortest_path(const MapConfig& config, int source, int target,
                         const std::map<int, int>& roads);
std::optional<std::string> simulate_team_day(
    const MapConfig& config, TeamState& team, const ActionPlan& plan,
    const std::map<int, int>& roads, std::map<int, int>& traffic,
    SimulationTrace* trace = nullptr);
ActionPlan wait_plan(std::size_t agents, int steps);
DayInfo make_day_info(const MapConfig& config,
                      const std::vector<TeamState>& teams,
                      std::size_t own_index, int day,
                      const std::map<int, int>& roads);
std::map<int, int> road_status_for_day(
    const MapConfig& config, const std::vector<std::map<int, int>>& history,
    int players);

bool is_routing_policy(const std::string& policy);

extern thread_local std::size_t alns_restart_worker_count;
std::size_t configured_workers(std::size_t tasks);

int path_fuel_cost(const MapConfig& config, int source,
                   const std::vector<int>& directions);

ForcedPaths coordinated_first_targets(const MapConfig& config,
                                      const DayInfo& day,
                                      const PolicyHistory& history,
                                      const AgentTypes& types);

ActionPlan build_routing_plan(const std::string& policy,
                              const MapConfig& config, const DayInfo& day,
                              const PolicyHistory& history,
                              const AgentTypes& fixed_types,
                              const ForcedPaths& forced = {},
                              const SearchLimits& limits = {});

std::optional<CandidateEvaluation> evaluate_candidate(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const ActionPlan& plan);
std::optional<CandidateValue> candidate_value(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const ActionPlan& plan);
std::vector<ActionPlan> refuel_staging_variants(
    const MapConfig& config, const DayInfo& day, const AgentTypes& types,
    const PolicyHistory& history, const ActionPlan& plan);

AcoGraph build_aco_graph(const MapConfig& config, const DayInfo& day,
                         const std::vector<int>& extra_nodes = {});
std::vector<AcoMeetingList> build_aco_meeting_cache(const AcoGraph& graph,
                                                    std::size_t capacity = 6);
std::vector<int> alns_transit_nodes(const MapConfig& config,
                                    const AcoGraph& graph,
                                    std::size_t limit = 8,
                                    bool include_path_variants = false);

std::tuple<int, int, int> alns_official_value(const CandidateValue& value);
std::int64_t alns_ordinal(const CandidateValue& value,
                          const MapConfig& config, const AgentTypes& types);
unsigned alns_features_for_policy(const std::string& policy);

int lns_path_time(const AcoGraph& graph, int from, int to);

AgentTypes select_lns_agent_types(const MapConfig& config);

std::optional<ActionPlan> decode_lns_skeleton(
    const MapConfig& config, const DayInfo& day, const AgentTypes& types,
    const AcoGraph& graph, const std::vector<AcoMeetingList>& meeting_cache,
    const LnsSkeleton& skeleton,
    const AlnsTravelChoices* travel_choices = nullptr,
    bool strict_travel = true);

// Algorithm entry points (defined in their own translation units).
ActionPlan build_local_search_plan(const MapConfig& config, const DayInfo& day,
                                   const PolicyHistory& history,
                                   const AgentTypes& types,
                                   const SearchLimits& limits = {});
ActionPlan improve_with_route_substitutions(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, ActionPlan best,
    const std::vector<ActionPlan>& alternatives, std::size_t passes,
    bool parallel_evaluation);

ActionPlan build_aco_plan(const MapConfig& config, const DayInfo& day,
                          const PolicyHistory& history,
                          const AgentTypes& types, bool apply_local_search,
                          const SearchLimits& limits = {});

ActionPlan build_lns_plan(const MapConfig& config, const DayInfo& day,
                          const PolicyHistory& history,
                          const AgentTypes& types, const SearchLimits& limits,
                          const AcoGraph* shared_graph = nullptr,
                          const std::vector<AcoMeetingList>* shared_meetings =
                              nullptr);
ActionPlan build_alns_plan(const MapConfig& config, const DayInfo& day,
                           const PolicyHistory& history,
                           const AgentTypes& types, const SearchLimits& limits,
                           unsigned features, bool allow_continuation = true,
                           std::uint64_t restart_salt = 0,
                           const ImprovementSink* on_improve = nullptr);
// Refuel-escort construction seed (empty when there is no refuel car).
ActionPlan build_escort_plan(const MapConfig& config, const DayInfo& day,
                             const PolicyHistory& history,
                             const AgentTypes& types);
ActionPlan build_alns_multirestart_plan(const MapConfig& config,
                                        const DayInfo& day,
                                        const PolicyHistory& history,
                                        const AgentTypes& types,
                                        const SearchLimits& limits,
                                        unsigned features);

ActionPlan build_stop_bp_plan(const MapConfig& config, const DayInfo& day,
                              const PolicyHistory& history,
                              const AgentTypes& types,
                              const SearchLimits& limits);

// Parallel helpers (templates, so defined inline here).
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

template <class Function>
auto parallel_alns_restarts(std::size_t count, Function function) {
  using Result = std::invoke_result_t<Function, std::size_t>;
  std::vector<std::optional<Result>> slots(count);
  std::vector<std::exception_ptr> errors(count);
  std::vector<std::thread> threads;
  threads.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    threads.emplace_back([&, index] {
      alns_restart_worker_count = count;
      try {
        slots[index] = function(index);
      } catch (...) {
        errors[index] = std::current_exception();
      }
      alns_restart_worker_count = 1;
    });
  }
  for (auto& thread : threads) thread.join();
  std::vector<Result> result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    if (errors[index]) std::rethrow_exception(errors[index]);
    result.push_back(std::move(*slots[index]));
  }
  return result;
}

}  // namespace hexudon
