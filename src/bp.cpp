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
using BpDeadline =
    std::optional<std::chrono::steady_clock::time_point>;

bool bp_expired(const BpDeadline& deadline) {
  return deadline && std::chrono::steady_clock::now() >= *deadline;
}

// ============================================================================
// Branch-and-price policy `stop_bp` / `bp`.
//
// Maps the STOP paper's branch-and-price onto the per-day hexudon problem:
//   brand   <-> cluster,  patrol agent <-> vehicle,  horizon <-> T_max.
// The master is a set-partition LP (one column per feasible fuel-relaxed
// route) solved with an in-house revised simplex; pricing is a label-setting
// DP with dominance; branching follows the lambda-variables (equivalent to
// the paper's arc/cluster rules but expressed purely through column removal,
// which keeps the master feasible at every node). When the instance is too
// large or the deadline expires the warm (short ALNS) solution is returned.
// ============================================================================

struct BpColumn {
  std::uint64_t brand_mask{};
  std::uint64_t spot_mask{};
  std::vector<int> spots;
  int time{};
  int servings{};
  int agent{};
  std::int64_t profit{};
};

struct BpModel {
  const MapConfig* config{};
  const DayInfo* day{};
  const AcoGraph* graph{};
  int horizon{};
  int spot_count{};
  int brand_count{};
  std::uint64_t known_brand_mask{};
  std::int64_t w_servings{1};
  std::int64_t w_daily{};
  std::int64_t w_distinct{};
  std::vector<int> brand_index;          // spot -> brand bit index
  std::vector<std::uint64_t> brand_bit;  // spot -> brand mask
  std::vector<int> stocks;               // spot -> stocks
  std::vector<std::size_t> patrols;      // agent indices that are patrols
};

std::int64_t bp_column_profit(const BpModel& m, std::uint64_t brand_mask,
                              int servings) {
  const int brands = static_cast<int>(std::popcount(brand_mask));
  const int new_brands =
      static_cast<int>(std::popcount(brand_mask & ~m.known_brand_mask));
  return m.w_distinct * new_brands + m.w_daily * brands +
         m.w_servings * servings;
}

BpModel bp_build_model(const MapConfig& config, const DayInfo& day,
                       const AgentTypes& types, const AcoGraph& graph,
                       const PolicyHistory& history) {
  BpModel m;
  m.config = &config;
  m.day = &day;
  m.graph = &graph;
  m.horizon = config.day_steps[day.day];
  m.spot_count = static_cast<int>(config.spots.size());
  std::map<int, int> brand_id;
  for (const auto& spot : config.spots) {
    if (!brand_id.contains(spot.brand)) {
      brand_id[spot.brand] = static_cast<int>(brand_id.size());
    }
  }
  m.brand_count = static_cast<int>(brand_id.size());
  m.brand_index.assign(m.spot_count, 0);
  m.brand_bit.assign(m.spot_count, 0);
  m.stocks.assign(m.spot_count, 0);
  for (int i = 0; i < m.spot_count; ++i) {
    const int b = brand_id.at(config.spots[i].brand);
    m.brand_index[i] = b;
    m.brand_bit[i] = std::uint64_t{1} << b;
    m.stocks[i] = config.spots[i].stocks;
  }
  for (int brand : history.distinct_brands) {
    if (auto it = brand_id.find(brand); it != brand_id.end()) {
      m.known_brand_mask |= std::uint64_t{1} << it->second;
    }
  }
  for (std::size_t a = 0; a < types.size(); ++a) {
    if (types[a] == AgentKind::Patrol) m.patrols.push_back(a);
  }
  int total_servings = 0;
  for (const auto& spot : config.spots) total_servings += spot.stocks;
  m.w_servings = 1;
  m.w_daily = static_cast<std::int64_t>(total_servings) + 1;
  m.w_distinct = m.w_daily * (m.brand_count + 1) + 1;
  return m;
}

// Pricing subproblem: label-setting DP for one patrol agent. Returns columns
// whose reduced cost (under the supplied duals) exceeds eps.
std::vector<BpColumn> bp_pricing(const BpModel& m, std::size_t agent,
                                 double gamma_a,
                                 const std::vector<double>& beta,
                                 std::size_t label_cap, double eps,
                                 const BpDeadline& deadline) {
  const auto& config = *m.config;
  const auto& day = *m.day;
  const auto& graph = *m.graph;
  const int S = m.spot_count;
  std::vector<BpColumn> result;
  if (S == 0 || bp_expired(deadline)) return result;
  std::vector<int> spot_node(S);
  for (int i = 0; i < S; ++i) {
    spot_node[i] = graph.node_for_pos.at(config.spots[i].pos);
  }
  const int start = graph.node_for_pos.at(day.agents[agent].pos);
  std::vector<int> start_to(S), node_to(S * S);
  for (int i = 0; i < S; ++i) {
    start_to[i] = (spot_node[i] == start) ? 1
                                          : lns_path_time(graph, start, spot_node[i]);
  }
  for (int u = 0; u < S; ++u) {
    if (bp_expired(deadline)) return result;
    for (int v = 0; v < S; ++v) {
      node_to[u * S + v] =
          (spot_node[u] == spot_node[v]) ? 1
                                         : lns_path_time(graph, spot_node[u], spot_node[v]);
    }
  }

  auto brand_gain = [&](int spot) {
    const int b = m.brand_index[spot];
    const bool is_new = (m.brand_bit[spot] & m.known_brand_mask) == 0;
    double g = static_cast<double>(m.w_daily);
    if (is_new) g += static_cast<double>(m.w_distinct);
    return g - beta[b];
  };

  struct Lbl {
    double rc;
    int time;
    int servings;
    std::uint64_t brand_mask;
    std::uint64_t spot_mask;
    std::vector<int> spots;
  };
  std::vector<std::vector<Lbl>> labels(S);

  auto dominates = [](const Lbl& a, const Lbl& b) {
    if (a.rc < b.rc - 1e-9) return false;
    if (a.time > b.time) return false;
    if ((a.brand_mask & ~b.brand_mask) != 0) return false;
    if ((a.spot_mask & ~b.spot_mask) != 0) return false;
    return true;
  };
  auto insert_label = [&](int spot, Lbl lbl) {
    auto& lst = labels[spot];
    for (const auto& e : lst) {
      if (dominates(e, lbl)) return false;
    }
    lst.erase(std::remove_if(lst.begin(), lst.end(),
                             [&](const Lbl& e) { return dominates(lbl, e); }),
              lst.end());
    lst.push_back(std::move(lbl));
    if (lst.size() > label_cap) {
      std::sort(lst.begin(), lst.end(),
                [](const Lbl& l, const Lbl& r) { return l.rc > r.rc; });
      lst.resize(label_cap);
    }
    return true;
  };

  for (int i = 0; i < S; ++i) {
    if (bp_expired(deadline)) return result;
    if (start_to[i] > m.horizon) continue;
    Lbl lbl{brand_gain(i) + static_cast<double>(m.w_servings * m.stocks[i]),
            start_to[i], m.stocks[i], m.brand_bit[i],
            std::uint64_t{1} << i, {i}};
    insert_label(i, std::move(lbl));
  }

  bool changed = true;
  int rounds = 0;
  while (changed && rounds <= S + 2 && !bp_expired(deadline)) {
    changed = false;
    ++rounds;
    // Snapshot only the size of each label list: labels added during this
    // round must not seed further extensions within the same round. Capturing
    // sizes (instead of deep-copying every Lbl and its spots vector) keeps the
    // extension loop allocation-free.
    std::vector<std::size_t> base_size(S);
    for (int u = 0; u < S; ++u) base_size[u] = labels[u].size();
    for (int u = 0; u < S; ++u) {
      if (bp_expired(deadline)) return result;
      for (std::size_t li = 0; li < base_size[u]; ++li) {
        const auto& lbl = labels[u][li];
        for (int v = 0; v < S; ++v) {
          if ((lbl.spot_mask & (std::uint64_t{1} << v)) != 0) continue;
          const int tt = node_to[u * S + v];
          if (tt >= std::numeric_limits<int>::max() / 8) continue;
          const int new_time = lbl.time + tt;
          if (new_time > m.horizon) continue;
          const bool brand_new =
              (lbl.brand_mask & m.brand_bit[v]) == 0;
          double delta = static_cast<double>(m.w_servings * m.stocks[v]);
          if (brand_new) delta += brand_gain(v);
          Lbl next{lbl.rc + delta, new_time, lbl.servings + m.stocks[v],
                   lbl.brand_mask | m.brand_bit[v],
                   lbl.spot_mask | (std::uint64_t{1} << v), lbl.spots};
          next.spots.push_back(v);
          if (insert_label(v, std::move(next))) changed = true;
        }
      }
    }
  }

  std::unordered_map<std::uint64_t, double> best_rc;
  std::unordered_map<std::uint64_t, const Lbl*> best_lbl;
  for (int i = 0; i < S; ++i) {
    for (const auto& lbl : labels[i]) {
      const double rc = lbl.rc - gamma_a;
      auto [it, ins] = best_rc.try_emplace(lbl.spot_mask, rc);
      if (ins || rc > it->second) {
        it->second = rc;
        best_lbl[lbl.spot_mask] = &lbl;
      }
    }
  }
  for (const auto& [mask, ptr] : best_lbl) {
    (void)mask;
    const double rc = ptr->rc - gamma_a;
    if (rc <= eps) continue;
    BpColumn col;
    col.brand_mask = ptr->brand_mask;
    col.spot_mask = ptr->spot_mask;
    col.spots = ptr->spots;
    col.time = ptr->time;
    col.servings = ptr->servings;
    col.agent = static_cast<int>(agent);
    col.profit = bp_column_profit(m, col.brand_mask, col.servings);
    result.push_back(std::move(col));
  }
  std::sort(result.begin(), result.end(),
            [](const BpColumn& l, const BpColumn& r) { return l.profit > r.profit; });
  return result;
}

// Revised simplex for max c^T x s.t. A x <= b, x >= 0 (b >= 0). Slacks form an
// immediately feasible basis, so no Phase-I/artificial variables are needed.
// Returns the primal solution, objective, and the dual of every <= row.
bool bp_simplex_le(const std::vector<double>& c,
                   const std::vector<std::vector<double>>& A,
                   const std::vector<double>& b, std::vector<double>& x,
                   double& obj, std::vector<double>& dual,
                   const BpDeadline& deadline) {
  const int n = static_cast<int>(c.size());
  const int m = static_cast<int>(b.size());
  x.assign(n, 0.0);
  dual.assign(m, 0.0);
  obj = 0.0;
  if (n == 0) return true;
  if (m == 0) {
    // Unconstrained apart from x>=0: take every positive-cost variable.
    for (int k = 0; k < n; ++k) {
      if (c[k] > 0.0) {
        x[k] = 1e9;
        obj = 1e18;
      }
    }
    return true;
  }
  const int ncols = n + m;
  std::vector<double> cfull(ncols, 0.0);
  for (int k = 0; k < n; ++k) cfull[k] = c[k];
  std::vector<std::vector<double>> T(m, std::vector<double>(ncols + 1, 0.0));
  for (int j = 0; j < m; ++j) {
    for (int k = 0; k < n; ++k) T[j][k] = A[j][k];
    T[j][n + j] = 1.0;
    T[j][ncols] = b[j];
  }
  std::vector<int> basis(m);
  for (int j = 0; j < m; ++j) basis[j] = n + j;
  std::vector<double> objrow(ncols + 1, 0.0);
  auto rebuild = [&]() {
    std::fill(objrow.begin(), objrow.end(), 0.0);
    for (int k = 0; k < ncols; ++k) objrow[k] = -cfull[k];
    for (int i = 0; i < m; ++i) {
      const double cb = cfull[basis[i]];
      if (cb == 0.0) continue;
      for (int k = 0; k <= ncols; ++k) objrow[k] += cb * T[i][k];
    }
  };
  auto pivot = [&](int r, int col) {
    const double piv = T[r][col];
    for (int k = 0; k <= ncols; ++k) T[r][k] /= piv;
    for (int i = 0; i < m; ++i) {
      if (i == r) continue;
      const double f = T[i][col];
      if (f == 0.0) continue;
      for (int k = 0; k <= ncols; ++k) T[i][k] -= f * T[r][k];
    }
    basis[r] = col;
  };
  rebuild();
  int guard = 0;
  while (++guard < 30000) {
    if (bp_expired(deadline)) return false;
    int ent = -1;
    for (int k = 0; k < ncols; ++k) {
      if (objrow[k] < -1e-9) {
        ent = k;
        break;
      }
    }
    if (ent < 0) break;
    int leav = -1;
    double best = 1e100;
    for (int i = 0; i < m; ++i) {
      if (T[i][ent] > 1e-9) {
        const double r = T[i][ncols] / T[i][ent];
        if (r < best - 1e-12) {
          best = r;
          leav = i;
        }
      }
    }
    if (leav < 0) {
      obj = 1e18;
      return true;
    }
    pivot(leav, ent);
    rebuild();
  }
  for (int i = 0; i < m; ++i) {
    const int col = basis[i];
    if (col < n) x[col] = T[i][ncols];
  }
  for (int j = 0; j < m; ++j) dual[j] = objrow[n + j];
  obj = objrow[ncols];
  return true;
}

struct BpLpSolution {
  std::vector<double> x;
  std::vector<double> agent_dual;  // per patrol (in m.patrols order)
  std::vector<double> brand_dual;  // per brand index
  double objective{};
  bool complete{true};
};

// Solve the restricted master LP over `pool` (filtered by `active`).
BpLpSolution bp_solve_master(const BpModel& m, const std::vector<BpColumn>& pool,
                             const std::vector<char>& active,
                             const BpDeadline& deadline) {
  BpLpSolution sol;
  const int P = static_cast<int>(pool.size());
  std::vector<int> col_index;
  col_index.reserve(P);
  for (int i = 0; i < P; ++i) {
    if (active[i]) col_index.push_back(i);
  }
  const int n = static_cast<int>(col_index.size());
  const int agent_rows = static_cast<int>(m.patrols.size());
  const int total_rows = agent_rows + m.brand_count;
  std::vector<double> c(n, 0.0);
  std::vector<std::vector<double>> A(total_rows, std::vector<double>(n, 0.0));
  std::vector<double> b(total_rows, 1.0);
  for (int j = 0; j < n; ++j) {
    const auto& col = pool[col_index[j]];
    c[j] = static_cast<double>(col.profit);
    for (int a = 0; a < agent_rows; ++a) {
      if (static_cast<int>(m.patrols[a]) == col.agent) A[a][j] = 1.0;
    }
    for (int bi = 0; bi < m.brand_count; ++bi) {
      if ((col.brand_mask & (std::uint64_t{1} << bi)) != 0) {
        A[agent_rows + bi][j] = 1.0;
      }
    }
  }
  std::vector<double> dual;
  sol.complete =
      bp_simplex_le(c, A, b, sol.x, sol.objective, dual, deadline);
  if (!sol.complete) return sol;
  sol.agent_dual.assign(agent_rows, 0.0);
  sol.brand_dual.assign(m.brand_count, 0.0);
  for (int a = 0; a < agent_rows; ++a) sol.agent_dual[a] = dual[a];
  for (int bi = 0; bi < m.brand_count; ++bi) {
    sol.brand_dual[bi] = dual[agent_rows + bi];
  }
  // Map x back to pool-indexed vector for convenience of callers.
  std::vector<double> xfull(P, 0.0);
  for (int j = 0; j < n; ++j) xfull[col_index[j]] = sol.x[j];
  sol.x = std::move(xfull);
  return sol;
}

// Reduced cost of a column under the current duals (for diagnostic use).
double bp_reduced_cost(const BpModel& m, const BpColumn& col, double gamma_a,
                       const std::vector<double>& beta) {
  double rc = static_cast<double>(col.profit) - gamma_a;
  for (int bi = 0; bi < m.brand_count; ++bi) {
    if ((col.brand_mask & (std::uint64_t{1} << bi)) != 0) rc -= beta[bi];
  }
  return rc;
}

// Column generation at the root. Seeds `pool` with single-spot columns and
// then prices multi-stop routes until no column with positive reduced cost
// remains (capped to keep the pool tractable).
bool bp_column_generation(const BpModel& m, std::vector<BpColumn>& pool,
                          std::size_t label_cap, int max_rounds,
                          const BpDeadline& deadline) {
  const auto& config = *m.config;
  const auto& day = *m.day;
  const auto& graph = *m.graph;
  std::unordered_map<std::uint64_t, int> present;
  const std::size_t pool_cap = 12000;
  auto register_col = [&](BpColumn col) {
    if (pool.size() >= pool_cap) return false;
    const std::uint64_t key =
        (static_cast<std::uint64_t>(col.agent) << 58) | col.spot_mask;
    auto [it, ins] = present.try_emplace(key, static_cast<int>(pool.size()));
    if (ins) {
      pool.push_back(std::move(col));
      return true;
    }
    return false;
  };

  // Initial columns: one per reachable (agent, spot) pair.
  for (std::size_t a : m.patrols) {
    if (bp_expired(deadline)) return false;
    const int start = graph.node_for_pos.at(day.agents[a].pos);
    for (int i = 0; i < m.spot_count; ++i) {
      const int target = graph.node_for_pos.at(config.spots[i].pos);
      const int added = (start == target) ? 1 : lns_path_time(graph, start, target);
      if (added > m.horizon) continue;
      BpColumn col;
      col.brand_mask = m.brand_bit[i];
      col.spot_mask = std::uint64_t{1} << i;
      col.spots = {i};
      col.time = added;
      col.servings = m.stocks[i];
      col.agent = static_cast<int>(a);
      col.profit = bp_column_profit(m, col.brand_mask, col.servings);
      register_col(std::move(col));
    }
  }
  if (pool.empty()) return true;

  for (int round = 0; round < max_rounds; ++round) {
    if (bp_expired(deadline)) return false;
    std::vector<char> active(pool.size(), 1);
    auto sol = bp_solve_master(m, pool, active, deadline);
    if (!sol.complete) return false;
    bool added = false;
    for (std::size_t ai = 0; ai < m.patrols.size(); ++ai) {
      if (bp_expired(deadline)) return false;
      const std::size_t a = m.patrols[ai];
      auto cols = bp_pricing(m, a, sol.agent_dual[ai], sol.brand_dual,
                             label_cap, 1e-7, deadline);
      // Keep the most promising priced columns to bound pool growth.
      if (cols.size() > 24) cols.resize(24);
      for (auto& col : cols) {
        if (bp_reduced_cost(m, col, sol.agent_dual[ai], sol.brand_dual) <= 1e-7)
          continue;
        if (register_col(std::move(col))) added = true;
      }
    }
    if (!added) break;
  }
  return !bp_expired(deadline);
}

std::int64_t bp_value_of(const BpModel& m, int distinct, int daily,
                         int servings) {
  const int initial_distinct =
      static_cast<int>(std::popcount(m.known_brand_mask));
  const int new_brands = std::max(0, distinct - initial_distinct);
  return m.w_distinct * new_brands + m.w_daily * daily +
         m.w_servings * servings;
}

struct BpSearch {
  const BpModel* model{};
  const MapConfig* config{};
  const DayInfo* day{};
  const PolicyHistory* history{};
  const AgentTypes* types{};
  const AcoGraph* graph{};
  const std::vector<AcoMeetingList>* meeting_cache{};
  const ActionPlan* incumbent_plan{};
  std::vector<BpColumn> pool;
  std::int64_t best_bigm{std::numeric_limits<std::int64_t>::min()};
  ActionPlan best_plan;
  bool has_best{false};
  std::int64_t node_budget{};
  std::int64_t nodes_used{0};
  BpDeadline deadline;
  bool stopped{false};

  bool expired() {
    if (stopped) return true;
    if (deadline && std::chrono::steady_clock::now() >= *deadline) {
      stopped = true;
      return true;
    }
    if (node_budget > 0 && nodes_used >= node_budget) {
      stopped = true;
      return true;
    }
    return false;
  }

  // Decode a set of selected columns (one per agent at most) into an action
  // plan and update the incumbent when it improves the official objective.
  void maybe_update(const std::vector<int>& selected) {
    LnsSkeleton skeleton;
    skeleton.routes.resize(types->size());
    for (int idx : selected) {
      const auto& col = pool[idx];
      if (col.spots.empty()) continue;
      skeleton.routes[col.agent] = col.spots;
    }
    auto plan = decode_lns_skeleton(*config, *day, *types, *graph,
                                    *meeting_cache, skeleton);
    if (!plan) return;
    if (incumbent_plan != nullptr && *plan == *incumbent_plan) return;
    auto eval = evaluate_candidate(*config, *day, *history, *plan);
    if (!eval) return;
    const std::int64_t bigm =
        bp_value_of(*model, std::get<0>(eval->value), std::get<1>(eval->value),
                    std::get<2>(eval->value));
    if (bigm < best_bigm) return;
    if (!has_best || bigm > best_bigm) {
      best_bigm = bigm;
      best_plan = std::move(*plan);
      has_best = true;
    }
  }

  void search(std::vector<char> active) {
    if (expired()) return;
    ++nodes_used;
    auto sol = bp_solve_master(*model, pool, active, deadline);
    if (!sol.complete) {
      stopped = true;
      return;
    }
    // Equal-objective routes can end at a different position/fuel state. They
    // are irrelevant to the standalone daily policy but valuable to MLNS's
    // suffix replay, so prune only a strict upper-bound loss.
    if (sol.objective + 1e-6 < static_cast<double>(best_bigm)) return;
    // Find fractional column closest to 0.5.
    int frac = -1;
    double dist = 1.0;
    for (int i = 0; i < static_cast<int>(pool.size()); ++i) {
      if (!active[i]) continue;
      const double v = sol.x[i];
      if (v > 1e-6 && v < 1.0 - 1e-6) {
        const double d = std::abs(v - 0.5);
        if (d < dist) {
          dist = d;
          frac = i;
        }
      }
    }
    if (frac < 0) {
      std::vector<int> selected;
      for (int i = 0; i < static_cast<int>(pool.size()); ++i) {
        if (active[i] && sol.x[i] > 0.5) selected.push_back(i);
      }
      maybe_update(selected);
      // An integral root is normally where branch-and-bound stops. For MLNS we
      // also need alternative optima, because an equal daily score with a
      // different ending state can improve later days. Exclude each selected
      // route in turn and enumerate the next integral solutions within the
      // same node/deadline budget.
      for (int index : selected) {
        if (expired()) break;
        auto alternate = active;
        alternate[index] = 0;
        search(std::move(alternate));
      }
      return;
    }
    // Branch lambda = 0: remove the column.
    auto child0 = active;
    child0[frac] = 0;
    search(std::move(child0));
    if (expired()) return;
    // Branch lambda = 1: keep only this column among those sharing its agent
    // or any of its brands.
    const auto& chosen = pool[frac];
    auto child1 = active;
    for (int i = 0; i < static_cast<int>(pool.size()); ++i) {
      if (!active[i] || i == frac) continue;
      const auto& col = pool[i];
      if (col.agent == chosen.agent ||
          (col.brand_mask & chosen.brand_mask) != 0) {
        child1[i] = 0;
      }
    }
    search(std::move(child1));
  }
};

ExactDayResult bp_branch_and_price(
    const BpModel& model, const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& types,
    const AcoGraph& graph, const std::vector<AcoMeetingList>& meeting_cache,
    const ActionPlan& incumbent_plan, const CandidateValue& incumbent_value,
    std::int64_t node_budget,
    const BpDeadline& deadline) {
  ExactDayResult result{incumbent_plan, incumbent_value, 0, false};
  if (model.patrols.empty()) {
    result.complete = true;
    return result;
  }
  BpSearch s;
  s.model = &model;
  s.config = &config;
  s.day = &day;
  s.history = &history;
  s.types = &types;
  s.graph = &graph;
  s.meeting_cache = &meeting_cache;
  s.incumbent_plan = &incumbent_plan;
  s.node_budget = node_budget;
  s.deadline = deadline;
  s.best_bigm =
      bp_value_of(model, std::get<0>(incumbent_value), std::get<1>(incumbent_value),
                  std::get<2>(incumbent_value));
  s.has_best = false;

  if (!bp_column_generation(model, s.pool, 64, 5, deadline)) {
    result.complete = false;
    return result;
  }
  if (s.pool.empty()) {
    result.complete = true;
    return result;
  }
  std::vector<char> active(s.pool.size(), 1);
  s.search(std::move(active));
  result.explored_nodes = s.nodes_used;
  result.complete = !s.stopped;
  if (s.has_best) {
    auto eval = evaluate_candidate(config, day, history, s.best_plan);
    if (eval && alns_official_value(eval->value) >=
                    alns_official_value(incumbent_value)) {
      result.plan = std::move(s.best_plan);
      result.value = eval->value;
    }
  }
  return result;
}

std::optional<ActionPlan> build_stop_bp_proposal(
    const MapConfig& config, const DayInfo& day,
    const PolicyHistory& history, const AgentTypes& types,
    const ActionPlan& incumbent, const SearchLimits& limits,
    bool allow_official_tie) {
  const auto started = std::chrono::steady_clock::now();
  const bool timed = limits.time_limit_ms >= 0;
  if (timed && limits.time_limit_ms <= 0) return std::nullopt;
  auto incumbent_eval = evaluate_candidate(config, day, history, incumbent);
  if (!incumbent_eval) return std::nullopt;

  std::set<int> brands;
  for (const auto& spot : config.spots) brands.insert(spot.brand);
  const bool unit_stock =
      std::all_of(config.spots.begin(), config.spots.end(),
                  [](const Spot& spot) { return spot.stocks == 1; });
  // The current STOP master treats each brand as a single cluster and each
  // route visit as one serving. Repeated-brand or multi-stock maps require
  // explicit activation/capacity variables; claiming an exact bound there is
  // incorrect, so use the warm incumbent until that richer master exists.
  if (config.spots.empty() || config.spots.size() > 20 ||
      brands.size() != config.spots.size() || !unit_stock) {
    return std::nullopt;
  }

  const auto graph = build_aco_graph(config, day);
  const auto meeting_cache = build_aco_meeting_cache(graph);
  const auto model = bp_build_model(config, day, types, graph, history);
  const BpDeadline deadline =
      timed ? BpDeadline(started + std::chrono::milliseconds(
                                      std::max(1, limits.time_limit_ms)))
            : std::nullopt;
  const std::int64_t node_budget =
      limits.exact_nodes > 0 ? limits.exact_nodes : 50;
  auto result = bp_branch_and_price(
      model, config, day, history, types, graph, meeting_cache, incumbent,
      incumbent_eval->value, node_budget, deadline);
  auto candidate_eval = evaluate_candidate(config, day, history, result.plan);
  if (!candidate_eval || result.plan == incumbent) {
    return std::nullopt;
  }
  const auto candidate_official = alns_official_value(candidate_eval->value);
  const auto incumbent_official = alns_official_value(incumbent_eval->value);
  if (candidate_official < incumbent_official ||
      (!allow_official_tie && candidate_official == incumbent_official)) {
    return std::nullopt;
  }
  return result.plan;
}

ActionPlan build_stop_bp_plan(const MapConfig& config, const DayInfo& day,
                              const PolicyHistory& history,
                              const AgentTypes& types,
                              const SearchLimits& limits) {
  const auto started = std::chrono::steady_clock::now();
  const bool timed = limits.time_limit_ms >= 0;

  // Warm ALNS supplies the incumbent. In timed mode it owns 70% of the request;
  // branch-and-price owns the actual remaining window instead of inheriting an
  // already-expired deadline.
  SearchLimits warm_limits = limits;
  if (timed) {
    warm_limits.time_limit_ms =
        std::max(1, limits.time_limit_ms * 70 / 100);
    warm_limits.min_iterations = 0;
    warm_limits.stagnation_iterations = 0;
  } else {
    const int warm_iters =
        limits.final_alns_iterations > 0 ? limits.final_alns_iterations : 96;
    warm_limits.min_iterations = warm_iters;
    warm_limits.max_iterations = warm_iters;
    warm_limits.stagnation_iterations = warm_iters;
  }
  warm_limits.alns_restarts = limits.alns_restarts > 0 ? limits.alns_restarts : 2;
  ActionPlan warm = build_alns_multirestart_plan(
      config, day, history, types, warm_limits, kProductionAlnsFeatures);
  auto warm_eval = evaluate_candidate(config, day, history, warm);
  if (!warm_eval) return warm;

  SearchLimits bp_limits = limits;
  if (timed) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                               started + std::chrono::milliseconds(
                                             std::max(0, limits.time_limit_ms)) -
                               std::chrono::steady_clock::now())
                               .count();
    if (remaining <= 0) return warm;
    bp_limits.time_limit_ms = static_cast<int>(remaining);
  }
  auto proposal =
      build_stop_bp_proposal(config, day, history, types, warm, bp_limits);
  if (!proposal) return warm;

  // Only replace the warm incumbent on a strict lexicographic improvement;
  // on non-final days retain the warm plan's brands and avoid a large fuel loss.
  auto warm_official = alns_official_value(warm_eval->value);
  auto candidate_eval = evaluate_candidate(config, day, history, *proposal);
  if (!candidate_eval) return warm;
  auto cand_official = alns_official_value(candidate_eval->value);
  if (cand_official <= warm_official) return warm;
  const bool final_day =
      day.day + 1 >= static_cast<int>(config.day_steps.size());
  if (final_day) return *proposal;
  auto resulting_brands = [&](const CandidateEvaluation& e) {
    std::set<int> b = history.distinct_brands;
    for (const auto& acq : e.trace.acquisitions) {
      if (const Spot* sp = spot_at(config, acq.spot_pos))
        b.insert(sp->brand);
    }
    return b;
  };
  for (std::size_t agent = 0; agent < candidate_eval->ending_fuel.size();
       ++agent) {
    if (candidate_eval->ending_fuel[agent] <
        warm_eval->ending_fuel[agent] - 10) {
      return warm;
    }
  }
  auto bp_brands = resulting_brands(*candidate_eval);
  auto warm_brands = resulting_brands(*warm_eval);
  for (int b : warm_brands) {
    if (!bp_brands.contains(b)) return warm;
  }
  return *proposal;
}
}  // namespace hexudon
