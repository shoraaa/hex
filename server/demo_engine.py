from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple


PLAIN, ROAD, MOUNTAIN, POND = 0, 1, 2, 3
SMOOTH, CONGESTED, JAM = 0, 1, 2

_PLAIN_STEPS    = 2
_MOUNTAIN_STEPS = 3
_ROAD_STEPS     = {SMOOTH: 1, CONGESTED: 2, JAM: 4}
_FUEL_COST      = {PLAIN: 1, ROAD: 2, MOUNTAIN: 2}


_EVEN_DELTAS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (0, -1)]
_ODD_DELTAS  = [(-1,-1), (-1, 0), (0, 1), (1, 0), (1,-1), (0, -1)]


_CHAR_TO_TERRAIN = {"P": PLAIN, "R": ROAD, "M": MOUNTAIN, "O": POND}


class HexGrid:
    def __init__(
        self,
        width: int,
        height: int,
        cells: List[int],
        road_cond: Dict[int, int],
    ) -> None:
        self.width = width
        self.height = height
        self.cells = cells
        self.road_cond = road_cond

    @classmethod
    def from_map_response(cls, resp: dict) -> "HexGrid":
        if "map" in resp:
            m = resp["map"]
            w, h = m["width"], m["height"]
            flat = [v for row in m["cells"] for v in row]
        else:
            w, h = resp["width"], resp["height"]
            raw = resp["cells"]
            flat = [
                _CHAR_TO_TERRAIN.get(str(v).upper(), int(v))
                if not isinstance(v, int) else v
                for v in raw
            ]
        return cls(width=w, height=h, cells=flat, road_cond={})


    def neighbors(self, pos: int) -> List[Tuple[int, int]]:
        row, col = divmod(pos, self.width)
        deltas = _EVEN_DELTAS if row % 2 == 0 else _ODD_DELTAS
        result = []
        for d, (dr, dc) in enumerate(deltas):
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                nb = nr * self.width + nc
                if self.cells[nb] != POND:
                    result.append((d, nb))
        return result


    def travel_steps(self, pos: int) -> int:
        t = self.cells[pos]
        if t == ROAD:
            return _ROAD_STEPS.get(self.road_cond.get(pos, SMOOTH), 1)
        return _PLAIN_STEPS if t == PLAIN else _MOUNTAIN_STEPS

    def fuel_cost(self, pos: int) -> int:
        return _FUEL_COST.get(self.cells[pos], 0)


def dijkstra_from(
    grid: HexGrid,
    start: int,
    fuel: int,
    optimize_fuel: bool = False
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, Tuple[int, int]]]:
    INF = 10 ** 9
    dist_steps: Dict[int, int] = {start: 0}
    dist_fuel:  Dict[int, int] = {start: 0}
    prev: Dict[int, Tuple[int, int]] = {}

    heap = [(0, 0, start)]

    while heap:
        c1, c2, pos = heapq.heappop(heap)

        if optimize_fuel:
            f, s = c1, c2
            if f > dist_fuel.get(pos, INF):
                continue
            if f == dist_fuel.get(pos, INF) and s > dist_steps.get(pos, INF):
                continue
        else:
            s, f = c1, c2
            if s > dist_steps.get(pos, INF):
                continue
            if s == dist_steps.get(pos, INF) and f > dist_fuel.get(pos, INF):
                continue

        step_cost = grid.travel_steps(pos)
        fuel_cost = grid.fuel_cost(pos)

        for d, nb in grid.neighbors(pos):
            ns = s + step_cost
            nf = f + fuel_cost
            if nf > fuel:
                continue

            if optimize_fuel:
                better = (
                    nf < dist_fuel.get(nb, INF)
                    or (nf == dist_fuel.get(nb, INF) and ns < dist_steps.get(nb, INF))
                )
            else:
                better = (
                    ns < dist_steps.get(nb, INF)
                    or (ns == dist_steps.get(nb, INF) and nf < dist_fuel.get(nb, INF))
                )

            if better:
                dist_steps[nb] = ns
                dist_fuel[nb]  = nf
                prev[nb] = (pos, d)
                if optimize_fuel:
                    heapq.heappush(heap, (nf, ns, nb))
                else:
                    heapq.heappush(heap, (ns, nf, nb))

    return dist_steps, dist_fuel, prev


def reconstruct_path(
    prev: Dict[int, Tuple[int, int]], start: int, goal: int
) -> List[int]:
    path: List[int] = []
    cur = goal
    while cur != start:
        parent, direction = prev[cur]
        path.append(direction)
        cur = parent
    path.reverse()
    return path


class DayPlanner:
    def plan(
        self,
        grid: HexGrid,
        patrol_agents: List[dict],
        refuel_agents: List[dict],
        spots: List[dict],
        day_steps: int,
        brands_seen: Set,
        current_day: int = 0,
        strategy: str = "lazy",
    ) -> Dict[str, List[int]]:
        all_actions = {}
        if current_day > 0:
            for ag in patrol_agents + refuel_agents:
                aid = ag["agent_id"]
                all_actions[aid] = [-day_steps]
            return all_actions

        spot_map = {s["pos"]: s for s in spots}
        assigned_spots = set()

        for ag in patrol_agents:
            aid = ag["agent_id"]
            pos = ag["pos"]
            fuel = ag["fuel"]
            
            ds, _, prev = dijkstra_from(grid, pos, fuel)
            
            best_spot = None
            best_dist = float('inf')
            
            for spos in spot_map:
                if spos in assigned_spots:
                    continue
                if spos in ds and ds[spos] < best_dist:
                    best_dist = ds[spos]
                    best_spot = spos
                    
            actions = []
            remaining_steps = day_steps
            
            if best_spot is not None:
                assigned_spots.add(best_spot)
                path_dirs = reconstruct_path(prev, pos, best_spot)
                
                cur_pos = pos
                for d in path_dirs:
                    step_cost = grid.travel_steps(cur_pos)
                    if remaining_steps < step_cost:
                        break
                    actions.append(d)
                    remaining_steps -= step_cost
                    for nd, nb in grid.neighbors(cur_pos):
                        if nd == d:
                            cur_pos = nb
                            break

            if remaining_steps > 0:
                actions.append(-remaining_steps)
                
            all_actions[aid] = actions
            
        for ag in refuel_agents:
            all_actions[ag["agent_id"]] = [-day_steps]

        return all_actions

