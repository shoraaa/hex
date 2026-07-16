from __future__ import annotations
import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Set, Tuple
from demo_engine import HexGrid, DayPlanner


class Client:
    def __init__(
        self,
        base_url: str,
        game_id: str,
        team_id: str,
        token: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.game_id  = game_id
        self.team_id  = str(team_id)
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "Hexudon/1.0",
        }


    def _get(self, path: str, **params) -> dict:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.base_url}{path}?{qs}" if qs else f"{self.base_url}{path}"
        req = urllib.request.Request(url, headers=self._headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"GET {path} -> HTTP {e.code}: {e.read().decode(errors='replace')}") from e

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=self._headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"POST {path} -> HTTP {e.code}: {e.read().decode(errors='replace')}") from e

    def get_map(self) -> dict:
        return self._get("/api/game/config", game_id=self.game_id)

    def get_state(self) -> dict:
        return self._get("/api/game/state", game_id=self.game_id)

    def get_day_info(self) -> Optional[dict]:
        return self._get("/api/game/day", game_id=self.game_id)

    def select_agent_types(self, assignments: List[dict]) -> dict:
        types = [
            0 if a["type"] == "patrol" else 1
            for a in sorted(assignments, key=lambda a: int(a["agent_id"]))
        ]
        return self._post("/api/game/agent-types", {
            "game_id": self.game_id,
            "types":   types,
        })

    def submit_actions(self, day: int, actions: List[List[int]]) -> dict:
        return self._post("/api/game/actions", {
            "game_id": self.game_id,
            "day":     day,
            "actions": actions,
        })

    def submit_practice(self, day: int, actions: List[List[int]]) -> dict:
        return self._post("/api/game/practice/actions", {
            "game_id": self.game_id,
            "day":     day,
            "actions": actions,
        })

    def get_result(self) -> dict:
        return self._get("/api/game/result", game_id=self.game_id)


def _parse_map_config(resp: dict) -> Tuple[List[dict], List[int], int]:
    spots          = resp.get("spots", [])
    day_steps_list = resp.get("daySteps", [])
    fuel_limits    = resp.get("fuelLimits", 100)
    return spots, day_steps_list, fuel_limits

def _count_agents(resp: dict) -> int:
    raw = resp.get("agents", [])
    return len(raw)

def _parse_agent_states(
    state: dict,
    team_id: str,
    type_assignments: Dict[str, str],
) -> Tuple[List[dict], List[dict], Dict[int, int], Set, int]:

    team_data = (
        state["teams"].get(team_id)
        or state["teams"].get(str(team_id))
        or {}
    )
    raw_agents = team_data.get("agents", [])
    brands_seen = set(team_data.get("distinct_types", []))
    current_day = state.get("day", 0)

    road_cond = {int(k): v for k, v in state.get("road_condition", {}).items()}

    patrol_agents = []
    refuel_agents = []
    for i, ag in enumerate(raw_agents):
        aid  = str(ag.get("agent_id", i))
        pos  = ag.get("pos", ag.get("cell"))
        fuel = ag.get("fuel") or 0

        kind = ag.get("type")

        if isinstance(kind, int):
            is_patrol = kind == 0
        elif isinstance(kind, str):
            is_patrol = kind == "patrol"
        else:
            is_patrol = type_assignments.get(aid, "patrol") == "patrol"

        entry = {"agent_id": aid, "pos": pos, "fuel": fuel}
        (patrol_agents if is_patrol else refuel_agents).append(entry)

    return patrol_agents, refuel_agents, road_cond, brands_seen, current_day


class Hexudon:
    def __init__(self, client: Client, practice: bool = False) -> None:
        self.client   = client
        self.practice = practice
        self._planner = DayPlanner()

    def fetch_initial_state(self) -> Tuple[HexGrid, List[dict], List[int], int, int, List[int]]:
        map_resp = self.client.get_map()
        spots, day_steps_list, fuel_limits = _parse_map_config(map_resp)
        n_agents = _count_agents(map_resp)
        agents_starts = map_resp.get("agents", [])
        grid = HexGrid.from_map_response(map_resp)
        return grid, spots, day_steps_list, fuel_limits, n_agents, agents_starts

    def auto_assign_agent_types(
        self, n_agents: int, fuel_limits: int, total_steps: int,
        agents_starts: List[int], grid: HexGrid
    ) -> List[dict]:
        use_refuel = n_agents >= 3 and fuel_limits < (total_steps * 1.5)
        refuel_agent_id = None
        
        if use_refuel and agents_starts and grid:
            from demo_engine import dijkstra_from
            center_pos = (grid.height // 2) * grid.width + (grid.width // 2)
            best_dist = float('inf')

            for i, start_pos in enumerate(agents_starts):
                ds, _, _ = dijkstra_from(grid, start_pos, 9999)
                dist = ds.get(center_pos, float('inf'))
                if dist < best_dist:
                    best_dist = dist
                    refuel_agent_id = str(i)

            if refuel_agent_id is None:
                refuel_agent_id = "0"

        assignments_body = []
        for i in range(n_agents):
            aid = str(i)
            t   = "refuel" if (use_refuel and aid == refuel_agent_id) else "patrol"
            assignments_body.append({"agent_id": aid, "type": t})
        return assignments_body

    def submit_agent_types(self, assignments_body: List[dict]) -> None:
        self.client.select_agent_types(assignments_body)

    def fetch_day_state(self, day: int) -> dict:
        if not self.practice:
            self._wait_for_day(day)
        state = self.client.get_state()
        return state

    def generate_day_plan(
        self,
        day: int,
        state: dict,
        grid: HexGrid,
        spots: List[dict],
        day_steps_list: List[int],
        n_agents: int,
        type_assignments: Dict[str, str],
        strategy: str = "lazy"
    ) -> Tuple[List[List[int]], List[dict], List[dict], Set]:
        patrol_agents, refuel_agents, road_cond, brands_seen, current_day = \
            _parse_agent_states(state, self.client.team_id, type_assignments)

        grid.road_cond = road_cond
        day_steps = (
            day_steps_list[day] if day < len(day_steps_list)
            else state.get("steps_today", 50)
        )

        planned_actions = self._planner.plan(
            grid, patrol_agents, refuel_agents, spots, day_steps, brands_seen, current_day=current_day, strategy=strategy
        )

        actions: List[List[int]] = []
        for i in range(n_agents):
            aid = str(i)
            if aid in planned_actions:
                actions.append(planned_actions[aid])
            else:
                actions.append([-day_steps])

        return actions, patrol_agents, refuel_agents, brands_seen

    def submit_day_plan(self, day: int, actions: List[List[int]]) -> dict:
        if self.practice:
            return self.client.submit_practice(day, actions)
        else:
            return self.client.submit_actions(day, actions)

    def _wait_for_day(
        self,
        expected_day: int,
        poll_interval: float = 5.0,
        timeout: float = 300.0,
    ) -> None:
        print(f"Waiting for day {expected_day}...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                state = self.client.get_state()
                status = state.get("status", "")
                current = state.get("day", -1)
                if status == "finished":
                    return
                if status == "in_progress" and current == expected_day:
                    return
            except RuntimeError as e:
                print(f"Poll error: {e}")
            time.sleep(poll_interval)
        print(f"WARNING: timed out waiting for day {expected_day}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hexudon demo (Procon 2026)")
    p.add_argument("--url",     required=True, metavar="URL",
                   help="Server base URL, e.g. http://localhost:8000")
    p.add_argument("--game-id", required=True, metavar="ID",  dest="game_id")
    p.add_argument("--team-id", required=True, metavar="ID",  dest="team_id")
    p.add_argument("--token",   required=True, metavar="JWT",
                   help="Bearer auth token")
    p.add_argument("--practice", action="store_true",
                   help="Use self-paced practice endpoint (local server)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    client = Client(
        base_url=args.url,
        game_id=args.game_id,
        team_id=args.team_id,
        token=args.token,
    )
    engine = Hexudon(client=client, practice=args.practice)
    engine.run()

if __name__ == "__main__":
    main()

