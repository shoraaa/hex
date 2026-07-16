# The 37th National KOSEN (College of Technology) Programming Contest
## About the Formats of the Competition Division “HEXUDON”

*Unofficial English translation of 競技部門「ヘキサうどん」のフォーマットについて.pdf (translated 2026-07-09). Terminology follows docs/hexudon_EN.md.*

### Map Configuration Format Before the Match Starts

Before the match starts, the map configuration is provided as JSON data in the following form.

For information about match timing, the `"startsAt"` key records the start time of the match (UNIX TIME).

The `"daySeconds"` key records the answer time in seconds for each day of the match, and the `"daySteps"` key records the number of steps for each day. The number of elements in the `"daySteps"` array indicates the number of days.

The `"map"` key records the terrain information of the map, and the `"spots"` key records the spot information.

In the terrain information, the `"height"` key is the vertical size of the map, the `"width"` key is the horizontal size of the map, and the `"cells"` key gives the terrain type of each cell of the map (0: Plain, 1: Road, 2: Mountain, 3: Pond).

In the spot information, the `"brand"` key is the spot’s franchise chain, the `"pos"` key is the spot’s position, and the `"stocks"` key is the maximum stock.

For information about the agents, the `"agents"` key records the initial positions of your own player’s agents. During the match, the order in this list is assigned to the agents, and the match information format, the agent type answer format, and the action plan answer format described below are all written in this same order.

The `"fuelLimits"` key records the maximum fuel capacity.

For information about traffic, the `"players"` key records the number of participating players, the `"busyThreshold"` key records the congestion threshold, and the `"jammedThreshold"` key records the traffic jam threshold.

```json
{
 "startsAt": 1778227200,
 "daySeconds": [5, 5, 5, 10],
 "daySteps": [50, 100, 150, 200],
 "map": {
 "height": 8,
 "width": 8,
 "cells": [
 [3, 0, 1, 2, 0, 1, 2, 0],
 [3, 0, 1, 2, 0, 1, 2, 0],
 [3, 0, 1, 2, 0, 1, 2, 0],
 [3, 0, 1, 2, 0, 1, 2, 0],
 [3, 0, 1, 2, 0, 1, 2, 0],
 [3, 0, 1, 2, 0, 1, 2, 0],
 [3, 0, 1, 2, 0, 1, 2, 0],
 [3, 0, 1, 2, 0, 1, 2, 0]
 ]
 },
 "spots": [
 {"brand": 0, "pos": 1, "stocks": 4},
 {"brand": 1, "pos": 9, "stocks": 1},
 {"brand": 0, "pos": 17, "stocks": 1},
 {"brand": 1, "pos": 25, "stocks": 3}
 ],
 "agents": [4, 12, 20, 28],
 "fuelLimits": 20,
 "players": 8,
 "busyThreshold": 2,
 "jammedThreshold": 4
}
```

### Match Information Format at the Start of Each Day

The map information provided at the start of each day is given as JSON data in the following form.

The `"endsAt"` key records the time (UNIX TIME) when answer submissions close for that day.

The `"day"` key records the day as of just before this map information is provided. On the first day this value is 0.

The `"agents"` key records your own team’s agent information. The `"kind"` key is the agent type (0: Patrol Car, 1: Refueling Car), the `"pos"` key is the agent’s position, and the `"fuel"` key is its fuel load.

The `"others"` key records the agent information of the other teams. The `"id"` key is the other team’s ID, and the `"agents"` key records the information in the same format as your own team’s.

The `"traffics"` key records the congestion information of the road cells. The `"pos"` key is the road’s position, and the `"status"` key is the state of each road cell (0: Smooth, 1: Congested, 2: Traffic Jam).

```json
{
 "endsAt": 1778227205,
 "day": 1,
 "agents": [
 {"kind": 0, "pos": 1, "fuel": 20},
 {"kind": 1, "pos": 1, "fuel": 20},
 {"kind": 0, "pos": 9, "fuel": 10},
 {"kind": 0, "pos": 9, "fuel": 0}
 ],
 "others": [
 {
 "id": 0,
 "agents": [
 {"kind": 0, "pos": 1, "fuel": 2},
 {"kind": 0, "pos": 1, "fuel": 3},
 {"kind": 0, "pos": 1, "fuel": 4},
 {"kind": 0, "pos": 1, "fuel": 5}
 ]
 },
 {
 "id": 1,
 "agents": [
 {"kind": 1, "pos": 1, "fuel": 20},
 {"kind": 1, "pos": 9, "fuel": 20},
 {"kind": 1, "pos": 17, "fuel": 20},
 {"kind": 1, "pos": 25, "fuel": 20}
 ]
 }
 ],
 "traffics": [
 {"pos": 1, "status": 0},
 {"pos": 9, "status": 0},
 {"pos": 17, "status": 1},
 {"pos": 25, "status": 2}
 ]
}
```

### Agent Type Answer Format

Before the match starts, the agent types are received as JSON data in the following form.

The agent types (0: Patrol Car, 1: Refueling Car) are written in an array. The number of elements in the array must match the number of agents.

Data in which the number of elements differs from the number of agents, or in which a value other than {0, 1} is specified as an agent type, is treated as invalid data and rejected. If no valid data has been accepted within the time limit, all agents are set to Patrol Cars.

```json
[0, 1, 0, 1]
```

### Action Plan Answer Format

As the answer, the action plan for each day is received as JSON data in the following form.

The action data for all agents is written in an array. The example describes the action sequences of two agents.

Values of -1 or less represent waiting. In the example, `[-15]` specifies waiting for 15 steps.

Values 0 to 5 represent movement directions: 0 is a move toward the upper left, 1 is a move toward the upper right, and so on clockwise.

In the example, `[0,1,-10]` specifies moving to the upper left, then moving to the upper right, and then waiting for 10 steps.

Each agent’s action plan must match the number of steps in the day. In addition, an answer that contains a move to an impassable cell (a Pond, or outside the map) is treated as invalid data. If the action plan of even one agent is invalid, the action plans of all agents are treated as invalid and rejected. If no valid data has been accepted within the time limit, all agents wait for the entire day.

```json
[
 [-15],
 [0,1,-10]
]
``
