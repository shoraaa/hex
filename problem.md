# NAPROCK 18th International Programming Contest, Takamatsu, Japan
## Competition Section
### "HEXUDON"

## *Competition Section Overview*
In Kagawa prefecture and Shikoku region, particularly in Takamatsu, a style of tourism involving visits to multiple locations is very popular. In addition to the pilgrimage to the 88 temples and tours of local “udon noodle” restaurants, “pilgrimages” to locations featured in movies and anime have also become popular in recent years, and “trips to visit various spots” have taken root as a regional culture in diverse forms. For this year’s competition, drawing on these region-specific tourism styles, we have set the theme as “How to efficiently visit as many spots as possible while refueling.”

In this competition, players control two types of agents—"patrol cars" and "refueling cars"—to move across a board called the "map," which is composed of hexagonal grids. There are spots across the map where players can collect “udon”; patrol cars move while consuming fuel to collect “udon”, and refueling cars can replenish the patrol cars’ fuel. The team that efficiently visits the spots and collects the most “udon” wins.

### *Time Units*
- The smallest unit of time for issuing instructions is called a "step."
- The match is structured into multiple "days."
- The active time for each day is determined by the number of steps.
- The number of steps per day may vary from day by day.
- A single match lasts 4 to 10 days.

### *Map*
- The entire board used in this problem is called a "map," and the hexagonal frames that make up the map are called "cells" (Figure 1). Each cell is adjacent to other cells in six directions.
- An example of a map is shown in Figure 1. Each cell is assigned an integer value, and the range of these integer values is 0 to (map height × map width – 1). For example, as shown in Figure 1, if the map size is 10 by 10, integer values from 0 to 99 are assigned.
- Each map has a maximum width and height of 32 cells and a minimum of 8 cells, respectively.

*(Figure 1: Map - Example Map Layout & Example Map Coordinates)*

### *Terrain Types*
- Each cell is assigned a terrain type. There are four types of terrain: "Plain," "Mountain," "Pond," and "Road."
- Agents can move across plains, mountains, and roads, but the number of steps required to move varies (Table 1). Furthermore, roads have three conditions: "Smooth," "Congested," and "Traffic Jam," and the number of steps required to move varies depending on the condition. Agents cannot move across ponds.
- Movable cells are connected, and there are no plains, mountains, or roads that an agent cannot reach.
- Movement requires fuel, and the amount of fuel consumed varies by terrain type (the details on the fuel is described later in the section on agents).
- The terrain type of each cell does not change during a match.

**Table 1: List of Cell Terrain Types** *(Transposed for formatting)*

| Terrain Type | Travel Time (Steps) | Fuel consumption |
| :--- | :--- | :--- |
| **Plain** | 2 | 1 |
| **Mountain** | 3 | 2 |
| **Pond** | Cannot Enter | – |
| **Road** | 1 (Smooth)<br>2 (Congested)<br>4 (Traffic jam) | 2 |

### *Road Cells and Traffic Volume*
- Road cells have three possible conditions: "Smooth," "Congested," and "Traffic Jam." These conditions are determined by the "traffic volume" over the previous two days (the day before and the day before that). On Day 1, all roads are in the "Smooth" condition, and on Day 2, the road condition is determined solely by the traffic volume from Day 1.
- Traffic volume is calculated by summing the total number of stay steps for all agents in each road cell over the previous two days (the day before and the day before that) for all teams, and then dividing that total by the number of teams.
- The condition of a road cell is determined by the predefined "congestion thresholds" and "jam thresholds." If traffic volume is less than the congestion threshold, the condition is "smooth"; if traffic volume is more than or equal to the congestion threshold and less than the jam threshold, the condition is "congested"; and if traffic volume is more than or equal to the jam threshold, the condition is "traffic jam."
- The values for the congestion threshold and jam threshold vary by match.
- Road conditions are provided by the server at the start of each day and do not change during the day.
- Road conditions are the same for all teams on each day.

## *Spots*

### *Basic Rules for Spots*
- "Spots" are set up on some “Plain” cells (Figure 2). When an agent called a patrol car (described later) arrives at a spot, it automatically acquires one serving of “udon.”
- Each patrol car can collect up to one serving of “udon” per spot per day. No matter how many times the same patrol car visits the same spot on the same day, it can only collect “udon” on its first visit.
- There is no limit to the number of servings of “udon” a patrol car can collect.

### *Franchise Chains and Udon Types*
- Spots are divided into several "franchise chains," and the “type of udon” differs for each chain (Figure 2).
- Each spot is assigned exactly one franchise chain.
- The number of franchise chains assigned to spots ranges from 1 to (the number of spots on the map).
- The location and the franchise chain for each spot do not change during a single match.

*(Figure 2: Spots and Franchise Chains - Example of Spot Placement & Examples of Franchise Chain)*

### *Stocks*
- A maximum stock level of “udon” is set for each spot, and the stock at each spot is replenished to the maximum level at the start of each day.
- The range of maximum stock levels set for each spot is 1 to (the number of agents in a team).
- When a patrol car acquires “udon”, the stock at that spot decreases by one serving. If the stock at a spot is zero, the patrol car cannot acquire “udon” even if it arrives.
- The stock at a spot is independent for each team; even if another team acquires “udon” at a spot, the stock at that spot on your team’s map will not decrease.

## *Agents*
- Each team manages multiple "agents."
- The number of agents ranges from 3 to 8.
- The agents' initial positions are predetermined and are designated as “Plains” where no spots have been placed.
- There are two types of agents: "Patrol Cars" and "Refueling Cars." Patrol cars travel around spots to collect “udon”, while refueling cars provide fuel to patrol cars (Figure 3). Patrol cars cannot supply fuel, and refueling cars cannot collect “udon”.
- Players can specify the type of each agent at the start of the match. Once specified, the type cannot be changed until the match ends.

*(Figure 3: Agent Types and Roles - Patrol Car (Collect "udon" at the spot) / Refueling Car (Supply fuel to patrol cars))*

## *Movement and Fuel*

### *Basic Movement Rules*
- Agents can move to cells adjacent in any of the six directions. They can also move to cells occupied by other agents. Additionally, agents can be instructed to "wait" at the current cell for the specified number of steps instead of moving to any cells. Instructions to move to non-adjacent cells or cells assigned as “Ponds” will result in an invalid answer.
- An agent that receives an instruction to move, moves to the next cell after the number of steps assigned to the terrain at its current location (Figure 4).

*(Figure 4: Movement of an Agent)*

### *Fuel Consumption and Halt Conditions*
- When moving, the patrol car consumes fuel (Table 1) based on the terrain at the time the movement instruction is received.
- Each patrol car has a maximum fuel capacity, and the same limit applies to all patrol cars across all teams.
- If a patrol car runs out of fuel, it must wait until it is refueled. If you issue an instruction to move when fuel is insufficient, your answer will be invalid.
- At the end of each day, or whenever the number of steps required for movement is insufficient, the patrol car must also wait. If an instruction to move is issued despite an insufficient number of steps, the answer will be invalid.

### *Refueling and Refueling Cars*
- If a refueling car remains in the same cell as a patrol car for one or more steps, the patrol car’s fuel is automatically refueled to its maximum capacity.
- Refueling cars do not have a maximum fuel capacity and can supply fuel infinitely.
- A refueling car can move to a different cell without consuming fuel. As with patrol cars, if an instruction to move is issued despite an insufficient number of steps, the answer will be invalid.

## *Match Progression*
(1) Before the match begins, the map configuration is provided, and players specify the type of each agent.
(2) At the start of the match, map information containing the agent selection details for all teams is provided from the server.
(3) Within the designated response time for Day 1, players must submit their actions for Day 1.
(4) After the response time for Day 1 passed, map information for Day 2—reflecting the agent positions and traffic condition data for all teams at the end of Day 1—is provided from the server.
(5) Similarly, on Day 2 or later, players submit their actions within the designated response time for each day.
(6) This procedure is repeated for the specified number of days.

### *Parameters at the start of Day 1*
- The initial positions of the agents on Day 1 are the positions specified in the map configuration.
- The fuel load of each patrol car on Day 1 is equal to the maximum limit.
- The condition of all road cells are “Smooth.”
- The stock of each spot is at its maximum capacity.

### *Parameters at the start of Day 2 and beyond*
- The initial positions of the agents at the start of Day 2 and beyond are the positions at the end of the previous day.
- The fuel load of each patrol car on Day 2 and beyond is the same as the load at the end of the previous day.
- The conditions of road cells are determined based on the traffic volume at the preceding two days for all teams. On Day 2 only, it is determined based on the traffic volume from the previous day.
- The stock of each spot is supplied to the maximum stock level.

## *Determining the Winner*
The winner will be determined based on the number of “udon” types, the number of servings, and the response time, in the following order of priority.
(1) The team that obtains the most **different types** of “udon” in a single match is declared the winner.
(2) The team with the highest **cumulative total of different types** of “udon” collected each day (the sum of the types collected daily) is declared the winner.
(3) The team with the highest **total number of “udon” servings** obtained in a single match is declared the winner.
(4) The team with the lowest **cumulative response time** (the sum of response times for each day) for the last valid answer submitted at the end of each day is declared the winner.
(5) The winner will be determined by rolling dice or the match will be declared a draw.

## *Notes on Matches*
(1) Each match will be played simultaneously by multiple teams. The number of opposing teams will vary depending on the matchups.
(2) The number of competing teams for each match will be announced in the final competition guidelines issued later.
(3) The number of days, the number of steps per day, and the response time for each day will be determined for each match.
(4) Once an answer is accepted by the server, you will be notified whether it is valid or invalid (format error etc.).
(5) Resubmissions are allowed within the time limit, but submitting an excessive number of answers or large file sizes that disrupt the progress of the competition may be considered disruptive behavior and result in disqualification.
(6) The last valid answer accepted will be adopted.
(7) There may be a slight delay in updating map information.

## *Regarding the Problem and Answer Formats*
- Competition map information and agent behavior information are planned to be in text format; details will be published on the Procon/NAPROCK official website and announced by early May.

## *Release of Competition Software*
- We plan to provide the answering protocol, a simplified version of the answering software, and its source code by late July.
- Information regarding the above software will be provided on the Procon/NAPROCK official website and announced as it becomes available.

## *Communication Method*
- We plan to connect each team’s PC to the wired LAN provided at the competition booth and enable data transmission and reception using HTTP POST and GET methods.
- Detailed information on the communication protocol and the answer submission system is scheduled to be published on the Procon/NAPROCK official website and be announced through MS Teams around early July.

## *Miscellaneous Notes*
- Participants may bring a maximum of three portable, programmable devices into the competition. At least one of these must be a device equipped with a 10BASE-T/100BASE-TX/1000BASE-T RJ45 wired LAN port and capable of TCP/IP connectivity for submitting answers.
- We plan to provide each team with at least four power outlets in the competition booth. Please ensure that the total power consumption does not exceed 500W.
- We plan to provide one LAN cable per team for connecting to the competition network. If you need to connect multiple computers to the competition network, please provide your own equipment, such as a switching hub.
- Wireless communication via Bluetooth or similar technologies between brought-in devices is permitted, but Wi-Fi communication is not allowed.
- For cooling your equipment, only air-cooling devices (such as fans) listed in the System Declaration Form are permitted. Items that may leak liquid (including plastic bottles, ice, and items prone to condensation) are not permitted.
- During the competition, teams may exchange information among themselves, but exchanging information with parties outside the team is not permitted. Furthermore, communication with devices other than those brought in is not permitted.
- Any actions that interfere with the server or the progress of other teams’ matche progresses are prohibited. Teams may be disqualified if they are deemed to have interfered with the progress of the match, obstructed referees or other teams, or committed other prohibited acts.
- Regarding the transmission and reception of data via the network, if a malfunction occurs in the organizer’s system, the event may be conducted offline. In this case, the match schedule and other details may be subject to change.
- If the organizers encounter any troubles, they may prepare a different problem and hold a rematch.
- Data used in the competition, as well as data submitted by each team to the server, may be published on the Procon official website or similar platforms after the competition concludes. Additionally, some response information, such as the number of data transmissions, may be displayed on the competition visualizer during the event.
- During the competition, players and their desks (including computer screens, operations, and notes on the desk) may be filmed or recorded using video cameras and displayed simultaneously on screens or other displays.
- During the competition, judges may view the players and their desks (computer screens, operations, notes on the desk, etc.) for evaluation purposes.
- You may be able to obtain additional information by accessing the problem server, which is scheduled to be made available on the official website or through MS Teams.

## *Inquiries*
Please direct all inquiries to the "Contact Information" listed below on the Procon official website. Responses to inquiries will be published on the Procon official website or through MS Teams as they become available. Please be aware that your questions may be made public. Even ideas that could influence the outcome of the competition will be disclosed.

**Inquiry Deadline:** Friday, May 8, 2026, at 5:00 PM (JST)  
**Email address for inquiries:** procon@naprock.jp

- ✓ Please be sure to submit inquiries through your faculty advisor. When doing so, please be sure to include the faculty advisor’s affiliation (name of the university/KOSEN/Institution, department, etc.) and his/her name. We may be unable to respond if the affiliation or the advisor’s name is not provided.
- ✓ We cannot answer questions received after the above deadline.
