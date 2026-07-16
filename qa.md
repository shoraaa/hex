# The 18th NAPROCK International Programming Contest, Takamatsu, Japan
## Competition Section “HEXUDON”
### Q&A

This document contains the questions (Q1–Q21) received by the Secretariat by 5:00 PM (JST) on Friday, April 17, 2026, along with their answers. Additional information, including the answers to these questions, is scheduled to be released after the question submission period ends on May 8. ***Details regarding the format will be released shortly.***

**Q1.** In the example shown in Figure 1 of CFP, if row 0 is defined as the row containing cells 0, 1 and 2, is the hexagon tiling fixed such that even-numbered rows are shifted to the right?
**A1.** It is fixed such that even rows are shifted to the right.

**Q2.** Is the map a torus?
**A2.** The map is not a torus.

**Q3.** Is the answer submission time for each day provided as input?
**A3.** The time available for answering each day is provided in the “daySeconds” key of the map configuration format before the match begins. Additionally, the deadline for submitting answers for each day is provided in the “endsAt” key of the match information format.

**Q4.** How much time is allotted for the phase where agent types are specified before the competition begins?
**A4.** We anticipate approximately 1 minute. Details will be provided in the official competition guidelines for the final.

**Q5.** How long is the answer submission time for each day, and what is the total competition time for the entire match?
**A5.** We anticipate approximately 1 minute per day, and the total competition time for the entire match will be the product of that duration multiplied by the number of days. Details will be provided in the official competition guidelines for the final.

**Q6.** During each step, in what order are the following processed: determining whether movement is possible, acquiring “udon”, refueling, applying movement, and fuel consumption?
**A6.** First, during the answer submission process, we determine the number of action steps and assess whether movement is possible. If the action plan does not match the specified number of steps (e.g., exceeding or falling short of the limit) or includes impossible movements (such as moving into a pond or running out of fuel), the entire answer will be rejected. In particular, if you lack the necessary number of steps for movement toward the end of a day, please specify "Wait" at the end to ensure the total matches the required number of steps.
Since the feasibility of movement is determined at the time of response acceptance, the "Reflection Phase" is performed first, assuming that all movements are possible during the processing of each step. In the Reflection Phase, the process is performed in the following order: (1) fuel consumption; (2) Movement update; (3) (If current location is a spot) “Udon” acquisition; (4) (if the refueling car and patrol car are in the same cell) Fuel replenishment; (5) Traffic volume update. 
Next, the "Action Phase" is processed. In the Action Phase, the system schedules the reflection of the next action (move or wait).
On the 0th step of each day, only the Action Phase is performed, and on the final step of each day, only the Reflection Phase is performed. On all other days, processing proceeds in the order of Reflection Phase, then Action Phase.
The processing flow is summarized below.

> * **Phase Types**
>   1. Reflection Phase (No processing at Step 0)
>      1.1. Fuel Consumption
>      1.2. Reflection of Movement
>      1.3. Acquiring “Udon”
>      1.4. Fuel Refueling
>      1.5. Update Traffic
>   2. Action Phase (No processing in the final step)
>      2.1. Action Acceptance
>         + Accept Movement
>         (Reserve movement; fuel is consumed during the Reflection Phase)
> * **Phase Flow**
>   Step 0: Action Phase
>   Step 1: Reflection Phase, then Action Phase
>   ...
>   Step N-1: Resolution Phase, then Action Phase
>   Step N (Final): Resolution Phase

***Please also refer to Appendix A in the end of this document, which include an explanation using a map.***

**Q7.** If a patrol car is at a spot at the start of the day, can it obtain “udon” at that spot?
**A7.** Since there is no Reflection Phase during Step 0 of each day, “udon” is not acquired during Step 0; however, if the patrol car remains at the spot starting from Step 1, it will acquire “udon”.

**Q8.** Regarding the "Wait" action, if a player waits in a “udon” shop cell, can they obtain “udon”?
**A8.** Since a single patrol car can only acquire up to one bowl of “udon” from the same spot on the same day, you cannot acquire it if you have already done so. However, you can acquire it if you wait after the day has changed (see **A7**).

**Q9.** Is the number of steps per day provided before the match begins?
**A9.** It is provided in the “daySteps” key of the map configuration format before the match begins. Additionally, the number of steps per day will be announced as match information during the participants’ briefing on the day of the contest.

**Q10.** How are movement steps counted?
**A10.** Movement steps are counted by consuming the number of steps required for the cell from which you move.

**Q11.** Does “Fuel consumption” in Table 1 refer to fuel consumption of cars?
**A11.** Yes, it refers to the fuel consumption required to move from each terrain type in a cell.

**Q12.** Is the daily traffic volume for each road cell provided?
**A12.** The match information format provided at the start of each day includes only road conditions; traffic volume for each road cell is not provided.

**Q13.** Regarding the determination of road congestion, what are the specific numerical values for the congestion threshold and traffic jam threshold? 
**A13.** The congestion threshold ranges from 1 to 5, and the traffic jam threshold ranges from 2 to 10.

**Q14.** What is the maximum amount of fuel a patrol car can carry?
**A14.** The maximum amount of fuel a patrol car can carry is 1 to 3 times the number of steps on Day 1.

**Q15.** What is the range of the number of spots?
**A15.** The minimum number of spots is equal to the number of agents, and the maximum number is equal to the maximum width or height of the map.

**Q16.** Regarding the "wait" action, which states "wait at the current cell for the specified number of steps," is fuel consumed while waiting?
**A16.** No fuel is consumed while waiting.

**Q17.** Regarding the "wait" action, if an agent waits at a cell containing a “udon” shop, can they obtain “udon”?
**A17.** Since each agent can obtain only one serving of “udon” from the same spot per day, you can obtain it if you are waiting after the day has changed.

**Q18.** Number of spots per cell: Can multiple (two or more) spots be placed within a single cell?
**A18.** Only one spot is placed within a single cell; multiple spots are never placed in the same cell.

**Q19.** How is fuel carried over? Is the previous day’s fuel managed and carried over by the player, or is it provided in the problem format?
**A19.** The remaining fuel at the end of the previous day is provided in the “fuel” key of each day’s match information format. 

**Q20.** What is the range of steps for each day?
**A20.** The minimum number of steps per day is the map’s width (W) plus height (H), that is, W+H, and the maximum number of steps is four times of W+H.

**Q21.** If a response to an inquiry is provided after the deadline, will further questions regarding that response be answered even after the deadline?
**A21.** As a general rule, we will not answer questions submitted after the deadline. As in previous years, we will set up another question submission period after the final round guidelines are released, so please submit your questions at that time. However, if an urgent matter arises, we will promptly provide information on the official website or other means such as MS Teams.

---

## Appendix A: Supplementary Material for Q6

**Q6.** During each step, in what order are the following processed: determining whether movement is possible, acquiring “udon”, refueling, applying movement, and fuel consumption?
**A6.** First, during the answer submission process, we determine the number of action steps and assess whether movement is possible. If the action plan does not match the specified number of steps (e.g., exceeding or falling short of the limit) or includes impossible movements (such as moving into a pond or running out of fuel), the entire answer will be rejected. In particular, if you lack the necessary number of steps for movement toward the end of a day, please specify "Wait" at the end to ensure the total matches the required number of steps.
Since the feasibility of movement is determined at the time of response acceptance, the "Reflection Phase" is performed first, assuming that all movements are possible during the processing of each step. In the Reflection Phase, the process is performed in the following order: (1) fuel consumption; (2) Movement update; (3) (If current location is a spot) “Udon” acquisition; (4) (if the refueling car and patrol car are in the same cell) Fuel replenishment; (5) Traffic volume update. 
Next, the "Action Phase" is processed. In the Action Phase, the system schedules the reflection of the next action (move or wait).
On the 0th step of each day, only the Action Phase is performed, and on the final step of each day, only the Reflection Phase is performed. On all other days, processing proceeds in the order of Reflection Phase, then Action Phase.
The processing flow is summarized below.

> * **Phase Types**
>   3. Reflection Phase (No processing at Step 0)
>      3.1. Fuel Consumption
>      3.2. Reflection of Movement
>      3.3. Acquiring “Udon”
>      3.4. Fuel Refueling
>      3.5. Update Traffic
>   4. Action Phase (No processing in the final step)
>      4.1. Action Acceptance
>         + Accept Movement
>         (Reserve movement; fuel is consumed during the Reflection Phase)
> * **Phase Flow**
>   Step 0: Action Phase
>   Step 1: Reflection Phase, then Action Phase
>   ...
>   Step N-1: Resolution Phase, then Action Phase
>   Step N (Final): Resolution Phase

*Figure 1: Processing flow.*

---

Here is an example using the following map.

*(Figure 2: Example of a map.)*

Assume the situation is as follows. Cells 0 and 1 are roads, and cells 2 and 3 are flat terrain (number of steps: 2, fuel: 1). Road conditions are smooth in cell 0 (number of steps: 1, fuel: 2) and congested in cell 1 (number of steps: 2, fuel: 2). Cell 2 contains a spot with a stock of 4. Four patrol cars (A, B, C, D) are stationed in cell 0, one patrol car (E) and one refueling car (A) are stationed in cell 2, and two patrol cars (F and D) are stationed in cell 3. The maximum fuel capacity of each patrol car is 3.

Suppose that the number of steps per day is 6 (0 to 6 steps) and the action plans for each agent are given in Table 1.

**Table 1: Example of the action plans for each agent.**

| Agent | Initial Cell | Action Plan | Reference |
| :--- | :--- | :--- | :--- |
| Patrol Car A | 0 | 2, 2, 2, -1 | RT, RT, RT, WT(1) |
| Patrol Car B | 0 | -1, 2, 2, 2 | WT(1), RT, RT, RT |
| Patrol Car C | 0 | -2, 2, 2, -1 | WT(2), RT, RT, WT(1) |
| Patrol Car D | 0 | -3, 2, -2 | WT(3), RT, WT(2) |
| Patrol Car E | 2 | 5, 5, -2 | LF, LF, WT(2) |
| Patrol Car F | 3 | 5, 5, -2 | LF, LF, WT(2) |
| Patrol Car G | 3 | -2, 5, 5 | WT(2), LF, LF |
| Refueling Car A | 2 | 5, 5, -2 | LF, LF, WT(2) |

In this table, note that, in the action plan, “2” indicates a move to the right, “5” indicates a move to the left, and “-3” to “-1” indicate waiting (the number of waiting steps is the absolute value). Additionally, “WT(2)” shown in the reference indicates waiting for 2 steps. 
Then, the changes in each phase are as shown in Fig. 3. In this figure, “T2” indicates moving right in Step 2, and “WT2” indicates waiting until Step 2. 

**Notes Regarding Each Agent’s Action Plan in this Example**
* **Patrol Car A:** Although it has reached the cell 3, there is still one day’s step remaining, so an error will occur unless “wait” is specified at the end.
* **Patrol Car B:** The day ends at the step where it reaches the cell 3, so it completes as is.
* **Patrol Car C:** Has not reached the cell 3, but since there are not enough steps remaining to reach it, specifying a move will result in an error. Additionally, since there are steps remaining after moving to the cell 2 (similar to Patrol Car A), an error will occur unless "wait" is specified at the end. Has reached the spot, but since the stock is depleted, “udon” cannot be obtained.
* **Patrol Car D:** Since the fuel capacity is below the amount required for movement, specifying a move afterward will result in an error. Use the remaining steps to specify "wait."
* **Patrol Car E:** Since it starts from the spot, it acquires “udon” on Step 1. (It cannot acquire “udon” if it moves to another cell on Step 1, but since the spot is flat terrain, it will not move on Step 1.) It moves simultaneously with the supply car, so its fuel load is always at maximum capacity.
* **Patrol Car F:** Since it always moves one cell behind the supply car, it does not receive a fuel supply. By the time it reaches the cell 1, its fuel capacity has fallen below the amount required for movement, so specifying a move after that will result in an error.
* **Patrol Car G:** It moves to the spot, but since it moves at the same time as Patrol Car B, and the stock is only one at this point, only Patrol Car B acquires it based on ID order, while Patrol Car G does not.

```text
Step Initial 0 1 2 3 4 5 6 Final
AC Reflected AC Reflected AC Reflect AC Reflect AC Reflect AC Reflect AC Reflect AC
Consump tion Moveme nt Replenis hment Gain Transpor tation Consum ption Transpor tation Supply Acquisiti on Transpo rtation Consum ption Movem ent Supply Acquire d Transpo rtation Consum ption Movem ent Supply Acquire d Transpor tation Consum ption Movem ent Supply Acquired Transpor tation Consum ption Movem ent Supply Acquired Transpor tation

Patrol Car A Coordinates 0 0 0 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3
Fuel 3 3 1 1 1 1 1 1 1 1 3 3 3 3 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
Udon 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
Booking Behavior > RT1 RT1 RT1 > > > RT3 RT3 RT3 RT3 RT3 RT3 RT3 RT3 RT3 > > > RT5 RT5 RT5 RT5 RT5 RT5 RT5 RT5 RT5 > > > WT6 WT6 WT6 > > > > >

Patrol Car B Coordinates 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3
Fuel 3 3 3 3 3 3 3 3 1 1 3 3 3 3 3 3 3 3 3 3 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
Udon 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
Booking Behavior > WT1 WT1 WT1 > > > RT2 RT2 RT2 > > > RT4 RT4 RT4 RT4 RT4 RT4 RT4 RT4 RT4 > > > RT6 RT6 RT6 RT6 RT6 RT6 RT6 RT6 RT6 > > > > >

Patrol Car C Coordinates 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2
Fuel 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 3 3 3 3 3 3 3 3 3 3 1 1 1 1 1 1 1 1 1 1 1 1 1
Udon 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Booking Behavior > WT2 WT2 WT2 WT2 WT2 WT2 WT2 WT2 WT2 > > > RT3 RT3 RT3 > > > RT5 RT5 RT5 RT5 RT5 RT5 RT5 RT5 RT5 > > > WT6 WT6 WT6 > > > > >

Patrol Car D Coordinates 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
Fuel 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
Udon 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Booking Behavior > WT3 WT3 WT3 WT3 WT3 WT3 WT3 WT3 WT3 WT3 WT3 WT3 WT3 WT3 WT3 WT3 > > > RT4 RT4 RT4 > > > WT6 WT6 WT6 WT6 WT6 WT6 WT6 WT6 WT6 > > > > >

Patrol Car E Coordinates 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Fuel 3 3 3 3 3 3 3 3 2 2 3 3 3 3 3 3 3 3 3 3 2 2 3 3 3 3 3 3 3 3 3 3 2 2 3 3 3 3 3
Udon 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
Booking Behavior > LF2 LF2 LF2 LF2 LF2 LF2 LF2 LF2 LF2 > > > LF4 LF4 LF4 LF4 LF4 LF4 LF4 LF4 LF4 > > > WT6 WT6 WT6 WT6 WT6 WT6 WT6 WT6 WT6 > > > > >

Patrol Car F Coordinates 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
Fuel 3 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
Udon 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
Booking Behavior > LF2 LF2 LF2 LF2 LF2 LF2 LF2 LF2 LF2 > > > LF4 LF4 LF4 LF4 LF4 LF4 LF4 LF4 LF4 LF4 > > > WT6 WT6 WT6 WT6 WT6 WT6 WT6 WT6 WT6 > > > > >

Patrol Car G Coordinates 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1
Fuel 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1
Udon 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Booking Behavior > WT2 WT2 WT2 WT2 WT2 WT2 WT2 WT2 WT2 > > > LF4 LF4 LF4 LF4 LF4 LF4 LF4 LF4 LF4 > > > LF6 LF6 LF6 LF6 LF6 LF6 LF6 LF6 LF6 LF6 > > > > >

Supply Vehicle A Coordinates 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Booking Behavior > LF2 LF2 LF2 LF2 LF2 LF2 LF2 LF2 LF2 > > > LF4 LF4 LF4 LF4 LF4 LF4 LF4 LF4 LF4 LF4 > > > WT6 WT6 WT6 WT6 WT6 WT6 WT6 WT6 WT6 WT6 > > > > >

Number of nights 0 0 0 0 0 0 0 3 3 3 3 3 3 5 5 5 5 5 5 6 6 6 6 6 6 8 8 8 8 8 8 10 10 10 10 10 10 12 12 12
```
*Figure 3: Changes in each phase.*
