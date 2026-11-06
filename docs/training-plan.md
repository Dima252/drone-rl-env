# Training plan and open decisions

Status on 2026-11-06. The decisions below are in the code, and the user left training
to the assistant's judgment. **The best lab model is `runs/scout/baro-lab`, with the
barometer: 100% holds and no crashes on 800 flights. The best realistic model is
`runs/scout/slow-s1`, which passes nominal and randomized but not the whole bar**: on 800
randomized flights it holds on 85.5% and crashes on 2.0%, exactly at the limit. It is the
only one of four seeds to pass those two. Worst case (about 41% crashes) and unseen (about
66%) still fail, mostly from altitude. Climb rate roughly halves those two, and
`runs/scout/climbslow-s3` matches `slow-s1` on randomized, but it has not been re-scored on
fresh flights and the node cannot fly it yet. No model meets the whole bar. See "Slower
approach: results" and "Climb rate with the slower approach" below.

## Lab results so far

| Run | Learning rate | Steps | Holds | Crashes | Notes |
| --- | --- | --- | --- | --- | --- |
| `lab-check` | 3e-4 | 2M | 48% | 2% | Final model. The rate fell to zero quickly |
| `lab` | 3e-4 | 20M | 12% | 7% | Best model. Failed: see below |
| `lab-lr1e4` | 1e-4 | 20M | 74% | 0% | Best model. Peaked near 3M, then circled |
| `lab-lr5e5` | 5e-5 | 20M | 57% | 0.5% | Best model. Final model fell to 16% |
| `lab-2m-s0` | 1e-4 | 20M | 74% | 0.5% | Best model. 2 m marker, `PROXIMITY_SPEED_MPS` 0.4 |
| `lab-2m-s1` | 1e-4 | 20M | 0% | 0% | Best model. Same, and circled at about 5 m/s |
| `lab-fix-s0` | 1e-4 | 20M | 72% | 0% | Best model. 2 m marker, progress fix |
| `lab-fix-s1` | 1e-4 | 20M | 55% | 1% | Best model. Same, with another seed |
| **`lab-fix-s0` at 2.5M** | 1e-4 | 2.5M | **90%** | **0%** | Checkpoint. Passes. 90% and 0.5% on a second set of flights |
| `lab-fix-s0` at 1.0M | 1e-4 | 1M | 92% | 1% | Checkpoint. Failed a second set of flights at 86% |

The first four used a 1 m marker. The first six had the progress bug below. Holds are
from 200 flights, except `lab-check` with 100.

- **`lab` failed because the rate stayed high for too long.** The policy change per
  update reached a KL over 0.1, half of all updates were clipped, the action std
  collapsed from 0.37 to 0.04, and the commands became jerky. With the small batch of
  the time, 1e-4 was better. The larger batch below made 3e-4 safe.
- **The later runs failed by circling the marker, because the reward paid for it.**
  Failed flights drifted sideways in one direction. `step()` moved the hold point to
  the vehicle's new position, then the reward measured progress from the old position
  to that moved point. So a slide along the ring was paid as progress: a 0.1 m slide
  earned 0.099. `tests/test_marker_env.py` now covers this, and `prev_waypoint` fixes it.
- **`PROXIMITY_SPEED_MPS` 0.4 was a wrong fix, and is reverted to 1.0.** It assumed the
  reward was only too weak against drifting. It took away reward for hovering while
  the exploit stayed, and `lab-2m-s1` then circled at 5 m/s.
- **`best_model` was misjudged by 20 flights.** One scored 90% in training and 74% on
  200 flights. It now uses 64.

## Sweeps: what made training work

Each run is scored on 200 flights. Seeds are in the run names.

| Change | Steps | Final model | Result |
|---|---|---|---|
| Batch 2048, 5 epochs, 3e-4, lambda 0.95 | 6M | 25% and 18% holds | No better alone |
| Same, lambda 0.98 | 6M | 92% and 98% holds | The breakthrough |
| Same, lambda 0.98 | 12M | 97% and 20% holds | One seed collapsed |
| Same, lambda 0.99 | 12M | 94 to 100% holds | Six of six held up, in three variants |
| Same, lambda 0.995 | 12M | 95% and 98% holds | More crashes, 5.0% on one seed |
| Lambda 0.98, batch 256, 1e-4 | 12M | 47% and 65% holds | Declined |
| Crash penalty 300, lambda 0.99 | 12M | 96% and 98% holds | No help, 2.5 to 4.5% crashes |
| Lambda 0.99, no entropy bonus | 24M | 96% and 32% holds | One of two collapsed |
| Same, reward with no flat region | 24M | 96, 96, 48, 52% holds | Two of four declined |
| Same, entropy bonus 0.001 | 24M | 98% and 30% holds | One of two collapsed |
| **Same, entropy bonus 0.003** | 24M | **96% and 98% holds** | **Two of two held, 30 holds a minute** |

Why these work:

- **Lambda.** At 0.95 an action gets credit for about 0.35 s. Stopping sideways drift pays
  off 1 to 2 s later, so the policy could not learn it.
- **Entropy bonus.** Lambda alone delays the decline. It starts as steady sideways drift,
  usually long before scores fall: in one run it grew from 0.07 to 0.86 m/s, while
  healthy runs stay near 0.1. Another run was healthy at 18M and collapsed by 24M. The
  policy ends up circling on the ring at about 1.7 m/s, too fast to count a hold, and can
  circle both ways, so it has lost its correction rather than gained a bias.
  - The likely cause is falling action noise: with little noise the vehicle is rarely
    knocked sideways, and the signal to correct it fades. At 10 m a 0.2 m/s drift is only
    about 0.02 rad/s of yaw rate. At 0.003 the std stays above 0.22.
  - This rests on two runs and a likely cause, not proof. Four of the eight 24M runs
    without enough bonus declined, against none of two with it.
- **A collapsed policy does not recover.** Continuing one for 3M steps failed under the
  current reward and two others, so a reward with no flat region is not the fix.

`best_model` is still picked by reward. At these settings it kept a 98 to 100% checkpoint
from three of the four collapsed 24M runs, and an 89% one from the other.

## Open problem: crashes at the start of a flight

Nearly every failure is now a crash in the first few seconds, from an awkward start:

- **Far and low**, 15 to 22 m out and 3 to 5 m up: the marker sits high in the frame, and
  a nose-down dash pushes it out of the top. Pitching forward about 0.26 rad is enough.
- **Close and high**: the marker sits low in the frame, and the vehicle drops too fast.

Crash rates over 800 flights differ by seed: 0.8 to 3.75%. The two entropy models crash
on 3.75% and 2.25%, so the stable settings do not yet meet the 2% bar on their own, and
`lab-best` came from a run without the bonus. The start positions are unchanged on
purpose: narrowing them would hide this.

## Realistic stage: stalled at about half difficulty

Evaluator scores, 100 flights per scenario. The bar for randomized flights is 80% holds
and 2% crashes.

| Run | Starts from | Gate, holds a flight | Level reached | Nominal holds | Randomized holds | Randomized crashes |
|---|---|---|---|---|---|---|
| `realistic` | `lab-best` | 1 | 0.65 | 48% | 6% | 42% |
| `realistic-gate` | `lab-best` | 5 | 0.30 | 33% | 6% | 37% |
| **`realistic-gate-ent`** | `r5-ent3-s3` | 5 | 0.55 | **97%** | 9% | 53% |
| `realistic-stage2` | `realistic-gate-ent` | 5 | 0.50 | 95% | 2% | 57% |

Worst case crashed on 99 to 100% of flights in all four, and unseen on 76 to 98%.

- **Start from a model with action noise left.** `lab-best` has a std of 0.13, which fell
  to 0.05 and lost the lab task, in spite of the entropy bonus. `r5-ent3-s3` has 0.22 and
  kept it. A better lab score did not mean a better start.
- **The gate matters.** At 1 hold a flight, difficulty rose to where the policy completes
  too few holds to learn from. It is now 5. This is uncommitted.
- **More training did not help.** `realistic-stage2` ran 48M steps with a fresh learning
  rate and stalled lower. It is a little better up to 0.4 (98% holds, 2% crashes against
  94% and 6%), and worse above it (62% against 83% at 0.6).

Flights with a hold, 48 flights per case, for all conditions together:

| Difficulty | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|
| `lab-best` | 98% | 71% | 19% | 4% | 0% |
| `realistic-gate-ent` | 96% | 94% | 83% | 40% | 6% |

The combination is the problem. With one group of conditions on alone at full strength,
`realistic-gate-ent` holds on 88 to 98% of flights. Removing one group from the full mix
gives, on 64 flights each:

| Removed | Nothing | Wind | Sensing | Vehicle | Detector | Timing |
|---|---|---|---|---|---|---|
| Holds | 9% | 9% | 16% | 30% | 31% | 41% |
| Crashes | 48% | 52% | 45% | 23% | 38% | 23% |
| Altitude crashes | 31% | 30% | 31% | 8% | 31% | 17% |

No single group is the cause, and none alone gets near the bar. Altitude is the largest
crash cause. These crashes come early, about 3 s in, and lean towards heavier vehicles
and longer action delays, on only 20 flights: 1.67 kg against 1.46 kg. Removing vehicle
variation cuts them most. A likely reason is that the policy cannot tell how a given
vehicle responds, and has no direct measure of height. That is untested.

Options, for the user to decide:

1. Add a barometer: height and climb rate as inputs. Flight controllers have one, it
   targets the largest crash cause, but it changes the camera and IMU only design. The
   user chose this one, and the results are in the next section.
2. Measure the real vehicle and narrow the ranges, which are placeholders so far.
3. Give the policy memory, to work out the vehicle's response in flight. This adds the
   most complexity.

## Realistic stage with the barometer

The policy now reads height and climb rate, 53 observations in all. In the lab, on 400
flights each, `baro-lab-s1` scored 100% holds and no crashes, with 0 crashes on a second
400, and `baro-lab-s0` 99% and 0.8%. `baro-lab` is a copy of the s1 model. Its action noise
stayed at about 0.23.

Two runs started from it, for 48M steps each, with the 60 to 100 ms window and a gate of 5
holds a flight. Evaluator scores, 100 flights per scenario, final models:

| Run | Level reached | Nominal holds | Randomized holds | Randomized crashes | Worst case crashes | Unseen crashes |
|---|---|---|---|---|---|---|
| `realistic-gate-ent`, for comparison | 0.55 | 97% | 9% | 53% | See above | See above |
| `real60-s0` | 0.65 | 97% | 43% | 16% | 100% | 85% |
| **`real60-s1`** | 0.75 | 85% | 57% | 8% | 100% | 72% |
| Bar | 1.0 | | 80% | 2% | 10% | 15% |

The saved best models are no better: 47% and 16% for s0, 50% and 9% for s1. The barometer
and the narrower latency window arrived together, so these runs cannot separate them.

Flights with a hold, 48 flights per case, for all conditions together:

| Difficulty | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|
| `realistic-gate-ent` | 96% | 94% | 83% | 40% | 6% |
| `real60-s0` | 98% | 94% | 85% | 65% | 40% |
| `real60-s1` | 85% | 94% | 94% | 77% | 56% |

Removing one group from the full mix, 64 flights each, s0 then s1:

| Removed | Nothing | Wind | Sensing | Vehicle | Detector | Timing |
|---|---|---|---|---|---|---|
| Holds | 36%, 58% | 48%, 70% | 39%, 59% | 55%, 72% | 64%, 84% | 56%, 72% |
| Crashes | 19%, 5% | 12%, 3% | 19%, 6% | 3%, 0% | 14%, 3% | 9%, 2% |

- **Crashes.** Removing vehicle variation cuts them most in both runs. Crashes still come
  early: the median is 4 to 7 s in, and about 2.5 s for altitude crashes. At full difficulty,
  altitude crashes fell from 31% of flights to 12% with s0 and 2% with s1. Altitude is still
  s0's largest cause, at 9 of 100 randomized flights, while s1 had 3 of 100.
- **Holds.** Removing detector errors adds the most. No group alone reaches the 2% and 80%
  bars together, though s1 without detector errors reaches 84% holds with 3% crashes.
- **Most flights with no hold end without a crash, and they fail in two ways.** At
  difficulty 0, s0's 5 misses in 64 flights stop still about 1.2 m from the hold range, just
  outside the 1.0 m radius. s1's 9 misses circle at about 1.2 m/s, like the collapse seen in
  the lab. At full difficulty both drift at 0.5 to 0.7 m/s, 1 to 2 m off the range.
- **Action noise fell again.** It went from 0.23 to 0.08 in s0 and 0.06 in s1, under the
  entropy bonus of 0.003 that held it at 0.22 in the lab. Difficulty stopped rising at about
  15M steps, so the last 33M steps did not reach any harder conditions. Low noise and the
  stall appear together, which does not show that one causes the other.

`--seed` was ignored with `--start-from` until this stage: loading reseeded from the saved
run, so two "seeds" were the same run. It is fixed, and `real60-s0` and `real60-s1` are
independent. They differ by 14 points on randomized holds and 12 on nominal, so a
difference that size between two single runs means little.

## Reference build

There is no hardware yet, so this is a software-only simulator for now. The user asked for
ranges from a published build in place of measurements. The reference is a
[Holybro X500 V2](https://docs.holybro.com/drone-development-kit/px4-development-kit-x500v2):
2216 KV920 motors, 1045 propellers and a 4S 5000 mAh battery.

| Range | Was | Now | Basis |
|---|---|---|---|
| `mass_kg` | 1.2 to 1.8 | 1.35 to 1.65 | About 1.5 kg all-up, ±10% |
| `inertia_scale` | 0.8 to 1.25 | 0.9 to 1.1 | One frame, loaded a little differently |
| `motor_lag_scale` | 0.7 to 2.0 | 0.7 to 1.7 | 21 to 51 ms, from the sources below |
| `hover_thrust` | 0.3 to 0.5 | 0.35 to 0.5 | 1.5 kg on 3.4 to 4.2 kg of maximum thrust |
| `battery_sag` | 0 to 0.08 | 0 to 0.02 | 18 minutes of hover is about 0.6% per 20 s flight, with 3x margin |

How firm these are:

- **Published:** the 610 g kit weight, about 1 kg of payload at 70% throttle, the 18 minute
  hover, and thrust at 30 to 40% throttle from a
  [similar motor and propeller](https://www.ligpower.com/product/air-gear-450II-combo-set.html).
  PX4's [simulated X500](https://github.com/PX4/PX4-gazebo-models/blob/main/models/x500/model.sdf)
  uses a 12.5 ms motor time constant going up and 25 ms going down, and a
  [test of a 2216/880 motor with a 1045 propeller](https://ar5iv.labs.arxiv.org/html/2312.09981)
  says "many 10's of milliseconds", without a time constant for that system.
- **Estimated:** the all-up weight, from the parts plus a companion computer and camera, and
  the maximum thrust, from the payload figure and PX4's model. No source gives either
  directly. The sag figure assumes thrust falls about 30% over a full battery.
- **Unchanged:** wind, which the user set, and detector errors. Published detectors report
  recall of 0.96 to 0.99, which fits the 5% dropout, but no source gave box jitter in pixels.
  Drag, attitude gain, pushes and motor faults are also unchanged, for lack of a source.
- **Known difference:** the simulated vehicle's inertia is 0.03, 0.03 and 0.05 kg m². PX4's
  model has 0.022, 0.022 and 0.04 at 2 kg. The rate loop scales with inertia, and changing
  it would mean new lab training, so it stays.

The bars did not change. `worst case` follows the training ranges, so it became easier with
them. `unseen` has fixed values and did not, so it now sits further outside training.

Flying the `real60` models on the new ranges, with no new training:

| Model | Randomized holds | Randomized crashes | Worst case crashes | Unseen crashes |
|---|---|---|---|---|
| `real60-s0` | 51% | 12% | 80% | 78% |
| `real60-s1` | 63% | 6% | 55% | 71% |

`--start-from` used to keep the saved model's entropy bonus, so changing the constant did
nothing there. `--ent-coef` now sets it for every run.

## Entropy and climb rate on the reference build

All runs took 24M steps from `baro-lab`, except climb rate, which needed its own lab model.
`x500-e03` used an entropy bonus of 0.003 and `x500-e10` 0.01. `climb-real` used 0.01, and
replaced the thrust output with a climb rate that the simulated autopilot holds.

Final models, 400 flights per scenario, except `x500-e03` at 100. Output change is the mean
largest step-to-step change on nominal flights, where the full range is 2.

| Run | Level | Nominal holds | Randomized holds | Randomized crashes | Worst case crashes | Unseen crashes | Output change |
|---|---|---|---|---|---|---|---|
| Bar | 1.0 | | 80% | 2% | 10% | 15% | None set |
| `x500-e03-s0` | 0.80 | 100% | 65% | 9% | 78% | 78% | 0.08 |
| `x500-e03-s1` | 0.75 | 100% | 63% | 14% | 74% | 85% | 0.09 |
| `x500-e10-s0` | 0.90 | 99% | 86% | 5.8% | 69% | 84% | 1.20 |
| `x500-e10-s1` | 0.85 | 98% | 82% | 8.0% | 55% | 70% | 1.31 |
| `climb-real-s0` | 0.80 | 96% | 71% | 14.2% | 47% | 80% | 1.32 |
| `climb-real-s1` | 0.85 | 92% | 72% | 11.8% | 68% | 80% | 1.72 |

No model meets the bar. `x500-e10-s0` scored 92% and 2% on the first 100 flights, which was
luck: it had been picked as the best of four on those flights.

- **Entropy 0.01 clears the holds bar but not the crash bar.** Both seeds beat both 0.003
  runs on randomized holds.
- **It also makes the outputs chatter.** Action noise rose from 0.23 to about 0.6, where 0.003
  let it fall to 0.11. On nominal flights the commands reverse on about 95% of steps, and
  `x500-e10-s1` holds roll at its limit 54% of the time. Motor lag and the autopilot filter
  this out, so body rates stay low, at 0.16 to 0.20 rad/s. On a real vehicle it would mean
  heat and vibration, which the simulator barely models, so these scores may not transfer.
  The evaluator has no limit on it yet, and it should.
- **Climb rate did not win, but the test was not fair.** Its lab models scored 92% and 70%
  holds, against 100% for attitude, and it inherited the chatter. Worst case altitude crashes
  fell from 154 to 215 of 400 flights to 44 to 79. Losing the marker is now the main cause
  there, at 135 to 181.

Climb mode is simulator only, selected by `DRONE_CONTROL_MODE=climb`. The node and exporter
accept attitude models alone.

A middle bonus of 0.005 did not keep the gain, on 400 flights per scenario:

| Run | Level | Action noise | Randomized holds | Randomized crashes | Worst case crashes | Unseen crashes | Output change |
|---|---|---|---|---|---|---|---|
| `x500-e05-s0` | 0.75 | 0.18 | 60% | 12.8% | 86% | 86% | 0.92 |
| `x500-e05-s1` | 0.80 | 0.18 | 70% | 10.8% | 58% | 74% | 0.18 |

That is about where 0.003 landed, and s0 chatters anyway, so entropy alone does not separate
the gain from the chatter. The smoothness penalty is small: at 0.05 it costs chatter about 20
points of a 450 point flight.

`x500-smooth` started from `x500-e10-s0` for 16M steps, with the penalty at 0.5 and a bonus
of 0.003. On 400 flights per scenario:

| Run | Level | Action noise | Randomized holds | Randomized crashes | Worst case crashes | Unseen crashes | Output change |
|---|---|---|---|---|---|---|---|
| `x500-smooth-s0` | 0.80 | 0.29 | 66% | 16.5% | 66% | 80% | 0.21 |
| `x500-smooth-s1` | 0.85 | 0.30 | 78% | 7.5% | 86% | 83% | 0.30 |

- **The penalty works on chatter in clean conditions, much less under randomization.**
  Nominal output change fell by about 4 to 6 times, and the share of time at a limit fell
  from up to 54% to 5% or less. Direction reversals stay near 90%, but they are now small,
  and `baro-lab` shows the same at 0.04. On randomized flights the change only fell from
  1.27 to 0.81 to 0.86, which is where the low-entropy models sit too, and yaw still spends
  38 to 51% of the time at its limit.
- **It costs some of the skill, and how much depends on the seed.** s1 is close to
  `x500-e10-s1`, while s0 lost 20 points of holds and tripled its crashes. Both started from
  the same model, so this is seed spread, and a single run proves little.
- **Worst case did not improve.** It rose to 86% for s1.
- **`x500-smooth-s1` is the best model that does not chatter**, and no model meets the bar.
  Losing the marker is its main crash cause on randomized flights, 25 of 30 crashes.

## Why the marker is lost, and whether worst case can be flown

`x500-smooth-s1` on the 400 randomized evaluation flights:

- **All 25 marker-lost crashes come in the first 7 seconds, before any hold.** The vehicle is
  usually 3 to 8 m from the marker, often well inside its hold range, and moving at 2 to 5 m/s.
- **The marker leaves by the bottom of the frame in 14 of 25**, by a side in 7, and in 4 it
  stays in frame while detections stop. Coming in fast, then braking nose-up, drops it out
  of a fixed camera. Backing away needs the same nose-up attitude.
- **Wind, noise, delay and faults look about the same in lost and safe flights**, so this is
  how the policy flies rather than one bad condition.

A hold pays 50 points and can repeat, so time is worth roughly 0.37 points a step, and nothing
priced speed. A plain controller given the true state shows what speed costs. It flies the
same physics, delays and action filter, and is ended by the same marker rules. On 400 flights:

| Speed limit | Nominal crashes | Randomized holds | Randomized crashes | Worst case holds | Worst case crashes |
|---|---|---|---|---|---|
| 0.75 m/s | 2.5% | 94% | 4.0% | 86% | 7.2% |
| 1.0 m/s | 2.5% | 96% | 4.5% | 92% | 8.8% |
| 1.5 m/s | 3.2% | 92% | 8.0% | 88% | 14.0% |
| 2.0 m/s | 7.2% | 88% | 12.0% | 82% | 22.2% |

- **Worst case can be flown, so its bar stays.** At 1 m/s the controller is inside the 10%
  limit, where the policy crashes on 86%. It has perfect state and no camera noise, so this
  is an upper bound, but the gap is the policy's and not the physics'. Unseen at 1.5 m/s was
  12% crashes on 160 flights.
- **Worst case is a corner of combined conditions.** Pinning one group at its hard end costs
  `x500-smooth-s1` 4 to 14% crashes on 160 flights. Pinning all of them costs 86%. Training
  almost never meets that combination.
- **The controller's remaining crashes come early**, with a median about 1 second in, often
  from starts with the marker low in the frame. `x500-smooth` crashes on 1% of nominal
  flights, so these starts can be survived, and they stay as they are.
- **Detector errors cause about 30% of randomized chatter.** Removing them takes output
  change from 0.85 to 0.59.
- **Training never reached full difficulty, though evaluation flies at 1.0.** The best level
  was 0.90, and `x500-smooth-s1` sat at 0.80 for 13M steps under the gate of 5 holds.

Changes that follow, all in the reward, the gate and the evaluator:

- Ground speed over 1 m/s now costs 0.5 a step per m/s, just above the value of time.
- The smoothness weight is 0.5, as the new default.
- The curriculum gate is 4 holds a flight, because a slower approach leaves time for fewer.
- The evaluator limits output change to 0.5 on nominal and 1.0 on randomized. The smooth
  models pass; `x500-e10` at 1.20 and 1.27, and climb rate at 1.3 to 1.7 on nominal, fail.
  Randomized has the higher limit because detector noise raises it for every model.

## Slower approach: results

`slow` started from `baro-lab` for 24M steps, with the reward, gate and limits above and an
entropy bonus of 0.01. All four seeds reached level 1.0, the first runs to do so. Each cell
gives 400 evaluation flights, then, for s0 and s1, 400 fresh flights that played no part in
any choice.

| Run | Nominal holds | Nominal crashes | Randomized holds | Randomized crashes | Worst case crashes | Unseen crashes | Output change, nominal and randomized |
|---|---|---|---|---|---|---|---|
| `slow-s0` | 95%, 96% | 0%, 0.2% | 79%, 82% | 3.0%, 4.0% | 56%, 56% | 69%, 69% | 0.05, 0.69 |
| **`slow-s1`** | 100%, 98% | 0%, 0% | 88%, 83% | 2.0%, 2.0% | 41%, 42% | 65%, 66% | 0.24, 0.75 |
| `slow-s2` | 100% | 0% | 80% | 3.2% | 77% | 76% | 0.07, 0.66 |
| `slow-s3` | 86% | 5.2% | 75% | 7.5% | 56% | 76% | 0.05, 0.72 |

- **Crashes fell for three seeds of four, with no chatter limit broken.** `x500-smooth-s1` had
  7.5%, and `x500-e10-s0` 5.8% with chatter. Marker-lost crashes on those evaluation flights
  fell from 25 to 4 for s1, and to 9 or 10 for the others.
- **`slow-s1` passes nominal and randomized with no margin, and the method does not pass
  reliably.** 16 crashes in 800 flights is exactly 2%. The other three seeds miss, at 3 to
  7.5%, and s3 fails nominal too, with 21 altitude crashes. The mean over the seeds is 80.5%
  holds and 3.9% crashes on the evaluation flights.
- **Altitude is now the main failure in the two hardest scenarios.** It causes 116 to 125 of
  about 165 worst case crashes for s1. Hovering at the corner needs about +0.3 on the thrust
  output, where randomized flights need about 0. Over 160 flights the policy sinks at about
  0.7 m/s and only raises thrust in the final second. 13 of 51 of those crashes came before
  the motor fault. Climb rate moves that offset into the autopilot's loop.

## Climb rate with the slower approach

The earlier comparison was unfair because the climb lab models were weak. With the new
reward both lab seeds pass: `climbslow-lab-s1` scored 100% holds and 0.2% crashes, and
`climbslow-lab-s0` 99% and 1.2%, on 400 flights. `climbslow` started from `climbslow-lab-s1`
with the same settings as `slow`, on four seeds. All reached level 1.0. On 400 flights per
scenario:

| Run | Nominal holds | Nominal crashes | Randomized holds | Randomized crashes | Worst case crashes | Unseen crashes | Output change, nominal and randomized |
|---|---|---|---|---|---|---|---|
| `climbslow-s0` | 97% | 0% | 76% | 4.2% | 22% | 56% | 0.04, 0.65 |
| `climbslow-s1` | 96% | 0.2% | 76% | 3.8% | 13% | 44% | 0.04, 0.56 |
| `climbslow-s2` | 99% | 0% | 82% | 2.8% | 25% | 52% | 0.04, 0.62 |
| **`climbslow-s3`** | 93% | 0% | 81% | 2.0% | 29% | 53% | 0.03, 0.61 |
| Mean of the four | 96% | 0.1% | 79% | 3.2% | 22% | 51% | |
| Mean of the four `slow` seeds | 95% | 1.3% | 80.5% | 3.9% | 57% | 71% | |

- **Climb rate clearly helps the hard scenarios.** Mean worst case crashes fell from 57% to
  22%, and unseen from 71% to 51%, on every seed. `climbslow-s1` at 13% is close to the 10%
  bar, though unseen is still far from 15%.
- **It is level on randomized.** The means are within the spread between seeds. Holds are 79%
  against 80.5%, and crashes 3.2% against 3.9%. Chatter is a little lower.
- **`climbslow-s3` passes nominal and randomized on these flights, with no margin.** It is the
  best of four on one set of flights and has not been re-scored on fresh ones.
- **No seed passes everything.** The seed best on worst case is not the one best on
  randomized.
- **Altitude fell, and other causes now share the crashes.** In worst case, altitude is 3 to
  63 crashes per seed, against 116 to 227 for `slow`. Leaving the geofence rose to 7 to 24.
  Unseen altitude crashes are still 69 to 139.

Climb rate is still a simulator-only mode: `DRONE_CONTROL_MODE=climb` is a temporary switch,
and `ros2_inference_node.py` and `export_onnx.py` accept attitude models only. Adopting it
means making it the default and changing both, as set out above. Next, in order: re-score
`climbslow-s3` and `climbslow-s1` on fresh flights, adopt climb rate if that holds, then look
at the geofence crashes and the remaining unseen altitude crashes.

## Latency

The user set processing and transport latency to 60 to 100 ms, from a realistic budget of
60 to 95 ms. Training uses exactly that from camera to motors: `camera_delay_steps` 2 to 3
and `action_delay_steps` 1 to 2, with no link jitter. It was 40 to 160 ms, and up to 200 ms
with jitter. `unseen` keeps longer delays and jitter, so it stays outside training.

Not part of that window: frame age, which `camera_frame_hz` already adds at 15 to 30 frames
a second, and the IMU and height links, which stay at 0 to 2 steps.

## Decisions

Made by the user on 2026-11-05. The lab-first approach is the user's suggestion.

| Decision | Choice |
| --- | --- |
| Lab stage | True lab: nominal vehicle, still air, sensor noise off. Start positions stay random. Stop and report before the realistic stage |
| Lab pass bar | At least 90% of flights complete a hold, at most 2% crash, on the evaluator's `nominal` scenario |
| Realistic stage | All conditions rise together in small steps, as the curriculum does now, starting from the lab model |
| Marker | 2 m across, changed by the user from 1 m on 2026-11-05 |
| Network memory | 8 boxes over 0.7 s |
| Hold tolerance | 1.0 m for the first model |
| Maximum wind | 5 m/s steady, unchanged |
| Rain, fog, poor light | Detector errors stay as they are, and are set from real camera tests later |

A 2 m marker is about 67 pixels at 14 m, so a few pixels of error is about 0.5 m of
range. Policy control now ends closer than about 3.7 m, not 1.8 m, which is still
inside the closest 6 m hold.

## Proposed process

| Stage | Steps | Rough time | Question it answers |
| --- | --- | --- | --- |
| 1. Lab check | About 2M | 3 minutes | Does it start completing holds? If not, stop and fix that. |
| 2. Lab run | Up to about 20M | Up to 25 minutes | Does it reach the lab pass bar? Stop and report. |
| 3. Realistic run | About 20M | 25 minutes | Does it earn its way up to full difficulty? |
| 4. Judge | - | - | Does it pass the evaluator? If so, export it and fly SITL. |

The times come from about 14,000 steps per second, built from timing each part of a
run separately. No complete run has been timed yet, and the longer box history may
change them slightly. Report back after each stage. If a run fails, diagnose it and
agree the next step before training again. If the realistic run stalls, test the model
against one condition at a time, which needs no retraining.

Evaluator gates, from `agents/evaluate.py`:

| Scenario | Crashes at most | Completes a hold | Output change at most |
| --- | --- | --- | --- |
| Nominal | 2% | At least 80% | 0.5 |
| Randomized | 2% | At least 80% | 1.0 |
| Worst case | 10% | Not gated | Not gated |
| Unseen conditions | 15% | Not gated | Not gated |

Crashes include losing the marker and coming too close. The lab pass bar is stricter
than the nominal row above, and applies before the realistic stage.

## What is known

- Box size and position, with the IMU, recover range and height closely on a perfect
  detector: for example 17.0 m against a true 17.0 m.
- The marker must be round. A square's box changes width by up to 41% with the viewing
  bearing, and box size is the only cue to range.
- A hand-written controller on the same inputs completed a hold in 43% of nominal
  episodes, with a 0.63 m median error late in flight. **Under full randomization it
  completed none**, with a 2.57 m error. This is the main risk.
- The last GPS waypoint policy settled 1.1 m from its target in perfect conditions. The
  proximity reward is now stronger and tighter because of that, but this is untested.

## Faster training: what changed for learning

The simulation is bit-identical, and PPO's update and batch size are unchanged. These
defaults did change, and each can be set back:

| Setting | Was | Now | Effect on learning |
| --- | --- | --- | --- |
| `--n-envs` | 8 | 64 | More varied experience per update, with the same amount of it |
| `n_steps` | 2048 | 256 | Follows from `--n-envs`. Rollouts end sooner, but GAE only looks about 18 steps ahead |
| `--eval-freq` | 50,000 | 250,000 | `best_model` is chosen from a fifth as many evaluations |

## Not yet verified

- `tests/test_pipeline.py`, and so a real `model.learn` with the new worker pool and
  GPU update, including resuming a run. The parts are tested separately, but that file
  trains tiny throwaway models, so leave it out until training is agreed:
  `python -m pytest tests/test_env.py tests/test_marker_env.py tests/test_node_parity.py tests/test_fast_training.py`.
- The exporter ran on the camera-only task with an untrained policy, and matched SB3
  to 2e-8. It has not yet exported a trained one.
- The "your policy" SITL checks with a real policy. A fixed placeholder scored 46 of 47:
  it failed the approach check, as expected. It passed the hold check while doing
  nothing, so only the approach check tests skill.

## Assumptions to revisit

- The camera is tilted 20 degrees down (`camera_pitch_down_rad`), because a level camera
  keeps a ground marker at the bottom edge of the frame.
- A forward camera cannot see a marker beneath the vehicle, so the hold is at a
  stand-off, facing the marker.
- The network ignores GPS, but reads height and climb rate from ArduPilot's estimate. The
  simulation assumes that height is measured from the marker's level, and that it keeps
  publishing without GPS. **Neither is checked on a vehicle or in SITL yet.**
- The node's flight limits and the `LOITER` fallback still need ArduPilot's position
  estimate. A vehicle with no GPS at all needs different `fallback_mode` and
  `required_mode` values.
- All camera, marker and airframe values are placeholders until calibrated. See "Before
  hardware" in the README.
- The marker is static. Nothing in this project chases or hits a moving object, and
  coming too close ends policy control.
