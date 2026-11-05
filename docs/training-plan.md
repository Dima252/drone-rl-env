# Training plan and open decisions

Status on 2026-11-05. The decisions below are in the code, and the user left training
to the assistant's judgment. **The best lab model is `runs/scout/lab-best`: 99% holds and
0.8% crashes on 800 flights. The realistic stage has not started**, because the user
chose to review the lab result first.

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

| Scenario | Crashes at most | Completes a hold |
| --- | --- | --- |
| Nominal, randomized | 2% | At least 80% |
| Worst case | 10% | Not gated |
| Unseen conditions | 15% | Not gated |

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
- The network ignores GPS, but the node's flight limits and the `LOITER` fallback still
  need ArduPilot's position estimate. A vehicle with no GPS at all needs different
  `fallback_mode` and `required_mode` values.
- All camera, marker and airframe values are placeholders until calibrated. See "Before
  hardware" in the README.
- The marker is static. Nothing in this project chases or hits a moving object, and
  coming too close ends policy control.
