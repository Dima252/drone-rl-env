# Training plan and open decisions

Status on 2026-11-05. The decisions below are in the code, and the user left training
to the assistant's judgment. **One lab model has passed, narrowly: `runs/scout/lab-pass`.
The realistic stage has not started**, because the user chose to review the lab result
first, and because of the open problem below.

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
  collapsed from 0.37 to 0.04, and the commands became jerky. The default is now 1e-4.
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

## Open problem: training gets worse after its peak

Both the pass and the fix are fragile:

- `lab-fix-s0` peaked between 1M and 3M steps, then fell from about 600 reward to about
  100, and sideways drift came back at 1.3 to 2.4 m/s. The reward falls, so this is not
  an exploit.
- `lab-fix-s1` never passed, so only one of two seeds did, and only at a checkpoint.
- The model at the bar is one result: 90% over 200 flights is about +/-4%.
- `best_model` is picked by reward, and picked a 72% model over the 90% one.

The likely cause is falling exploration. There is no entropy bonus, so the action std
falls from 0.37 to about 0.05. As it falls, the same weight change moves the policy
further: the decline starts near a std of 0.15, as KL and clipping rise. This is a
correlation, and has not been tested.

A 20M realistic run on these settings would probably decline the same way. Before it:

1. Add `--ent-coef`, and test whether a small entropy bonus holds the std and stops the
   decline. `GraphedPPO` already supports `ent_coef`.
2. Pick `best_model` by holds completed, not by reward.

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
