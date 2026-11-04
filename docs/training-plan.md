# Training plan and open decisions

Status at the end of the session on 2026-11-04. **No camera-only model has been
trained, and no training should start until the decisions below are made.**

## Open decisions

1. **Approve the training plan?** The recommendation is to start with Stage 1 only.
2. **How big is the real marker?** The simulation assumes a round marker 1 m across
   (`marker_size_m`). At 14 m that is about 33 pixels, so a few pixels of error is
   about 1 m of range. A 2 m marker roughly halves that error.
3. **How much memory should the network have?** It sees 4 boxes over 0.3 s, but a
   hand-written controller needed about 0.5 s of smoothing to fly.
   - Longer box history, about 8 boxes over 0.7 s. **Recommended:** a small change
     that stays simple on the drone. `BOX_LAGS` is defined in both `marker_env.py` and
     the node, and `OBS_DIM` in `make_const.py` must change with it.
   - An LSTM network: stronger memory, but slower to train and harder to export and
     run on the drone.
4. **What hold tolerance?** It is 0.75 m (`hold_radius_m`). The recommendation is
   1.0 m for the first model, because 0.75 m may be unreachable with a 1 m marker.

## Proposed process

| Stage | Steps | Rough time | Question it answers |
| --- | --- | --- | --- |
| 1. Short check | About 2M | 15 minutes | Does it start completing holds at the easiest level? If not, stop and fix that. |
| 2. Full run | About 20M | 2 hours | Does it earn its way up to full difficulty? |
| 3. Judge | - | - | Does it pass the evaluator? If so, export it and fly SITL. |

The times are estimates: training speed on this task has not been measured. Report
back after each stage. If a run fails, diagnose it and agree the next step before
training again.

Evaluator gates, from `agents/evaluate.py`:

| Scenario | Crashes at most | Completes a hold |
| --- | --- | --- |
| Nominal, randomized | 2% | At least 80% |
| Worst case | 10% | Not gated |
| Unseen conditions | 15% | Not gated |

Crashes include losing the marker and coming too close.

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

## Not yet verified

- The exporter on the camera-only task, and `tests/test_pipeline.py`. That test file
  trains tiny throwaway models, so leave it out until training is agreed:
  `python -m pytest tests/test_env.py tests/test_marker_env.py tests/test_node_parity.py`.
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
