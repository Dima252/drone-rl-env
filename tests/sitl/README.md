# SITL test

Flies `ros2_inference_node.py` on ArduCopter SITL through real MAVROS, on ROS 2
Humble, inside Docker. Run it on every policy before it goes near hardware.

## Run

You need Docker and an exported policy. From the repository root, in bash (Git Bash
on Windows):

```bash
tests/sitl/run.sh drone_rl_workspace/runs/scout/<run-name>/export
```

The first run builds the image, which takes a few minutes and about 1.7 GB. A run
takes about 7 minutes and exits non-zero if any check fails. Logs go to
`tests/sitl/out/`, with one `node_<scenario>.log` per scenario.

To hold for longer than the default 10 seconds, or allow more than the default 20
seconds to reach the stand-off:

```bash
POLICY_SECONDS=60 WAYPOINT_SECONDS=30 tests/sitl/run.sh <export-dir>
```

To measure only the control loop, against a synthetic MAVROS:

```bash
tests/sitl/run.sh <export-dir> --bench
```

## What it checks

A scripted pilot arms, takes off and hands over by selecting GUIDED. SITL starts at
a 60 degree heading, because frame errors can cancel out at 0 degrees. Attitude is
decoded independently of the node.

| Area | Checks |
| --- | --- |
| Link | 50 Hz sensor streams and a 50 Hz control loop with no late ticks |
| Frames | Each axis moves the vehicle the right way, using constant-output policies |
| Takeover | Pilot takeover, handing back, and a restart in GUIDED not seizing control |
| Faults | Tilt, position, altitude, speed and stale sensors all hand over to `LOITER` |
| Failures | Wrong `GUID_OPTIONS`, the node killed in flight, and MAVROS frozen |
| Marker | Refuses to engage without one, levels out and hands over when it is lost or too close |
| Your policy | Runs on real telemetry, holds the stand-off on the camera alone, and approaches it and stops |

## Reading the result

Everything except the policy checks uses built-in constant policies, so a failure
there is a problem in the node, the setup or your change to them.

The policy checks fly your exported network on ArduCopter's own dynamics, which it
never trained on. A failure there is a sim-to-sim gap worth understanding before
hardware. A pass covers one calm-air hold and one approach, so also read the errors
they report, and fly for longer than the default.

## If you change the observation or action space

The node, `make_const.py` and the exporter all assume 53 observations and 4 actions.
Change all three together, and update `build_observation` in the node so that each
new value is built the same way as in training. `tests/test_node_parity.py` compares
the two.

## Limits

SITL does not cover the real serial link, companion computer timing, vibration or
real sensors. It has no camera either: a fake detector projects a marker from the
true pose with no noise, which proves the plumbing and not your real detector. Passing here is required before hardware, but it is not enough on its
own: re-measure the loop rate on the real companion computer.

ArduCopter is pinned to 4.7.2 in the `Dockerfile`. Change `COPTER_VERSION` to match
your flight controller, because parameter names change between releases.
