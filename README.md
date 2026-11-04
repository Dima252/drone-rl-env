# drone-rl-env

A modular reinforcement learning workspace for training several drones with
different objectives on one shared environment. The end goal is sim-to-real
transfer to physical hardware through ArduPilot and ROS 2.

The main task is `MarkerHoldEnv`: a single quadcopter in MuJoCo holds a set range and
height from a static ground marker, flying on a forward camera's detector boxes and
its IMU, and never GPS. `BaseDroneEnv` underneath it also offers a GPS waypoint task.
The scout and payload drones share the marker task
and differ only in their reward weights. Both train with PPO from Stable Baselines3.

```text
drone-rl-env/
├── requirements.txt
├── pytest.ini
├── tests/
│   ├── test_env.py               # Dynamics, sensors, latency, waypoints, reward
│   ├── test_marker_env.py        # Field of view, detector, the stand-off hold
│   ├── test_node_parity.py       # Training and the flight node must agree
│   ├── test_pipeline.py          # Train, resume, evaluate and export for real
│   └── sitl/                     # Flies policies on ArduCopter SITL in Docker
└── drone_rl_workspace/
    ├── assets/
    │   └── sandbox.xml           # MuJoCo quadcopter model
    ├── envs/
    │   ├── __init__.py
    │   ├── base_drone_env.py     # BaseDroneEnv and its configuration
    │   └── marker_env.py         # MarkerHoldEnv: camera and IMU only
    ├── ros2_inference_node.py    # Flies a policy through MAVROS (ROS 2 Humble)
    └── agents/
        ├── __init__.py
        ├── waypoint_reward.py    # Reward shared by every drone
        ├── evaluate.py           # Measures a trained policy
        ├── scout/
        │   ├── __init__.py
        │   ├── reward.py         # scout_reward: fast and smooth
        │   ├── train.py          # PPO training script
        │   └── export_onnx.py    # Exports a policy for the drone
        └── payload/
            ├── __init__.py
            ├── reward.py         # payload_reward: steady airframe
            └── train.py          # PPO training script
```

No camera-only model has been trained yet. See the [training plan and open
decisions](docs/training-plan.md) before training.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run code from inside `drone_rl_workspace/` so that `envs` and `agents` import. Run
the tests from the repository root:

```bash
python -m pytest
```

## Usage

```python
from agents.scout import scout_reward
from envs import DroneEnvConfig, MarkerHoldEnv

env = MarkerHoldEnv(reward_func=scout_reward, config=DroneEnvConfig())
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

A new drone needs a new reward function and nothing else. To share the marker task,
it only needs new `WaypointRewardWeights`.

## Training

Run these from inside `drone_rl_workspace/`:

```bash
python -m agents.scout.train       # or agents.payload.train
tensorboard --logdir runs
```

Each run writes to `runs/<agent>/<run-name>/`. A saved model only works together
with the `vecnormalize.pkl` saved beside it. `training_config.json` records the full
environment configuration, the commit and the package versions.

To continue an interrupted run, repeat its name with `--resume`. It restarts from
the latest checkpoint, and `--total-timesteps` still means the length of the whole
run.

Watch `rollout/crash_rate`, `rollout/waypoints_reached` (completed holds) and `eval/success_rate` in
TensorBoard. Reward alone cannot show what a policy is doing. `rollout/difficulty`
shows the curriculum: randomization starts at the nominal vehicle and rises only
while the policy completes waypoints without crashing. If it stalls below 1, the run
never trained on the full ranges, even though evaluation always uses them.

## Evaluation

Measure a run, from inside `drone_rl_workspace/`:

```bash
python -m agents.evaluate runs/scout/<run-name>
```

| Scenario | What it flies | Gate |
| --- | --- | --- |
| `nominal` | The nominal vehicle, perfect sensors, still air | `--max-crash-rate` (2%), `--min-waypoint-rate` (80%) |
| `randomized` | The full training ranges | The same |
| `worst case` | Every range pinned at its hard end at once | `--max-worst-case-crash-rate` (10%) |
| `unseen` | Conditions beyond the training ranges | `--max-unseen-crash-rate` (15%) |

The gap between `randomized` and `unseen` measures overfitting to the training
ranges. `hold m` is the true distance from the hold point while settled there. The
command exits non-zero if any gate fails.

## Deployment

Export a trained run, from inside `drone_rl_workspace/`:

```bash
python -m agents.scout.export_onnx runs/<agent>/<run-name>
```

Copy `model.onnx`, `policy_meta.npz` and `ros2_inference_node.py` to the companion
computer, which needs ROS 2 Humble, MAVROS, `numpy` and `onnxruntime`. The node's
docstring lists the required vehicle setup. Several of those settings fail
silently, so check each one.

You must supply a detector, which publishes the marker's box on `marker_topic`. The
node's docstring gives the message. Set the stand-off with `hold_range_m` and
`hold_height_m`.

Point the camera at the marker before selecting GUIDED, or the node refuses to
engage. If the marker is lost for `lost_timeout_s`, or the box grows past
`max_box_width`, the node commands level flight and requests `LOITER`.

The network never reads GPS. The node's flight limits and the `LOITER` fallback still
need ArduPilot's position estimate, so a vehicle without GPS must change
`fallback_mode` and `required_mode`.

Set up ArduPilot's own protection as well, because the node cannot replace it:
`FENCE_ENABLE`, `BATT_FS_*`, `FS_THR_ENABLE` and `FS_EKF_ACTION`. Keep a pilot on the
sticks, ready to leave GUIDED.

Fly every policy in [SITL](tests/sitl/README.md) before it goes near hardware.

## How a step runs

```text
action -> latency -> simulated autopilot -> 4 motors -> MuJoCo -> noise -> latency -> observation
```

Actions are setpoints, so a cascaded controller stands in for ArduPilot between
the policy and the motors. Its gains live in `FlightControllerConfig` and are not
tuned to any real vehicle.

The model is written directly in NED, so the MuJoCo viewer shows it upside down.
`render_mode="rgb_array"` turns the picture the right way up.

## Environment summary

| Concern | Where | Status |
| --- | --- | --- |
| Reward injection | `reward_func` argument | Done |
| Task | `_next_waypoint`, `_update_hold` | Static waypoints: arrive slowly, hold for a second, and the next appears |
| Physics | `_physics_reset`, `_physics_step` | MuJoCo at 400 Hz with first-order motor lag and quadratic drag |
| Flight envelope | `_termination_reason` | Equal to the node's defaults, and tested to stay so |
| Airframe randomization | `_sample_domain_params`, `_physics_reset` | Mass, inertia, centre of mass, drag, motor lag and efficiency, thrust curve, autopilot gain |
| Weather and events | `_apply_wind`, `_update_push`, `_update_thrust_loss` | Wind with height, updrafts, gusts, sudden shoves, battery sag, a motor weakening mid-flight |
| Sensor noise | `_read_imu`, `_read_gps` | Per-episode noise level, gyro bias walk and vibration, trim and heading error, EKF-style drift at the GPS fix rate, baro drift, GPS glitches |
| Camera and detector | `MarkerTaskConfig`, `_project_marker`, `_detect_box` | Field of view, bounding box from marker corners, pixel error, blur, small-box dropouts, false detections, frame rate, delay, mount error |
| Communication latency | `DelayLine` | Separate uplink, GPS, IMU and camera delays with jitter and packet loss |
| Curriculum | `set_difficulty`, `next_difficulty` | Scales all randomization, raised as the policy earns it |
| Action filter | `action_filter_alpha` | Keeps command chatter out of the motors; the node applies the same filter |

The action space is `Box(-1, 1, (4,))`. It holds velocity or attitude targets,
selected by `DroneEnvConfig.control_mode`. The observation space is a 24-value
`Box`. The module docstring in `base_drone_env.py` documents both layouts and
the NED/FRD frame conventions.

## Before hardware

Simulation cannot supply these. Until they are measured, the randomization ranges
are deliberately wide so that a policy tolerates the unknown values, which costs
performance. Narrow each range around its measured value.

| Value | Where | How to measure it |
| --- | --- | --- |
| Mass and inertia | `sandbox.xml`, `mass_kg`, `inertia_scale` | Scales; a bifilar pendulum, or CAD |
| Thrust curve, motor lag | `hover_thrust`, `motor_lag_scale` | Thrust stand, or logged step responses |
| Drag | `drag_scale` | Terminal speed at a fixed tilt |
| Autopilot response | `FlightControllerConfig`, `attitude_gain_scale` | Step responses from flight logs |
| Gyro noise, bias and vibration | `SensorNoiseConfig` | Bench and in-flight IMU logs |
| Position and velocity error | `SensorNoiseConfig` | EKF output against RTK or a static log |
| Link delay, jitter and loss | `action_delay_steps`, `gps_delay_steps`, `imu_delay_steps`, `link_*` | Timestamped telemetry on the real link |
| Camera and marker geometry | `MarkerTaskConfig` | Lens data sheet, or a checkerboard calibration; measure the mount angle and marker size |
| Camera timing | `camera_frame_hz`, `camera_delay_steps` | Timestamps from the running detector |
| Detection quality | `box_std_px`, `min_box_px`, `vision_*` | Fly near the real marker and log boxes against RTK |

Check the action filter too: on the real vehicle, confirm that commands are smooth
and that the motors do not run hot in a long hover.

## Not modelled

The marker is static, round and on the ground. It has to be round: box size is the
only cue to range, and a square's box changes with the viewing bearing. Moving
objects, obstacles, visual odometry
and images are not simulated: the policy receives a detector's output, not pixels.
A forward camera cannot see a marker beneath the vehicle, so the hold is at a
stand-off, and a box that grows too large ends policy control.
