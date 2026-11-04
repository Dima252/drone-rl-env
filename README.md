# drone-rl-env

A modular reinforcement learning workspace for training several drones with
different objectives on one shared environment. The end goal is sim-to-real
transfer to physical hardware through ArduPilot and ROS 2.

The environment is a working MuJoCo simulation of a single quadcopter, intended
for hover stability, waypoint following and disturbance rejection. The scout
drone has a PPO training script built on Stable Baselines3.

```text
drone-rl-env/
├── requirements.txt
└── drone_rl_workspace/
    ├── assets/
    │   └── sandbox.xml           # MuJoCo quadcopter model
    ├── envs/
    │   ├── __init__.py
    │   └── base_drone_env.py     # BaseDroneEnv and its configuration
    └── agents/
        ├── __init__.py
        ├── scout/
        │   ├── __init__.py
        │   ├── reward.py         # scout_reward (placeholder)
        │   └── train.py          # PPO training script
        └── payload/
            ├── __init__.py
            └── reward.py         # payload_reward (placeholder)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run code from inside `drone_rl_workspace/` so that `envs` and `agents` import.

## Usage

```python
from agents.scout import scout_reward
from envs import BaseDroneEnv, DroneEnvConfig

env = BaseDroneEnv(reward_func=scout_reward, config=DroneEnvConfig())
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

A new drone needs a new reward function and nothing else.

## Training

Run these from inside `drone_rl_workspace/`:

```bash
python -m agents.scout.train
tensorboard --logdir runs/scout
```

Each run writes to `runs/scout/<run-name>/`. A saved model only works together
with the `vecnormalize.pkl` saved beside it.

## How a step runs

```text
action -> latency -> simulated autopilot -> 4 motors -> MuJoCo -> noise -> latency -> observation
```

Actions are setpoints, so a cascaded controller stands in for ArduPilot between
the policy and the motors. Its gains live in `FlightControllerConfig` and are not
tuned to any real vehicle.

The model is written directly in NED, so the MuJoCo viewer shows it upside down.

## Environment summary

| Concern | Where | Status |
| --- | --- | --- |
| Reward injection | `reward_func` argument | Done |
| Physics | `_physics_reset`, `_physics_step` | MuJoCo at 400 Hz with first-order motor lag and quadratic drag |
| Domain randomization | `_sample_domain_params`, `_physics_reset` | Mass, inertia, steady wind and per-motor efficiency reach MuJoCo |
| Sensor noise | `_read_imu`, `_read_gps`, `_read_vision` | Simple Gaussian models, placeholder values |
| Communication latency | `_delay` | Fixed per-episode delay; no jitter or packet loss yet |

The action space is `Box(-1, 1, (4,))`. It holds velocity or attitude targets,
selected by `DroneEnvConfig.control_mode`. The observation space is a 21-value
`Box`. The module docstring in `base_drone_env.py` documents both layouts and
the NED/FRD frame conventions.

The noise, latency, airframe and controller values are placeholders. Replace them
with measurements from the real vehicle before relying on sim-to-real results.
