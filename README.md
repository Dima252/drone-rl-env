# drone-rl-env

A modular reinforcement learning workspace for training several drones with
different objectives on one shared environment. The end goal is sim-to-real
transfer to physical hardware through ArduPilot and ROS 2.

This repository currently holds the environment architecture only. There is no
physics backend or training algorithm yet.

```text
drone-rl-env/
├── requirements.txt
└── drone_rl_workspace/
    ├── envs/
    │   ├── __init__.py
    │   └── base_drone_env.py     # BaseDroneEnv and its configuration
    └── agents/
        ├── __init__.py
        ├── scout/
        │   ├── __init__.py
        │   └── reward.py         # scout_reward (placeholder)
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

## Adding a physics backend

`BaseDroneEnv` raises `NotImplementedError` from `reset()` until a subclass
implements the two physics hooks:

```python
from agents.scout import scout_reward
from envs import BaseDroneEnv, DomainParams, DroneEnvConfig, DroneState


class SimDroneEnv(BaseDroneEnv):
    def _physics_reset(self, domain: DomainParams) -> DroneState:
        ...  # Apply mass, wind and motor efficiency; return the first state.

    def _physics_step(self, command, dt: float) -> DroneState:
        ...  # Advance the simulator by dt; return the new state.


env = SimDroneEnv(reward_func=scout_reward, config=DroneEnvConfig())
```

A new drone needs a new reward function and nothing else.

## Environment summary

| Concern | Where | Status |
| --- | --- | --- |
| Reward injection | `reward_func` argument | Done |
| Domain randomization | `_sample_domain_params` | Sampling done; applying it is the backend's job |
| Sensor noise | `_read_imu`, `_read_gps`, `_read_vision` | Simple Gaussian models, placeholder values |
| Communication latency | `_delay` | Fixed per-episode delay; no jitter or packet loss yet |
| Physics | `_physics_reset`, `_physics_step` | Not implemented |

The action space is `Box(-1, 1, (4,))`. It holds velocity or attitude targets,
selected by `DroneEnvConfig.control_mode`. The observation space is a 21-value
`Box`. The module docstring in `base_drone_env.py` documents both layouts and
the NED/FRD frame conventions.

The noise and latency values are placeholders. Replace them with measurements
from the real vehicle before relying on sim-to-real results.
