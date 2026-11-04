"""Train the payload drone with PPO. Takes the same arguments as the scout.

Run from inside ``drone_rl_workspace``:

    python -m agents.payload.train
    tensorboard --logdir runs/payload
"""

from __future__ import annotations

from agents.payload.reward import payload_reward
from agents.scout.train import train

# Required: each spawned worker re-imports this module.
if __name__ == "__main__":
    train("payload", payload_reward)
