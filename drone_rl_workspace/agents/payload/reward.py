"""Reward for the payload drone: keep the airframe steady under load."""

from __future__ import annotations

from typing import Any

import numpy as np

from envs import DroneState

CRASH_PENALTY = 100.0
TILT_WEIGHT = 1.0
RATE_WEIGHT = 0.1


def payload_reward(
    state: DroneState,
    action: np.ndarray,
    next_state: DroneState,
    info: dict[str, Any],
) -> float:
    """Placeholder objective, untuned: penalize tilt and body rates."""
    rate = float(np.linalg.norm(next_state.angular_velocity))
    crash = CRASH_PENALTY if info["terminated"] else 0.0
    return -TILT_WEIGHT * next_state.tilt - RATE_WEIGHT * rate - crash
