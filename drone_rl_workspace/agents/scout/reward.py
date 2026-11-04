"""Reward for the scout drone: cover ground quickly with smooth commands."""

from __future__ import annotations

from typing import Any

import numpy as np

from envs import DroneState

CRASH_PENALTY = 100.0
SMOOTHNESS_WEIGHT = 0.05


def scout_reward(
    state: DroneState,
    action: np.ndarray,
    next_state: DroneState,
    info: dict[str, Any],
) -> float:
    """Placeholder objective, untuned: reward horizontal speed."""
    speed = float(np.linalg.norm(next_state.velocity[:2]))
    jerk = float(np.sum((action - info["prev_action"]) ** 2))
    crash = CRASH_PENALTY if info["terminated"] else 0.0
    return speed - SMOOTHNESS_WEIGHT * jerk - crash
