"""Reward for the scout drone: reach the hold quickly with smooth commands."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from agents.waypoint_reward import WaypointRewardWeights, waypoint_reward
from envs import DroneState

WEIGHTS = WaypointRewardWeights(
    progress=5.0,
    proximity=1.0,
    arrival=50.0,
    smoothness=float(os.environ.get("DRONE_SMOOTHNESS", "0.05")),
    tilt=0.0,
    rate=0.02,
    facing=0.3,
)


def scout_reward(
    state: DroneState,
    action: np.ndarray,
    next_state: DroneState,
    info: dict[str, Any],
) -> float:
    return waypoint_reward(WEIGHTS, state, action, next_state, info)
