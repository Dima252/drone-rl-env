"""Reward for the payload drone: reach the hold with a steady airframe.

Same task as the scout, traded towards steadiness: progress pays less, and tilt,
body rates and abrupt commands cost more.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.waypoint_reward import WaypointRewardWeights, waypoint_reward
from envs import DroneState

WEIGHTS = WaypointRewardWeights(
    progress=2.0,
    proximity=1.0,
    arrival=50.0,
    smoothness=0.2,
    tilt=0.5,
    rate=0.1,
    facing=0.3,
)


def payload_reward(
    state: DroneState,
    action: np.ndarray,
    next_state: DroneState,
    info: dict[str, Any],
) -> float:
    return waypoint_reward(WEIGHTS, state, action, next_state, info)
