"""Shared reward for the waypoint task. Each drone chooses its own weights.

Progress is the difference of a potential, so it pays nothing once the vehicle
stops. Proximity is what pulls a stationary vehicle in for the last metre: when it
was weak, a policy settled 1.1 m off in perfect conditions. Arrival has to pay more
than loitering just outside the arrival gate.

Repeated arrivals make time valuable, so without a cost on speed a policy rushes in
and overshoots. See docs/training-plan.md for why that loses a forward camera's view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from envs import DroneState
from envs.base_drone_env import _norm

PROXIMITY_RADIUS_M = 0.7
PROXIMITY_SPEED_MPS = 1.0
SPEED_LIMIT_MPS = 1.0


@dataclass(frozen=True)
class WaypointRewardWeights:
    progress: float  # Per metre closed on the waypoint.
    proximity: float  # Per step, at the waypoint and stationary.
    arrival: float  # Once, on completing the hold.
    smoothness: float  # Per step, on the squared change in action.
    tilt: float  # Per step, per radian.
    rate: float  # Per step, per rad/s.
    facing: float = 0.0  # Per step, per radian off pointing at the marker.
    speed: float = 0.0  # Per step, per m/s of ground speed over SPEED_LIMIT_MPS.
    crash: float = 100.0  # Once, on leaving the flight envelope.


def waypoint_reward(
    weights: WaypointRewardWeights,
    state: DroneState,
    action: np.ndarray,
    next_state: DroneState,
    info: dict[str, Any],
) -> float:
    waypoint = info["waypoint"]
    # Against the waypoint as it stood. A hold point follows the vehicle, so measuring
    # from where it moved to would pay for sliding around the marker as progress.
    before = _norm(info["prev_waypoint"] - state.position)
    after = _norm(waypoint - next_state.position)
    speed = _norm(next_state.velocity)

    proximity = math.exp(
        -((after / PROXIMITY_RADIUS_M) ** 2) - (speed / PROXIMITY_SPEED_MPS) ** 2
    )
    jerk = float(np.sum((action - info["prev_action"]) ** 2))
    rate = _norm(next_state.angular_velocity)
    excess_speed = max(_norm(next_state.velocity[:2]) - SPEED_LIMIT_MPS, 0.0)

    return (
        weights.progress * (before - after)
        + weights.proximity * proximity
        + (weights.arrival if info["arrived"] else 0.0)
        - weights.smoothness * jerk
        - weights.tilt * next_state.tilt
        - weights.rate * rate
        - weights.facing * info.get("bearing_error", 0.0)
        - weights.speed * excess_speed
        - (weights.crash if info["terminated"] else 0.0)
    )
