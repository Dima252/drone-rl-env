"""Hold a steady position relative to a static marker, on camera and IMU alone.

A forward-facing camera, tilted down a little, feeds a bounding-box detector. The
policy sees that detector's output and its own IMU, and never GPS. The camera and
the detector are the only source of where the marker is, so:

* If the vehicle yaws or pitches far enough for the marker to leave the field of
  view, the detections stop. Losing them for ``lost_timeout_s`` ends the episode, as
  it ends policy control on the vehicle.
* Velocity is not measured. The policy infers motion from how the box changes, so
  the observation holds a short history of boxes.

The job is to hold a set range and height from the marker while facing it. A forward
camera cannot see a marker beneath the vehicle, so this is a stand-off hold rather
than a hold overhead. Any bearing around the marker is acceptable. The marker is
static, and a box that grows past ``max_box_width`` ends the episode, so approaching
it closely is always a failure.

Observation (31 values, float32):

    ======  ===============================  ===================================
    Index   Field                            Source
    ======  ===============================  ===================================
    0-19    4 boxes: cx, cy, w, h, visible   Detector, newest first, 0.1 s apart
    20-21   roll, pitch (rad)                IMU
    22-24   body rates (rad/s)               IMU
    25-28   previous action                  Policy, after the action filter
    29-30   hold range, hold height (m)      Mission
    ======  ===============================  ===================================

Boxes are in normalized image coordinates: x right and y down, -1 to 1 across the
frame. ``w`` and ``h`` are full sizes in the same units, so 2 fills the frame. A box
of all zeros means no detection.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from envs.base_drone_env import (
    ACTION_DIM,
    VISION_START,
    BaseDroneEnv,
    DroneState,
    _euler_to_quaternion,
    _quaternion_to_euler,
    _world_to_body,
)

BOX_DIM = 5
BOX_LAGS = (0, 5, 10, 15)  # Control steps back from now.
BOX_OBS_DIM = BOX_DIM * len(BOX_LAGS)
OBS_ATTITUDE = BOX_OBS_DIM
OBS_RATES = OBS_ATTITUDE + 2
OBS_ACTION = OBS_RATES + 3
OBS_HOLD = OBS_ACTION + ACTION_DIM
MARKER_OBS_DIM = OBS_HOLD + 2

# A round marker, because box size is the only cue to range. A square's box changes
# width by up to 41% with the viewing bearing, which is ambiguous by 2 m at 10 m.
_ANGLES = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
MARKER_RIM = np.stack([np.cos(_ANGLES), np.sin(_ANGLES), 0.0 * _ANGLES], axis=1) / 2.0


class MarkerHoldEnv(BaseDroneEnv):
    """Stand-off hold on a static marker. See the module docstring."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._boxes: deque[np.ndarray] = deque(maxlen=BOX_LAGS[-1] + 1)
        self._lost_steps = 0
        self._latest_box = np.zeros(4)
        self._hold_range = 0.0
        self._hold_height = 0.0

    # ------------------------------------------------------------------
    # Start of an episode
    # ------------------------------------------------------------------

    def _physics_reset(self, domain) -> DroneState:
        """Start somewhere the marker is in view, as the pilot must before engaging."""
        super()._physics_reset(domain)
        task = self.config.marker_task
        model, data = self._model, self._data
        rng = self.np_random
        self._marker = np.array([*rng.uniform(-task.marker_xy_m, task.marker_xy_m, 2), 0.0])
        self._sample_hold()
        self._lost_steps = 0

        for _ in range(200):
            bearing = rng.uniform(-np.pi, np.pi)
            distance = rng.uniform(*task.spawn_range_m)
            data.qpos[:2] = self._marker[:2] + distance * np.array(
                [math.cos(bearing), math.sin(bearing)]
            )
            data.qpos[2] = -rng.uniform(*task.spawn_height_m)
            tilt = self.config.spawn_tilt_rad
            data.qpos[3:7] = _euler_to_quaternion(
                rng.uniform(-tilt, tilt),
                rng.uniform(-tilt, tilt),
                bearing + np.pi + rng.uniform(-task.spawn_heading_error_rad, task.spawn_heading_error_rad),
            )
            mujoco.mj_forward(model, data)
            box = self._project_marker(data.qpos[:3], data.qpos[3:7])
            if box.any() and max(abs(box[0]), abs(box[1])) < 0.8 and box[2] < task.max_box_width:
                break
        else:
            raise RuntimeError("No spawn point sees the marker. Check marker_task.")

        self._apply_wind()
        return self._read_state()

    def _sample_hold(self) -> None:
        """Pick a stand-off that puts the marker near the middle of the frame."""
        task = self.config.marker_task
        rng = self.np_random
        self._hold_range = float(rng.uniform(*task.hold_range_m))
        depression = task.camera_pitch_down_rad + rng.uniform(
            -task.hold_depression_spread_rad, task.hold_depression_spread_rad
        )
        height = self._hold_range * math.tan(depression)
        self._hold_height = float(np.clip(height, *task.hold_height_m))

    # ------------------------------------------------------------------
    # Camera and detector
    # ------------------------------------------------------------------

    def _to_camera(self, offset: np.ndarray) -> tuple[float, float, float]:
        """Return a body-frame vector as depth, and image x and y before scaling."""
        task = self.config.marker_task
        pitch = task.camera_pitch_down_rad + self.domain.camera_tilt_error_rad
        depth = math.cos(pitch) * offset[0] + math.sin(pitch) * offset[2]
        down = -math.sin(pitch) * offset[0] + math.cos(pitch) * offset[2]
        return depth, offset[1], down

    def _in_camera_view(self, offset: np.ndarray) -> bool:
        task = self.config.marker_task
        depth, right, down = self._to_camera(offset)
        return bool(
            depth > 0.0
            and abs(right) < depth * math.tan(task.horizontal_fov_rad / 2.0)
            and abs(down) < depth * math.tan(task.vertical_fov_rad / 2.0)
        )

    def _project_marker(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """Return the true box [cx, cy, w, h], or zeros if the marker is out of view.

        The marker drops out as soon as its centre leaves the frame. Until then a
        marker at the edge gives a box cut off by the frame, as a detector does.
        """
        task = self.config.marker_task
        half_width = math.tan(task.horizontal_fov_rad / 2.0)
        half_height = math.tan(task.vertical_fov_rad / 2.0)

        w, i, j, k = quaternion
        body_to_world = np.array(
            [
                [1 - 2 * (j * j + k * k), 2 * (i * j - w * k), 2 * (i * k + w * j)],
                [2 * (i * j + w * k), 1 - 2 * (i * i + k * k), 2 * (j * k - w * i)],
                [2 * (i * k - w * j), 2 * (j * k + w * i), 1 - 2 * (i * i + j * j)],
            ]
        )
        rim = (self._marker + MARKER_RIM * task.marker_size_m - position) @ body_to_world
        depth, right, down = self._to_camera(rim.T)
        if depth.min() < 0.1:
            return np.zeros(4)
        x = right / depth / half_width
        y = down / depth / half_height

        if abs((x.min() + x.max()) / 2) > 1.0 or abs((y.min() + y.max()) / 2) > 1.0:
            return np.zeros(4)
        left, right_edge = max(x.min(), -1.0), min(x.max(), 1.0)
        top, bottom = max(y.min(), -1.0), min(y.max(), 1.0)
        width, height = right_edge - left, bottom - top
        min_width = 2.0 * task.min_box_px / task.image_width_px
        if width < min_width or height <= 0.0:
            return np.zeros(4)
        return np.array([(left + right_edge) / 2, (top + bottom) / 2, width, height])

    def _camera_slots(self, state: DroneState) -> np.ndarray:
        """Return the newest detection, which only changes on each frame."""
        frame_steps = max(round(self.config.control_hz / self.domain.camera_frame_hz), 1)
        if self._step_count % frame_steps == 0:
            self._camera_frame = self._detect_box(state)
        return self._camera_frame

    def _detect_box(self, state: DroneState) -> np.ndarray:
        """Process one frame: pixel error, blur, small-box dropouts, false alarms."""
        cfg = self.config.noise
        task = self.config.marker_task
        box = self._project_marker(state.position, state.quaternion)
        if not cfg.enabled:
            return box

        rng = self.np_random
        if rng.random() < cfg.vision_false_positive_prob:
            size = rng.uniform(0.03, 0.3, 2)
            return np.array([*rng.uniform(-1.0, 1.0, 2), *size])
        if not box.any():
            return box

        min_width = 2.0 * task.min_box_px / task.image_width_px
        fade = (box[2] - min_width) / min_width
        blur = float(np.linalg.norm(state.angular_velocity)) / cfg.vision_blur_rate_rps
        detect_prob = (
            (1.0 - cfg.vision_dropout_prob) * min(fade + 0.3, 1.0) * math.exp(-blur * blur)
        )
        if rng.random() >= detect_prob:
            return np.zeros(4)

        pixel = 2.0 / task.image_width_px
        noisy = box + rng.normal(0.0, task.box_std_px * pixel * self.domain.noise_scale, 4)
        noisy[2:] = np.maximum(noisy[2:], pixel)
        return noisy

    def _on_sensors(self, sensors: np.ndarray) -> None:
        """Track what the vehicle receives, because that is what ends policy control."""
        self._latest_box = sensors[VISION_START:].astype(np.float64)
        self._lost_steps = 0 if self._latest_box[2] > 0.0 else self._lost_steps + 1

    # ------------------------------------------------------------------
    # The hold
    # ------------------------------------------------------------------

    def _ring_point(self, position: np.ndarray) -> np.ndarray:
        """Return the nearest point at the hold range and height."""
        away = position[:2] - self._marker[:2]
        distance = float(np.linalg.norm(away))
        direction = away / distance if distance > 1e-6 else np.array([1.0, 0.0])
        return np.array([*(self._marker[:2] + self._hold_range * direction), -self._hold_height])

    def _next_waypoint(self, position: np.ndarray) -> np.ndarray:
        """After a completed hold, sometimes move to a new stand-off."""
        if self.np_random.random() < self.config.marker_task.new_hold_prob:
            self._sample_hold()
        return self._ring_point(position)

    def _update_hold(self, state: DroneState) -> bool:
        """Count the steps spent holding; True once a hold is complete.

        This uses true position: the camera sees the marker directly, so unlike GPS
        drift, its error does not move the place the vehicle is trying to reach.
        """
        task = self.config.marker_task
        self._waypoint = self._ring_point(state.position)
        holding = (
            np.linalg.norm(self._waypoint - state.position) < task.hold_radius_m
            and np.linalg.norm(state.velocity) < self.config.waypoints.arrival_speed_mps
        )
        self._hold_steps = self._hold_steps + 1 if holding else 0
        if self._hold_steps < round(self.config.waypoints.hold_s / self.dt):
            return False
        self._hold_steps = 0
        return True

    def _termination_reason(self, state: DroneState) -> str:
        task = self.config.marker_task
        reason = super()._termination_reason(state)
        if reason:
            return reason
        if self._latest_box[2] > task.max_box_width:
            return "too_close"
        if self._lost_steps >= round(task.lost_timeout_s / self.dt):
            return "marker_lost"
        return ""

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _make_observation_space(self) -> spaces.Box:
        cfg = self.config
        task = cfg.marker_task
        box_low = np.tile([-1.0, -1.0, 0.0, 0.0, 0.0], len(BOX_LAGS))
        box_high = np.tile([1.0, 1.0, 2.0, 2.0, 1.0], len(BOX_LAGS))
        rates = np.full(3, cfg.max_angular_rate_rps)
        ones = np.ones(ACTION_DIM)
        low = np.concatenate([box_low, [-np.pi, -np.pi], -rates, -ones, [0.0, 0.0]])
        high = np.concatenate(
            [box_high, [np.pi, np.pi], rates, ones, [task.hold_range_m[1], task.hold_height_m[1]]]
        )
        return spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)

    def _make_observation(self, sensors: np.ndarray) -> np.ndarray:
        box = sensors[VISION_START:]
        slots = np.array([*box, float(box[2] > 0.0)], dtype=np.float32)
        if self._step_count == 0:
            self._boxes.clear()
            self._boxes.extend([slots] * self._boxes.maxlen)
        else:
            self._boxes.append(slots)

        roll, pitch, _ = _quaternion_to_euler(sensors[6:10])
        observation = np.concatenate(
            [
                *[self._boxes[-1 - lag] for lag in BOX_LAGS],
                [roll, pitch],
                sensors[10:13],
                self._last_action,
                [self._hold_range, self._hold_height],
            ]
        )
        return np.clip(
            observation, self.observation_space.low, self.observation_space.high
        ).astype(np.float32)

    def _make_info(self, state: DroneState, termination: str = "") -> dict[str, Any]:
        info = super()._make_info(state, termination)
        offset = _world_to_body(state.quaternion, self._marker - state.position)
        info["bearing_error"] = abs(math.atan2(offset[1], offset[0]))
        info["marker_visible"] = bool(self._latest_box[2] > 0.0)
        return info
