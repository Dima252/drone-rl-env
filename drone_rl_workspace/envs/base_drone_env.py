"""Base Gymnasium environment shared by every drone in the workspace.

Each drone differs only in the ``reward_func`` it injects, so all of them train
on identical dynamics, sensors and latency. Dynamics come from MuJoCo, using the
single-quadcopter model in ``assets/sandbox.xml``.

Policies output setpoints, not motor commands. On hardware ArduPilot closes the
loop from those setpoints to the motors, so the simulation needs a stand-in for
it. A cascaded controller provides that here: velocity to attitude at the
control rate, then attitude to body rates to motor thrusts at the physics rate.
It reads true state, standing in for the autopilot's own state estimate, and it
knows only the nominal vehicle, so randomized parameters are real disturbances.

Conventions match ArduPilot/MAVLink so policies need no remapping on hardware
(MAVROS converts to ROS ENU/FLU at its own boundary):

* World frame: NED (x north, y east, z down), so altitude is ``-position[2]``.
* Body frame: FRD (x forward, y right, z down).
* Quaternions: ``[w, x, y, z]``, rotating body to world.

Action (normalized to [-1, 1], 4 values; all zeros is the neutral command):

* ``ControlMode.VELOCITY``: ``[vx, vy, vz, yaw_rate]``, velocities in the world
  frame. Maps to MAVLink ``SET_POSITION_TARGET_LOCAL_NED``.
* ``ControlMode.ATTITUDE``: ``[roll, pitch, yaw_rate, thrust]``. Maps to MAVLink
  ``SET_ATTITUDE_TARGET``. ArduPilot reads that thrust field as a climb rate
  unless ``GUID_OPTIONS`` bit 3 is set.

Every episode is a series of static waypoints: reach one, hold it slowly, and the
next appears. A waypoint at the current position is a hover task, and wind is the
disturbance, so this one task covers hover, waypoint following and disturbance
rejection.

Observation (24 values, float32):

    ======  ==================  =====================================
    Index   Field               Source
    ======  ==================  =====================================
    0-2     position (m)        GPS
    3-5     velocity (m/s)      GPS
    6-9     attitude quaternion IMU
    10-12   body rates (rad/s)  IMU
    13-15   marker offset (m)   Camera, body frame
    16      marker visible      Camera, 0 or 1
    17-20   previous action     Policy, undelayed
    21-23   waypoint offset (m) Mission, body frame, from sensed pose
    ======  ==================  =====================================

The previous action is part of the observation because the policy cannot infer
the commands still in flight under latency without it. It also lets a memoryless
policy trim out a wrong hover thrust.

A downward camera looks for one static marker on the ground, as in precision
landing. Indices 13-16 are zero unless the marker is inside the camera's field of
view and was detected. The camera measures the marker directly, so it is free of
the GPS drift in the waypoint offset, but it is slower, later and often absent.

The position and velocity channels model the autopilot's EKF output, which is what
the flight node reads, rather than raw GPS. That output is smooth at the control
rate and its error changes in small steps at each GPS fix.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Callable, SupportsFloat

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ACTION_DIM = 4
SENSOR_DIM = 17
WAYPOINT_DIM = 3
OBS_DIM = SENSOR_DIM + ACTION_DIM + WAYPOINT_DIM
OBS_TARGET_VISIBLE = 16

# Sensor vector layout: position and velocity, attitude and rates, then the camera.
GPS_DIM = 6
IMU_DIM = 7
VISION_START = GPS_DIM + IMU_DIM

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "sandbox.xml"


class ControlMode(str, Enum):
    """High-level setpoint type sent to the flight controller."""

    VELOCITY = "velocity"
    ATTITUDE = "attitude"


@dataclass(frozen=True, eq=False)
class DroneState:
    """Ground-truth vehicle state. Never shown to the policy directly."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Accelerometer output in the body frame, so level hover reads [0, 0, -g].
    specific_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_visible: bool = False

    @property
    def altitude(self) -> float:
        """Height above the origin in metres."""
        return float(-self.position[2])

    @property
    def tilt(self) -> float:
        """Angle between body z and world z in radians, 0 when level."""
        _, x, y, _ = self.quaternion.tolist()
        return math.acos(min(max(1.0 - 2.0 * (x * x + y * y), -1.0), 1.0))


# (state, action, next_state, info) -> reward. Reward is computed from ground
# truth, which is available in simulation and is only needed during training.
RewardFunc = Callable[
    [DroneState, np.ndarray, DroneState, dict[str, Any]], SupportsFloat
]


@dataclass(frozen=True, eq=False)
class DomainParams:
    """Hidden parameters of one episode. The defaults are the nominal vehicle."""

    mass_kg: float = 1.5
    inertia_scale: float = 1.0
    drag_scale: float = 1.0
    motor_lag_scale: float = 1.0
    attitude_gain_scale: float = 1.0
    # Throttle that hovers the nominal mass, which sets the thrust curve's slope.
    hover_thrust: float = 0.5
    com_offset_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wind_ned_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wind_shear_exponent: float = 0.0
    gust_std_mps: float = 0.0
    push_rate_hz: float = 0.0
    push_force_n: float = 0.0
    battery_sag: float = 0.0  # Share of thrust lost by the end of an episode.
    motor_fault_loss: float = 0.0  # Share of one motor's thrust lost mid-flight.
    motor_efficiency: np.ndarray = field(default_factory=lambda: np.ones(4))
    noise_scale: float = 1.0
    gyro_bias_rps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Roll and pitch trim error, and heading error, of the autopilot's estimate.
    attitude_error_rad: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gps_update_hz: float = 10.0
    camera_tilt_error_rad: float = 0.0
    camera_frame_hz: float = 25.0
    camera_delay_steps: int = 0
    action_delay_steps: int = 0
    gps_delay_steps: int = 0
    imu_delay_steps: int = 0
    link_jitter_steps: int = 0
    link_drop_prob: float = 0.0


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Ranges sampled on every reset. Disable for nominal-vehicle evaluation."""

    enabled: bool = True
    mass_kg: tuple[float, float] = (1.2, 1.8)
    # Applied on top of the scaling with mass.
    inertia_scale: tuple[float, float] = (0.8, 1.25)
    drag_scale: tuple[float, float] = (0.5, 1.5)
    motor_lag_scale: tuple[float, float] = (0.7, 2.0)
    attitude_gain_scale: tuple[float, float] = (0.7, 1.3)
    hover_thrust: tuple[float, float] = (0.3, 0.5)
    com_offset_std_m: float = 0.01
    wind_speed_mps: tuple[float, float] = (0.0, 5.0)
    updraft_mps: tuple[float, float] = (-1.0, 1.0)
    wind_shear_exponent: tuple[float, float] = (0.0, 0.3)
    gust_std_mps: tuple[float, float] = (0.0, 1.5)
    gust_time_constant_s: float = 2.0
    push_rate_hz: tuple[float, float] = (0.0, 0.2)
    push_force_n: tuple[float, float] = (0.0, 6.0)
    push_duration_s: float = 0.2
    battery_sag: tuple[float, float] = (0.0, 0.08)
    motor_fault_prob: float = 0.1
    motor_fault_loss: tuple[float, float] = (0.05, 0.2)
    motor_efficiency: tuple[float, float] = (0.9, 1.0)
    noise_scale: tuple[float, float] = (0.5, 2.0)
    gyro_bias_std_rps: float = 0.01
    trim_error_std_rad: float = 0.01
    heading_error_std_rad: float = 0.05
    gps_update_hz: tuple[float, float] = (5.0, 10.0)
    camera_tilt_error_std_rad: float = 0.03
    camera_frame_hz: tuple[float, float] = (15.0, 30.0)
    camera_delay_steps: tuple[int, int] = (2, 5)
    action_delay_steps: tuple[int, int] = (0, 3)
    gps_delay_steps: tuple[int, int] = (0, 2)
    imu_delay_steps: tuple[int, int] = (0, 2)
    link_jitter_steps: tuple[int, int] = (0, 1)
    link_drop_prob: tuple[float, float] = (0.0, 0.05)


@dataclass(frozen=True)
class SensorNoiseConfig:
    """Per-step noise standard deviations. Placeholders until bench-measured."""

    enabled: bool = True
    gyro_std_rps: float = 0.02
    gyro_bias_walk_rps: float = 0.002  # Per square root of a second.
    gyro_vibration_std_rps: float = 0.03  # Added at full thrust, scaling with it.
    attitude_std: float = 0.005
    gps_position_std_m: float = 0.02  # Per step, on top of the drift.
    gps_drift_std_m: float = 0.3
    baro_drift_std_m: float = 0.5  # Vertical, where the barometer dominates.
    gps_drift_time_constant_s: float = 30.0
    gps_glitch_rate_hz: float = 0.02
    gps_glitch_m: float = 2.0
    gps_glitch_time_constant_s: float = 3.0
    gps_velocity_std_mps: float = 0.05
    vision_angle_std_rad: float = 0.003  # Lateral error grows with range.
    vision_range_std: float = 0.02  # As a share of range, along the line of sight.
    vision_dropout_prob: float = 0.05
    vision_false_positive_prob: float = 0.002  # Per frame.
    vision_blur_rate_rps: float = 3.0  # Body rate that blurs detection away.


@dataclass(frozen=True)
class FlightControllerConfig:
    """Gains of the simulated autopilot. Retune to match the real vehicle."""

    velocity_p: float = 4.0
    velocity_i: float = 3.0
    max_velocity_integral: float = 5.0  # m/s^2
    attitude_p: float = 6.0
    rate_p: tuple[float, float, float] = (15.0, 15.0, 4.0)  # roll, pitch, yaw
    rate_i: float = 5.0
    max_rate_integral: float = 10.0  # rad/s^2
    max_yaw_torque_nm: float = 0.1


@dataclass(frozen=True)
class CameraConfig:
    """Downward camera and the static ground marker it detects."""

    enabled: bool = True
    horizontal_fov_rad: float = 1.2  # Across the body, about 69 degrees.
    vertical_fov_rad: float = 0.9  # Along the body, about 52 degrees.
    mount_tilt_rad: float = 0.0  # Forward of straight down.
    max_range_m: float = 30.0
    fade_start: float = 0.7  # Share of max_range_m where detection starts to fade.
    marker_spawn_range_m: float = 10.0


@dataclass(frozen=True)
class MarkerTaskConfig:
    """Forward camera, detector and stand-off hold. Fill in after calibration."""

    # Camera. Pitch is below the horizon: a level camera would keep a ground marker
    # at the bottom edge of the frame, where any nose-up motion loses it.
    horizontal_fov_rad: float = 1.2
    vertical_fov_rad: float = 0.9
    camera_pitch_down_rad: float = 0.35
    image_width_px: int = 640

    # Detector.
    marker_size_m: float = 2.0  # Diameter of a round marker.
    min_box_px: float = 8.0
    box_std_px: float = 2.0
    lost_timeout_s: float = 0.5
    max_box_width: float = 0.8  # Closer than this ends policy control.

    # Mission.
    hold_range_m: tuple[float, float] = (6.0, 14.0)
    hold_height_m: tuple[float, float] = (3.0, 8.0)
    hold_depression_spread_rad: float = 0.1
    hold_radius_m: float = 1.0
    new_hold_prob: float = 0.3

    # Start of an episode.
    marker_xy_m: float = 15.0
    spawn_range_m: tuple[float, float] = (6.0, 22.0)
    spawn_height_m: tuple[float, float] = (3.0, 12.0)
    spawn_heading_error_rad: float = 0.3


@dataclass(frozen=True)
class WaypointConfig:
    """How the static waypoints of an episode are placed and when one is reached."""

    hover_prob: float = 0.25  # Chance a waypoint is simply the current position.
    marker_prob: float = 0.3  # Chance it is above the marker, where the camera helps.
    marker_altitude_m: tuple[float, float] = (5.0, 15.0)
    distance_m: tuple[float, float] = (2.0, 15.0)
    altitude_m: tuple[float, float] = (3.0, 40.0)
    max_xy_m: float = 40.0
    arrival_radius_m: float = 1.0
    arrival_speed_mps: float = 0.5
    hold_s: float = 1.0
    # The observed offset saturates here, per axis, so that a distant waypoint
    # looks like a direction to fly in rather than an unfamiliar input.
    max_offset_m: float = 20.0


@dataclass(frozen=True)
class DroneEnvConfig:
    """Everything that is shared across drones and independent of the reward."""

    control_mode: ControlMode = ControlMode.VELOCITY
    control_hz: float = 50.0
    max_episode_steps: int = 1000
    model_path: Path = DEFAULT_MODEL_PATH

    # Initial state. Position covers the flight envelope because the policy reads
    # absolute position, and motion is non-zero because a pilot can hand over a
    # moving vehicle. Heading is uniform.
    spawn_xy_range_m: float = 35.0
    spawn_altitude_m: tuple[float, float] = (5.0, 35.0)
    spawn_speed_mps: float = 2.0
    spawn_tilt_rad: float = 0.2
    spawn_rate_rps: float = 0.5

    # Command limits: what a normalized action of +/-1 means.
    max_horizontal_speed_mps: float = 5.0
    max_vertical_speed_mps: float = 2.0
    max_yaw_rate_rps: float = 1.0
    max_attitude_cmd_rad: float = 0.35
    min_thrust: float = 0.1
    max_thrust: float = 0.9

    # Share of each new action that passes to the vehicle, 1 being no filter. This
    # keeps command chatter out of the motors. The flight node applies the same one.
    action_filter_alpha: float = 0.5

    # Flight envelope: the episode ends beyond these. They equal the defaults in
    # ros2_inference_node.py, so a policy learns the limits it will really fly
    # under. max_position_m also bounds the observation.
    min_altitude_m: float = 1.0
    max_position_m: float = 50.0
    max_speed_mps: float = 15.0
    max_tilt_rad: float = 0.7

    # Sensor saturation limits.
    sensor_max_speed_mps: float = 20.0
    max_angular_rate_rps: float = 10.0
    max_vision_range_m: float = 50.0

    randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    noise: SensorNoiseConfig = field(default_factory=SensorNoiseConfig)
    controller: FlightControllerConfig = field(
        default_factory=FlightControllerConfig
    )
    waypoints: WaypointConfig = field(default_factory=WaypointConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    marker_task: MarkerTaskConfig = field(default_factory=MarkerTaskConfig)


# The environment steps millions of times, on vectors of three or four values, where
# general NumPy routines cost far more than the arithmetic. The helpers below, and
# the plain floats in the flight controller, return exactly what those routines do.


def _norm(vector: np.ndarray) -> float:
    """Return the length of a vector, as np.linalg.norm does but faster."""
    return math.sqrt(vector.dot(vector))


def _fields_to_dict(item: Any) -> dict[str, Any]:
    """Copy a flat dataclass to a dict, as dataclasses.asdict does but faster."""
    return {
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in vars(item).items()
    }


def _quaternion_to_euler(quaternion: np.ndarray) -> tuple[float, float, float]:
    """Return roll, pitch and yaw in radians (aerospace ZYX order)."""
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a quaternion from aerospace ZYX angles."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def _quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a.tolist()
    bw, bx, by, bz = b.tolist()
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _blend_domains(easy: DomainParams, hard: DomainParams, amount: float) -> DomainParams:
    """Move every parameter from ``easy`` towards ``hard``."""
    blended = {}
    for item in fields(DomainParams):
        low, high = getattr(easy, item.name), getattr(hard, item.name)
        value = low + amount * (high - low)
        blended[item.name] = round(value) if isinstance(low, int) else value
    return DomainParams(**blended)


def _world_to_body(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the body frame."""
    w, x, y, z = quaternion.tolist()
    x, y, z = -x, -y, -z
    a, b, c = vector.tolist()
    # Two cross products: first u x v, then u x (u x v).
    p, q, r = y * c - z * b, z * a - x * c, x * b - y * a
    s, t, u = y * r - z * q, z * p - x * r, x * q - y * p
    return np.array(
        [
            a + 2.0 * (w * p + s),
            b + 2.0 * (w * q + t),
            c + 2.0 * (w * r + u),
        ]
    )


class DelayLine:
    """One direction of a link: fixed delay, jitter and dropped packets, in steps.

    A dropped packet leaves the receiver holding the last value it got, as an
    autopilot does with its last setpoint. Packets never arrive out of order.
    """

    def __init__(
        self, initial: np.ndarray, delay: int, jitter: int, drop_prob: float
    ) -> None:
        self._delay = delay
        self._jitter = jitter
        self._drop_prob = drop_prob
        self._history = deque([initial] * (delay + jitter + 1), maxlen=delay + jitter + 1)
        self._held = initial
        self._held_age = delay

    def __call__(self, value: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Send ``value`` and return what the far end holds after this step."""
        self._history.append(value)
        self._held_age += 1
        dropped = rng.random() < self._drop_prob
        age = self._delay
        if self._jitter:
            age = max(age + int(rng.integers(-self._jitter, self._jitter + 1)), 0)
        if not dropped and age < self._held_age:
            self._held = self._history[-1 - age]
            self._held_age = age
        return self._held


class BaseDroneEnv(gym.Env):
    """Shared drone environment with an injected reward function.

    Args:
        reward_func: Called as ``reward_func(state, action, next_state, info)``
            after every step. ``info`` holds ``prev_action``, ``prev_waypoint``,
            ``terminated``, ``truncated`` and ``arrived`` in addition to the keys
            returned from ``step``. Its ``waypoint`` is the one this step was flown
            towards, and ``prev_waypoint`` is where that stood before the step.
        config: Shared environment settings. Defaults to ``DroneEnvConfig()``.
        render_mode: ``None``, or ``"rgb_array"`` for a camera following the drone.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        reward_func: RewardFunc,
        config: DroneEnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not callable(reward_func):
            raise TypeError(f"reward_func must be callable, got {reward_func!r}")

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")

        self.reward_func = reward_func
        self.config = config or DroneEnvConfig()
        self.render_mode = render_mode
        self.dt = 1.0 / self.config.control_hz

        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.observation_space = self._make_observation_space()

        self.domain = DomainParams()
        self.difficulty = 1.0
        self._state: DroneState | None = None
        self._step_count = 0
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._action_link: DelayLine | None = None
        self._gps_link: DelayLine | None = None
        self._imu_link: DelayLine | None = None
        self._camera_link: DelayLine | None = None

        # Slowly varying sensor and wind errors, redrawn on every reset.
        self._gust = np.zeros(3)
        self._gyro_bias = np.zeros(3)
        self._gps_position_error = np.zeros(3)
        self._gps_velocity_error = np.zeros(3)
        self._gps_glitch = np.zeros(3)
        self._heading_error = np.array([1.0, 0.0, 0.0, 0.0])
        self._trim_error = np.array([1.0, 0.0, 0.0, 0.0])
        self._camera_frame = np.zeros(4)

        # Events within an episode.
        self._push_steps_left = 0
        self._motor_fault_step = 0
        self._faulty_motor = 0

        self._marker = np.zeros(3)

        self._waypoint = np.zeros(3)
        self._waypoints_reached = 0
        self._hold_steps = 0

        self._renderer: mujoco.Renderer | None = None
        self._camera: mujoco.MjvCamera | None = None

        self._load_model()

    def _load_model(self) -> None:
        """Compile the MJCF and cache what the controller needs from it."""
        model = mujoco.MjModel.from_xml_path(str(self.config.model_path))
        self._model = model
        self._data = mujoco.MjData(model)

        substeps = self.dt / model.opt.timestep
        if substeps < 1.0 or abs(substeps - round(substeps)) > 1e-9:
            raise ValueError(
                f"control_hz={self.config.control_hz} must divide the physics "
                f"rate of {1.0 / model.opt.timestep:g} Hz."
            )
        self._substeps = round(substeps)

        self._body_id = model.body("drone").id
        self._nominal_mass = float(model.body_mass[self._body_id])
        self._nominal_inertia = model.body_inertia[self._body_id].copy()
        self._nominal_density = float(model.opt.density)
        self._nominal_motor_lag = model.actuator_dynprm[:, 0].copy()
        self._gravity = float(model.opt.gravity[2])
        self._max_motor_thrust = float(model.actuator_ctrlrange[0, 1])
        self._physics_dt = float(model.opt.timestep)
        self._inertia_values = self._nominal_inertia.tolist()
        self._gyro = self._data.sensor("gyro")
        self._accelerometer = self._data.sensor("accelerometer")

        # Mixer: invert the map from motor thrusts to [thrust, torque xyz], built
        # from the model so that motor layout and spin directions live in one place.
        rotor_positions = model.site_pos[model.actuator_trnid[:, 0]]
        gears = model.actuator_gear
        torques = np.cross(rotor_positions, gears[:, :3]) + gears[:, 3:]
        self._mixer = np.linalg.inv(np.vstack([-gears[:, 2], torques.T]))

        self._velocity_integral = np.zeros(3)
        self._rate_integral = [0.0, 0.0, 0.0]

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Domain randomization: draw this episode's hidden parameters, then let
        # the simulator apply them. See _sample_domain_params.
        self.domain = self._sample_domain_params()
        self._state = self._physics_reset(self.domain)

        self._step_count = 0
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._waypoints_reached = 0
        self._hold_steps = 0
        self._waypoint = self._next_waypoint(self._state.position)
        self._reset_sensor_errors()
        self._push_steps_left = 0
        self._motor_fault_step = int(
            self.np_random.integers(1, self.config.max_episode_steps)
        )
        self._faulty_motor = int(self.np_random.integers(4))

        # Prefill the links so a delay of N is in effect from step one.
        neutral = np.zeros(ACTION_DIM, dtype=np.float32)
        sensors = self._apply_sensor_noise(self._state)
        self._action_link = self._make_link(neutral, self.domain.action_delay_steps)
        self._gps_link = self._make_link(
            sensors[:GPS_DIM], self.domain.gps_delay_steps
        )
        self._imu_link = self._make_link(
            sensors[GPS_DIM:VISION_START], self.domain.imu_delay_steps
        )
        self._camera_link = self._make_link(
            sensors[VISION_START:], self.domain.camera_delay_steps
        )

        return self._make_observation(sensors), self._make_info(self._state)

    def set_difficulty(self, difficulty: float) -> None:
        """Scale randomization and sensor errors from none (0) to the full range (1).

        Takes effect from the next reset. Training ramps this up, and evaluation
        always runs at 1.
        """
        self.difficulty = min(max(float(difficulty), 0.0), 1.0)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        if not np.all(np.isfinite(action)):
            raise ValueError(f"Action contains non-finite values: {action}")
        action = np.clip(action, -1.0, 1.0)
        alpha = self.config.action_filter_alpha
        action = alpha * action + (1.0 - alpha) * self._last_action

        # Uplink latency: the vehicle executes a command issued earlier.
        delayed_action = self._action_link(action, self.np_random)
        command = self._action_to_command(delayed_action)

        state = self._state
        next_state = self._physics_step(command, self.dt)
        self._state = next_state
        self._step_count += 1

        # Sensor noise, then downlink latency: the policy sees stale, noisy data.
        measured = self._apply_sensor_noise(next_state)
        sensors = np.concatenate(
            [
                self._gps_link(measured[:GPS_DIM], self.np_random),
                self._imu_link(measured[GPS_DIM:VISION_START], self.np_random),
                self._camera_link(measured[VISION_START:], self.np_random),
            ]
        )
        self._on_sensors(sensors)

        termination = self._termination_reason(next_state)
        terminated = bool(termination)
        truncated = self._step_count >= self.config.max_episode_steps
        previous_waypoint = self._waypoint.copy()
        arrived = not terminated and self._update_hold(next_state)

        info = self._make_info(next_state, termination)
        reward_info = {
            **info,
            "prev_action": self._last_action,
            "prev_waypoint": previous_waypoint,
            "terminated": terminated,
            "truncated": truncated,
            "arrived": arrived,
        }
        reward = float(self.reward_func(state, action, next_state, reward_info))

        if arrived:
            self._waypoints_reached += 1
            self._waypoint = self._next_waypoint(next_state.position)
            info = self._make_info(next_state, termination)

        self._last_action = action
        return self._make_observation(sensors), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Return the view from a camera following the drone, if rendering is on."""
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)
            self._camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self._model, self._camera)
            self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._camera.trackbodyid = self._body_id
            self._camera.distance = 3.0
        self._renderer.update_scene(self._data, self._camera)
        # NED puts the ground overhead, so turn the picture the right way up.
        return self._renderer.render()[::-1, ::-1].copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # MuJoCo physics
    # ------------------------------------------------------------------

    def _physics_reset(self, domain: DomainParams) -> DroneState:
        """Restart the simulation with ``domain`` applied; return the first state."""
        model, data = self._model, self._data

        # Inertia scales with mass, so the airframe keeps its shape.
        model.body_mass[self._body_id] = domain.mass_kg
        model.body_inertia[self._body_id] = (
            self._nominal_inertia
            * (domain.mass_kg / self._nominal_mass)
            * domain.inertia_scale
        )
        model.actuator_gainprm[:, 0] = domain.motor_efficiency
        model.actuator_dynprm[:, 0] = self._nominal_motor_lag * domain.motor_lag_scale
        model.opt.density = self._nominal_density * domain.drag_scale
        model.body_ipos[self._body_id] = domain.com_offset_m
        mujoco.mj_setConst(model, data)
        mujoco.mj_resetData(model, data)

        cfg = self.config
        rng = self.np_random
        data.qpos[:2] = rng.uniform(-cfg.spawn_xy_range_m, cfg.spawn_xy_range_m, 2)
        data.qpos[2] = -rng.uniform(*cfg.spawn_altitude_m)
        data.qpos[3:7] = _euler_to_quaternion(
            rng.uniform(-cfg.spawn_tilt_rad, cfg.spawn_tilt_rad),
            rng.uniform(-cfg.spawn_tilt_rad, cfg.spawn_tilt_rad),
            rng.uniform(-np.pi, np.pi),
        )
        direction = rng.normal(size=3)
        data.qvel[:3] = (
            rng.uniform(0.0, cfg.spawn_speed_mps) * direction / np.linalg.norm(direction)
        )
        data.qvel[3:6] = rng.uniform(-cfg.spawn_rate_rps, cfg.spawn_rate_rps, 3)

        # Start in flight with the motors already spinning at nominal hover.
        data.act[:] = self._nominal_mass * self._gravity / 4.0
        self._velocity_integral = np.zeros(3)
        self._rate_integral = [0.0, 0.0, 0.0]

        self._gust = rng.normal(0.0, domain.gust_std_mps, 3)
        self._apply_wind()

        reach = cfg.camera.marker_spawn_range_m
        self._marker = np.array([*(data.qpos[:2] + rng.uniform(-reach, reach, 2)), 0.0])

        mujoco.mj_forward(model, data)
        return self._read_state()

    def _physics_step(self, command: np.ndarray, dt: float) -> DroneState:
        """Advance the simulation by ``dt`` seconds and return the new state.

        ``command`` holds physical units from ``_action_to_command``.
        """
        model, data = self._model, self._data

        if self.config.control_mode is ControlMode.VELOCITY:
            roll, pitch, thrust = self._velocity_to_attitude(command[:3], dt)
            yaw_rate = float(command[3])
        else:
            roll, pitch, yaw_rate, throttle = command.tolist()
            weight = self._nominal_mass * self._gravity
            thrust = throttle * weight / self.domain.hover_thrust

        self._update_gust(dt)
        self._update_push()
        self._update_thrust_loss()
        ctrl = data.ctrl
        for _ in range(self._substeps):
            ctrl[:] = self._attitude_to_motors(roll, pitch, yaw_rate, thrust)
            mujoco.mj_step(model, data)

        # mj_step leaves sensors one substep behind the integrated state.
        mujoco.mj_forward(model, data)
        return self._read_state()

    def _update_gust(self, dt: float) -> None:
        """Advance the gust, a first-order Gauss-Markov process, on the steady wind."""
        decay = math.exp(-dt / self.config.randomization.gust_time_constant_s)
        noise = self.domain.gust_std_mps * math.sqrt(1.0 - decay * decay)
        self._gust = decay * self._gust + self.np_random.normal(0.0, noise, 3)
        self._apply_wind()

    def _apply_wind(self) -> None:
        """Set the wind at the vehicle: stronger with height, as near real ground."""
        steady = self.domain.wind_ned_mps.copy()
        altitude = max(-float(self._data.qpos[2]), 1.0)
        steady[:2] *= (altitude / 10.0) ** self.domain.wind_shear_exponent
        self._model.opt.wind[:] = steady + self._gust

    def _update_push(self) -> None:
        """Apply short random shoves, as from a sudden gust or a tug on the airframe."""
        rng = self.np_random
        cfg = self.config.randomization
        force = self._data.xfrc_applied[self._body_id, :3]
        if self._push_steps_left > 0:
            self._push_steps_left -= 1
            if self._push_steps_left == 0:
                force[:] = 0.0
        elif rng.random() < self.domain.push_rate_hz * self.dt:
            direction = rng.normal(size=3)
            force[:] = self.domain.push_force_n * direction / np.linalg.norm(direction)
            self._push_steps_left = max(round(cfg.push_duration_s / self.dt), 1)

    def _update_thrust_loss(self) -> None:
        """Lose thrust to the sagging battery and, in some flights, a failing motor."""
        steps = self.config.max_episode_steps
        gain = self.domain.motor_efficiency * (
            1.0 - self.domain.battery_sag * self._step_count / steps
        )
        if self._step_count >= self._motor_fault_step:
            gain[self._faulty_motor] *= 1.0 - self.domain.motor_fault_loss
        self._model.actuator_gainprm[:, 0] = gain

    def _read_state(self) -> DroneState:
        """Copy the true state out of MuJoCo.

        A free joint stores linear velocity in the world frame and angular
        velocity in the body frame, which are the frames ``DroneState`` uses.
        """
        data = self._data
        quaternion = data.qpos[3:7].copy()
        target_offset = _world_to_body(quaternion, self._marker - data.qpos[:3])
        return DroneState(
            position=data.qpos[:3].copy(),
            velocity=data.qvel[:3].copy(),
            quaternion=quaternion,
            angular_velocity=self._gyro.data.copy(),
            specific_force=self._accelerometer.data.copy(),
            target_offset=target_offset,
            target_visible=self._in_camera_view(target_offset),
        )

    def _in_camera_view(self, offset: np.ndarray) -> bool:
        """Return whether a body-frame point falls inside the camera's frustum.

        This uses the body frame, so tilting the vehicle swings the view: a marker
        directly below can leave the frame during hard manoeuvres.
        """
        cfg = self.config.camera
        if not cfg.enabled or _norm(offset) > cfg.max_range_m:
            return False
        tilt = cfg.mount_tilt_rad + self.domain.camera_tilt_error_rad
        depth = math.sin(tilt) * offset[0] + math.cos(tilt) * offset[2]
        along = math.cos(tilt) * offset[0] - math.sin(tilt) * offset[2]
        return bool(
            depth > 0.0
            and abs(offset[1]) < depth * math.tan(cfg.horizontal_fov_rad / 2.0)
            and abs(along) < depth * math.tan(cfg.vertical_fov_rad / 2.0)
        )

    # ------------------------------------------------------------------
    # Simulated flight controller
    # ------------------------------------------------------------------

    def _velocity_to_attitude(
        self, velocity_cmd: np.ndarray, dt: float
    ) -> tuple[float, float, float]:
        """PI velocity loop: return roll, pitch and total thrust in newtons."""
        cfg = self.config
        gains = cfg.controller
        data = self._data

        error = velocity_cmd - data.qvel[:3]
        self._velocity_integral = np.clip(
            self._velocity_integral + gains.velocity_i * error * dt,
            -gains.max_velocity_integral,
            gains.max_velocity_integral,
        )
        accel_north, accel_east, accel_down = (
            gains.velocity_p * error + self._velocity_integral
        )

        # Rotate the horizontal demand into the heading frame, then tilt towards
        # it. In FRD, accelerating forward takes a negative pitch.
        roll, pitch, yaw = _quaternion_to_euler(data.qpos[3:7])
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        accel_forward = cos_yaw * accel_north + sin_yaw * accel_east
        accel_right = cos_yaw * accel_east - sin_yaw * accel_north
        limit = cfg.max_attitude_cmd_rad
        pitch_cmd = float(
            np.clip(-math.atan2(accel_forward, self._gravity), -limit, limit)
        )
        roll_cmd = float(
            np.clip(math.atan2(accel_right, self._gravity), -limit, limit)
        )

        # Divide by the tilt so the vertical component of thrust stays correct.
        tilt_factor = max(math.cos(roll) * math.cos(pitch), 0.5)
        thrust = self._nominal_mass * (self._gravity - accel_down) / tilt_factor
        return roll_cmd, pitch_cmd, float(thrust)

    def _attitude_to_motors(
        self, roll_cmd: float, pitch_cmd: float, yaw_rate_cmd: float, thrust: float
    ) -> list[float]:
        """Attitude P and rate PI loops, then the mixer: return motor thrusts."""
        gains = self.config.controller
        data = self._data

        roll, pitch, _ = _quaternion_to_euler(data.qpos[3:7].tolist())
        attitude_p = gains.attitude_p * self.domain.attitude_gain_scale
        rate_x, rate_y, rate_z = data.qvel[3:6].tolist()
        rate_error = (
            attitude_p * (roll_cmd - roll) - rate_x,
            attitude_p * (pitch_cmd - pitch) - rate_y,
            yaw_rate_cmd - rate_z,
        )
        limit = gains.max_rate_integral
        integral = [
            min(max(held + gains.rate_i * error * self._physics_dt, -limit), limit)
            for held, error in zip(self._rate_integral, rate_error)
        ]
        self._rate_integral = integral

        torque = [
            inertia * (gain * error + held)
            for inertia, gain, error, held in zip(
                self._inertia_values, gains.rate_p, rate_error, integral
            )
        ]
        # Yaw authority is weak, so an unlimited demand would saturate the motors
        # and take away the roll and pitch control that keeps the vehicle upright.
        yaw_limit = gains.max_yaw_torque_nm
        torque[2] = min(max(torque[2], -yaw_limit), yaw_limit)
        motors = (self._mixer @ np.array([thrust, *torque])).tolist()

        # Attitude takes priority over thrust, as in ArduPilot: shift all motors
        # together to stay in range, because clipping them one by one would
        # remove the thrust differences that hold attitude.
        raise_by = max(-min(motors), 0.0)
        motors = [motor + raise_by for motor in motors]
        lower_by = max(max(motors) - self._max_motor_thrust, 0.0)
        return [motor - lower_by for motor in motors]

    # ------------------------------------------------------------------
    # Domain randomization (reset)
    # ------------------------------------------------------------------

    def _sample_domain_params(self) -> DomainParams:
        """Draw the hidden parameters for one episode."""
        cfg = self.config.randomization
        if not cfg.enabled:
            return DomainParams()

        rng = self.np_random
        faulty = rng.random() < cfg.motor_fault_prob
        wind_speed = rng.uniform(*cfg.wind_speed_mps)
        wind_heading = rng.uniform(0.0, 2.0 * np.pi)
        full = DomainParams(
            mass_kg=float(rng.uniform(*cfg.mass_kg)),
            inertia_scale=float(rng.uniform(*cfg.inertia_scale)),
            drag_scale=float(rng.uniform(*cfg.drag_scale)),
            motor_lag_scale=float(rng.uniform(*cfg.motor_lag_scale)),
            attitude_gain_scale=float(rng.uniform(*cfg.attitude_gain_scale)),
            hover_thrust=float(rng.uniform(*cfg.hover_thrust)),
            com_offset_m=rng.normal(0.0, cfg.com_offset_std_m, size=3),
            wind_ned_mps=np.array(
                [
                    wind_speed * np.cos(wind_heading),
                    wind_speed * np.sin(wind_heading),
                    rng.uniform(*cfg.updraft_mps),
                ]
            ),
            wind_shear_exponent=float(rng.uniform(*cfg.wind_shear_exponent)),
            gust_std_mps=float(rng.uniform(*cfg.gust_std_mps)),
            push_rate_hz=float(rng.uniform(*cfg.push_rate_hz)),
            push_force_n=float(rng.uniform(*cfg.push_force_n)),
            battery_sag=float(rng.uniform(*cfg.battery_sag)),
            motor_fault_loss=(
                float(rng.uniform(*cfg.motor_fault_loss)) if faulty else 0.0
            ),
            # Drawn per motor, so that thrust asymmetry is represented too.
            motor_efficiency=rng.uniform(*cfg.motor_efficiency, size=4),
            noise_scale=float(rng.uniform(*cfg.noise_scale)),
            gyro_bias_rps=rng.normal(0.0, cfg.gyro_bias_std_rps, size=3),
            attitude_error_rad=rng.normal(
                0.0,
                [
                    cfg.trim_error_std_rad,
                    cfg.trim_error_std_rad,
                    cfg.heading_error_std_rad,
                ],
            ),
            gps_update_hz=float(rng.uniform(*cfg.gps_update_hz)),
            camera_tilt_error_rad=float(rng.normal(0.0, cfg.camera_tilt_error_std_rad)),
            camera_frame_hz=float(rng.uniform(*cfg.camera_frame_hz)),
            camera_delay_steps=self._sample_delay(cfg.camera_delay_steps),
            action_delay_steps=self._sample_delay(cfg.action_delay_steps),
            gps_delay_steps=self._sample_delay(cfg.gps_delay_steps),
            imu_delay_steps=self._sample_delay(cfg.imu_delay_steps),
            link_jitter_steps=self._sample_delay(cfg.link_jitter_steps),
            link_drop_prob=float(rng.uniform(*cfg.link_drop_prob)),
        )
        return _blend_domains(DomainParams(noise_scale=0.0), full, self.difficulty)

    def _sample_delay(self, delay_range: tuple[int, int]) -> int:
        low, high = delay_range
        return int(self.np_random.integers(low, high + 1))

    # ------------------------------------------------------------------
    # Sensor noise (step)
    # ------------------------------------------------------------------

    def _reset_sensor_errors(self) -> None:
        """Draw the slow errors from their steady state, as in a running EKF."""
        cfg = self.config.noise
        rng = self.np_random
        scale = self.domain.noise_scale
        self._gyro_bias = self.domain.gyro_bias_rps.copy()
        self._gps_position_error = (
            rng.normal(0.0, self._drift_std() * scale) if cfg.enabled else np.zeros(3)
        )
        self._gps_velocity_error = rng.normal(0.0, cfg.gps_velocity_std_mps * scale, 3)
        self._gps_glitch = np.zeros(3)
        self._camera_frame = np.zeros(4)

        trim_roll, trim_pitch, heading = self.domain.attitude_error_rad
        self._trim_error = _euler_to_quaternion(trim_roll, trim_pitch, 0.0)
        self._heading_error = _euler_to_quaternion(0.0, 0.0, heading)

    def _drift_std(self) -> np.ndarray:
        cfg = self.config.noise
        return np.array([cfg.gps_drift_std_m, cfg.gps_drift_std_m, cfg.baro_drift_std_m])

    def _apply_sensor_noise(self, state: DroneState) -> np.ndarray:
        """Turn ground truth into the 17-value sensor vector the policy sees."""
        if self.config.noise.enabled:
            quaternion, gyro = self._read_imu(state)
            position, velocity = self._read_gps(state)
        else:
            quaternion, gyro = state.quaternion, state.angular_velocity
            position, velocity = state.position, state.velocity

        # q and -q are the same attitude. The flight node sends w >= 0, so the
        # policy must never have to learn the other half of the sphere.
        if quaternion[0] < 0.0:
            quaternion = -quaternion

        return np.concatenate(
            [
                position,
                velocity,
                quaternion,
                gyro,
                self._camera_slots(state),
            ],
            dtype=np.float32,
        )

    def _read_imu(self, state: DroneState) -> tuple[np.ndarray, np.ndarray]:
        """Return attitude and body rates with a wandering bias and vibration.

        The attitude also carries a fixed trim and heading error. A heading error
        rotates the waypoint offset, because the policy can only turn that offset
        into the body frame with the attitude it believes.
        """
        cfg = self.config.noise
        rng = self.np_random
        scale = self.domain.noise_scale
        self._gyro_bias = self._gyro_bias + rng.normal(
            0.0, cfg.gyro_bias_walk_rps * self.difficulty * math.sqrt(self.dt), size=3
        )
        motors = self._data.act.tolist()
        throttle = sum(motors) / len(motors) / self._max_motor_thrust
        gyro_std = scale * (cfg.gyro_std_rps + cfg.gyro_vibration_std_rps * throttle)
        gyro = (
            state.angular_velocity
            + self._gyro_bias
            + rng.normal(0.0, gyro_std, size=3)
        )
        quaternion = _quaternion_multiply(
            self._heading_error, _quaternion_multiply(state.quaternion, self._trim_error)
        )
        quaternion = quaternion + rng.normal(0.0, cfg.attitude_std * scale, size=4)
        return quaternion / _norm(quaternion), gyro

    def _read_gps(self, state: DroneState) -> tuple[np.ndarray, np.ndarray]:
        """Return position and velocity as the autopilot's EKF reports them.

        The EKF propagates on the IMU between GPS fixes, so its output keeps moving
        at the control rate while its error only changes at each fix. Holding the
        error, rather than the position, is what makes that a fix rate and not a
        stale estimate.
        """
        cfg = self.config.noise
        rng = self.np_random
        scale = self.domain.noise_scale
        fix_steps = max(round(self.config.control_hz / self.domain.gps_update_hz), 1)
        if self._step_count % fix_steps == 0:
            fix_s = fix_steps * self.dt
            decay = math.exp(-fix_s / cfg.gps_drift_time_constant_s)
            self._gps_position_error = decay * self._gps_position_error + rng.normal(
                0.0, self._drift_std() * scale * math.sqrt(1.0 - decay * decay)
            )
            self._gps_velocity_error = rng.normal(
                0.0, cfg.gps_velocity_std_mps * scale, size=3
            )
            # A glitch is a sudden jump that the EKF then pulls back out.
            self._gps_glitch = self._gps_glitch * math.exp(
                -fix_s / cfg.gps_glitch_time_constant_s
            )
            if rng.random() < cfg.gps_glitch_rate_hz * fix_s * self.difficulty:
                direction = rng.normal(size=3)
                self._gps_glitch = (
                    cfg.gps_glitch_m * direction / np.linalg.norm(direction)
                )
        position = (
            state.position
            + self._gps_position_error
            + self._gps_glitch
            + rng.normal(0.0, cfg.gps_position_std_m * scale, size=3)
        )
        return position, state.velocity + self._gps_velocity_error

    def _camera_slots(self, state: DroneState) -> np.ndarray:
        """Return the four camera values of the sensor vector."""
        if self.config.noise.enabled:
            offset, visible = self._read_camera(state)
        else:
            offset, visible = state.target_offset, state.target_visible
        return np.array([*offset, float(visible)])

    def _on_sensors(self, sensors: np.ndarray) -> None:
        """Called with what the policy receives, before termination is checked."""

    def _read_camera(self, state: DroneState) -> tuple[np.ndarray, bool]:
        """Return the newest camera result, which only changes on each frame."""
        frame_steps = max(round(self.config.control_hz / self.domain.camera_frame_hz), 1)
        if self._step_count % frame_steps == 0:
            offset, visible = self._detect_marker(state)
            self._camera_frame = np.array([*offset, float(visible)])
        return self._camera_frame[:3], bool(self._camera_frame[3])

    def _detect_marker(self, state: DroneState) -> tuple[np.ndarray, bool]:
        """Process one frame: range-dependent error, blur, dropouts, false alarms."""
        cfg = self.config.noise
        camera = self.config.camera
        rng = self.np_random
        scale = self.domain.noise_scale

        if camera.enabled and rng.random() < cfg.vision_false_positive_prob * self.difficulty:
            # Something else on the ground looks like the marker.
            half_width = math.tan(camera.horizontal_fov_rad / 2.0)
            half_height = math.tan(camera.vertical_fov_rad / 2.0)
            depth = rng.uniform(2.0, camera.max_range_m)
            x = rng.uniform(-half_height, half_height) * depth
            y = rng.uniform(-half_width, half_width) * depth
            return np.array([x, y, depth]), True

        if not state.target_visible:
            return np.zeros(3), False

        distance = _norm(state.target_offset)
        fade_start = camera.fade_start * camera.max_range_m
        fade = (camera.max_range_m - distance) / (camera.max_range_m - fade_start)
        blur = _norm(state.angular_velocity) / cfg.vision_blur_rate_rps
        detect_prob = (
            (1.0 - cfg.vision_dropout_prob) * min(fade, 1.0) * math.exp(-blur * blur)
        )
        if rng.random() >= 1.0 - self.difficulty * (1.0 - detect_prob):
            return np.zeros(3), False

        direction = state.target_offset / max(distance, 1e-6)
        lateral = rng.normal(0.0, cfg.vision_angle_std_rad * distance * scale, size=3)
        along = rng.normal(0.0, cfg.vision_range_std * distance * scale)
        return state.target_offset + lateral + along * direction, True

    # ------------------------------------------------------------------
    # Hardware communication latency (step)
    # ------------------------------------------------------------------

    def _make_link(self, initial: np.ndarray, delay_steps: int) -> DelayLine:
        return DelayLine(
            initial,
            delay_steps,
            self.domain.link_jitter_steps,
            self.domain.link_drop_prob,
        )

    # ------------------------------------------------------------------
    # Waypoints
    # ------------------------------------------------------------------

    def _next_waypoint(self, position: np.ndarray) -> np.ndarray:
        """Return a static waypoint near ``position``, inside the flight envelope."""
        cfg = self.config.waypoints
        rng = self.np_random
        if rng.random() < cfg.marker_prob:
            return np.array(
                [*self._marker[:2], -rng.uniform(*cfg.marker_altitude_m)]
            ).clip(-cfg.max_xy_m, cfg.max_xy_m)
        if rng.random() < cfg.hover_prob:
            offset = np.zeros(3)
        else:
            direction = rng.normal(size=3)
            offset = (
                rng.uniform(*cfg.distance_m) * direction / np.linalg.norm(direction)
            )
        waypoint = position + offset
        waypoint[:2] = np.clip(waypoint[:2], -cfg.max_xy_m, cfg.max_xy_m)
        waypoint[2] = -np.clip(-waypoint[2], *cfg.altitude_m)
        return waypoint

    def _update_hold(self, state: DroneState) -> bool:
        """Count the steps spent holding the waypoint; True once it is reached.

        Arrival is judged on the position estimate, because that is all the vehicle
        has. Judging it on true position would fail a perfect pilot whenever the
        drift alone exceeds the radius, which starves the policy of its arrival
        reward. The reward's proximity term still uses the truth.
        """
        cfg = self.config.waypoints
        estimate = state.position + self._gps_position_error + self._gps_glitch
        holding = (
            _norm(self._waypoint - estimate) < cfg.arrival_radius_m
            and _norm(state.velocity) < cfg.arrival_speed_mps
        )
        self._hold_steps = self._hold_steps + 1 if holding else 0
        if self._hold_steps < round(cfg.hold_s / self.dt):
            return False
        self._hold_steps = 0
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _action_to_command(self, action: np.ndarray) -> np.ndarray:
        """Scale a normalized action to the physical setpoint for this mode."""
        cfg = self.config
        if cfg.control_mode is ControlMode.VELOCITY:
            return action * np.array(
                [
                    cfg.max_horizontal_speed_mps,
                    cfg.max_horizontal_speed_mps,
                    cfg.max_vertical_speed_mps,
                    cfg.max_yaw_rate_rps,
                ]
            )

        # Thrust is piecewise linear so that 0 is hover and +/-1 are the limits,
        # then clamped as the flight node clamps it.
        thrust = np.interp(
            action[3], [-1.0, 0.0, 1.0], [0.0, self.domain.hover_thrust, 1.0]
        )
        thrust = min(max(thrust, cfg.min_thrust), cfg.max_thrust)
        return np.array(
            [
                action[0] * cfg.max_attitude_cmd_rad,
                action[1] * cfg.max_attitude_cmd_rad,
                action[2] * cfg.max_yaw_rate_rps,
                thrust,
            ]
        )

    def _termination_reason(self, state: DroneState) -> str:
        """Return which flight limit was broken, or "" if none was."""
        cfg = self.config
        if state.altitude < cfg.min_altitude_m:
            return "altitude"
        if max(map(abs, state.position.tolist())) > cfg.max_position_m:
            return "geofence"
        if _norm(state.velocity) > cfg.max_speed_mps:
            return "speed"
        if state.tilt > cfg.max_tilt_rad:
            return "tilt"
        return ""

    def _make_observation_space(self) -> spaces.Box:
        cfg = self.config
        high = np.concatenate(
            [
                np.full(3, cfg.max_position_m),
                np.full(3, cfg.sensor_max_speed_mps),
                np.ones(4),
                np.full(3, cfg.max_angular_rate_rps),
                np.full(3, cfg.max_vision_range_m),
                np.ones(1),
                np.ones(ACTION_DIM),
                np.full(WAYPOINT_DIM, cfg.waypoints.max_offset_m),
            ],
            dtype=np.float32,
        )
        low = -high
        low[OBS_TARGET_VISIBLE] = 0.0
        return spaces.Box(low, high, dtype=np.float32)

    def _make_observation(self, sensors: np.ndarray) -> np.ndarray:
        """Append the previous action and waypoint, and saturate to the space.

        The offset comes from the sensed, delayed pose, as it does on the vehicle.
        """
        position, attitude = sensors[:3], sensors[GPS_DIM : GPS_DIM + 4]
        offset = _world_to_body(attitude, self._waypoint - position)
        observation = np.concatenate([sensors, self._last_action, offset])
        return np.clip(
            observation, self.observation_space.low, self.observation_space.high
        ).astype(np.float32)

    def _make_info(self, state: DroneState, termination: str = "") -> dict[str, Any]:
        """Privileged data for logging and asymmetric critics, not for the actor."""
        return {
            "true_state": _fields_to_dict(state),
            "domain": _fields_to_dict(self.domain),
            "step": self._step_count,
            "waypoint": self._waypoint.copy(),
            "waypoints_reached": self._waypoints_reached,
            "termination": termination,
            "is_success": self._waypoints_reached > 0 and not termination,
        }
