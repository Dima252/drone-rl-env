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

Observation (21 values, float32):

    ======  ==================  =====================================
    Index   Field               Source
    ======  ==================  =====================================
    0-2     position (m)        GPS
    3-5     velocity (m/s)      GPS
    6-9     attitude quaternion IMU
    10-12   body rates (rad/s)  IMU
    13-15   target offset (m)   Vision, body frame
    16      target visible      Vision, 0 or 1
    17-20   previous action     Policy, undelayed
    ======  ==================  =====================================

The previous action is part of the observation because the policy cannot infer
the commands still in flight under latency without it. The sandbox has no vision
target, so indices 13-16 are always zero.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, SupportsFloat

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ACTION_DIM = 4
SENSOR_DIM = 17
OBS_DIM = SENSOR_DIM + ACTION_DIM
OBS_TARGET_VISIBLE = 16

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
        _, x, y, _ = self.quaternion
        return float(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


# (state, action, next_state, info) -> reward. Reward is computed from ground
# truth, which is available in simulation and is only needed during training.
RewardFunc = Callable[
    [DroneState, np.ndarray, DroneState, dict[str, Any]], SupportsFloat
]


@dataclass(frozen=True, eq=False)
class DomainParams:
    """Hidden parameters of one episode. The defaults are the nominal vehicle."""

    mass_kg: float = 1.5
    wind_ned_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    motor_efficiency: np.ndarray = field(default_factory=lambda: np.ones(4))
    gyro_bias_rps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    action_delay_steps: int = 0
    obs_delay_steps: int = 0


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Ranges sampled on every reset. Disable for nominal-vehicle evaluation."""

    enabled: bool = True
    mass_kg: tuple[float, float] = (1.2, 1.8)
    wind_speed_mps: tuple[float, float] = (0.0, 5.0)
    motor_efficiency: tuple[float, float] = (0.9, 1.0)
    gyro_bias_std_rps: float = 0.01
    action_delay_steps: tuple[int, int] = (0, 3)
    obs_delay_steps: tuple[int, int] = (0, 2)


@dataclass(frozen=True)
class SensorNoiseConfig:
    """Per-step noise standard deviations. Placeholders until bench-measured."""

    enabled: bool = True
    gyro_std_rps: float = 0.02
    attitude_std: float = 0.005
    gps_position_std_m: float = 0.3
    gps_velocity_std_mps: float = 0.1
    vision_std_m: float = 0.05
    vision_dropout_prob: float = 0.05


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
class DroneEnvConfig:
    """Everything that is shared across drones and independent of the reward."""

    control_mode: ControlMode = ControlMode.VELOCITY
    control_hz: float = 50.0
    max_episode_steps: int = 1000
    model_path: Path = DEFAULT_MODEL_PATH

    # Initial pose: uniform over this box, with a uniform random heading.
    spawn_xy_range_m: float = 5.0
    spawn_altitude_m: tuple[float, float] = (5.0, 15.0)

    # Command limits: what a normalized action of +/-1 means.
    max_horizontal_speed_mps: float = 5.0
    max_vertical_speed_mps: float = 2.0
    max_yaw_rate_rps: float = 1.0
    max_attitude_cmd_rad: float = 0.35
    hover_thrust: float = 0.5

    # Episode ends beyond these. max_position_m also bounds the observation.
    max_position_m: float = 100.0
    max_tilt_rad: float = 1.2

    # Sensor saturation limits.
    max_speed_mps: float = 20.0
    max_angular_rate_rps: float = 10.0
    max_vision_range_m: float = 50.0

    randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    noise: SensorNoiseConfig = field(default_factory=SensorNoiseConfig)
    controller: FlightControllerConfig = field(
        default_factory=FlightControllerConfig
    )


def _quaternion_to_euler(quaternion: np.ndarray) -> tuple[float, float, float]:
    """Return roll, pitch and yaw in radians (aerospace ZYX order)."""
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class BaseDroneEnv(gym.Env):
    """Shared drone environment with an injected reward function.

    Args:
        reward_func: Called as ``reward_func(state, action, next_state, info)``
            after every step. ``info`` holds ``prev_action``, ``terminated`` and
            ``truncated`` in addition to the keys returned from ``step``.
        config: Shared environment settings. Defaults to ``DroneEnvConfig()``.
        render_mode: Must be ``None``; no renderer exists yet.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_func: RewardFunc,
        config: DroneEnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not callable(reward_func):
            raise TypeError(f"reward_func must be callable, got {reward_func!r}")

        self.reward_func = reward_func
        self.config = config or DroneEnvConfig()
        self.render_mode = render_mode
        self.dt = 1.0 / self.config.control_hz

        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.observation_space = self._make_observation_space()

        self.domain = DomainParams()
        self._state: DroneState | None = None
        self._step_count = 0
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._action_queue: deque[np.ndarray] = deque()
        self._sensor_queue: deque[np.ndarray] = deque()

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
        self._gravity = float(model.opt.gravity[2])
        self._max_motor_thrust = float(model.actuator_ctrlrange[0, 1])
        self._gyro = self._data.sensor("gyro")
        self._accelerometer = self._data.sensor("accelerometer")

        # Mixer: invert the map from motor thrusts to [thrust, torque xyz], built
        # from the model so that motor layout and spin directions live in one place.
        rotor_positions = model.site_pos[model.actuator_trnid[:, 0]]
        gears = model.actuator_gear
        torques = np.cross(rotor_positions, gears[:, :3]) + gears[:, 3:]
        self._mixer = np.linalg.inv(np.vstack([-gears[:, 2], torques.T]))

        self._velocity_integral = np.zeros(3)
        self._rate_integral = np.zeros(3)

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

        # Prefill the latency queues so a delay of N is in effect from step one.
        neutral = np.zeros(ACTION_DIM, dtype=np.float32)
        sensors = self._apply_sensor_noise(self._state)
        self._action_queue = self._make_delay_queue(
            neutral, self.domain.action_delay_steps
        )
        self._sensor_queue = self._make_delay_queue(
            sensors, self.domain.obs_delay_steps
        )

        return self._make_observation(sensors), self._make_info(self._state)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        if not np.all(np.isfinite(action)):
            raise ValueError(f"Action contains non-finite values: {action}")
        action = np.clip(action, -1.0, 1.0)

        # Uplink latency: the vehicle executes a command issued earlier.
        delayed_action = self._delay(self._action_queue, action)
        command = self._action_to_command(delayed_action)

        state = self._state
        next_state = self._physics_step(command, self.dt)
        self._state = next_state
        self._step_count += 1

        # Sensor noise, then downlink latency: the policy sees stale, noisy data.
        sensors = self._delay(
            self._sensor_queue, self._apply_sensor_noise(next_state)
        )

        terminated = self._check_terminated(next_state)
        truncated = self._step_count >= self.config.max_episode_steps

        info = self._make_info(next_state)
        reward_info = {
            **info,
            "prev_action": self._last_action,
            "terminated": terminated,
            "truncated": truncated,
        }
        reward = float(self.reward_func(state, action, next_state, reward_info))

        self._last_action = action
        return self._make_observation(sensors), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # MuJoCo physics
    # ------------------------------------------------------------------

    def _physics_reset(self, domain: DomainParams) -> DroneState:
        """Restart the simulation with ``domain`` applied; return the first state."""
        model, data = self._model, self._data

        # Inertia scales with mass, so the airframe keeps its shape.
        model.body_mass[self._body_id] = domain.mass_kg
        model.body_inertia[self._body_id] = self._nominal_inertia * (
            domain.mass_kg / self._nominal_mass
        )
        model.actuator_gainprm[:, 0] = domain.motor_efficiency
        model.opt.wind[:] = domain.wind_ned_mps
        mujoco.mj_setConst(model, data)
        mujoco.mj_resetData(model, data)

        cfg = self.config
        rng = self.np_random
        heading = rng.uniform(-np.pi, np.pi)
        data.qpos[:2] = rng.uniform(-cfg.spawn_xy_range_m, cfg.spawn_xy_range_m, 2)
        data.qpos[2] = -rng.uniform(*cfg.spawn_altitude_m)
        data.qpos[3:7] = [math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)]

        # Start in flight with the motors already spinning at nominal hover.
        data.act[:] = self._nominal_mass * self._gravity / 4.0
        self._velocity_integral = np.zeros(3)
        self._rate_integral = np.zeros(3)

        mujoco.mj_forward(model, data)
        return self._read_state()

    def _physics_step(self, command: np.ndarray, dt: float) -> DroneState:
        """Advance the simulation by ``dt`` seconds and return the new state.

        ``command`` holds physical units from ``_action_to_command``.
        """
        model, data = self._model, self._data

        if self.config.control_mode is ControlMode.VELOCITY:
            roll, pitch, thrust = self._velocity_to_attitude(command[:3], dt)
            yaw_rate = command[3]
        else:
            roll, pitch, yaw_rate = command[:3]
            thrust = command[3] * 4.0 * self._max_motor_thrust

        for _ in range(self._substeps):
            data.ctrl[:] = self._attitude_to_motors(roll, pitch, yaw_rate, thrust)
            mujoco.mj_step(model, data)

        # mj_step leaves sensors one substep behind the integrated state.
        mujoco.mj_forward(model, data)
        return self._read_state()

    def _read_state(self) -> DroneState:
        """Copy the true state out of MuJoCo.

        A free joint stores linear velocity in the world frame and angular
        velocity in the body frame, which are the frames ``DroneState`` uses.
        """
        data = self._data
        return DroneState(
            position=data.qpos[:3].copy(),
            velocity=data.qvel[:3].copy(),
            quaternion=data.qpos[3:7].copy(),
            angular_velocity=self._gyro.data.copy(),
            specific_force=self._accelerometer.data.copy(),
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
        return roll_cmd, pitch_cmd, thrust

    def _attitude_to_motors(
        self, roll_cmd: float, pitch_cmd: float, yaw_rate_cmd: float, thrust: float
    ) -> np.ndarray:
        """Attitude P and rate PI loops, then the mixer: return motor thrusts."""
        gains = self.config.controller
        data = self._data

        roll, pitch, _ = _quaternion_to_euler(data.qpos[3:7])
        rate_cmd = np.array(
            [
                gains.attitude_p * (roll_cmd - roll),
                gains.attitude_p * (pitch_cmd - pitch),
                yaw_rate_cmd,
            ]
        )
        rate_error = rate_cmd - data.qvel[3:6]
        self._rate_integral = np.clip(
            self._rate_integral
            + gains.rate_i * rate_error * self._model.opt.timestep,
            -gains.max_rate_integral,
            gains.max_rate_integral,
        )

        torque = self._nominal_inertia * (
            np.array(gains.rate_p) * rate_error + self._rate_integral
        )
        # Yaw authority is weak, so an unlimited demand would saturate the motors
        # and take away the roll and pitch control that keeps the vehicle upright.
        torque[2] = np.clip(
            torque[2], -gains.max_yaw_torque_nm, gains.max_yaw_torque_nm
        )
        motors = self._mixer @ np.array([thrust, *torque])

        # Attitude takes priority over thrust, as in ArduPilot: shift all motors
        # together to stay in range, because clipping them one by one would
        # remove the thrust differences that hold attitude.
        motors += max(-motors.min(), 0.0)
        motors -= max(motors.max() - self._max_motor_thrust, 0.0)
        return motors

    # ------------------------------------------------------------------
    # Domain randomization (reset)
    # ------------------------------------------------------------------

    def _sample_domain_params(self) -> DomainParams:
        """Draw the hidden parameters for one episode."""
        cfg = self.config.randomization
        if not cfg.enabled:
            return DomainParams()

        rng = self.np_random
        # TODO: add gusts as a time-varying process on top of this steady wind.
        wind_speed = rng.uniform(*cfg.wind_speed_mps)
        wind_heading = rng.uniform(0.0, 2.0 * np.pi)
        return DomainParams(
            mass_kg=float(rng.uniform(*cfg.mass_kg)),
            wind_ned_mps=np.array(
                [
                    wind_speed * np.cos(wind_heading),
                    wind_speed * np.sin(wind_heading),
                    0.0,
                ]
            ),
            # Drawn per motor, so that thrust asymmetry is represented too.
            motor_efficiency=rng.uniform(*cfg.motor_efficiency, size=4),
            gyro_bias_rps=rng.normal(0.0, cfg.gyro_bias_std_rps, size=3),
            action_delay_steps=self._sample_delay(cfg.action_delay_steps),
            obs_delay_steps=self._sample_delay(cfg.obs_delay_steps),
        )

    def _sample_delay(self, delay_range: tuple[int, int]) -> int:
        low, high = delay_range
        return int(self.np_random.integers(low, high + 1))

    # ------------------------------------------------------------------
    # Sensor noise (step)
    # ------------------------------------------------------------------

    def _apply_sensor_noise(self, state: DroneState) -> np.ndarray:
        """Turn ground truth into the 17-value sensor vector the policy sees."""
        if self.config.noise.enabled:
            quaternion, gyro = self._read_imu(state)
            position, velocity = self._read_gps(state)
            target_offset, target_visible = self._read_vision(state)
        else:
            quaternion, gyro = state.quaternion, state.angular_velocity
            position, velocity = state.position, state.velocity
            target_offset, target_visible = state.target_offset, state.target_visible

        return np.concatenate(
            [
                position,
                velocity,
                quaternion,
                gyro,
                target_offset,
                [float(target_visible)],
            ],
            dtype=np.float32,
        )

    def _read_imu(self, state: DroneState) -> tuple[np.ndarray, np.ndarray]:
        """Return attitude and body rates with bias and white noise."""
        # TODO: replace with a calibrated model (bias random walk, vibration).
        cfg = self.config.noise
        rng = self.np_random
        gyro = (
            state.angular_velocity
            + self.domain.gyro_bias_rps
            + rng.normal(0.0, cfg.gyro_std_rps, size=3)
        )
        quaternion = state.quaternion + rng.normal(0.0, cfg.attitude_std, size=4)
        return quaternion / np.linalg.norm(quaternion), gyro

    def _read_gps(self, state: DroneState) -> tuple[np.ndarray, np.ndarray]:
        """Return position and velocity with white noise."""
        # TODO: model the real update rate (5-10 Hz, held between fixes) and
        # the slowly drifting position error.
        cfg = self.config.noise
        rng = self.np_random
        position = state.position + rng.normal(0.0, cfg.gps_position_std_m, size=3)
        velocity = state.velocity + rng.normal(0.0, cfg.gps_velocity_std_mps, size=3)
        return position, velocity

    def _read_vision(self, state: DroneState) -> tuple[np.ndarray, bool]:
        """Return the target offset with white noise and random dropouts."""
        # TODO: scale noise with range and model field-of-view limits.
        cfg = self.config.noise
        rng = self.np_random
        if not state.target_visible or rng.random() < cfg.vision_dropout_prob:
            return np.zeros(3), False
        return state.target_offset + rng.normal(0.0, cfg.vision_std_m, size=3), True

    # ------------------------------------------------------------------
    # Hardware communication latency (step)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_delay_queue(initial: np.ndarray, delay_steps: int) -> deque[np.ndarray]:
        return deque([initial] * (delay_steps + 1), maxlen=delay_steps + 1)

    @staticmethod
    def _delay(queue: deque[np.ndarray], value: np.ndarray) -> np.ndarray:
        """Push ``value`` and return the one from ``delay_steps`` steps ago."""
        # TODO: add per-step jitter and dropped packets, and separate delays per
        # sensor once the real telemetry link has been measured.
        queue.append(value)
        return queue[0]

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

        # Thrust is piecewise linear so that 0 is hover and +/-1 are the limits.
        thrust = np.interp(action[3], [-1.0, 0.0, 1.0], [0.0, cfg.hover_thrust, 1.0])
        return np.array(
            [
                action[0] * cfg.max_attitude_cmd_rad,
                action[1] * cfg.max_attitude_cmd_rad,
                action[2] * cfg.max_yaw_rate_rps,
                thrust,
            ]
        )

    def _check_terminated(self, state: DroneState) -> bool:
        """Return True on a ground strike, a geofence breach or excessive tilt."""
        cfg = self.config
        return bool(
            state.altitude < 0.0
            or np.any(np.abs(state.position) > cfg.max_position_m)
            or state.tilt > cfg.max_tilt_rad
        )

    def _make_observation_space(self) -> spaces.Box:
        cfg = self.config
        high = np.concatenate(
            [
                np.full(3, cfg.max_position_m),
                np.full(3, cfg.max_speed_mps),
                np.ones(4),
                np.full(3, cfg.max_angular_rate_rps),
                np.full(3, cfg.max_vision_range_m),
                np.ones(1),
                np.ones(ACTION_DIM),
            ],
            dtype=np.float32,
        )
        low = -high
        low[OBS_TARGET_VISIBLE] = 0.0
        return spaces.Box(low, high, dtype=np.float32)

    def _make_observation(self, sensors: np.ndarray) -> np.ndarray:
        """Append the undelayed previous action and saturate to the space."""
        observation = np.concatenate([sensors, self._last_action])
        return np.clip(
            observation, self.observation_space.low, self.observation_space.high
        )

    def _make_info(self, state: DroneState) -> dict[str, Any]:
        """Privileged data for logging and asymmetric critics, not for the actor."""
        return {
            "true_state": asdict(state),
            "domain": asdict(self.domain),
            "step": self._step_count,
        }
