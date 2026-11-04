#!/usr/bin/env python3
"""Fly a trained attitude policy through MAVROS and ArduPilot (ROS 2 Humble).

The node never arms the vehicle or takes off. A pilot flies it to a safe altitude
and selects GUIDED, which hands control to the policy. Wait for the "Ready" log line
first: selecting GUIDED before it is ignored. Leaving GUIDED takes control back at
once. On any fault the node stops publishing and switches to
``fallback_mode``, and stays locked out until the pilot leaves GUIDED and selects
it again.

    ros2 run <package> ros2_inference_node.py --ros-args \\
        -p model_path:=model.onnx -p policy_meta_path:=policy_meta.npz \\
        -p hover_thrust:=0.42 -p hold_range_m:=10.0 -p hold_height_m:=4.0

The policy holds ``hold_range_m`` from a static, round marker at ``hold_height_m``,
facing it. It flies on the detector and the IMU alone. ``/mavros/local_position/odom`` never
reaches the network: it only feeds the flight limits, and ArduPilot needs it for the
``LOITER`` fallback. Without GPS, set ``fallback_mode`` to a mode that does not need a
position, and ``required_mode`` to ``GUIDED_NOGPS``.

A detector must publish ``std_msgs/Float32MultiArray`` on ``marker_topic``, holding
``[cx, cy, w, h]`` in normalized image coordinates (x right and y down, -1 to 1 across
the frame, so 2 fills it), only when it sees the marker. Keep the real camera close
to ``MarkerTaskConfig`` in training.

The pilot must point the camera at the marker before selecting GUIDED, or the node
refuses to engage. If detections stop for ``lost_timeout_s``, the node commands level
flight at hover thrust and asks for ``fallback_mode``. A box wider than
``max_box_width`` means the vehicle is too close, and also ends policy control.

Inputs:

* ``model.onnx``: the deterministic actor. Input ``[1, 31]`` normalized observation
  in float32, output ``[1, 4]`` action.
* ``policy_meta.npz``: what training used, which the network file cannot carry.
  Holds ``control_mode`` (must be ``attitude``), ``control_hz`` (must equal
  ``rate_hz``), ``action_filter_alpha``, ``obs_mean``, ``obs_var``, ``epsilon`` and
  ``clip_obs`` from
  VecNormalize, ``obs_low`` and ``obs_high`` from the observation space, and
  ``max_attitude_cmd_rad`` and ``max_yaw_rate_rps``, and ``lost_timeout_s`` and
  ``max_box_width`` from the task. The camera values in it are for the SITL suite.

Vehicle setup. Each of these fails without an obvious error:

* ``GUID_OPTIONS`` bit 3 set, or ArduPilot reads thrust as a climb rate. The node
  checks this and will not engage without it.
* MAVROS started with ``apm_config.yaml`` (``thrust_scaling: 1.0``), or MAVROS
  silently drops every attitude message.
* Telemetry at 50 Hz or more for the link MAVROS uses: ``MAVn_EXTRA1`` for attitude
  and ``MAVn_POSITION`` for local position. Older firmware names these ``SRn_EXTRA1``
  and ``SRn_POSITION``. Slower streams trip the sensor timeout, and at the default
  of 0 there is no sensor data at all.
* ``hover_thrust`` set to the throttle the vehicle really hovers at, read from
  ``/mavros/vfr_hud`` in a steady LOITER. ``MOT_THST_HOVER`` can be out of date: in
  SITL it read 0.39 against a true 0.36, which alone caused a 1.6 m/s climb.
* ``GUID_TIMEOUT`` lowered to about 0.5 s, the time ArduPilot waits before levelling
  out if this node stops.

Frames: MAVROS uses ENU world and FLU body, while the policy was trained in NED
world and FRD body. All conversion happens here. Note that the linear velocity in
``/mavros/local_position/odom`` is in the body frame, not the world frame.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
import onnxruntime as ort
import rclpy
from mavros_msgs.msg import AttitudeTarget, State
from mavros_msgs.srv import SetMode
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

ACTION_DIM = 4
SENSOR_DIM = 17
BOX_DIM = 5
BOX_LAGS = (0, 5, 10, 15)  # Control steps back from now.
OBS_DIM = BOX_DIM * len(BOX_LAGS) + 2 + 3 + ACTION_DIM + 2

# 180 degree rotations relating the two conventions. Each is its own inverse, so
# the same conversion works in both directions.
ENU_TO_NED = np.array([0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0])
FLU_TO_FRD = np.array([0.0, 1.0, 0.0, 0.0])

THRUST_AS_THRUST_BIT = 1 << 3


# ----------------------------------------------------------------------
# Frame conversion. Quaternions are [w, x, y, z], rotating body to world.
# ----------------------------------------------------------------------


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def convert_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Convert an orientation between ENU/FLU and NED/FRD, in either direction."""
    converted = quaternion_multiply(
        quaternion_multiply(ENU_TO_NED, quaternion), FLU_TO_FRD
    )
    converted /= np.linalg.norm(converted)
    return -converted if converted[0] < 0.0 else converted


def rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a body-frame vector into the world frame."""
    w, u = quaternion[0], quaternion[1:]
    return vector + 2.0 * (w * np.cross(u, vector) + np.cross(u, np.cross(u, vector)))


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
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


def yaw_of(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def roll_pitch_of(quaternion: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


def tilt_of(quaternion: np.ndarray) -> float:
    _, x, y, _ = quaternion
    return math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def pack_sensors(imu: Imu, odom: Odometry) -> np.ndarray:
    """Pack MAVROS messages into observation indices 0-16, in NED and FRD."""
    position = odom.pose.pose.position
    twist = odom.twist.twist.linear
    gyro = imu.angular_velocity
    imu_q = imu.orientation
    odom_q = odom.pose.pose.orientation

    attitude = convert_quaternion(np.array([imu_q.w, imu_q.x, imu_q.y, imu_q.z]))
    # Rotate with the attitude that MAVROS used to put velocity in the body frame.
    odom_attitude = convert_quaternion(
        np.array([odom_q.w, odom_q.x, odom_q.y, odom_q.z])
    )
    velocity = rotate_vector(odom_attitude, np.array([twist.x, -twist.y, -twist.z]))

    return np.concatenate(
        [
            [position.y, position.x, -position.z],
            velocity,
            attitude,
            [gyro.x, -gyro.y, -gyro.z],
            np.zeros(4),  # Unused: the detector arrives separately.
        ],
        dtype=np.float32,
    )


def box_slots(message: Float32MultiArray | None, age_s: float, timeout_s: float) -> np.ndarray:
    """Return [cx, cy, w, h, visible] from the newest detection, or all zeros."""
    empty = np.zeros(BOX_DIM, dtype=np.float32)
    if message is None or age_s > timeout_s or len(message.data) != 4:
        return empty
    box = np.array(message.data, dtype=np.float32)
    if not np.all(np.isfinite(box)) or box[2] <= 0.0:
        return empty
    return np.append(box, np.float32(1.0))


def build_observation(
    boxes: list[np.ndarray],
    sensors: np.ndarray,
    prev_action: np.ndarray,
    hold_range_m: float,
    hold_height_m: float,
) -> np.ndarray:
    """Assemble the observation from the detector, the IMU and the mission only."""
    return np.concatenate(
        [
            *[boxes[-1 - lag] for lag in BOX_LAGS],
            roll_pitch_of(sensors[6:10]),
            sensors[10:13],
            prev_action,
            [hold_range_m, hold_height_m],
        ]
    )


def smooth_action(action: np.ndarray, previous: np.ndarray, alpha: float) -> np.ndarray:
    """Apply the action filter the policy trained with."""
    return (alpha * action + (1.0 - alpha) * previous).astype(np.float32)


# ----------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------


class Policy:
    """Observation normalization and ONNX inference, matching training exactly."""

    def __init__(self, model_path: str, meta_path: str) -> None:
        with np.load(meta_path, allow_pickle=False) as meta:
            control_mode = str(meta["control_mode"])
            if control_mode != "attitude":
                raise ValueError(
                    f"Policy was trained for {control_mode!r} actions; this node "
                    "only publishes attitude targets."
                )
            self.control_hz = float(meta["control_hz"])
            self.action_filter_alpha = float(meta["action_filter_alpha"])
            self.lost_timeout_s = float(meta["lost_timeout_s"])
            self.max_box_width = float(meta["max_box_width"])
            self.obs_mean = meta["obs_mean"].astype(np.float64)
            self.obs_scale = np.sqrt(meta["obs_var"].astype(np.float64) + meta["epsilon"])
            self.obs_low = meta["obs_low"].astype(np.float32)
            self.obs_high = meta["obs_high"].astype(np.float32)
            self.clip_obs = float(meta["clip_obs"])
            self.max_attitude_cmd_rad = float(meta["max_attitude_cmd_rad"])
            self.max_yaw_rate_rps = float(meta["max_yaw_rate_rps"])

        for array in (self.obs_mean, self.obs_scale, self.obs_low, self.obs_high):
            if array.shape != (OBS_DIM,):
                raise ValueError(f"Expected {OBS_DIM} values, got {array.shape}.")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # Lower jitter for a network this small.
        self.session = ort.InferenceSession(
            model_path, options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # Also warms up the session, so the first real step is not slow.
        action = self(np.zeros(OBS_DIM, dtype=np.float32))
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Model produced an invalid action: {action}")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Return the action for a raw observation, in [-1, 1]."""
        saturated = np.clip(observation, self.obs_low, self.obs_high)
        normalized = np.clip(
            (saturated - self.obs_mean) / self.obs_scale, -self.clip_obs, self.clip_obs
        )
        batch = normalized.astype(np.float32).reshape(1, OBS_DIM)
        action = self.session.run(None, {self.input_name: batch})[0].reshape(-1)
        return np.clip(action, -1.0, 1.0)


# ----------------------------------------------------------------------
# Safety
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FlightLimits:
    """Flight envelope. Leaving it ends policy control."""

    min_altitude_m: float
    max_position_m: float
    max_speed_mps: float
    max_tilt_rad: float


def find_fault(sensors: np.ndarray, limits: FlightLimits) -> str | None:
    """Return why the vehicle is outside its envelope, or None."""
    if not np.all(np.isfinite(sensors)):
        return "non-finite sensor data"
    position, velocity, attitude = sensors[:3], sensors[3:6], sensors[6:10]
    altitude = -float(position[2])
    speed = float(np.linalg.norm(velocity))
    tilt = tilt_of(attitude)
    if altitude < limits.min_altitude_m:
        return f"altitude {altitude:.2f} m below {limits.min_altitude_m:.2f} m"
    if np.max(np.abs(position)) > limits.max_position_m:
        return f"position {np.round(position, 1)} outside {limits.max_position_m} m"
    if speed > limits.max_speed_mps:
        return f"speed {speed:.1f} m/s above {limits.max_speed_mps:.1f} m/s"
    if tilt > limits.max_tilt_rad:
        return f"tilt {tilt:.2f} rad above {limits.max_tilt_rad:.2f} rad"
    return None


class Status(Enum):
    IDLE = "idle"  # Ready for the pilot to select GUIDED.
    ACTIVE = "active"  # The policy is flying.
    LOCKED = "locked"  # Silent until the pilot leaves GUIDED.


# ----------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------


class InferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("rl_inference_node")

        self.declare_parameter("model_path", "model.onnx")
        self.declare_parameter("policy_meta_path", "policy_meta.npz")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("required_mode", "GUIDED")
        self.declare_parameter("fallback_mode", "LOITER")  # Empty: GUID_TIMEOUT only.
        self.declare_parameter("hover_thrust", 0.5)
        self.declare_parameter("min_thrust", 0.1)
        self.declare_parameter("max_thrust", 0.9)
        self.declare_parameter("max_yaw_lead_rad", 0.5)
        self.declare_parameter("sensor_sync_slop_s", 0.02)
        self.declare_parameter("sensor_timeout_s", 0.1)
        self.declare_parameter("state_timeout_s", 2.5)
        self.declare_parameter("min_altitude_m", 1.0)
        self.declare_parameter("max_position_m", 50.0)
        self.declare_parameter("max_speed_mps", 15.0)
        self.declare_parameter("max_tilt_rad", 0.7)
        self.declare_parameter("hold_range_m", 10.0)
        self.declare_parameter("hold_height_m", 4.0)
        self.declare_parameter("marker_topic", "/vision/marker_box")
        self.declare_parameter("marker_timeout_s", 0.2)

        param = lambda name: self.get_parameter(name).value  # noqa: E731
        self.period_s = 1.0 / param("rate_hz")
        self.required_mode = param("required_mode")
        self.fallback_mode = param("fallback_mode")
        self.hover_thrust = param("hover_thrust")
        self.min_thrust = param("min_thrust")
        self.max_thrust = param("max_thrust")
        self.max_yaw_lead_rad = param("max_yaw_lead_rad")
        self.sensor_timeout_s = param("sensor_timeout_s")
        self.state_timeout_s = param("state_timeout_s")
        self.limits = FlightLimits(
            min_altitude_m=param("min_altitude_m"),
            max_position_m=param("max_position_m"),
            max_speed_mps=param("max_speed_mps"),
            max_tilt_rad=param("max_tilt_rad"),
        )
        self.marker_timeout_s = param("marker_timeout_s")
        self.hold_range_m = param("hold_range_m")
        self.hold_height_m = param("hold_height_m")
        if self.hold_height_m < self.limits.min_altitude_m:
            raise ValueError("hold_height_m is below min_altitude_m.")
        self.policy = Policy(param("model_path"), param("policy_meta_path"))
        if param("rate_hz") != self.policy.control_hz:
            raise ValueError(
                f"rate_hz is {param('rate_hz')} but the policy trained at "
                f"{self.policy.control_hz} Hz, so its dynamics would be wrong."
            )

        # Starting locked means a restart in mid-flight cannot seize control: the
        # pilot must be seen leaving GUIDED first.
        self.status = Status.LOCKED
        self.handover_pending = False
        self.last_handover_time = -math.inf
        self.thrust_mode_verified = False

        self.state: State | None = None
        self.state_time = -math.inf
        self.sensors: np.ndarray | None = None
        self.sensor_time = -math.inf
        self.marker: Float32MultiArray | None = None
        self.marker_time = -math.inf
        self.boxes = [np.zeros(BOX_DIM, dtype=np.float32)] * (BOX_LAGS[-1] + 1)
        self.levelling = False

        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.yaw_setpoint = 0.0
        self.last_tick_time: float | None = None
        self.overruns = 0

        # MAVROS publishes best-effort, so a reliable subscriber gets nothing.
        qos = qos_profile_sensor_data
        self.create_subscription(State, "/mavros/state", self.on_state, qos)
        self.create_subscription(
            Float32MultiArray, param("marker_topic"), self.on_marker, qos
        )
        self.imu_sub = Subscriber(self, Imu, "/mavros/imu/data", qos_profile=qos)
        self.odom_sub = Subscriber(
            self, Odometry, "/mavros/local_position/odom", qos_profile=qos
        )
        self.synchronizer = ApproximateTimeSynchronizer(
            [self.imu_sub, self.odom_sub],
            queue_size=10,
            slop=param("sensor_sync_slop_s"),
        )
        self.synchronizer.registerCallback(self.on_sensors)

        self.attitude_pub = self.create_publisher(
            AttitudeTarget, "/mavros/setpoint_raw/attitude", qos
        )
        self.set_mode_client = self.create_client(SetMode, "/mavros/set_mode")
        self.get_params_client = self.create_client(
            GetParameters, "/mavros/param/get_parameters"
        )

        self.params_timer = self.create_timer(2.0, self.poll_guid_options)

        # A steady clock keeps system time changes out of the control loop.
        self.control_timer = self.create_timer(
            self.period_s,
            self.on_control_tick,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        self.get_logger().info(
            f"Policy loaded. Waiting for the pilot to select {self.required_mode}."
        )

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def on_state(self, message: State) -> None:
        self.state = message
        self.state_time = time.monotonic()

    def on_marker(self, message: Float32MultiArray) -> None:
        self.marker = message
        self.marker_time = time.monotonic()

    def on_sensors(self, imu: Imu, odom: Odometry) -> None:
        """Synchronized callback: pack the 17 sensor values of the observation.

        Only the IMU part reaches the policy. Position and velocity feed the flight
        limits, in find_fault.
        """
        self.sensors = pack_sensors(imu, odom)
        self.sensor_time = time.monotonic()

    def poll_guid_options(self) -> None:
        """Check GUID_OPTIONS until bit 3 is confirmed set."""
        if not self.get_params_client.service_is_ready():
            return
        request = GetParameters.Request()
        request.names = ["GUID_OPTIONS"]
        self.get_params_client.call_async(request).add_done_callback(
            self.on_guid_options
        )

    def on_guid_options(self, future) -> None:
        values = future.result().values
        if not values or values[0].type == ParameterType.PARAMETER_NOT_SET:
            return  # MAVROS is still pulling parameters from the autopilot.

        value = values[0]
        options = (
            value.integer_value
            if value.type == ParameterType.PARAMETER_INTEGER
            else int(value.double_value)
        )
        if options & THRUST_AS_THRUST_BIT:
            self.thrust_mode_verified = True
            self.params_timer.cancel()
            self.get_logger().info("GUID_OPTIONS verified: thrust is thrust.")
        else:
            self.get_logger().error(
                f"GUID_OPTIONS={options} lacks bit 3. ArduPilot would read thrust as "
                "a climb rate, so the policy will not engage.",
                throttle_duration_sec=10.0,
            )

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def on_control_tick(self) -> None:
        now = time.monotonic()
        self.track_timing(now)

        state, sensors = self.state, self.sensors

        link_up = (
            state is not None
            and state.connected
            and now - self.state_time <= self.state_timeout_s
        )
        if not link_up:
            if self.status is Status.ACTIVE:
                self.lock_out("lost connection to the autopilot", handover=True)
            return

        if state.mode != self.required_mode or not state.armed:
            if self.status is Status.ACTIVE:
                self.get_logger().warning(
                    f"Pilot took control (mode {state.mode}, armed {state.armed})."
                )
            elif self.status is Status.LOCKED:
                self.get_logger().info(f"Ready: select {self.required_mode} to engage.")
            self.status = Status.IDLE
            self.handover_pending = False
            self.levelling = False
            return

        if self.status is Status.LOCKED:
            if self.levelling and sensors is not None:
                level = np.zeros(ACTION_DIM, dtype=np.float32)
                self.attitude_pub.publish(self.make_setpoint(level, sensors))
            self.retry_handover(now)
            return

        fault = self.find_any_fault(sensors, now - self.sensor_time)
        if fault is not None:
            # Before engaging, the pilot is still flying: only refuse control.
            self.lock_out(fault, handover=self.status is Status.ACTIVE)
            return

        marker_age = now - self.marker_time
        box = box_slots(self.marker, marker_age, self.marker_timeout_s)

        if self.status is Status.IDLE:
            if not box[4]:
                self.lock_out("no marker in view", handover=False)
                return
            self.boxes = [box] * (BOX_LAGS[-1] + 1)
            self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
            self.yaw_setpoint = yaw_of(sensors[6:10])
            self.status = Status.ACTIVE
            self.get_logger().info("Policy engaged.")

        if marker_age > self.policy.lost_timeout_s:
            self.levelling = True
            self.lock_out(f"marker lost for {marker_age:.2f} s", handover=True)
            return
        if box[2] > self.policy.max_box_width:
            self.levelling = True
            self.lock_out(f"too close to the marker: box width {box[2]:.2f}", handover=True)
            return

        self.boxes = self.boxes[1:] + [box]
        observation = build_observation(
            self.boxes, sensors, self.prev_action, self.hold_range_m, self.hold_height_m
        )
        action = self.policy(observation)
        if not np.all(np.isfinite(action)):
            self.lock_out("non-finite policy output", handover=True)
            return

        action = smooth_action(action, self.prev_action, self.policy.action_filter_alpha)
        self.attitude_pub.publish(self.make_setpoint(action, sensors))
        self.prev_action = action

    def find_any_fault(self, sensors: np.ndarray | None, age_s: float) -> str | None:
        if not self.thrust_mode_verified:
            return "GUID_OPTIONS bit 3 not verified"
        if sensors is None or age_s > self.sensor_timeout_s:
            return f"sensor data is {age_s:.3f} s old"
        return find_fault(sensors, self.limits)

    def make_setpoint(self, action: np.ndarray, sensors: np.ndarray) -> AttitudeTarget:
        """Scale an action as training did and express it in MAVROS frames."""
        policy = self.policy
        roll = float(action[0]) * policy.max_attitude_cmd_rad
        pitch = float(action[1]) * policy.max_attitude_cmd_rad
        yaw_rate = float(action[2]) * policy.max_yaw_rate_rps
        thrust = float(np.interp(action[3], [-1.0, 0.0, 1.0], [0.0, self.hover_thrust, 1.0]))

        # ArduPilot takes an absolute attitude but the policy commands a yaw rate,
        # so integrate a heading that may only lead the vehicle by a bounded amount.
        yaw = yaw_of(sensors[6:10])
        lead = wrap_angle(self.yaw_setpoint + yaw_rate * self.period_s - yaw)
        lead = max(-self.max_yaw_lead_rad, min(self.max_yaw_lead_rad, lead))
        self.yaw_setpoint = wrap_angle(yaw + lead)

        orientation = convert_quaternion(
            euler_to_quaternion(roll, pitch, self.yaw_setpoint)
        )

        message = AttitudeTarget()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        # Send all three body rates. ArduPilot rejects a mask that ignores only some
        # of them and holds position instead of following the policy.
        message.type_mask = 0
        message.orientation.w = float(orientation[0])
        message.orientation.x = float(orientation[1])
        message.orientation.y = float(orientation[2])
        message.orientation.z = float(orientation[3])
        message.body_rate.x = 0.0
        message.body_rate.y = 0.0
        message.body_rate.z = -yaw_rate  # FRD yaw rate, expressed in FLU.
        message.thrust = max(self.min_thrust, min(self.max_thrust, thrust))
        return message

    def lock_out(self, reason: str, handover: bool) -> None:
        self.status = Status.LOCKED
        self.handover_pending = handover and bool(self.fallback_mode)
        self.last_handover_time = -math.inf
        self.get_logger().error(
            f"Policy not in control: {reason}. Leave {self.required_mode} to reset."
        )
        self.retry_handover(time.monotonic())

    def retry_handover(self, now: float) -> None:
        """Ask for the fallback mode once a second until the mode changes."""
        if not self.handover_pending or now - self.last_handover_time < 1.0:
            return
        if not self.set_mode_client.service_is_ready():
            self.get_logger().error("Set-mode service unavailable.")
            return
        self.last_handover_time = now
        request = SetMode.Request()
        request.custom_mode = self.fallback_mode
        self.set_mode_client.call_async(request)
        self.get_logger().warning(f"Requesting {self.fallback_mode}.")

    def track_timing(self, now: float) -> None:
        """Report late ticks. Python on stock Linux cannot guarantee the period."""
        if self.last_tick_time is not None:
            if now - self.last_tick_time > 1.5 * self.period_s:
                self.overruns += 1
                self.get_logger().warning(
                    f"Control loop late: {self.overruns} late ticks so far.",
                    throttle_duration_sec=5.0,
                )
        self.last_tick_time = now


def main() -> None:
    rclpy.init()
    node = InferenceNode()
    # Keep this single-threaded. Every callback takes well under a millisecond, and
    # measured on MAVROS topics, MultiThreadedExecutor dropped the loop to 37 Hz with
    # 160 ms gaps where this holds 50 Hz to within 0.1 ms.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # With no setpoints, ArduPilot levels out after GUID_TIMEOUT.
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
