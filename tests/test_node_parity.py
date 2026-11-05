"""Training and the flight node must build the same observation and command.

The node is copied alone to the companion computer, so it cannot import the
environment and repeats its logic. These tests load it with ROS stubbed out and
compare the two, because a mismatch here fails silently in the air.
"""

import ast
import dataclasses
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from agents.scout import scout_reward
from agents.scout.train import make_config
from envs import (
    MARKER_OBS_DIM,
    DomainRandomizationConfig,
    DroneEnvConfig,
    MarkerHoldEnv,
    SensorNoiseConfig,
)
from envs.base_drone_env import (
    ACTION_DIM,
    SENSOR_DIM,
    _quaternion_to_euler,
    _world_to_body,
)
from envs.marker_env import BOX_DIM, BOX_LAGS, OBS_ACTION

NODE_PATH = Path(__file__).parents[1] / "drone_rl_workspace" / "ros2_inference_node.py"
ROS_MODULES = (
    "rclpy",
    "rclpy.clock",
    "rclpy.executors",
    "rclpy.node",
    "rclpy.qos",
    "mavros_msgs",
    "mavros_msgs.msg",
    "mavros_msgs.srv",
    "message_filters",
    "nav_msgs",
    "nav_msgs.msg",
    "rcl_interfaces",
    "rcl_interfaces.msg",
    "rcl_interfaces.srv",
    "sensor_msgs",
    "sensor_msgs.msg",
    "std_msgs",
    "std_msgs.msg",
)


@pytest.fixture(scope="module")
def node():
    stubs = {name: mock.MagicMock() for name in ROS_MODULES}
    stubs["rclpy.node"].Node = object
    with mock.patch.dict(sys.modules, stubs):
        sys.modules.pop("ros2_inference_node", None)
        yield importlib.import_module("ros2_inference_node")
    sys.modules.pop("ros2_inference_node", None)


def node_defaults() -> dict:
    """Read the node's declared parameters without running any ROS."""
    tree = ast.parse(NODE_PATH.read_text(encoding="utf-8"))
    return {
        call.args[0].value: ast.literal_eval(call.args[1])
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "declare_parameter"
    }


def to_ros(node, position, velocity, quaternion, rates):
    """Express a NED/FRD state as MAVROS publishes it, in ENU/FLU."""
    w, x, y, z = node.convert_quaternion(quaternion)
    orientation = SimpleNamespace(w=w, x=x, y=y, z=z)
    body_frd = _world_to_body(quaternion, velocity)
    twist = SimpleNamespace(x=body_frd[0], y=-body_frd[1], z=-body_frd[2])
    pose = SimpleNamespace(
        position=SimpleNamespace(x=position[1], y=position[0], z=-position[2]),
        orientation=orientation,
    )
    odom = SimpleNamespace(
        pose=SimpleNamespace(pose=pose),
        twist=SimpleNamespace(twist=SimpleNamespace(linear=twist)),
    )
    imu = SimpleNamespace(
        orientation=orientation,
        angular_velocity=SimpleNamespace(x=rates[0], y=-rates[1], z=-rates[2]),
    )
    return imu, odom


def perfect_env() -> MarkerHoldEnv:
    config = dataclasses.replace(
        make_config(),
        randomization=DomainRandomizationConfig(enabled=False),
        noise=SensorNoiseConfig(enabled=False),
    )
    env = MarkerHoldEnv(scout_reward, config)
    env.reset(seed=4)
    return env


def test_dimensions_agree(node):
    assert node.OBS_DIM == MARKER_OBS_DIM
    assert node.ACTION_DIM == ACTION_DIM
    assert node.SENSOR_DIM == SENSOR_DIM
    assert node.BOX_DIM == BOX_DIM
    assert node.BOX_LAGS == BOX_LAGS


def test_training_envelope_equals_the_flight_envelope():
    defaults = node_defaults()
    config = DroneEnvConfig()
    for name in ("min_altitude_m", "max_position_m", "max_speed_mps", "max_tilt_rad"):
        assert defaults[name] == getattr(config, name), name
    assert defaults["min_thrust"] == config.min_thrust
    assert defaults["max_thrust"] == config.max_thrust
    assert defaults["rate_hz"] == config.control_hz


def test_default_hold_is_inside_the_training_range():
    defaults = node_defaults()
    task = DroneEnvConfig().marker_task
    assert task.hold_range_m[0] <= defaults["hold_range_m"] <= task.hold_range_m[1]
    assert task.hold_height_m[0] <= defaults["hold_height_m"] <= task.hold_height_m[1]


def test_node_builds_the_same_observation_as_training(node):
    env = perfect_env()
    rng = np.random.default_rng(1)
    alpha = env.config.action_filter_alpha
    previous = np.zeros(4, dtype=np.float32)
    first = np.array([*env._camera_frame, 1.0], dtype=np.float32)
    boxes = [first] * (BOX_LAGS[-1] + 1)

    compared = 0
    for _ in range(120):
        action = rng.uniform(-0.1, 0.1, 4).astype(np.float32)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
        previous = node.smooth_action(action, previous, alpha)
        state = env._state

        imu, odom = to_ros(
            node, state.position, state.velocity, state.quaternion, state.angular_velocity
        )
        sensors = node.pack_sensors(imu, odom)
        detection = SimpleNamespace(data=[float(v) for v in env._latest_box])
        box = node.box_slots(detection if env._latest_box[2] > 0 else None, 0.0, 0.2)
        boxes = boxes[1:] + [box]
        built = node.build_observation(
            boxes, sensors, previous, env._hold_range, env._hold_height
        )
        assert built == pytest.approx(observation, abs=2e-3)
        compared += 1
    assert compared > 2 * BOX_LAGS[-1]


def test_node_observation_ignores_position_and_velocity(node):
    boxes = [np.zeros(BOX_DIM, dtype=np.float32)] * (BOX_LAGS[-1] + 1)
    sensors = np.zeros(SENSOR_DIM)
    sensors[6] = 1.0
    moved = sensors.copy()
    moved[:6] = [40, -30, -20, 5, 5, 5]
    action = np.zeros(4, dtype=np.float32)
    assert np.array_equal(
        node.build_observation(boxes, sensors, action, 10.0, 4.0),
        node.build_observation(boxes, moved, action, 10.0, 4.0),
    )


def test_a_detection_becomes_a_visible_box(node):
    message = SimpleNamespace(data=[0.1, -0.2, 0.3, 0.2])
    assert node.box_slots(message, 0.05, 0.2) == pytest.approx([0.1, -0.2, 0.3, 0.2, 1])


@pytest.mark.parametrize(
    "message, age",
    [
        (None, 0.0),  # No detector.
        (SimpleNamespace(data=[0.1, 0.2, 0.3, 0.2]), 0.21),  # Stale.
        (SimpleNamespace(data=[float("nan"), 0.2, 0.3, 0.2]), 0.0),  # Broken.
        (SimpleNamespace(data=[0.1, 0.2, 0.0, 0.2]), 0.0),  # Empty box.
        (SimpleNamespace(data=[0.1, 0.2]), 0.0),  # Wrong length.
    ],
)
def test_an_unusable_detection_is_not_visible(node, message, age):
    assert not node.box_slots(message, age, 0.2).any()


def test_node_action_filter_equals_training(node):
    env = perfect_env()
    rng = np.random.default_rng(0)
    previous = np.zeros(4, dtype=np.float32)
    for _ in range(10):
        raw = rng.uniform(-0.1, 0.1, 4).astype(np.float32)
        observation, *_ = env.step(raw)
        previous = node.smooth_action(raw, previous, env.config.action_filter_alpha)
        assert previous == pytest.approx(observation[OBS_ACTION : OBS_ACTION + 4], abs=1e-6)


def setpoint(node, action, hover_thrust, yaw=0.0):
    """Run the node's real make_setpoint on a level vehicle."""
    defaults = node_defaults()
    fake_node = SimpleNamespace(
        policy=SimpleNamespace(max_attitude_cmd_rad=0.35, max_yaw_rate_rps=1.0),
        hover_thrust=hover_thrust,
        min_thrust=defaults["min_thrust"],
        max_thrust=defaults["max_thrust"],
        max_yaw_lead_rad=defaults["max_yaw_lead_rad"],
        period_s=0.02,
        yaw_setpoint=yaw,
        get_clock=mock.MagicMock(),
    )
    sensors = np.zeros(SENSOR_DIM)
    sensors[6:10] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    action = np.array(action, dtype=np.float32)
    return node.InferenceNode.make_setpoint(fake_node, action, sensors)


@pytest.mark.parametrize("hover_thrust", [0.36, 0.5, 0.6])
def test_node_thrust_equals_training(node, hover_thrust):
    env = perfect_env()
    env.domain = dataclasses.replace(env.domain, hover_thrust=hover_thrust)
    for value in np.linspace(-1, 1, 21):
        action = np.array([0, 0, 0, value], dtype=np.float32)
        message = setpoint(node, action, hover_thrust)
        assert message.thrust == pytest.approx(env._action_to_command(action)[3])


def test_a_level_action_commands_level_flight_at_hover(node):
    message = setpoint(node, np.zeros(4), 0.42, yaw=1.0)
    q = message.orientation
    decoded = node.convert_quaternion(np.array([q.w, q.x, q.y, q.z]))
    roll, pitch, yaw = _quaternion_to_euler(decoded)
    assert (roll, pitch) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert yaw == pytest.approx(1.0, abs=1e-6)
    assert message.thrust == pytest.approx(0.42)


@pytest.mark.parametrize("yaw", [0.0, 1.0, -2.5])
def test_node_attitude_setpoint_equals_training(node, yaw):
    """Decode the ENU/FLU message back to the roll and pitch training commands."""
    env = perfect_env()
    action = np.array([0.6, -0.4, 0.0, 0.0], dtype=np.float32)
    message = setpoint(node, action, 0.5, yaw)
    q = message.orientation
    decoded = node.convert_quaternion(np.array([q.w, q.x, q.y, q.z]))
    roll, pitch, decoded_yaw = _quaternion_to_euler(decoded)
    expected = env._action_to_command(action)
    assert roll == pytest.approx(expected[0], abs=1e-6)
    assert pitch == pytest.approx(expected[1], abs=1e-6)
    assert decoded_yaw == pytest.approx(yaw, abs=1e-6)


def test_quaternion_conversion_is_its_own_inverse(node):
    rng = np.random.default_rng(0)
    for _ in range(100):
        quaternion = rng.normal(size=4)
        quaternion /= np.linalg.norm(quaternion)
        quaternion *= np.sign(quaternion[0])
        back = node.convert_quaternion(node.convert_quaternion(quaternion))
        assert back == pytest.approx(quaternion, abs=1e-9)


def save_meta(path, mean, var, epsilon, clip_obs, low, high):
    np.savez(
        path,
        control_mode="attitude",
        control_hz=50.0,
        action_filter_alpha=0.5,
        lost_timeout_s=0.5,
        max_box_width=0.8,
        obs_mean=mean,
        obs_var=var,
        epsilon=epsilon,
        clip_obs=clip_obs,
        obs_low=low,
        obs_high=high,
        max_attitude_cmd_rad=0.35,
        max_yaw_rate_rps=1.0,
    )


def test_node_policy_normalizes_like_vecnormalize(node, tmp_path):
    """Run the node's real Policy on a network that echoes its first four inputs."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    dim = MARKER_OBS_DIM
    echo = np.zeros((dim, ACTION_DIM), np.float32)
    echo[:ACTION_DIM, :ACTION_DIM] = np.eye(ACTION_DIM)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["observation", "W"], ["action"])],
        "echo",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [1, dim])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, ACTION_DIM])],
        [numpy_helper.from_array(echo, "W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, tmp_path / "echo.onnx")

    env = MarkerHoldEnv(scout_reward, make_config())
    stats = VecNormalize(DummyVecEnv([lambda: env]), clip_obs=10.0)
    rng = np.random.default_rng(0)
    stats.obs_rms.mean = rng.normal(size=dim) * 0.1
    stats.obs_rms.var = rng.uniform(1.0, 4.0, dim)
    low, high = env.observation_space.low, env.observation_space.high
    save_meta(
        tmp_path / "meta.npz",
        stats.obs_rms.mean,
        stats.obs_rms.var,
        stats.epsilon,
        stats.clip_obs,
        low,
        high,
    )
    policy = node.Policy(str(tmp_path / "echo.onnx"), str(tmp_path / "meta.npz"))

    for _ in range(50):
        observation = rng.uniform(low, high).astype(np.float32)
        expected = np.clip(stats.normalize_obs(observation)[:ACTION_DIM], -1, 1)
        assert policy(observation) == pytest.approx(expected, abs=1e-5)


def test_node_refuses_a_policy_with_the_wrong_observation_size(node, tmp_path):
    save_meta(tmp_path / "meta.npz", np.zeros(24), np.ones(24), 1e-8, 10.0, np.zeros(24), np.ones(24))
    with pytest.raises(ValueError, match=f"Expected {MARKER_OBS_DIM}"):
        node.Policy("unused.onnx", str(tmp_path / "meta.npz"))


def tick_node(node, **overrides):
    """Run the real control tick with the policy engaged and a fake everything."""
    published = []
    sensors = np.zeros(SENSOR_DIM, dtype=np.float32)
    sensors[2] = -10.0
    sensors[6] = 1.0
    now = node.time.monotonic()
    fields = dict(
        state=SimpleNamespace(connected=True, mode="GUIDED", armed=True),
        state_time=now,
        sensors=sensors,
        sensor_time=now,
        status=node.Status.ACTIVE,
        required_mode="GUIDED",
        state_timeout_s=2.5,
        sensor_timeout_s=0.1,
        marker=SimpleNamespace(data=[0.0, 0.0, 0.2, 0.1]),
        marker_time=now,
        marker_timeout_s=0.2,
        thrust_mode_verified=True,
        limits=node.FlightLimits(1.0, 50.0, 15.0, 0.7),
        policy=mock.MagicMock(
            lost_timeout_s=0.5,
            max_box_width=0.8,
            action_filter_alpha=0.5,
            return_value=np.zeros(4, dtype=np.float32),
        ),
        boxes=[np.zeros(BOX_DIM, dtype=np.float32)] * (BOX_LAGS[-1] + 1),
        prev_action=np.zeros(4, dtype=np.float32),
        levelling=False,
        hold_range_m=10.0,
        hold_height_m=4.0,
        handover_pending=False,
        last_handover_time=-np.inf,
        fallback_mode="LOITER",
        attitude_pub=SimpleNamespace(publish=published.append),
        yaw_setpoint=0.0,
    )
    fields.update(overrides)
    fake = mock.MagicMock()
    for key, value in fields.items():
        setattr(fake, key, value)
    cls = node.InferenceNode
    fake.find_any_fault = lambda *args: cls.find_any_fault(fake, *args)
    fake.lock_out = lambda *args, **kwargs: cls.lock_out(fake, *args, **kwargs)
    fake.make_setpoint = mock.MagicMock(return_value="setpoint")
    cls.on_control_tick(fake)
    return fake, published


def test_a_visible_marker_keeps_the_policy_flying(node):
    fake, published = tick_node(node)
    assert fake.status is node.Status.ACTIVE
    assert published == ["setpoint"]


def test_a_short_dropout_keeps_the_policy_flying(node):
    fake, published = tick_node(node, marker_time=node.time.monotonic() - 0.3)
    assert fake.status is node.Status.ACTIVE
    assert published == ["setpoint"]


def test_losing_the_marker_levels_out_and_hands_over(node):
    fake, _ = tick_node(node, marker_time=node.time.monotonic() - 0.6)
    assert fake.status is node.Status.LOCKED
    assert fake.levelling is True
    assert fake.handover_pending is True


def test_a_locked_node_keeps_commanding_level_until_the_mode_changes(node):
    fake, published = tick_node(node, status=node.Status.LOCKED, levelling=True)
    assert published == ["setpoint"]
    action = fake.make_setpoint.call_args[0][0]
    assert not action.any()


def test_coming_too_close_levels_out_and_hands_over(node):
    fake, _ = tick_node(node, marker=SimpleNamespace(data=[0.0, 0.0, 0.9, 0.5]))
    assert fake.status is node.Status.LOCKED
    assert fake.levelling is True


def test_engaging_needs_the_marker_in_view(node):
    fake, published = tick_node(node, status=node.Status.IDLE, marker=None)
    assert fake.status is node.Status.LOCKED
    assert fake.handover_pending is False
    assert published == []


def test_engaging_with_the_marker_in_view_starts_the_policy(node):
    fake, published = tick_node(node, status=node.Status.IDLE)
    assert fake.status is node.Status.ACTIVE
    assert published == ["setpoint"]
