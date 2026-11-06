"""Checks on the simulation. These protect what a training run silently relies on."""

import dataclasses

import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from agents.payload import payload_reward
from agents.scout import scout_reward
from agents.scout.train import make_config, next_difficulty
from envs import (
    BaseDroneEnv,
    CameraConfig,
    ControlMode,
    DomainParams,
    DomainRandomizationConfig,
    DroneEnvConfig,
    DroneState,
    SensorNoiseConfig,
    WaypointConfig,
)
from envs.base_drone_env import (
    OBS_DIM,
    DelayLine,
    _blend_domains,
    _euler_to_quaternion,
    _quaternion_to_euler,
    _world_to_body,
)

HOVER = np.zeros(4, dtype=np.float32)


def nominal(**overrides) -> DroneEnvConfig:
    """The nominal vehicle in still air with perfect sensors."""
    return dataclasses.replace(
        make_config(),
        randomization=DomainRandomizationConfig(enabled=False),
        noise=SensorNoiseConfig(enabled=False),
        spawn_speed_mps=0.0,
        spawn_tilt_rad=0.0,
        spawn_rate_rps=0.0,
        **overrides,
    )


def make(config: DroneEnvConfig | None = None, seed: int = 0) -> BaseDroneEnv:
    env = BaseDroneEnv(scout_reward, config or make_config())
    env.reset(seed=seed)
    return env


def fly(env: BaseDroneEnv, action, steps: int) -> None:
    for _ in range(steps):
        env.step(np.array(action, dtype=np.float32))


def body_displacement(action, steps: int = 100) -> np.ndarray:
    """Return how far an action moves the vehicle: forward, right, down."""
    env = make(nominal())
    start = env._state
    fly(env, action, steps)
    yaw = _quaternion_to_euler(start.quaternion)[2]
    level = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    return _world_to_body(level, env._state.position - start.position)


# ---------------------------------------------------------------------- API


def test_gymnasium_contract():
    check_env(BaseDroneEnv(scout_reward, make_config()), skip_render_check=True)


def test_observation_stays_in_its_space():
    env = make()
    rng = np.random.default_rng(0)
    observation, _ = env.reset(seed=1)
    assert observation.shape == (OBS_DIM,)
    for _ in range(500):
        action = rng.uniform(-1, 1, 4).astype(np.float32)
        observation, _, terminated, truncated, _ = env.step(action)
        assert observation.dtype == np.float32
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            observation, _ = env.reset()


def test_same_seed_same_episode():
    def rollout():
        env = make(seed=3)
        return [env.step(HOVER)[0] for _ in range(100)]

    assert np.array_equal(rollout(), rollout())


# ---------------------------------------------------------------------- Dynamics


def test_neutral_action_hovers():
    env = make(nominal())
    start = env._state.position.copy()
    fly(env, HOVER, 250)
    assert np.linalg.norm(env._state.position - start) < 0.1
    assert env._state.tilt < 0.01


@pytest.mark.parametrize(
    "action, axis, sign",
    [
        ([0.3, 0, 0, 0], 1, +1),  # Roll right moves right.
        ([-0.3, 0, 0, 0], 1, -1),
        ([0, -0.3, 0, 0], 0, +1),  # Nose down moves forward.
        ([0, 0.3, 0, 0], 0, -1),
        ([0, 0, 0, 0.2], 2, -1),  # More thrust climbs, and down is positive.
        ([0, 0, 0, -0.2], 2, +1),
    ],
)
def test_each_axis_moves_the_right_way(action, axis, sign):
    moved = body_displacement(action)
    assert sign * moved[axis] > 0.5
    assert abs(moved[axis]) > 3 * np.delete(np.abs(moved), axis).max() or axis == 2


def test_positive_yaw_rate_turns_clockwise_from_above():
    env = make(nominal())
    before = _quaternion_to_euler(env._state.quaternion)[2]
    fly(env, [0, 0, 0.5, 0], 50)
    after = _quaternion_to_euler(env._state.quaternion)[2]
    assert np.sin(after - before) > 0.2


def test_thrust_is_clamped_like_the_flight_node():
    env = make(nominal())
    command = env._action_to_command(np.array([0, 0, 0, 1.0], dtype=np.float32))
    assert command[3] == pytest.approx(env.config.max_thrust)
    command = env._action_to_command(np.array([0, 0, 0, -1.0], dtype=np.float32))
    assert command[3] == pytest.approx(env.config.min_thrust)


def test_steady_wind_pushes_the_vehicle_downwind():
    pinned = DomainRandomizationConfig(
        mass_kg=(1.5, 1.5),
        inertia_scale=(1, 1),
        drag_scale=(1, 1),
        motor_lag_scale=(1, 1),
        attitude_gain_scale=(1, 1),
        hover_thrust=(0.5, 0.5),
        wind_speed_mps=(5, 5),
        gust_std_mps=(0, 0),
        motor_efficiency=(1, 1),
        gyro_bias_std_rps=0,
        push_rate_hz=(0, 0),
        motor_fault_prob=0,
        battery_sag=(0, 0),
        com_offset_std_m=0,
        action_delay_steps=(0, 0),
        gps_delay_steps=(0, 0),
        imu_delay_steps=(0, 0),
        link_jitter_steps=(0, 0),
        link_drop_prob=(0, 0),
    )
    env = make(dataclasses.replace(nominal(), randomization=pinned))
    start = env._state.position.copy()
    fly(env, HOVER, 200)
    wind = env.domain.wind_ned_mps
    assert (env._state.position - start) @ wind / np.linalg.norm(wind) > 1.0


def test_gusts_vary_about_the_steady_wind_with_the_configured_size():
    env = make()
    env.domain = dataclasses.replace(
        env.domain,
        gust_std_mps=1.5,
        wind_ned_mps=np.array([3.0, 0.0, 0.0]),
        wind_shear_exponent=0.0,
    )
    samples = []
    for _ in range(20000):
        env._update_gust(env.dt)
        samples.append(env._model.opt.wind.copy())
    samples = np.array(samples)
    assert samples[:, 0].mean() == pytest.approx(3.0, abs=0.5)
    assert samples.std(axis=0) == pytest.approx([1.5] * 3, rel=0.35)


def test_nominal_config_has_no_gusts():
    env = make(nominal())
    winds = []
    for _ in range(50):
        env.step(HOVER)
        winds.append(env._model.opt.wind.copy())
    assert np.array_equal(winds, np.zeros((50, 3)))


# ---------------------------------------------------------------------- Randomization


def test_randomized_parameters_reach_mujoco():
    env = make(seed=5)
    domain = env.domain
    model = env._model
    body = env._body_id

    assert model.body_mass[body] == pytest.approx(domain.mass_kg)
    assert model.body_inertia[body] == pytest.approx(
        env._nominal_inertia
        * domain.mass_kg
        / env._nominal_mass
        * domain.inertia_scale
    )
    assert model.opt.density == pytest.approx(env._nominal_density * domain.drag_scale)
    assert model.actuator_dynprm[:, 0] == pytest.approx(
        env._nominal_motor_lag * domain.motor_lag_scale
    )
    assert model.actuator_gainprm[:, 0] == pytest.approx(domain.motor_efficiency)
    assert model.body_ipos[body] == pytest.approx(domain.com_offset_m)


def test_every_randomized_range_is_sampled():
    env = BaseDroneEnv(scout_reward, make_config())
    domains = []
    for seed in range(40):
        env.reset(seed=seed)
        domains.append(env.domain)
    cfg = make_config().randomization
    for name in (
        "mass_kg",
        "inertia_scale",
        "drag_scale",
        "motor_lag_scale",
        "attitude_gain_scale",
        "hover_thrust",
        "gust_std_mps",
        "gps_update_hz",
        "link_drop_prob",
        "wind_shear_exponent",
        "push_rate_hz",
        "push_force_n",
        "battery_sag",
        "noise_scale",
        "camera_frame_hz",
    ):
        values = [getattr(d, name) for d in domains]
        low, high = getattr(cfg, name)
        assert low <= min(values) < max(values) <= high, name


def test_disabling_randomization_restores_the_nominal_vehicle():
    env = make()
    nominal_env_mass = env._nominal_mass
    env.config = nominal()
    env.reset(seed=1)
    assert env._model.body_mass[env._body_id] == nominal_env_mass
    assert env._model.opt.density == env._nominal_density
    assert np.array_equal(env._model.actuator_dynprm[:, 0], env._nominal_motor_lag)
    assert np.array_equal(env._model.body_ipos[env._body_id], np.zeros(3))


def test_a_heavy_weak_vehicle_can_still_climb():
    """The hardest draw must not be unflyable, or those episodes teach nothing."""
    cfg = make_config()
    env = make(nominal())
    env.domain = dataclasses.replace(
        env.domain,
        mass_kg=cfg.randomization.mass_kg[1],
        hover_thrust=cfg.randomization.hover_thrust[1],
        motor_efficiency=np.full(4, cfg.randomization.motor_efficiency[0]),
    )
    env._physics_reset(env.domain)
    start = -env._data.qpos[2]
    fly(env, [0, 0, 0, 1], 100)
    assert -env._data.qpos[2] > start


# ---------------------------------------------------------------------- Sensors


def test_perfect_sensors_report_the_truth():
    env = make(nominal())
    observation, *_ = env.step(HOVER)
    state = env._state
    assert observation[:3] == pytest.approx(state.position, abs=1e-4)
    assert observation[3:6] == pytest.approx(state.velocity, abs=1e-4)


def test_quaternion_sign_matches_the_flight_node():
    """The node sends w >= 0, and spinning past 180 degrees flips the sign in MuJoCo."""
    for noise_enabled in (True, False):
        config = dataclasses.replace(
            make_config(),
            noise=SensorNoiseConfig(enabled=noise_enabled),
        )
        env = make(config)
        raw_negative = 0
        for _ in range(500):
            observation, _, terminated, truncated, _ = env.step(
                np.array([0, 0, 1, 0], dtype=np.float32)
            )
            assert observation[6] >= 0.0
            raw_negative += env._state.quaternion[0] < 0
            if terminated or truncated:
                env.reset()
        assert raw_negative > 0


def test_gps_error_is_held_between_fixes():
    env = make(
        dataclasses.replace(
            make_config(),
            noise=SensorNoiseConfig(gps_glitch_rate_hz=0.0),
        )
    )
    env.domain = dataclasses.replace(env.domain, gps_update_hz=10.0)
    state = env._state
    errors = []
    for step in range(1, 21):
        env._step_count = step
        env._read_gps(state)
        errors.append(env._gps_position_error.copy())
    changes = [not np.array_equal(a, b) for a, b in zip(errors, errors[1:])]
    assert sum(changes) == 4  # 20 steps at 50 Hz is 0.4 s of 10 Hz fixes.


def test_position_noise_is_slow_not_white():
    """The node reads the EKF, so position must not jump by the drift each step."""
    env = make()
    state = env._state
    samples = []
    for step in range(200):
        env._step_count = step
        samples.append(env._read_gps(state)[0].copy())
    steps = np.diff(samples, axis=0)
    assert steps.std() < 0.1
    cfg = make_config().noise
    assert np.abs(env._gps_position_error).max() < 8 * cfg.baro_drift_std_m


def test_gyro_bias_wanders():
    env = make()
    start = env._gyro_bias.copy()
    for _ in range(100):
        env._read_imu(env._state)
    assert not np.array_equal(env._gyro_bias, start)


def test_vibration_grows_with_thrust():
    def gyro_noise(throttle: float) -> float:
        env = make(dataclasses.replace(make_config()))
        env._data.act[:] = throttle * env._max_motor_thrust
        env.domain = dataclasses.replace(
            env.domain, gyro_bias_rps=np.zeros(3), noise_scale=1.0
        )
        state = env._state
        samples = [env._read_imu(state)[1] for _ in range(2000)]
        return float(np.std(samples, axis=0).mean())

    assert gyro_noise(1.0) > 1.5 * gyro_noise(0.0)


# ---------------------------------------------------------------------- Field of view


def camera_env(**overrides) -> BaseDroneEnv:
    return make(dataclasses.replace(nominal(), camera=CameraConfig(**overrides)))


@pytest.mark.parametrize(
    "offset, visible",
    [
        ([0.0, 0.0, 10.0], True),  # Straight below.
        ([0.0, 6.0, 10.0], True),  # Inside the wide axis: tan(0.6) = 0.68.
        ([0.0, 7.5, 10.0], False),  # Outside it.
        ([4.0, 0.0, 10.0], True),  # Inside the narrow axis: tan(0.45) = 0.48.
        ([5.5, 0.0, 10.0], False),  # Outside it, though inside the wide axis.
        ([0.0, 0.0, -10.0], False),  # Behind the camera.
        ([0.0, 0.0, 31.0], False),  # Beyond the camera's range.
    ],
)
def test_field_of_view_limits(offset, visible):
    assert camera_env()._in_camera_view(np.array(offset)) is visible


def test_mount_tilt_moves_the_view_forward():
    level = camera_env()
    tilted = camera_env(mount_tilt_rad=0.5)
    ahead = np.array([5.5, 0.0, 10.0])
    assert not level._in_camera_view(ahead)
    assert tilted._in_camera_view(ahead)


def test_disabled_camera_sees_nothing():
    assert not camera_env(enabled=False)._in_camera_view(np.array([0.0, 0.0, 10.0]))


def place_over_marker(env: BaseDroneEnv, altitude: float, roll: float = 0.0) -> None:
    env._data.qpos[:2] = env._marker[:2]
    env._data.qpos[2] = -altitude
    env._data.qpos[3:7] = _euler_to_quaternion(roll, 0.0, 0.0)
    mujoco.mj_forward(env._model, env._data)
    env._state = env._read_state()


def test_marker_below_is_seen_in_the_body_frame():
    env = camera_env()
    place_over_marker(env, 10.0)
    assert env._state.target_visible
    assert env._state.target_offset == pytest.approx([0.0, 0.0, 10.0], abs=1e-6)


def test_banking_hard_swings_the_marker_out_of_view():
    env = camera_env()
    place_over_marker(env, 10.0, roll=0.7)
    assert not env._state.target_visible


def test_marker_does_not_move_during_an_episode():
    env = make()
    start = env._marker.copy()
    rng = np.random.default_rng(0)
    for _ in range(300):
        env.step(rng.uniform(-0.3, 0.3, 4).astype(np.float32))
    assert np.array_equal(env._marker, start)
    assert env._marker[2] == 0.0


def detection_errors(altitude: float) -> tuple[float, float]:
    """Return the detection rate and error of a marker straight below."""
    env = make()
    env.domain = dataclasses.replace(env.domain, noise_scale=1.0)
    place_over_marker(env, altitude)
    state = dataclasses.replace(env._state, angular_velocity=np.zeros(3))
    results = [env._detect_marker(state) for _ in range(3000)]
    seen = [offset for offset, visible in results if visible]
    errors = np.array(seen) - state.target_offset
    return len(seen) / len(results), float(np.median(np.abs(errors)))


def test_error_grows_with_range():
    assert detection_errors(25.0)[1] > 2 * detection_errors(6.0)[1]


def test_detection_fades_near_maximum_range():
    assert detection_errors(28.0)[0] < 0.5 * detection_errors(6.0)[0]


def test_fast_rotation_blurs_the_marker_away():
    env = make()
    place_over_marker(env, 8.0)

    def rate(rps: float) -> float:
        state = dataclasses.replace(
            env._state, angular_velocity=np.array([rps, 0.0, 0.0])
        )
        return np.mean([env._detect_marker(state)[1] for _ in range(1000)])

    assert rate(4.0) < 0.5 * rate(0.0)


def test_false_positives_happen_when_nothing_is_in_view():
    noise = SensorNoiseConfig(vision_false_positive_prob=0.2)
    env = make(dataclasses.replace(make_config(), noise=noise))
    state = dataclasses.replace(env._state, target_visible=False)
    results = [env._detect_marker(state)[1] for _ in range(2000)]
    assert np.mean(results) == pytest.approx(0.2, abs=0.05)


def test_camera_result_only_changes_on_a_new_frame():
    env = make()
    env.domain = dataclasses.replace(env.domain, camera_frame_hz=10.0)
    place_over_marker(env, 8.0)
    frames = []
    for step in range(20):
        env._step_count = step
        frames.append(env._read_camera(env._state)[0].copy())
    changes = sum(not np.array_equal(a, b) for a, b in zip(frames, frames[1:]))
    assert changes <= 3  # 20 steps at 50 Hz holds four 10 Hz frames.


def test_observation_carries_the_camera_with_its_delay():
    config = dataclasses.replace(nominal(), camera=CameraConfig())
    env = make(config)
    place_over_marker(env, 10.0)
    env._action_link = DelayLine(HOVER, 0, 0, 0.0)
    seen = None
    for _ in range(4):
        seen, *_ = env.step(HOVER)
    assert seen[16] == 1.0
    assert seen[13:16] == pytest.approx([0.0, 0.0, 10.0], abs=0.2)


# ---------------------------------------------------------------------- Latency


def link(delay: int, jitter: int = 0, drop: float = 0.0) -> DelayLine:
    return DelayLine(np.array(0.0), delay, jitter, drop)


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_fixed_delay_is_exact(delay):
    rng = np.random.default_rng(0)
    line = link(delay)
    received = [float(line(np.array(float(i)), rng)) for i in range(1, 11)]
    expected = [max(i - delay, 0) for i in range(1, 11)]
    assert received == expected


def test_jitter_stays_in_range_and_never_reorders():
    rng = np.random.default_rng(0)
    line = link(2, jitter=1)
    received = [float(line(np.array(float(i)), rng)) for i in range(1, 2001)]
    ages = np.arange(1, 2001) - np.array(received)
    assert np.all(np.diff(received) >= 0)
    assert ages[10:].min() >= 1 and ages[10:].max() <= 3
    assert len(set(ages[10:])) > 1


def test_a_dropped_packet_holds_the_last_value():
    rng = np.random.default_rng(0)
    line = link(0, drop=1.0)
    assert float(line(np.array(5.0), rng)) == 0.0
    assert float(line(np.array(6.0), rng)) == 0.0


def test_packet_loss_happens_at_the_configured_rate():
    rng = np.random.default_rng(0)
    line = link(0, drop=0.2)
    received = [float(line(np.array(float(i)), rng)) for i in range(1, 5001)]
    assert np.mean(np.diff(received) == 0) == pytest.approx(0.2, abs=0.03)


def test_action_delay_reaches_the_vehicle():
    def climb_after(delay: int) -> float:
        env = make(nominal())
        env._action_link = DelayLine(HOVER, delay, 0, 0.0)
        start = -env._data.qpos[2]
        fly(env, [0, 0, 0, 0.5], 10)
        return -env._data.qpos[2] - start

    assert climb_after(5) < climb_after(0)


# ---------------------------------------------------------------------- Conditions


def test_noise_scale_widens_sensor_noise():
    def spread(scale: float) -> float:
        env = make()
        env.domain = dataclasses.replace(env.domain, noise_scale=scale)
        return float(np.std([env._read_imu(env._state)[1][0] for _ in range(2000)]))

    assert spread(2.0) > 1.5 * spread(1.0)


def test_heading_error_rotates_what_the_policy_believes():
    config = dataclasses.replace(make_config(), noise=SensorNoiseConfig(attitude_std=0.0))
    env = make(config)
    env._heading_error = _euler_to_quaternion(0.0, 0.0, 0.2)
    sensed, _ = env._read_imu(env._state)
    true_yaw = _quaternion_to_euler(env._state.quaternion)[2]
    error = _quaternion_to_euler(sensed)[2] - true_yaw
    assert np.sin(error) == pytest.approx(np.sin(0.2), abs=0.02)


def test_gps_glitch_jumps_then_fades():
    noise = SensorNoiseConfig(gps_glitch_rate_hz=1e6)
    env = make(dataclasses.replace(make_config(), noise=noise))
    env._step_count = 0
    env._read_gps(env._state)
    assert np.linalg.norm(env._gps_glitch) == pytest.approx(noise.gps_glitch_m)

    env.config = dataclasses.replace(
        make_config(), noise=SensorNoiseConfig(gps_glitch_rate_hz=0.0)
    )
    fix_steps = max(round(50 / env.domain.gps_update_hz), 1)
    for fix in range(1, 200):
        env._step_count = fix * fix_steps
        env._read_gps(env._state)
    assert np.linalg.norm(env._gps_glitch) < 0.5 * noise.gps_glitch_m


def test_wind_is_stronger_higher_up():
    env = make(nominal())
    env.domain = dataclasses.replace(
        env.domain,
        wind_ned_mps=np.array([4.0, 0.0, 0.0]),
        wind_shear_exponent=0.3,
    )
    env._gust = np.zeros(3)

    def wind_at(altitude: float) -> float:
        env._data.qpos[2] = -altitude
        env._apply_wind()
        return float(env._model.opt.wind[0])

    assert wind_at(40.0) > 4.0 > wind_at(3.0)


def test_battery_sag_loses_thrust_during_the_flight():
    env = make(nominal())
    env.domain = dataclasses.replace(env.domain, battery_sag=0.1)
    env._step_count = 0
    env._update_thrust_loss()
    start = env._model.actuator_gainprm[:, 0].copy()
    env._motor_fault_step = 10**6
    env._step_count = env.config.max_episode_steps
    env._update_thrust_loss()
    assert env._model.actuator_gainprm[:, 0] == pytest.approx(start * 0.9)


def test_motor_fault_weakens_one_motor_at_its_step():
    env = make(nominal())
    env.domain = dataclasses.replace(env.domain, motor_fault_loss=0.2)
    env._motor_fault_step, env._faulty_motor = 100, 2
    env._step_count = 99
    env._update_thrust_loss()
    assert env._model.actuator_gainprm[:, 0] == pytest.approx(np.ones(4))
    env._step_count = 100
    env._update_thrust_loss()
    assert env._model.actuator_gainprm[:, 0] == pytest.approx([1, 1, 0.8, 1])


def test_a_motor_fault_is_survivable():
    env = make(nominal())
    env.domain = dataclasses.replace(env.domain, motor_fault_loss=0.2)
    env._motor_fault_step = 1
    for _ in range(200):
        _, _, terminated, _, _ = env.step(HOVER)
        assert not terminated


def test_a_push_shoves_the_vehicle_then_stops():
    env = make(nominal())
    env.domain = dataclasses.replace(env.domain, push_rate_hz=1e6, push_force_n=6.0)
    start = env._state.position.copy()
    env.step(HOVER)
    force = env._data.xfrc_applied[env._body_id, :3].copy()
    assert np.linalg.norm(force) == pytest.approx(6.0)

    env.domain = dataclasses.replace(env.domain, push_rate_hz=0.0)
    fly(env, HOVER, 30)
    assert np.array_equal(env._data.xfrc_applied[env._body_id, :3], np.zeros(3))
    assert np.linalg.norm(env._state.position - start) > 0.05


def test_com_offset_is_a_real_torque():
    env = make(nominal())
    env.domain = dataclasses.replace(
        env.domain, com_offset_m=np.array([0.0, 0.03, 0.0])
    )
    env._physics_reset(env.domain)
    env._data.ctrl[:] = env._nominal_mass * env._gravity / 4.0
    env._data.act[:] = env._data.ctrl
    mujoco.mj_forward(env._model, env._data)
    assert abs(env._data.qacc[3]) > 0.1


def test_action_filter_smooths_the_command():
    env = make(nominal())
    env.step(np.array([1, 0, 0, 0], dtype=np.float32))
    assert env._last_action[0] == pytest.approx(0.5)
    env.step(np.array([1, 0, 0, 0], dtype=np.float32))
    assert env._last_action[0] == pytest.approx(0.75)


# ---------------------------------------------------------------------- Curriculum


def test_difficulty_zero_is_the_nominal_vehicle():
    env = BaseDroneEnv(scout_reward, make_config())
    env.set_difficulty(0.0)
    env.reset(seed=1)
    nominal_params = DomainParams()
    assert env.domain.mass_kg == nominal_params.mass_kg
    assert env.domain.gust_std_mps == 0.0
    assert env.domain.action_delay_steps == 0
    assert env.domain.push_force_n == 0.0


def test_difficulty_zero_has_clean_sensors():
    env = BaseDroneEnv(scout_reward, make_config())
    env.set_difficulty(0.0)
    env.reset(seed=1)
    assert env.domain.noise_scale == 0.0
    state = env._state
    position, velocity = env._read_gps(state)
    quaternion, gyro = env._read_imu(state)
    assert position == pytest.approx(state.position)
    assert velocity == pytest.approx(state.velocity)
    assert quaternion == pytest.approx(state.quaternion)
    assert gyro == pytest.approx(state.angular_velocity)


def test_full_difficulty_keeps_the_full_noise_range():
    env = BaseDroneEnv(scout_reward, make_config())
    scales = []
    for seed in range(50):
        env.reset(seed=seed)
        scales.append(env.domain.noise_scale)
    low, high = make_config().randomization.noise_scale
    assert low <= min(scales) < 0.7 and 1.8 < max(scales) <= high


def test_difficulty_scales_every_parameter_between_the_ends():
    easy = DomainParams()
    hard = DomainParams(
        mass_kg=1.8, action_delay_steps=3, wind_ned_mps=np.array([4.0, 0.0, 0.0])
    )
    half = _blend_domains(easy, hard, 0.5)
    assert half.mass_kg == pytest.approx(1.65)
    assert half.action_delay_steps == 2
    assert isinstance(half.action_delay_steps, int)
    assert half.wind_ned_mps == pytest.approx([2.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "waypoints, crashes, expected",
    [
        ([4] * 50, [False] * 50, 0.55),  # Earned.
        ([1] * 50, [False] * 50, 0.5),  # Survives, but barely does the job.
        ([4] * 49, [False] * 49, 0.5),  # Too few episodes to judge.
        ([6] * 50, [True] * 10 + [False] * 40, 0.5),  # Crashing too often.
    ],
)
def test_difficulty_rises_only_when_earned(waypoints, crashes, expected):
    assert next_difficulty(0.5, waypoints, crashes) == pytest.approx(expected)


def test_camera_to_motor_latency_fits_the_budget():
    cfg = DomainRandomizationConfig()
    step_s = 1.0 / make_config().control_hz
    fastest = (cfg.camera_delay_steps[0] + cfg.action_delay_steps[0]) * step_s
    slowest = (cfg.camera_delay_steps[1] + cfg.action_delay_steps[1]) * step_s
    assert fastest == pytest.approx(0.06)
    assert slowest == pytest.approx(0.10)
    assert cfg.link_jitter_steps == (0, 0)  # Any jitter breaks the window.


def test_difficulty_stops_at_one():
    assert next_difficulty(0.98, [6] * 60, [False] * 60) == 1.0


def test_environments_default_to_full_difficulty():
    assert BaseDroneEnv(scout_reward, make_config()).difficulty == 1.0


# ---------------------------------------------------------------------- Waypoints


def waypoint_env(**overrides) -> BaseDroneEnv:
    overrides.setdefault("marker_prob", 0.0)
    return make(
        dataclasses.replace(nominal(), waypoints=WaypointConfig(**overrides)), seed=2
    )


def test_observation_holds_the_waypoint_in_the_body_frame():
    env = make(nominal())
    observation, *_ = env.step(HOVER)
    offset = env._waypoint - env._state.position
    expected = _world_to_body(env._state.quaternion, offset)
    clipped = np.clip(expected, -20, 20)
    assert observation[21:24] == pytest.approx(clipped, abs=1e-3)


def test_offset_points_forward_when_the_waypoint_is_ahead():
    env = make(nominal())
    yaw = _quaternion_to_euler(env._state.quaternion)[2]
    env._waypoint = env._state.position + 5.0 * np.array(
        [np.cos(yaw), np.sin(yaw), 0.0]
    )
    observation, *_ = env.step(HOVER)
    assert observation[21:24] == pytest.approx([5.0, 0.0, 0.0], abs=0.05)


def test_some_waypoints_sit_above_the_marker():
    env = make()
    above = 0
    for _ in range(500):
        waypoint = env._next_waypoint(env._state.position)
        above += np.array_equal(waypoint[:2], env._marker[:2])
    assert above / 500 == pytest.approx(env.config.waypoints.marker_prob, abs=0.08)


def test_waypoints_stay_inside_the_flight_envelope():
    env = make()
    cfg = env.config
    for seed in range(300):
        env.reset(seed=seed)
        for _ in range(3):
            waypoint = env._next_waypoint(env._state.position)
            assert np.abs(waypoint[:2]).max() < cfg.max_position_m
            assert cfg.min_altitude_m < -waypoint[2] < cfg.max_position_m


def test_holding_the_waypoint_completes_it_and_a_new_one_appears():
    env = waypoint_env(hover_prob=1.0)
    first = env._waypoint.copy()
    for step in range(60):
        _, _, _, _, info = env.step(HOVER)
        if info["waypoints_reached"]:
            break
    assert info["waypoints_reached"] == 1
    assert step + 1 == 50  # A one second hold at 50 Hz.
    assert np.linalg.norm(first - env._state.position) < 1.0


@pytest.mark.parametrize("speed, reached", [(0.2, True), (3.0, False)])
def test_only_a_slow_hold_completes_a_waypoint(speed, reached):
    """Staying in the radius is not enough: the vehicle has to have stopped."""
    env = waypoint_env()
    inside = DroneState(
        position=env._waypoint.copy(), velocity=np.array([speed, 0.0, 0.0])
    )
    assert any(env._update_hold(inside) for _ in range(100)) is reached


def test_arrival_is_judged_on_the_position_estimate():
    """A perfect pilot must be able to arrive however large the drift is."""
    env = waypoint_env()
    env._gps_position_error = np.array([3.0, 0.0, 0.0])
    believes_arrived = DroneState(position=env._waypoint - env._gps_position_error)
    truly_there = DroneState(position=env._waypoint.copy())
    assert any(env._update_hold(believes_arrived) for _ in range(60))
    assert not any(env._update_hold(truly_there) for _ in range(60))


def test_perfect_sensors_have_no_drift():
    assert not make(nominal())._gps_position_error.any()


def test_leaving_the_radius_restarts_the_hold():
    env = waypoint_env()
    inside = DroneState(position=env._waypoint.copy())
    outside = DroneState(position=env._waypoint + np.array([2.0, 0.0, 0.0]))
    for _ in range(49):
        assert not env._update_hold(inside)
    assert not env._update_hold(outside)
    assert not any(env._update_hold(inside) for _ in range(49))
    assert env._update_hold(inside)


def test_reward_sees_the_waypoint_it_flew_towards():
    seen = []

    def spy(state, action, next_state, info):
        seen.append((info["waypoint"].copy(), info["arrived"]))
        return 0.0

    env = BaseDroneEnv(spy, dataclasses.replace(
        nominal(), waypoints=WaypointConfig(hover_prob=0.0, marker_prob=0.0)
    ))
    env.reset(seed=1)
    env._waypoint = env._state.position.copy()
    target = env._waypoint.copy()
    for _ in range(50):
        env.step(HOVER)
    assert seen[-1][1] is True
    assert np.array_equal(seen[-1][0], target)
    assert not np.array_equal(env._waypoint, target)


# ---------------------------------------------------------------------- Termination


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"position": [0.0, 0.0, -0.5]}, "altitude"),
        ({"position": [51.0, 0.0, -10.0]}, "geofence"),
        ({"position": [0.0, 0.0, -51.0]}, "geofence"),
        ({"position": [0.0, 0.0, -10.0], "velocity": [16.0, 0.0, 0.0]}, "speed"),
        (
            {
                "position": [0.0, 0.0, -10.0],
                "quaternion": [np.cos(0.4), np.sin(0.4), 0.0, 0.0],
            },
            "tilt",
        ),
        ({"position": [0.0, 0.0, -10.0]}, ""),
    ],
)
def test_termination_reasons(change, reason):
    env = make()
    state = DroneState(**{key: np.array(value) for key, value in change.items()})
    assert env._termination_reason(state) == reason


def test_spawn_stays_clear_of_every_limit():
    env = BaseDroneEnv(scout_reward, make_config())
    for seed in range(300):
        env.reset(seed=seed)
        assert env._termination_reason(env._state) == ""


def test_spawn_covers_motion():
    env = BaseDroneEnv(scout_reward, make_config())
    env.reset(seed=1)
    assert np.linalg.norm(env._state.velocity) > 0.0
    assert env._state.tilt > 0.0


def test_info_reports_why_the_episode_ended():
    env = make(nominal())
    for _ in range(600):
        _, _, terminated, _, info = env.step(np.array([0, 0, 0, -1], dtype=np.float32))
        if terminated:
            break
    assert info["termination"] == "altitude"
    assert info["is_success"] is False


# ---------------------------------------------------------------------- Reward


def info_for(waypoint, **overrides) -> dict:
    return {
        "waypoint": np.array(waypoint, dtype=float),
        "prev_action": np.zeros(4),
        "prev_waypoint": np.array(waypoint, dtype=float),
        "terminated": False,
        "arrived": False,
        **overrides,
    }


def at(position, velocity=(0, 0, 0), rates=(0, 0, 0)) -> DroneState:
    return DroneState(
        position=np.array(position, dtype=float),
        velocity=np.array(velocity, dtype=float),
        angular_velocity=np.array(rates, dtype=float),
    )


@pytest.mark.parametrize("reward", [scout_reward, payload_reward])
def test_closing_on_the_waypoint_pays(reward):
    info = info_for([10, 0, -5])
    closing = reward(at([0, 0, -5]), HOVER, at([0.1, 0, -5], [5, 0, 0]), info)
    opening = reward(at([0, 0, -5]), HOVER, at([-0.1, 0, -5], [-5, 0, 0]), info)
    assert closing > opening


@pytest.mark.parametrize("reward", [scout_reward, payload_reward])
def test_a_round_trip_earns_nothing_from_progress(reward):
    """Otherwise the policy can farm reward by flying back and forth."""
    info = info_for([10, 0, -5])
    out = reward(at([0, 0, -5]), HOVER, at([1, 0, -5]), info)
    back = reward(at([1, 0, -5]), HOVER, at([0, 0, -5]), info)
    still = reward(at([0, 0, -5]), HOVER, at([0, 0, -5]), info)
    assert out + back == pytest.approx(2 * still, abs=1e-6)


@pytest.mark.parametrize("reward", [scout_reward, payload_reward])
def test_stationary_at_the_waypoint_beats_moving_through_it(reward):
    info = info_for([0, 0, -5])
    still = reward(at([0, 0, -5]), HOVER, at([0, 0, -5]), info)
    moving = reward(at([0, 0, -5]), HOVER, at([0, 0, -5], [3, 0, 0]), info)
    assert still > moving


@pytest.mark.parametrize("reward", [scout_reward, payload_reward])
def test_arriving_beats_loitering_for_the_whole_discount_horizon(reward):
    """If it did not, the policy would learn never to complete a waypoint."""
    info = info_for([0, 0, -5])
    state = at([0, 0, -5])
    still = reward(state, HOVER, state, info)
    arrived = reward(state, HOVER, state, {**info, "arrived": True})
    horizon_steps = 200
    assert arrived - still > 0.2 * horizon_steps * 0.5


@pytest.mark.parametrize("reward", [scout_reward, payload_reward])
def test_crashing_costs_more_than_any_arrival_pays(reward):
    info = info_for([0, 0, -5])
    state = at([0, 0, -5])
    crash = reward(state, HOVER, state, {**info, "terminated": True})
    arrived = reward(state, HOVER, state, {**info, "arrived": True})
    assert crash < -arrived


def test_payload_cares_more_about_steadiness_than_scout():
    info = info_for([0, 0, -5])
    calm = at([0, 0, -5])
    busy = at([0, 0, -5], rates=[2, 2, 0])

    def cost(reward):
        return reward(calm, HOVER, calm, info) - reward(calm, HOVER, busy, info)

    assert cost(payload_reward) > cost(scout_reward)


# ---------------------------------------------------------------------- Rendering


def test_render_is_off_by_default():
    assert make().render() is None


def test_unknown_render_mode_is_refused():
    with pytest.raises(ValueError):
        BaseDroneEnv(scout_reward, make_config(), render_mode="human")


def test_rgb_array_shows_the_drone():
    env = BaseDroneEnv(scout_reward, make_config(), render_mode="rgb_array")
    env.reset(seed=0)
    try:
        frame = env.render()
    except Exception as error:  # No OpenGL, as on a headless machine.
        pytest.skip(f"Rendering unavailable: {error}")
    finally:
        env.close()
    assert frame.shape == (480, 640, 3)
    assert frame.std() > 1.0


def climb_env(mass_kg=1.5):
    env = BaseDroneEnv(
        reward_func=lambda *_: 0.0,
        config=dataclasses.replace(
            make_config(),
            control_mode=ControlMode.CLIMB,
            randomization=dataclasses.replace(make_config().randomization, enabled=False),
            noise=dataclasses.replace(make_config().noise, enabled=False),
        ),
    )
    env.reset(seed=0)
    env.domain = dataclasses.replace(env.domain, mass_kg=mass_kg)
    env._model.body_mass[env._body_id] = mass_kg
    return env


def test_climb_mode_scales_the_fourth_action_to_a_climb_rate():
    env = climb_env()
    command = env._action_to_command(np.array([0, 0, 0, 1.0], dtype=np.float32))
    assert command[3] == pytest.approx(env.config.max_vertical_speed_mps)


@pytest.mark.parametrize("mass_kg", [1.35, 1.5, 1.65])
def test_climb_mode_holds_height_whatever_the_mass(mass_kg):
    env = climb_env(mass_kg)
    command = np.zeros(4)
    for _ in range(400):
        state = env._physics_step(command, env.dt)
    assert abs(state.velocity[2] + env._gps_velocity_error[2]) < 0.02


def test_climb_mode_climbs_at_the_commanded_rate():
    env = climb_env()
    command = np.array([0.0, 0.0, 0.0, 1.5])
    for _ in range(150):
        state = env._physics_step(command, env.dt)
    assert -state.velocity[2] == pytest.approx(1.5, abs=0.05)


def test_climb_mode_stays_inside_the_throttle_limits():
    env = climb_env()
    weight = env._nominal_mass * env._gravity
    full = weight / env.domain.hover_thrust
    assert env._climb_to_thrust(100.0, env.dt) == pytest.approx(env.config.max_thrust * full)
    assert env._climb_to_thrust(-100.0, env.dt) == pytest.approx(env.config.min_thrust * full)
