"""Checks on the camera-only task: field of view, detector output and the hold."""

import dataclasses
import math

import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from agents.scout import scout_reward
from agents.scout.train import make_config, make_env
from envs import (
    MARKER_OBS_DIM,
    DomainRandomizationConfig,
    MarkerHoldEnv,
    MarkerTaskConfig,
    SensorNoiseConfig,
)
from envs.base_drone_env import _euler_to_quaternion
from envs.marker_env import BOX_LAGS, OBS_ACTION, OBS_ATTITUDE, OBS_HOLD, OBS_RATES

HOVER = np.zeros(4, dtype=np.float32)


def perfect(**task) -> MarkerHoldEnv:
    """The nominal vehicle with a perfect detector."""
    config = dataclasses.replace(
        make_config(),
        randomization=DomainRandomizationConfig(enabled=False),
        noise=SensorNoiseConfig(enabled=False),
        spawn_speed_mps=0.0,
        spawn_tilt_rad=0.0,
        spawn_rate_rps=0.0,
        marker_task=MarkerTaskConfig(**task),
    )
    env = MarkerHoldEnv(scout_reward, config)
    env.reset(seed=0)
    return env


def pose(env, distance=10.0, height=4.0, roll=0.0, pitch=0.0, yaw_off=0.0):
    """Place the vehicle north of the marker, looking at it, and return the box."""
    position = env._marker + np.array([distance, 0.0, -height])
    quaternion = _euler_to_quaternion(roll, pitch, math.pi + yaw_off)
    env._data.qpos[:3] = position
    env._data.qpos[3:7] = quaternion
    env._data.qvel[:] = 0.0
    mujoco.mj_forward(env._model, env._data)
    env._state = env._read_state()
    return env._project_marker(position, quaternion)


# ---------------------------------------------------------------------- Contract


def test_gymnasium_contract():
    check_env(make_env(), skip_render_check=True)


def test_observation_stays_in_its_space():
    env = make_env()
    observation, _ = env.reset(seed=1)
    assert observation.shape == (MARKER_OBS_DIM,)
    rng = np.random.default_rng(0)
    for _ in range(500):
        observation, _, terminated, truncated, _ = env.step(
            rng.uniform(-1, 1, 4).astype(np.float32)
        )
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            observation, _ = env.reset()


def test_same_seed_same_episode():
    def rollout():
        env = make_env()
        env.reset(seed=3)
        return [env.step(HOVER)[0] for _ in range(50)]

    assert np.array_equal(rollout(), rollout())


def test_gps_never_reaches_the_policy():
    """Change only the position estimate: the observation must not move."""
    def rollout(error):
        env = make_env()
        env.reset(seed=3)
        env._gps_position_error = np.array(error, dtype=float)
        env._gps_glitch = np.zeros(3)
        return [env.step(HOVER)[0] for _ in range(30)]

    assert np.array_equal(rollout([0, 0, 0]), rollout([30, -20, 10]))


def test_velocity_never_reaches_the_policy():
    def observation_with(velocity_error):
        env = perfect()
        env._gps_velocity_error = np.array(velocity_error, dtype=float)
        return env.step(HOVER)[0]

    assert np.array_equal(observation_with([0, 0, 0]), observation_with([5, 5, 5]))


# ---------------------------------------------------------------------- Field of view


def test_marker_ahead_gives_a_box():
    box = pose(perfect(), distance=10.0, height=3.64)
    assert box.any()
    assert abs(box[0]) < 0.01  # Dead ahead.
    assert abs(box[1]) < 0.05  # On the camera axis: tan(0.35) * 10 = 3.65 m.


def test_yawing_too_far_loses_the_marker():
    env = perfect()
    assert pose(env, yaw_off=0.3).any()
    assert not pose(env, yaw_off=0.8).any()
    assert not pose(env, yaw_off=math.pi).any()


def test_pitching_too_far_loses_the_marker():
    env = perfect()
    assert pose(env, height=3.64, pitch=0.2).any()
    assert not pose(env, height=3.64, pitch=0.6).any()  # Nose up: view over it.
    assert not pose(env, height=3.64, pitch=-0.6).any()  # Nose down: view short of it.


def test_a_marker_underneath_is_out_of_view():
    assert not pose(perfect(), distance=0.5, height=5.0).any()


def test_yawing_right_moves_the_marker_left_in_the_image():
    env = perfect()
    assert pose(env, yaw_off=0.2)[0] < -0.1
    assert pose(env, yaw_off=-0.2)[0] > 0.1


def test_nose_up_moves_the_marker_down_in_the_image():
    env = perfect()
    level = pose(env, height=3.64)[1]
    assert pose(env, height=3.64, pitch=0.15)[1] > level + 0.1


def test_box_shrinks_with_range():
    env = perfect()
    near, far = pose(env, distance=6.0, height=2.2), pose(env, distance=12.0, height=4.4)
    assert near[2] == pytest.approx(2 * far[2], rel=0.25)


def test_box_width_does_not_depend_on_the_viewing_bearing():
    """Box size is the only cue to range, so it must not change around the marker."""
    env = perfect()
    widths = []
    for bearing in np.linspace(0.0, math.pi / 2, 7):
        position = env._marker + np.array(
            [10 * math.cos(bearing), 10 * math.sin(bearing), -3.64]
        )
        widths.append(
            env._project_marker(position, _euler_to_quaternion(0.0, 0.0, bearing + math.pi))[2]
        )
    assert max(widths) == pytest.approx(min(widths), rel=0.03)


def test_too_small_a_box_is_not_detected():
    env = perfect(min_box_px=8.0)
    assert pose(env, distance=20.0, height=7.0).any()
    assert not pose(env, distance=120.0, height=43.0).any()


def test_box_is_cut_off_at_the_edge_of_the_frame():
    env = perfect(marker_size_m=4.0)
    box = pose(env, yaw_off=0.5)
    assert box.any()
    assert box[0] - box[2] / 2 == pytest.approx(-1.0, abs=1e-6)


def test_marker_drops_out_when_its_centre_leaves_the_frame():
    """Part of a large marker is still in frame here, but its centre is not."""
    env = perfect(marker_size_m=4.0)
    assert pose(env, yaw_off=0.5).any()
    assert not pose(env, yaw_off=0.7).any()


def test_camera_mount_setting_moves_the_view():
    level, tilted = perfect(camera_pitch_down_rad=0.0), perfect(camera_pitch_down_rad=0.5)
    assert pose(level, distance=8.0, height=6.0).sum() == 0.0
    assert pose(tilted, distance=8.0, height=6.0).any()


# ---------------------------------------------------------------------- Detector


def noisy_env() -> MarkerHoldEnv:
    env = make_env()
    env.reset(seed=1)
    env.domain = dataclasses.replace(env.domain, noise_scale=1.0, camera_tilt_error_rad=0.0)
    return env


def test_detector_error_is_a_few_pixels():
    env = noisy_env()
    truth = pose(env, height=3.6)
    state = dataclasses.replace(env._state, angular_velocity=np.zeros(3))
    boxes = np.array([env._detect_box(state) for _ in range(2000)])
    # Leave out false detections, which land anywhere in the frame.
    seen = boxes[np.abs(boxes[:, :2] - truth[:2]).max(axis=1) < 0.05]
    assert seen[:, :2].std(axis=0) * 320 == pytest.approx([2.0, 2.0], rel=0.25)


def test_detections_drop_out():
    env = noisy_env()
    pose(env, height=3.6)
    state = dataclasses.replace(env._state, angular_velocity=np.zeros(3))
    seen = np.mean([env._detect_box(state)[2] > 0 for _ in range(2000)])
    assert 0.85 < seen < 0.99


def test_fast_rotation_blurs_detections_away():
    env = noisy_env()
    pose(env, height=3.6)

    def seen(rate):
        state = dataclasses.replace(env._state, angular_velocity=np.array([rate, 0.0, 0.0]))
        return np.mean([env._detect_box(state)[2] > 0 for _ in range(1000)])

    assert seen(4.0) < 0.5 * seen(0.0)


def test_false_detections_happen_out_of_view():
    noise = SensorNoiseConfig(vision_false_positive_prob=0.2)
    env = MarkerHoldEnv(scout_reward, dataclasses.replace(make_config(), noise=noise))
    env.reset(seed=1)
    pose(env, yaw_off=math.pi)
    assert np.mean([env._detect_box(env._state)[2] > 0 for _ in range(2000)]) == pytest.approx(
        0.2, abs=0.05
    )


def test_box_only_changes_on_a_new_frame():
    env = noisy_env()
    env.domain = dataclasses.replace(env.domain, camera_frame_hz=10.0)
    pose(env, height=3.6)
    frames = []
    for step in range(20):
        env._step_count = step
        frames.append(env._camera_slots(env._state).copy())
    assert sum(not np.array_equal(a, b) for a, b in zip(frames, frames[1:])) <= 3


# ---------------------------------------------------------------------- Observation


def test_observation_layout():
    env = perfect()
    pose(env, height=3.64)
    env._camera_frame = env._project_marker(env._state.position, env._state.quaternion)
    observation, *_ = env.step(HOVER)
    assert observation[4] == 1.0  # Newest box is visible.
    assert observation[:4] == pytest.approx(env._latest_box, abs=1e-5)
    assert observation[OBS_ATTITUDE : OBS_ATTITUDE + 2] == pytest.approx([0, 0], abs=0.02)
    assert observation[OBS_RATES : OBS_RATES + 3] == pytest.approx(
        env._state.angular_velocity, abs=1e-4
    )
    assert observation[OBS_ACTION : OBS_ACTION + 4] == pytest.approx(env._last_action)
    assert observation[OBS_HOLD:] == pytest.approx([env._hold_range, env._hold_height])


def test_history_holds_older_boxes_at_the_right_lags():
    env = perfect()
    newest = []
    observation = None
    for step in range(40):
        pose(env, distance=20.0 - 0.2 * step, height=6.0)
        observation, *_ = env.step(HOVER)
        newest.append(observation[:5].copy())
    for slot, lag in enumerate(BOX_LAGS):
        assert observation[5 * slot : 5 * slot + 5] == pytest.approx(newest[-1 - lag])


def test_a_missing_detection_is_all_zeros():
    env = perfect()
    pose(env, yaw_off=math.pi)
    env._step_count = 1  # The next step is a new frame at 25 Hz.
    observation, *_ = env.step(HOVER)
    assert not observation[:5].any()


# ---------------------------------------------------------------------- Ending an episode


def test_losing_the_marker_ends_the_episode_after_the_timeout():
    env = perfect(lost_timeout_s=0.5)
    pose(env, yaw_off=math.pi)
    for step in range(1, 30):
        _, _, terminated, _, info = env.step(HOVER)
        pose(env, yaw_off=math.pi)
        if terminated:
            break
    assert info["termination"] == "marker_lost"
    assert step in (25, 26)  # 0.5 s at 50 Hz, after the frame already in hand.


def test_a_brief_dropout_does_not_end_the_episode():
    env = perfect(lost_timeout_s=0.5)
    env._lost_steps = 20
    pose(env, height=3.64)
    _, _, terminated, _, _ = env.step(HOVER)
    assert not terminated
    assert env._lost_steps == 0


def test_coming_too_close_ends_the_episode():
    env = perfect(max_box_width=0.8)
    pose(env, distance=3.0, height=2.0)
    env._latest_box = np.array([0.0, 0.0, 0.9, 0.5])
    assert env._termination_reason(env._state) == "too_close"


def test_every_spawn_starts_with_the_marker_in_view():
    env = make_env()
    for seed in range(200):
        env.reset(seed=seed)
        assert env._project_marker(env._state.position, env._state.quaternion).any()
        assert env._termination_reason(env._state) == ""


def test_the_marker_never_moves():
    env = make_env()
    env.reset(seed=0)
    start = env._marker.copy()
    rng = np.random.default_rng(0)
    for _ in range(200):
        _, _, terminated, truncated, _ = env.step(rng.uniform(-0.3, 0.3, 4).astype(np.float32))
        assert np.array_equal(env._marker, start)
        if terminated or truncated:
            break
    assert env._marker[2] == 0.0


# ---------------------------------------------------------------------- The hold


def test_hold_is_any_bearing_at_the_set_range_and_height():
    env = perfect()
    env._hold_range, env._hold_height = 10.0, 4.0
    for bearing in (0.0, 1.0, 2.5):
        position = env._marker + np.array([20 * math.cos(bearing), 20 * math.sin(bearing), -9.0])
        point = env._ring_point(position)
        assert np.linalg.norm(point[:2] - env._marker[:2]) == pytest.approx(10.0)
        assert point[2] == -4.0


def test_every_hold_is_reachable_and_in_view():
    env = make_env()
    task = env.config.marker_task
    for seed in range(100):
        env.reset(seed=seed)
        assert task.hold_range_m[0] <= env._hold_range <= task.hold_range_m[1]
        box = pose(env, distance=env._hold_range, height=env._hold_height)
        assert box.any()
        assert box[2] < task.max_box_width
        assert abs(box[1]) < 0.7


def test_holding_still_completes_a_hold():
    env = perfect(new_hold_prob=0.0)
    env._hold_range, env._hold_height = 10.0, 3.64
    state = dataclasses.replace(
        env._state,
        position=env._marker + np.array([10.0, 0.0, -3.64]),
        velocity=np.zeros(3),
    )
    assert sum(env._update_hold(state) for _ in range(50)) == 1


def test_a_fast_pass_does_not_complete_a_hold():
    env = perfect()
    env._hold_range, env._hold_height = 10.0, 3.64
    state = dataclasses.replace(
        env._state,
        position=env._marker + np.array([10.0, 0.0, -3.64]),
        velocity=np.array([3.0, 0.0, 0.0]),
    )
    assert not any(env._update_hold(state) for _ in range(100))


def test_reward_prefers_facing_the_marker():
    def step_reward(yaw_off):
        env = perfect()
        pose(env, height=3.64, yaw_off=yaw_off)
        return env.step(HOVER)[1]

    assert step_reward(0.0) > step_reward(0.4)


def test_nothing_pays_for_flying_past_the_hold_towards_the_marker():
    env = perfect()
    env._hold_range, env._hold_height = 10.0, 3.64
    at_hold = env._marker + np.array([10.0, 0.0, -3.64])
    closer = env._marker + np.array([5.0, 0.0, -3.64])
    assert np.linalg.norm(env._ring_point(at_hold) - at_hold) == pytest.approx(0.0)
    assert np.linalg.norm(env._ring_point(closer) - closer) == pytest.approx(5.0)
