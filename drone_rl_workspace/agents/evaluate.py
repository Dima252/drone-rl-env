"""Measure a trained policy, in simulation, on the job it was trained for.

Run from inside ``drone_rl_workspace``:

    python -m agents.evaluate runs/scout/<run-name>

Each scenario flies the policy deterministically with the observation handling it
trained with. The exit code is non-zero if a scenario breaks its limits, so this can
gate an export. A policy has to do the job as well as survive, because hovering in
place never crashes. The worst case only limits crashes: it pins every range at its
hard end at once, where staying safe is the job.

``--lab`` judges a lab run on ``nominal`` alone, with a higher bar.

``unseen`` flies conditions beyond every training range. The gap between it and
``randomized`` is the direct measure of overfitting to those ranges. ``hold m`` is
the true distance from the hold point while settled there.

The reward is not reported because it differs between drones.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.scout.train import lab_config, make_config, make_env
from envs import DroneEnvConfig, MarkerHoldEnv


def worst_case(config: DroneEnvConfig) -> DroneEnvConfig:
    """Every training range pinned at its hard end at once."""
    cfg = config.randomization
    low_end = {"attitude_gain_scale", "motor_efficiency", "gps_update_hz"}
    pinned = {
        item.name: (getattr(cfg, item.name)[0 if item.name in low_end else 1],) * 2
        for item in dataclasses.fields(cfg)
        if isinstance(getattr(cfg, item.name), tuple) and item.name != "updraft_mps"
    }
    return dataclasses.replace(
        config,
        randomization=dataclasses.replace(cfg, motor_fault_prob=1.0, **pinned),
    )


def unseen(config: DroneEnvConfig) -> DroneEnvConfig:
    """Conditions outside every training range, none of which the policy has met."""
    cfg = config.randomization
    return dataclasses.replace(
        config,
        randomization=dataclasses.replace(
            cfg,
            mass_kg=(1.1, 1.95),
            wind_speed_mps=(5.0, 7.0),
            gust_std_mps=(1.5, 2.2),
            motor_efficiency=(0.85, 0.9),
            noise_scale=(2.0, 3.0),
            motor_lag_scale=(2.0, 2.5),
            action_delay_steps=(3, 4),
            link_jitter_steps=(0, 1),
            link_drop_prob=(0.05, 0.1),
            push_force_n=(6.0, 8.0),
        ),
    )


SCENARIOS = {
    "nominal": lab_config,
    "randomized": lambda config: config,
    "worst case": worst_case,
    "unseen": unseen,
}
# These only have to stay safe, because each is harder than the job as specified.
SAFETY_ONLY = {"worst case": "max_worst_case_crash_rate", "unseen": "max_unseen_crash_rate"}
# A lab run must beat the final bar, because the realistic stage will cost it some.
LAB_MIN_WAYPOINT_RATE = 0.9


def fly(
    model: PPO, stats: VecNormalize, config: DroneEnvConfig, episodes: int, seed: int
) -> dict:
    """Fly ``episodes`` episodes and return the summary of one scenario."""
    env = MarkerHoldEnv(reward_func=lambda *_: 0.0, config=config)
    terminations: Counter[str] = Counter()
    reached, seconds, errors, hold_errors, tilts, jerks = [], [], [], [], [], []

    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        done = False
        previous = np.zeros(4, dtype=np.float32)
        while not done:
            action, _ = model.predict(
                stats.normalize_obs(observation), deterministic=True
            )
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = env._state
            error = float(np.linalg.norm(info["waypoint"] - state.position))
            errors.append(error)
            if error < 3.0 and np.linalg.norm(state.velocity) < 1.0:
                hold_errors.append(error)
            tilts.append(state.tilt)
            jerks.append(np.abs(action - previous).max())
            previous = action
        terminations[info["termination"]] += 1
        reached.append(info["waypoints_reached"])
        seconds.append(info["step"] * env.dt)

    env.close()
    crashes = episodes - terminations[""]
    return {
        "crash_rate": crashes / episodes,
        "causes": {k: v for k, v in terminations.items() if k},
        "waypoints_per_minute": 60.0 * sum(reached) / sum(seconds),
        "any_waypoint": float(np.mean(np.array(reached) > 0)),
        "median_error_m": float(np.median(errors)),
        "hold_error_m": float(np.median(hold_errors)) if hold_errors else float("nan"),
        "p95_tilt_rad": float(np.percentile(tilts, 95)),
        "mean_action_change": float(np.mean(jerks)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", type=Path, help="default: best_model/best_model.zip")
    parser.add_argument(
        "--vecnormalize", type=Path, help="default: best_model/vecnormalize.pkl"
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--max-crash-rate", type=float, default=0.02)
    parser.add_argument("--max-worst-case-crash-rate", type=float, default=0.10)
    parser.add_argument("--max-unseen-crash-rate", type=float, default=0.15)
    parser.add_argument(
        "--min-waypoint-rate",
        type=float,
        default=0.8,
        help="share of episodes completing a hold, except the last two scenarios",
    )
    parser.add_argument(
        "--lab",
        action="store_true",
        help="judge a lab run: the nominal scenario alone, at LAB_MIN_WAYPOINT_RATE",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model or args.run_dir / "best_model" / "best_model.zip"
    stats_path = args.vecnormalize or args.run_dir / "best_model" / "vecnormalize.pkl"

    stats = VecNormalize.load(str(stats_path), DummyVecEnv([make_env]))
    stats.training = False
    stats.norm_reward = False
    model = PPO.load(model_path, device="cpu")

    passed = True
    print(
        f"{'scenario':12s} {'crashes':>8s} {'wp/min':>7s} {'any wp':>7s} "
        f"{'err m':>6s} {'hold m':>7s} {'tilt95':>7s} {'jerk':>6s}  causes"
    )
    scenarios = {"nominal": lab_config} if args.lab else SCENARIOS
    min_waypoint_rate = LAB_MIN_WAYPOINT_RATE if args.lab else args.min_waypoint_rate
    for name, build in scenarios.items():
        result = fly(model, stats, build(make_config()), args.episodes, args.seed)
        if name in SAFETY_ONLY:
            ok = result["crash_rate"] <= getattr(args, SAFETY_ONLY[name])
        else:
            ok = (
                result["crash_rate"] <= args.max_crash_rate
                and result["any_waypoint"] >= min_waypoint_rate
            )
        passed = passed and ok
        print(
            f"{name:12s} {result['crash_rate']:8.1%} "
            f"{result['waypoints_per_minute']:7.1f} {result['any_waypoint']:7.0%} "
            f"{result['median_error_m']:6.2f} {result['hold_error_m']:7.2f} "
            f"{result['p95_tilt_rad']:7.3f} "
            f"{result['mean_action_change']:6.3f}  {result['causes'] or '-'}"
        )

    print("PASS" if passed else "FAIL: a scenario broke its limits")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
