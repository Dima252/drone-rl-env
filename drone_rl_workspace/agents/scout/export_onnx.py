"""Export a trained policy for ros2_inference_node.py.

Run from inside ``drone_rl_workspace``:

    python -m agents.scout.export_onnx runs/<agent>/<run-name>

By default this exports ``best_model/`` from the run into ``<run>/export/``. Use
``--model`` and ``--vecnormalize`` to export a checkpoint instead.

Two files are written, and both must be copied to the companion computer:

* ``model.onnx``: the deterministic actor, normalized observation in, action out.
* ``policy_meta.npz``: the VecNormalize statistics and the training limits.

The export is checked before it is accepted: ONNX Runtime must reproduce the
actions Stable Baselines3 gives across the whole observation range.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.scout.train import make_env
from envs import ControlMode
from envs.base_drone_env import ACTION_DIM
from envs.marker_env import MARKER_OBS_DIM as OBS_DIM

MAX_ACTION_ERROR = 1e-4
CHECK_SAMPLES = 2000


class DeterministicActor(torch.nn.Module):
    """The part of the SB3 policy that maps an observation to its mean action."""

    def __init__(self, policy: torch.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(
            observation, self.policy.pi_features_extractor
        )
        return self.policy.action_net(self.policy.mlp_extractor.forward_actor(features))


def load_training_config(run_dir: Path) -> dict:
    """Read how this run was trained, refusing anything the node cannot fly."""
    path = run_dir / "training_config.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}, so the control mode of this run is unknown. Retrain "
            "with the current train.py."
        )
    config = json.loads(path.read_text())
    if config["control_mode"] != ControlMode.ATTITUDE.value:
        raise ValueError(
            f"This run trained {config['control_mode']!r} actions; the node only "
            "publishes attitude targets."
        )
    return config


def load_policy(model_path: Path, stats_path: Path) -> tuple[PPO, VecNormalize]:
    """Load a policy with its statistics, frozen as they were at save time."""
    for path in (model_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. A policy needs both files.")

    stats = VecNormalize.load(str(stats_path), DummyVecEnv([make_env]))
    stats.training = False
    stats.norm_reward = False
    model = PPO.load(model_path, device="cpu")

    if model.observation_space.shape != (OBS_DIM,):
        raise ValueError(f"Expected {OBS_DIM} observations: {model.observation_space}")
    if model.action_space.shape != (ACTION_DIM,):
        raise ValueError(f"Expected {ACTION_DIM} actions: {model.action_space}")
    if stats.obs_rms.count < 2:
        raise ValueError(f"{stats_path} holds no training statistics.")
    return model, stats


def export_model(model: PPO, path: Path) -> None:
    actor = DeterministicActor(model.policy).eval()
    torch.onnx.export(
        actor,
        (torch.zeros(1, OBS_DIM),),
        str(path),
        input_names=["observation"],
        output_names=["action"],
        opset_version=17,
        dynamo=False,
        external_data=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))


def export_meta(stats: VecNormalize, config: dict, path: Path) -> None:
    """Save everything the node must reproduce, which the network cannot carry."""
    space = stats.observation_space
    task = config["env"]["marker_task"]
    np.savez(
        path,
        horizontal_fov_rad=task["horizontal_fov_rad"],
        vertical_fov_rad=task["vertical_fov_rad"],
        camera_pitch_down_rad=task["camera_pitch_down_rad"],
        marker_size_m=task["marker_size_m"],
        lost_timeout_s=task["lost_timeout_s"],
        max_box_width=task["max_box_width"],
        control_mode=config["control_mode"],
        control_hz=config["control_hz"],
        action_filter_alpha=config["action_filter_alpha"],
        obs_mean=stats.obs_rms.mean,
        obs_var=stats.obs_rms.var,
        epsilon=stats.epsilon,
        clip_obs=stats.clip_obs,
        obs_low=space.low,
        obs_high=space.high,
        max_attitude_cmd_rad=config["max_attitude_cmd_rad"],
        max_yaw_rate_rps=config["max_yaw_rate_rps"],
    )


def verify_export(model: PPO, stats: VecNormalize, onnx_path: Path) -> float:
    """Return the worst action difference between SB3 and ONNX Runtime."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    space = stats.observation_space

    # Scaling down covers normal flight as well as the edges of the space.
    scale = rng.choice([0.02, 0.2, 1.0], size=(CHECK_SAMPLES, 1))
    observations = rng.uniform(space.low, space.high, (CHECK_SAMPLES, OBS_DIM)) * scale
    normalized = stats.normalize_obs(observations.astype(np.float32))

    worst = 0.0
    for sample in normalized:
        batch = sample.astype(np.float32).reshape(1, OBS_DIM)
        expected, _ = model.predict(batch, deterministic=True)
        actual = np.clip(session.run(None, {"observation": batch})[0], -1.0, 1.0)
        worst = max(worst, float(np.abs(actual - expected).max()))
    return worst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", type=Path, help="default: best_model/best_model.zip")
    parser.add_argument(
        "--vecnormalize", type=Path, help="default: best_model/vecnormalize.pkl"
    )
    parser.add_argument("--output", type=Path, help="default: <run_dir>/export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model or args.run_dir / "best_model" / "best_model.zip"
    stats_path = args.vecnormalize or args.run_dir / "best_model" / "vecnormalize.pkl"
    output = args.output or args.run_dir / "export"

    config = load_training_config(args.run_dir)
    model, stats = load_policy(model_path, stats_path)
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "model.onnx"
    meta_path = output / "policy_meta.npz"
    export_model(model, onnx_path)
    export_meta(stats, config, meta_path)

    error = verify_export(model, stats, onnx_path)
    if error > MAX_ACTION_ERROR:
        onnx_path.unlink()
        meta_path.unlink()
        raise RuntimeError(
            f"Export rejected: ONNX differs from the policy by {error:.2e}."
        )

    print(f"Worst action difference from SB3: {error:.2e}")
    print(f"Copy to the drone:\n  {onnx_path}\n  {meta_path}")


if __name__ == "__main__":
    main()
