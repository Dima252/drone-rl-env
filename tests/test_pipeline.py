"""Run the real train, resume, evaluate and export commands on a tiny budget."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

WORKSPACE = Path(__file__).parents[1] / "drone_rl_workspace"
OBS_SIZE = 51
TINY = [
    "--n-envs", "2",
    "--n-workers", "2",
    "--n-steps", "2048",
    "--eval-freq", "2048",
    "--n-eval-episodes", "1",
    "--checkpoint-freq", "2048",
]


def run(*command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", *command],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train, then resume, a throwaway scout run; remove it afterwards."""
    name = f"_pytest_{tmp_path_factory.getbasetemp().name}"
    run_dir = WORKSPACE / "runs" / "scout" / name
    first = run(
        "agents.scout.train", "--total-timesteps", "4096", "--run-name", name, *TINY
    )
    assert first.returncode == 0, first.stderr[-2000:]
    yield run_dir
    import shutil

    shutil.rmtree(run_dir, ignore_errors=True)


def test_a_run_records_how_it_was_trained(trained):
    config = json.loads((trained / "training_config.json").read_text())
    assert config["control_mode"] == "attitude"
    assert config["control_hz"] == 50.0
    assert config["agent"] == "scout"
    assert config["env"]["max_position_m"] == 50.0
    assert config["packages"]["stable-baselines3"]
    assert config["git_commit"]


def test_a_run_name_cannot_be_overwritten_by_accident(trained):
    result = run(
        "agents.scout.train", "--total-timesteps", "2048",
        "--run-name", trained.name, *TINY,
    )
    assert result.returncode != 0
    assert "--resume" in result.stderr


def test_resume_continues_from_the_latest_checkpoint(trained):
    result = run(
        "agents.scout.train", "--total-timesteps", "12288",
        "--run-name", trained.name, "--resume", *TINY,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "Resuming from" in result.stdout
    assert "Traceback" not in result.stderr
    assert list((trained / "checkpoints").glob("scout_12288_steps.zip"))


def test_evaluate_reports_every_scenario(trained):
    result = run(
        "agents.evaluate", str(trained), "--episodes", "1",
        "--max-crash-rate", "1.0", "--max-worst-case-crash-rate", "1.0",
        "--max-unseen-crash-rate", "1.0", "--min-waypoint-rate", "0.0",
    )
    assert result.returncode == 0, result.stderr[-2000:]
    for scenario in ("nominal", "randomized", "worst case", "unseen", "PASS"):
        assert scenario in result.stdout


def test_evaluate_fails_a_policy_above_the_crash_limit(trained):
    result = run(
        "agents.evaluate", str(trained), "--episodes", "1",
        "--max-crash-rate", "-1", "--max-worst-case-crash-rate", "-1",
    )
    assert result.returncode == 1
    assert "FAIL" in result.stdout


def test_evaluate_fails_a_policy_that_survives_without_doing_the_job(trained):
    result = run(
        "agents.evaluate", str(trained), "--episodes", "1",
        "--max-crash-rate", "1.0", "--max-worst-case-crash-rate", "1.0",
        "--max-unseen-crash-rate", "1.0", "--min-waypoint-rate", "1.1",
    )
    assert result.returncode == 1
    assert "FAIL" in result.stdout


def test_export_matches_what_the_node_expects(trained):
    result = run("agents.scout.export_onnx", str(trained))
    assert result.returncode == 0, result.stderr[-2000:]

    session = ort.InferenceSession(str(trained / "export" / "model.onnx"))
    assert session.get_inputs()[0].shape == [1, OBS_SIZE]
    action = session.run(None, {"observation": np.zeros((1, OBS_SIZE), np.float32)})[0]
    assert action.shape == (1, 4)

    with np.load(trained / "export" / "policy_meta.npz") as meta:
        assert float(meta["control_hz"]) == 50.0
        assert float(meta["action_filter_alpha"]) == 0.5
        assert str(meta["control_mode"]) == "attitude"
        assert meta["obs_mean"].shape == (OBS_SIZE,)
        assert float(meta["max_box_width"]) == 0.8
        assert float(meta["lost_timeout_s"]) == 0.5


def test_a_lab_run_hands_on_to_a_realistic_run(trained):
    import shutil

    runs = trained.parent
    lab = f"{trained.name}_lab"
    realistic = f"{trained.name}_realistic"
    try:
        result = run(
            "agents.scout.train", "--lab", "--total-timesteps", "4096",
            "--run-name", lab, *TINY,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        config = json.loads((runs / lab / "training_config.json").read_text())
        assert config["env"]["noise"]["enabled"] is False
        assert config["env"]["randomization"]["enabled"] is False
        assert not (runs / lab / "curriculum.txt").exists()

        result = run(
            "agents.evaluate", str(runs / lab), "--lab", "--episodes", "1",
            "--max-crash-rate", "1.0",
        )
        assert "nominal" in result.stdout
        assert "randomized" not in result.stdout

        result = run(
            "agents.scout.train", "--start-from", str(runs / lab),
            "--total-timesteps", "4096", "--run-name", realistic, *TINY,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "Starting from" in result.stdout
        config = json.loads((runs / realistic / "training_config.json").read_text())
        assert config["env"]["noise"]["enabled"] is True
    finally:
        shutil.rmtree(runs / lab, ignore_errors=True)
        shutil.rmtree(runs / realistic, ignore_errors=True)


def test_payload_trains_with_the_same_machinery():
    name = "_pytest_payload"
    run_dir = WORKSPACE / "runs" / "payload" / name
    import shutil

    shutil.rmtree(run_dir, ignore_errors=True)
    try:
        result = run(
            "agents.payload.train", "--total-timesteps", "2048",
            "--run-name", name, *TINY,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert (run_dir / "final" / "model.zip").is_file()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
