"""The fast training path must change the speed and nothing else.

Nothing here trains on the task. The worker pool is driven with random actions, and
the GPU update runs on a made-up rollout, to compare it with stock PPO.
"""

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from agents.batched_vec_env import BatchedVecEnv
from agents.graphed_ppo import GraphedPPO, GraphedUpdate
from agents.scout.train import make_env

N_ENVS = 6
STEPS = 1200
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def batched():
    envs = BatchedVecEnv(make_env, N_ENVS, 2, seed=11)
    yield envs
    envs.close()


def test_the_pool_flies_exactly_as_subproc_vec_env_does():
    """Same seeds and actions, including through crashes and automatic resets."""
    fast = BatchedVecEnv(make_env, N_ENVS, 2, seed=5)
    stock = make_vec_env(
        make_env,
        n_envs=N_ENVS,
        seed=5,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )
    try:
        assert np.array_equal(fast.reset(), stock.reset())
        rng = np.random.default_rng(0)
        finished = 0
        for _ in range(STEPS):
            actions = rng.uniform(-1.0, 1.0, (N_ENVS, 4)).astype(np.float32)
            fast_result, stock_result = fast.step(actions), stock.step(actions)
            for ours, theirs in zip(fast_result[:3], stock_result[:3]):
                assert np.array_equal(ours, theirs)
                assert ours.dtype == theirs.dtype
            for done, ours, theirs in zip(fast_result[2], fast_result[3], stock_result[3]):
                if done:
                    finished += 1
                    assert ours["waypoints_reached"] == theirs["waypoints_reached"]
                    assert ours["termination"] == theirs["termination"]
                    assert ours["is_success"] == theirs["is_success"]
                    assert ours["episode"]["r"] == theirs["episode"]["r"]
                    assert ours["TimeLimit.truncated"] == theirs["TimeLimit.truncated"]
                    assert np.array_equal(
                        ours["terminal_observation"], theirs["terminal_observation"]
                    )
        assert finished > 0
    finally:
        fast.close()
        stock.close()


def test_the_pool_reaches_every_environment(batched):
    batched.env_method("set_difficulty", 0.25)
    assert batched.get_attr("difficulty") == [0.25] * N_ENVS
    batched.env_method("set_difficulty", 0.75, indices=[1, 4])
    assert batched.get_attr("difficulty", indices=[4, 0, 1]) == [0.75, 0.25, 0.75]
    assert batched.env_is_wrapped(Monitor) == [True] * N_ENVS


def test_every_callback_finds_what_it_reads_when_an_episode_ends(batched):
    """CurriculumCallback, EpisodeMetricsCallback, EvalCallback and PPO itself."""
    batched.reset()
    actions = np.ones((N_ENVS, 4), dtype=np.float32)
    for _ in range(1000):
        _, _, dones, infos = batched.step(actions)
        if dones.any():
            break
    assert dones.any()
    for done, info in zip(dones, infos):
        if done:
            for key in (
                "waypoints_reached", "termination", "is_success", "episode",
                "terminal_observation", "TimeLimit.truncated",
            ):
                assert key in info
        else:
            assert info == {}


def fake_rollout(model: PPO) -> None:
    """Fill the buffer with made-up experience; no environment is involved."""
    rng = np.random.default_rng(3)
    buffer = model.rollout_buffer
    buffer.reset()
    shape = (model.n_steps, model.n_envs)
    buffer.observations[:] = rng.normal(size=buffer.observations.shape)
    buffer.actions[:] = rng.uniform(-1.0, 1.0, buffer.actions.shape)
    buffer.advantages[:] = rng.normal(size=shape)
    buffer.returns[:] = rng.normal(size=shape)
    buffer.values[:] = rng.normal(size=shape)
    buffer.log_probs[:] = rng.normal(-5.0, 0.5, shape)
    buffer.pos = model.n_steps
    buffer.full = True


def tiny_model(cls: type, **kwargs) -> PPO:
    model = cls(
        "MlpPolicy",
        DummyVecEnv([make_env]),
        n_steps=512,
        batch_size=128,
        n_epochs=3,
        device="cpu",
        seed=1,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}, "log_std_init": -1.0},
        **kwargs,
    )
    model.set_logger(configure(None, []))
    model._current_progress_remaining = 0.7
    fake_rollout(model)
    return model


def weights(model: PPO) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in model.policy.parameters()])


@needs_cuda
def test_the_graphed_update_matches_stock_ppo(monkeypatch):
    """The same batches in the same order must give the same weights."""
    monkeypatch.setattr(np.random, "permutation", lambda n: np.arange(n))
    monkeypatch.setattr(
        GraphedUpdate,
        "_batch_order",
        lambda self: torch.arange(self.rollout_size, device=self.device),
    )
    stock = tiny_model(PPO)
    graphed = tiny_model(GraphedPPO, update_device="cuda")
    start = weights(stock)
    assert torch.equal(start, weights(graphed))

    stock.train()
    graphed.train()

    moved = (weights(stock) - start).abs().max()
    assert moved > 1e-3
    assert (weights(stock) - weights(graphed)).abs().max() < 1e-3 * moved + 1e-5
    assert stock._n_updates == graphed._n_updates
    for key in (
        "train/entropy_loss", "train/policy_gradient_loss", "train/value_loss",
        "train/approx_kl", "train/clip_fraction", "train/loss", "train/std",
    ):
        expected = stock.logger.name_to_value[key]
        assert graphed.logger.name_to_value[key] == pytest.approx(expected, rel=1e-3, abs=1e-5)


@needs_cuda
def test_a_second_update_continues_adam_rather_than_restarting_it(monkeypatch):
    monkeypatch.setattr(np.random, "permutation", lambda n: np.arange(n))
    monkeypatch.setattr(
        GraphedUpdate,
        "_batch_order",
        lambda self: torch.arange(self.rollout_size, device=self.device),
    )
    stock = tiny_model(PPO)
    graphed = tiny_model(GraphedPPO, update_device="cuda")
    for model in (stock, graphed):
        model.train()
        fake_rollout(model)
        model.train()
    drift = (weights(stock) - weights(graphed)).abs().max()
    assert drift < 1e-4


@needs_cuda
def test_a_saved_graphed_model_is_a_plain_ppo_model_and_resumes(tmp_path):
    graphed = tiny_model(GraphedPPO, update_device="cuda")
    graphed.train()
    path = tmp_path / "model"
    graphed.save(path)

    plain = PPO.load(path, device="cpu")
    assert torch.equal(weights(graphed), weights(plain))

    resumed = GraphedPPO.load(path, env=DummyVecEnv([make_env]), device="cpu", update_device="cuda")
    state = resumed.policy.optimizer.state_dict()["state"]
    assert float(state[0]["step"]) == graphed.n_epochs * 4
    resumed.set_logger(graphed.logger)
    fake_rollout(resumed)
    before = weights(resumed)
    resumed.train()
    assert not torch.equal(before, weights(resumed))
    assert float(resumed.policy.optimizer.state_dict()["state"][0]["step"]) == 2 * graphed.n_epochs * 4


@needs_cuda
def test_the_graph_refuses_a_batch_size_it_cannot_replay():
    model = tiny_model(GraphedPPO, update_device="cuda")
    model.batch_size = 100
    with pytest.raises(ValueError, match="batch_size"):
        GraphedUpdate(model, "cuda")


def test_cpu_updates_are_the_stock_algorithm():
    stock = tiny_model(PPO)
    plain = tiny_model(GraphedPPO, update_device="cpu")
    state = np.random.get_state()
    stock.train()
    np.random.set_state(state)
    plain.train()
    assert torch.equal(weights(stock), weights(plain))
