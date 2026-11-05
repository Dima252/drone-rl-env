"""PPO with its gradient steps on the GPU, replayed from a CUDA graph.

The network is small, so a gradient step is thousands of tiny GPU launches. Launched
one by one those are slower than the CPU, especially on Windows. A CUDA graph records
one step and replays it with a single launch, with the same maths and batch size.

Collecting a rollout stays on the CPU, where this network answers faster than a
round trip to the GPU. So the policy stays on the CPU, a copy on the GPU does the
updating, and the weights and Adam state come back after every update. Saved models
are therefore ordinary PPO models, which ``PPO.load`` reads.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBufferSamples
from stable_baselines3.common.utils import explained_variance
from torch.distributions import Distribution

# Running sums kept on the GPU, because reading one back would stall every step.
POLICY_LOSS, VALUE_LOSS, ENTROPY_LOSS, CLIP_FRACTION, APPROX_KL, TOTAL_LOSS = range(6)


class GraphedUpdate:
    """The GPU copy of a policy and the recorded gradient step that trains it."""

    def __init__(self, model: PPO, device: str) -> None:
        rollout_size = model.n_steps * model.n_envs
        if model.clip_range_vf is not None or model.target_kl is not None:
            raise ValueError("A CUDA graph cannot use clip_range_vf or target_kl.")
        if model.use_sde or not isinstance(model.action_space, spaces.Box):
            raise ValueError("A CUDA graph needs continuous actions, without gSDE.")
        if rollout_size % model.batch_size:
            raise ValueError(
                f"batch_size {model.batch_size} must divide the rollout size of "
                f"{rollout_size}, because a graph has one fixed batch size."
            )

        self.device = torch.device(device)
        self.source = model.policy
        self.rollout_size = rollout_size
        self.batch_size = model.batch_size
        self.normalize_advantage = model.normalize_advantage
        self.ent_coef = model.ent_coef
        self.vf_coef = model.vf_coef
        self.max_grad_norm = model.max_grad_norm

        self.policy = copy.deepcopy(model.policy).to(self.device)
        group = model.policy.optimizer.param_groups[0]
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=torch.tensor(group["lr"], device=self.device),
            betas=group["betas"],
            eps=group["eps"],
            capturable=True,
            foreach=True,
        )

        def zeros(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, device=self.device)

        self.observations = zeros(rollout_size, *model.observation_space.shape)
        self.actions = zeros(rollout_size, *model.action_space.shape)
        self.old_log_prob = zeros(rollout_size)
        self.advantages = zeros(rollout_size)
        self.returns = zeros(rollout_size)
        self.batch = zeros(self.batch_size, dtype=torch.long)
        self.clip_range = zeros()
        self.totals = zeros(6)

        self.graph = self._record()
        self._load_from_source()

    def _gradient_step(self) -> None:
        """Mirror the inner loop of PPO.train, without reading anything back."""
        batch = self.batch
        values, log_prob, entropy = self.policy.evaluate_actions(
            self.observations[batch], self.actions[batch]
        )
        advantages = self.advantages[batch]
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_ratio = log_prob - self.old_log_prob[batch]
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -torch.min(advantages * ratio, advantages * clipped).mean()
        value_loss = F.mse_loss(self.returns[batch], values.flatten())
        entropy_loss = -torch.mean(entropy)
        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        self.optimizer.zero_grad(set_to_none=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            self.totals[POLICY_LOSS] += policy_loss
            self.totals[VALUE_LOSS] += value_loss
            self.totals[ENTROPY_LOSS] += entropy_loss
            self.totals[CLIP_FRACTION] += ((ratio - 1).abs() > self.clip_range).float().mean()
            self.totals[APPROX_KL] += ((ratio - 1) - log_ratio).mean()
            self.totals[TOTAL_LOSS] = loss

    def _record(self) -> torch.cuda.CUDAGraph:
        """Record one gradient step. This changes the GPU weights, which are reloaded."""
        # Checking a distribution's arguments reads them back, which a graph forbids.
        # This only applies while recording: rollouts on the CPU keep their checks.
        validate = Distribution._validate_args
        Distribution.set_default_validate_args(False)
        try:
            stream = torch.cuda.Stream(self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._gradient_step()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self._gradient_step()
        finally:
            Distribution.set_default_validate_args(validate)
        return graph

    def _pairs(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return list(zip(self.source.parameters(), self.policy.parameters()))

    @torch.no_grad()
    def _load_from_source(self) -> None:
        """Take the weights and Adam state from the CPU policy, as a resume needs."""
        saved = self.source.optimizer.state
        for source, copied in self._pairs():
            copied.copy_(source)
            state = self.optimizer.state[copied]
            for name in ("step", "exp_avg", "exp_avg_sq"):
                if source in saved:
                    state[name].copy_(saved[source][name])
                else:
                    state[name].zero_()

    @torch.no_grad()
    def _save_to_source(self) -> None:
        """Give the CPU policy the new weights, and the Adam state a checkpoint saves."""
        for source, copied in self._pairs():
            source.copy_(copied)
            state = self.optimizer.state[copied]
            self.source.optimizer.state[source] = {
                name: state[name].cpu() for name in ("step", "exp_avg", "exp_avg_sq")
            }

    def _batch_order(self) -> torch.Tensor:
        return torch.randperm(self.rollout_size, device=self.device)

    def run(
        self,
        rollout: RolloutBufferSamples,
        n_epochs: int,
        learning_rate: float,
        clip_range: float,
    ) -> dict[str, float]:
        """Train on one rollout and return the mean of each logged quantity."""
        self.observations.copy_(rollout.observations)
        self.actions.copy_(rollout.actions)
        self.old_log_prob.copy_(rollout.old_log_prob)
        self.advantages.copy_(rollout.advantages)
        self.returns.copy_(rollout.returns)
        self.optimizer.param_groups[0]["lr"].fill_(learning_rate)
        self.clip_range.fill_(clip_range)
        self.totals.zero_()

        last_epoch_kl = self.totals[APPROX_KL].clone()
        for epoch in range(n_epochs):
            if epoch == n_epochs - 1:
                last_epoch_kl.copy_(self.totals[APPROX_KL])
            order = self._batch_order()
            for start in range(0, self.rollout_size, self.batch_size):
                self.batch.copy_(order[start : start + self.batch_size])
                self.graph.replay()

        self._save_to_source()
        totals = self.totals.tolist()
        batches = self.rollout_size // self.batch_size
        return {
            "train/entropy_loss": totals[ENTROPY_LOSS] / (batches * n_epochs),
            "train/policy_gradient_loss": totals[POLICY_LOSS] / (batches * n_epochs),
            "train/value_loss": totals[VALUE_LOSS] / (batches * n_epochs),
            # PPO.train reports the last epoch alone for this one.
            "train/approx_kl": (totals[APPROX_KL] - last_epoch_kl.item()) / batches,
            "train/clip_fraction": totals[CLIP_FRACTION] / (batches * n_epochs),
            "train/loss": totals[TOTAL_LOSS],
        }


class GraphedPPO(PPO):
    """PPO that updates on ``update_device``. "cpu" is exactly the stock algorithm.

    ``device`` must stay "cpu", as that is where rollouts run.
    """

    def __init__(self, *args: Any, update_device: str = "cpu", **kwargs: Any) -> None:
        self.update_device = update_device
        self._graphed_update: GraphedUpdate | None = None
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "update_device", "_graphed_update"]

    def train(self) -> None:
        if self.update_device == "cpu":
            super().train()
            return

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self._graphed_update is None:
            self._graphed_update = GraphedUpdate(self, self.update_device)

        # With no batch size this yields the whole rollout at once.
        rollout = next(self.rollout_buffer.get())
        results = self._graphed_update.run(
            rollout,
            self.n_epochs,
            self.policy.optimizer.param_groups[0]["lr"],
            clip_range,
        )
        self._n_updates += self.n_epochs

        for name, value in results.items():
            self.logger.record(name, value)
        self.logger.record(
            "train/explained_variance",
            explained_variance(
                self.rollout_buffer.values.flatten(),
                self.rollout_buffer.returns.flatten(),
            ),
        )
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
