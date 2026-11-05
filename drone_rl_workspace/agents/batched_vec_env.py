"""Run many environments on a few worker processes.

SubprocVecEnv gives every environment its own process and pays one round trip per
environment per step. A step here is short enough for that round trip to cost more
than the step, so this class puts several environments on each worker and pays one
round trip per worker instead.

It behaves as SubprocVecEnv does: environment ``i`` gets seed ``seed + i``, and the
same actions give the same observations, rewards and dones. The one difference is
that ``info`` is only filled in on the step an episode ends, and then only with
``info_keys``, because sending every info on every step is a large share of the cost.
Everything here reads info only when an episode ends: PPO, its callbacks and
``evaluate_policy``.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Iterable, Sequence
from multiprocessing.connection import Connection
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import CloudpickleWrapper, VecEnv

INFO_KEYS = ("episode", "is_success", "termination", "waypoints_reached")


def _step_all(
    envs: list[gym.Env], actions: np.ndarray, info_keys: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
    """Step each environment, resetting any that finish, as a VecEnv must."""
    observations, rewards, dones = [], [], []
    finished = {}
    for index, (env, action) in enumerate(zip(envs, actions)):
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            finished[index] = {key: info[key] for key in info_keys if key in info}
            finished[index]["TimeLimit.truncated"] = truncated and not terminated
            finished[index]["terminal_observation"] = observation
            observation, _ = env.reset()
        observations.append(observation)
        rewards.append(reward)
        dones.append(done)
    return np.stack(observations), np.array(rewards), np.array(dones), finished


def _worker(pipe: Connection, build: CloudpickleWrapper, count: int, info_keys: Sequence[str]) -> None:
    envs = [Monitor(build.var()) for _ in range(count)]
    try:
        while True:
            command, data = pipe.recv()
            if command == "step":
                pipe.send(_step_all(envs, data, info_keys))
            elif command == "reset":
                results = [
                    env.reset(seed=seed, **({"options": options} if options else {}))
                    for env, (seed, options) in zip(envs, data)
                ]
                pipe.send(np.stack([observation for observation, _ in results]))
            elif command == "spaces":
                pipe.send((envs[0].observation_space, envs[0].action_space))
            elif command == "env_method":
                name, args, kwargs, local = data
                pipe.send([envs[i].get_wrapper_attr(name)(*args, **kwargs) for i in local])
            elif command == "get_attr":
                name, local = data
                pipe.send([envs[i].get_wrapper_attr(name) for i in local])
            elif command == "set_attr":
                name, value, local = data
                for i in local:
                    setattr(envs[i], name, value)
                pipe.send(None)
            elif command == "is_wrapped":
                wrapper, local = data
                pipe.send([is_wrapped(envs[i], wrapper) for i in local])
            elif command == "close":
                break
            else:
                raise NotImplementedError(command)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        for env in envs:
            env.close()
        pipe.close()


class BatchedVecEnv(VecEnv):
    """A VecEnv with ``n_envs`` environments spread over ``n_workers`` processes.

    Args:
        env_fn: Builds one environment. Each is wrapped in a Monitor.
        n_envs: Total number of environments.
        n_workers: Number of processes. Capped at ``n_envs``.
        seed: Environment ``i`` is seeded with ``seed + i`` at the first reset.
        info_keys: Which info entries to pass back when an episode ends.
    """

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        n_envs: int,
        n_workers: int,
        seed: int | None = None,
        info_keys: Sequence[str] = INFO_KEYS,
    ) -> None:
        n_workers = min(n_workers, n_envs)
        ranks = np.array_split(np.arange(n_envs), n_workers)
        self._slices = [slice(int(part[0]), int(part[-1]) + 1) for part in ranks]
        self._owner = [
            (worker, local)
            for worker, part in enumerate(ranks)
            for local in range(len(part))
        ]
        self._waiting = False
        self._closed = False

        # spawn is the only start method on Windows, and is safe on every platform.
        context = mp.get_context("spawn")
        self._pipes: list[Connection] = []
        self._workers = []
        build = CloudpickleWrapper(env_fn)
        for part in ranks:
            ours, theirs = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(theirs, build, len(part), tuple(info_keys)),
                daemon=True,
            )
            process.start()
            theirs.close()
            self._pipes.append(ours)
            self._workers.append(process)

        self._pipes[0].send(("spaces", None))
        observation_space, action_space = self._pipes[0].recv()
        super().__init__(n_envs, observation_space, action_space)
        if seed is not None:
            self.seed(seed)

    def reset(self) -> np.ndarray:
        requests = list(zip(self._seeds, self._options))
        for pipe, part in zip(self._pipes, self._slices):
            pipe.send(("reset", requests[part]))
        observations = np.concatenate([pipe.recv() for pipe in self._pipes])
        self._reset_seeds()
        self._reset_options()
        return observations

    def step_async(self, actions: np.ndarray) -> None:
        for pipe, part in zip(self._pipes, self._slices):
            pipe.send(("step", actions[part]))
        self._waiting = True

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        results = [pipe.recv() for pipe in self._pipes]
        self._waiting = False
        infos: list[dict] = [{} for _ in range(self.num_envs)]
        for part, (_, _, _, finished) in zip(self._slices, results):
            for index, info in finished.items():
                infos[part.start + index] = info
        observations, rewards, dones, _ = zip(*results)
        return (
            np.concatenate(observations),
            np.concatenate(rewards),
            np.concatenate(dones),
            infos,
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._waiting:
            for pipe in self._pipes:
                pipe.recv()
        for pipe in self._pipes:
            pipe.send(("close", None))
        for process in self._workers:
            process.join()
        self._closed = True

    def _ask(self, command: str, indices: Any, *data: Any) -> list[Any]:
        """Send a command to the workers owning ``indices``, in order of index."""
        chosen = list(self._get_indices(indices))
        by_worker: dict[int, list[int]] = {}
        for index in chosen:
            worker, local = self._owner[index]
            by_worker.setdefault(worker, []).append(local)
        for worker, local in by_worker.items():
            self._pipes[worker].send((command, (*data, local)))
        answers = {
            worker: iter(self._pipes[worker].recv() or [None] * len(local))
            for worker, local in by_worker.items()
        }
        return [next(answers[self._owner[index][0]]) for index in chosen]

    def _get_indices(self, indices: Any) -> Iterable[int]:
        if indices is None:
            return range(self.num_envs)
        return [indices] if isinstance(indices, int) else indices

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        return self._ask("get_attr", indices, attr_name)

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        self._ask("set_attr", indices, attr_name, value)

    def env_method(
        self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any
    ) -> list[Any]:
        return self._ask("env_method", indices, method_name, method_args, method_kwargs)

    def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
        return self._ask("is_wrapped", indices, wrapper_class)
