"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

import random
import time
from typing import List, Optional, Union

import numpy as np
import torch
from tianshou.env.venvs import BaseVectorEnv
from torch.utils.tensorboard import SummaryWriter

__all__ = [
    "to_onehot",
    "Episode",
    "Logger",
    "collect_episode",
    "collect_episodes_parallel",
    "ReplayMemory",
    "EpisodeMemory",
]


def to_onehot(label, n_labels):
    """
    Compute one-hot representation of label tensor

    Parameters
    ----------
    label : int or `np.ndarray` or `torch.LongTensor`, shape [<bs>]
        Labels
    n_labels : int
        Total number of labels (classes) available

    Returns
    -------
    label_onehot: `np.ndarray` or `torch.Tensor`, shape [<bs> x `n_labels`]
        One-hot-representation of labels
    """
    if isinstance(label, np.ndarray):
        bs = label.shape
        label_onehot = np.zeros(bs + (n_labels,))
        np.put_along_axis(label_onehot, label[..., None], values=1, axis=-1)
    elif isinstance(label, torch.Tensor):
        label_shape = label.shape + (n_labels,)
        with torch.no_grad():
            label_onehot = torch.empty(size=label_shape, device=label.device)
            label_onehot.fill_(0)
            label_onehot.scatter_(-1, label.unsqueeze(-1), 1)
    else:
        label_onehot = np.zeros(n_labels)
        label_onehot[label] = 1
    return label_onehot


class Episode:
    def __init__(self):
        self.data = []
        self.flags = {}

    def add_transition(self, transition_dict):
        self.data.append(transition_dict)

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def limit_length(self, length):
        self.data = self.data[:length]


class Logger:
    def __init__(self, run, run_directory, log_every=1):
        self.run = run
        self.summary_writer = SummaryWriter(run_directory)
        self._log_every = 1

    @property
    def log_every(self):
        return self._log_every

    @log_every.setter
    def log_every(self, value):
        self._log_every = value

    def log_scalar(self, tag, value, step):
        self.run.log_scalar(tag, value, step)
        self.summary_writer.add_scalar(tag, value, step)

    def log_array(self, tag, value, step):
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.item()
        if isinstance(value, (int, float)):
            self.log_scalar(tag, value, step)
        else:
            for k in range(value.numel()):
                self.log_scalar(f"{tag}_{k}", value[k].item(), step)

    def maybe_log_scalar(self, tag, value, step):
        if step % self.log_every == 0:
            self.log_scalar(tag, value, step)

    def maybe_log_array(self, tag, value, step):
        if step % self.log_every == 0:
            self.log_array(tag, value, step)


def collect_episode(
    env,
    hybrid_agent,
    symbolic_action,
    random_actions,
    evaluate,
    reset=True,
    render=False,
):
    if reset:
        state = env.reset()
    else:
        state = env.get_current_observation()
    initial_state = np.copy(state)
    episode = Episode()

    done = False
    while not done:
        if random_actions:
            action = env.action_space.sample()
        else:
            action = hybrid_agent.select_action(
                state,
                symbolic_action,
                evaluate=evaluate,
            )

        next_state, _, done, info = env.step(action)

        transition = {
            "initial_state": initial_state,
            "state": state,
            "action": action,
            "next_state": next_state,
            "symbolic_action": symbolic_action,
            "done": done,
            "info": info,
        }

        if render:
            rendering = env.render(mode="rgb_array")
            transition["rendering"] = rendering

        episode.add_transition(transition)

        state = next_state

    return episode


def collect_episodes_parallel(
    vector_env: BaseVectorEnv,
    symbolic_actions: Union[List[int], np.ndarray],
    hybrid_agent,
    evaluate: bool,
    random_action_prob_list: List[Optional[float]],
):
    # random_action_probs: Sample random action with given probability

    assert len(symbolic_actions) == len(random_action_prob_list)

    start = time.time()
    """ 'finished_episodes' may be in different ordering than 'symbolic_actions'! """
    env_num = vector_env.env_num
    # symbolic action currently evaluated on each env
    env_sym_action = [
        None,
    ] * env_num
    # environments free to start a new episode
    env_free = [
        True,
    ] * env_num
    # current step of each env
    env_steps = [
        0,
    ] * env_num
    # buffers for current episodes
    episode_buffers = [
        None,
    ] * env_num
    # buffers for initial states of running episodes
    initial_obs = [
        None,
    ] * env_num
    # buffers for current states of running episodes
    current_obs = [
        None,
    ] * env_num
    # buffers for finished episodes
    cur_env_random_action_prob = [
        None,
    ] * env_num
    # prob to sample random action in current env
    finished_episodes = []

    symbolic_action_queue = list(symbolic_actions)
    random_action_probs_queue = list(random_action_prob_list)

    while len(finished_episodes) < len(symbolic_actions):
        # Reset particular environments / start new episodes
        if len(symbolic_action_queue) > 0:
            envs_free = [e for e in range(env_num) if env_free[e]]
            envs_to_reset = envs_free[: len(symbolic_action_queue)]
            if len(envs_to_reset) > 0:
                vector_obs = vector_env.reset(envs_to_reset)
                for env_idx, obs in zip(envs_to_reset, vector_obs):
                    symbolic_action = symbolic_action_queue[0]
                    symbolic_action_queue = symbolic_action_queue[1:]
                    env_sym_action[env_idx] = symbolic_action

                    random_action_prob = random_action_probs_queue[0]
                    random_action_probs_queue = random_action_probs_queue[1:]
                    cur_env_random_action_prob[env_idx] = random_action_prob

                    env_free[env_idx] = False
                    env_steps[env_idx] = 0
                    episode_buffers[env_idx] = Episode()
                    initial_obs[env_idx] = obs
                    current_obs[env_idx] = obs

        # Compute action for each active environment
        envs_active = [e for e in range(env_num) if not env_free[e]]
        current_obs_batch = np.stack([current_obs[e] for e in envs_active])
        sym_actions_batch = np.stack([env_sym_action[e] for e in envs_active])
        random_action_prob_batch = [cur_env_random_action_prob[e] for e in envs_active]

        action_batch = hybrid_agent.select_action(
            state=current_obs_batch,
            symbolic_action=sym_actions_batch,
            evaluate=evaluate,
            random_action_prob=random_action_prob_batch,
        )

        # Perform step in each active environment
        next_obs_batch, _, done_batch, info_batch = vector_env.step(
            action_batch, id=envs_active
        )
        for rel_idx, env_idx in enumerate(envs_active):
            transition = {
                "initial_state": initial_obs[env_idx],
                "state": current_obs[env_idx],
                "action": action_batch[rel_idx],
                "next_state": next_obs_batch[rel_idx],
                "symbolic_action": sym_actions_batch[rel_idx],
                "done": done_batch[rel_idx].item(),
                "info": info_batch[rel_idx],
            }
            episode_buffers[env_idx].add_transition(transition)
            current_obs[env_idx] = next_obs_batch[rel_idx]

            env_steps[env_idx] += 1

            if done_batch[rel_idx]:
                finished_episodes.append(episode_buffers[env_idx])
                env_sym_action[env_idx] = None
                env_free[env_idx] = True
                episode_buffers[env_idx] = None
                env_steps[env_idx] = 0

    elapsed_time = time.time() - start
    print(f"{len(symbolic_actions)} episodes collected in {elapsed_time}s")
    return finished_episodes


class ReplayMemory:
    def __init__(self, capacity, seed, batch_process_fcn=None, is_torch=False):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = None
        self.size = 0
        self.position = 0
        self.indices = None
        self.batch_process_fcn = batch_process_fcn
        self.is_torch = is_torch

    def add_episode(self, episode):
        for transition in episode:
            self.add_transition(transition)

    def add_transition(self, transition):
        if self.buffer is None:
            keys = transition.keys()
            self.buffer = {k: [] for k in keys}
        else:
            if transition.keys() != self.buffer.keys():
                raise ValueError("Inconsistent keys")
        if self.size < self.capacity:
            for list_ in self.buffer.values():
                list_.append(None)
            self.size += 1
            self.indices = np.arange(0, self.size)
        for key, list_ in self.buffer.items():
            v = transition[key]
            if isinstance(v, (list, tuple)):
                v = np.concatenate([np.atleast_1d(a) for a in v])
            if isinstance(v, torch.Tensor) and not self.is_torch:
                v = v.cpu().detach().numpy()
            list_[self.position] = v
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_indices = np.random.choice(self.indices, batch_size)
        stack_fcn = np.stack if not self.is_torch else torch.stack
        batch = {
            k: stack_fcn([v[idx] for idx in batch_indices])
            for k, v in self.buffer.items()
        }
        if self.batch_process_fcn:
            batch = self.batch_process_fcn(batch)
        return batch

    def __len__(self):
        return self.size


class EpisodeMemory:
    def __init__(self, capacity, seed, batch_process_fcn=None):
        self.rng = np.random.RandomState(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.batch_process_fcn = batch_process_fcn

    def add_episode(self, episode):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = episode
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_indices = self.rng.randint(0, len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in batch_indices]
        if self.batch_process_fcn:
            batch = self.batch_process_fcn(batch)
        return batch

    @property
    def size(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def save_dict(self):
        return {
            "rng": self.rng,
            "capacity": self.capacity,
            "buffer": self.buffer,
            "position": self.position,
        }

    def load_dict(self, serialized_dict):
        assert self.capacity == serialized_dict["capacity"]
        assert len(self.buffer) == 0
        self.rng = serialized_dict["rng"]
        self.buffer = serialized_dict["buffer"]
        self.position = serialized_dict["position"]
