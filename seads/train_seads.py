"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

import logging
import math
import os
import pickle as pkl
import random
import time
import warnings
from copy import deepcopy
from threading import Event, Thread

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from seads_envs import load_env
from tianshou.env.venvs import DummyVectorEnv, ShmemVectorEnv
from tqdm import tqdm

from seads import EXPERIMENT_DIR, SRC_DIR
from seads.agent import HybridAgent
from seads.common import Episode, EpisodeMemory, Logger, collect_episodes_parallel
from seads.skill_model.common import compute_symbolic_state_changed
from seads.skill_model.forward_model import compute_binary_match, plot_predicates
from seads.utils import torch_load
from seads.utils.sacred_utils import (
    InitializationFinishedInterrupt,
    WaitingForRestartInterrupt,
    create_experiment,
)

# fmt: off
matplotlib.use("Agg")
# fmt: on

# 'numpy' emits lots of DeprecationWarnings for aliased
# datatypes (see https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations)
# Ignore these for now...
warnings.filterwarnings("ignore", category=DeprecationWarning)

VERBOSE_LOGGING = True

experiment_name = "train_seads"
experiment = create_experiment(
    experiment_name,
    EXPERIMENT_DIR.joinpath("seads"),
    SRC_DIR,
    mongodb_observer=False,
    zip_src_dir=True,
    restartable=True,
    add_src_to_sacred_recursively=False,
    add_caller_src_to_sacred=True,
)


# noinspection PyUnusedLocal
@experiment.config
def config_fcn():
    sac_args = {
        "gamma": 0.99,  # discount factor for reward (default: 0.99)
        "tau": 0.005,  # target smoothing coefficient(τ) (default: 0.005)
        "alpha": 0.1,  # Temperature parameter α (default: 0.1)
        "policy": "Gaussian",
        "qnet_sigmoid_squash": False,
        "target_update_interval": 1,
        "automatic_entropy_tuning": False,
        "cuda": True,
        "hidden_size": [256, 256],
        "lr": 0.0003,  # learning rate (default: 0.0003)
    }
    ext_timelimit_obs = "full"

    n_collect_workers = 4

    episode_buffer_capacity = 2048  # on-policy: buffer is overwritten
    n_new_episodes_per_epoch = 32  # 'DADS': 10
    new_episode_random_action_prob = 0
    new_episode_all_random_prob = 0

    n_validation_episodes = 32
    n_sampled_episodes_per_epoch = 256
    evaluate_symbolic_state_change_sac = True
    evaluate_symbolic_state_change_skm = False
    relabel_sac = 0.5
    relabel_skm = 1.0
    relabel_specifier_skm = None
    relabel_specifier_sac = None

    rollout_datasource = "both"  # buffer, recent, both
    relabel_episode_batchsize = 32
    non_final_reward = 0
    reward_type = "reward"
    reward_options = {
        # available components:
        # log_q_k, neg_log_q_k_baseline, log_novb, log_K
        "reward_components": [
            "log_q_k",
            "log_novb",
            "neg_log_q_k_baseline",
        ],
    }
    batchsize_sac = 128
    n_steps_sac = 16
    batchsize_skm = 32
    n_steps_skm = 4
    n_steps_skm_outer = 1
    exclude_nop_episodes_skm = False

    loss_n_z_samples = 1
    reward_n_z_samples = 1

    relabel_type = "assignment"
    relabel_opts = {}

    save_all_episodes = False
    evaluate_every = 50
    log_every = 50
    n_epochs_per_leap = 5000
    evaluate_when_finished = True

    use_groundtruth_phi = True

    train_sac = True

    load_sac_experiment_name = None
    load_sac_run_id = None
    load_sac_ckpt_step = None


# noinspection PyUnusedLocal
@experiment.named_config
def lightsout_cursor_env():
    env_name = "LightsOutCursorEnvBs5"
    env_kwargs = {
        "max_steps": 10,
        "max_solution_depth": 5,
        "random_solution_depth": True,
    }
    num_skills = 25
    max_env_interactions = 500_000
    sac_args = {"alpha": 0.1}
    checkpoint_every = 100


# noinspection PyUnusedLocal
@experiment.named_config
def lightsout_reacher_env():
    env_name = "LightsOutReacherEnvBs5"
    env_kwargs = {
        "max_steps": 50,
        "max_solution_depth": 5,
        "random_solution_depth": True,
        "action_repeat": 2,
    }
    num_skills = 25
    max_env_interactions = 10_000_000
    sac_args = {"alpha": 0.01}
    checkpoint_every = 500


# noinspection PyUnusedLocal
@experiment.named_config
def lightsout_jaco_env():
    env_name = "LightsOutJacoEnvBs5"
    env_kwargs = {
        "max_steps": 50,
        "max_solution_depth": 5,
        "random_solution_depth": True,
        "control_timestep": 0.1,
        "stairs": False,
    }
    num_skills = 25
    max_env_interactions = 10_000_000
    sac_args = {"alpha": 0.01}
    checkpoint_every = 500


# noinspection PyUnusedLocal
@experiment.named_config
def lightsout_jaco_stairs_env():
    env_name = "LightsOutJacoEnvBs5"
    env_kwargs = {
        "max_steps": 50,
        "max_solution_depth": 5,
        "random_solution_depth": True,
        "control_timestep": 0.1,
        "stairs": True,
    }
    num_skills = 25
    max_env_interactions = 10_000_000
    sac_args = {"alpha": 0.01}
    checkpoint_every = 500


# noinspection PyUnusedLocal
@experiment.named_config
def tileswap_cursor_env():
    env_name = "TileSwapCursorEnvBs3"
    env_kwargs = {
        "max_steps": 10,
        "max_solution_depth": 5,
        "random_solution_depth": True,
    }
    num_skills = 12
    max_env_interactions = 500_000
    sac_args = {"alpha": 0.1}
    checkpoint_every = 100


# noinspection PyUnusedLocal
@experiment.named_config
def tileswap_reacher_env():
    env_name = "TileSwapReacherEnvBs3"
    env_kwargs = {
        "max_steps": 50,
        "max_solution_depth": 5,
        "random_solution_depth": True,
        "action_repeat": 2,
    }
    num_skills = 12
    max_env_interactions = 10_000_000
    sac_args = {"alpha": 0.01}
    checkpoint_every = 500


# noinspection PyUnusedLocal
@experiment.named_config
def tileswap_jaco_env():
    env_name = "TileSwapJacoEnvBs3"
    env_kwargs = {
        "max_steps": 50,
        "max_solution_depth": 5,
        "random_solution_depth": True,
        "control_timestep": 0.1,
    }
    num_skills = 12
    max_env_interactions = 10_000_000
    sac_args = {"alpha": 0.01}
    checkpoint_every = 500


# noinspection PyUnusedLocal
@experiment.named_config
def skill_forward_model():
    skill_model_type = "forward_model"
    skill_model_kwargs = dict(
        train_pfm=True,
        optim_pfm_classname="Adam",
        optim_pfm_kwargs={"lr": 1e-3},
        pfm_kwargs={"hidden_dim": 256},
    )


# noinspection PyUnusedLocal
@experiment.named_config
def skill_discriminator():
    skill_model_type = "discriminator"
    skill_model_kwargs = {
        "train_discriminator": True,
        "optim_classname": "Adam",
        "optim_kwargs": {"lr": 1e-3},
        "hidden_dim": 256,
        "input_type": "concatxor",
    }


def set_final_reward(episode, final_reward, non_final_reward):
    """
    Set reward to `final_reward` for last transition
    and to `non_final_reward` otherwise
    """
    # 'discount=False' yields better results when passing
    # pre-trained dynamics model
    for idx in range(len(episode)):
        # start from final transition (idx == 0)
        transition = episode[len(episode) - idx - 1]
        reward = final_reward if idx == 0 else non_final_reward
        transition["reward"] = reward


def compute_max_relabelling(episode_list):
    symbolic_action_assignment = dict()
    for episode_idx, episode in enumerate(episode_list):
        last_transition = episode[-1]
        log_prob_k_all = last_transition["log_prob_k_all"]
        relabeled_action = np.argmax(log_prob_k_all)
        symbolic_action_assignment[episode_idx] = relabeled_action
    return symbolic_action_assignment


def compute_assignment_relabelling(episode_list, num_skills):
    symbolic_action_assignment = dict()
    n_episodes = len(episode_list)

    # log-prob that skill i was executed in episode j,
    # determined via maximization over timesteps.
    log_prob_matrix = np.zeros((num_skills, n_episodes))

    for episode_idx, episode in enumerate(episode_list):
        last_transition = episode[-1]
        log_prob_matrix[:, episode_idx] = last_transition["log_prob_k_all"]

    target_relabeled_actions = np.array([e[0]["symbolic_action"] for e in episode_list])
    cost_matrix = -log_prob_matrix[target_relabeled_actions, :]

    optimal_assignment = linear_sum_assignment(cost_matrix)

    for row_idx, col_idx in zip(*optimal_assignment):
        relabeled_action = target_relabeled_actions[row_idx]
        episode_idx = col_idx
        symbolic_action_assignment[episode_idx] = relabeled_action
    return symbolic_action_assignment


def relabel_processed_episodes(episode_list, num_skills, relabel_type):
    if relabel_type == "assignment":
        relabeled_action_dict = compute_assignment_relabelling(episode_list, num_skills)
    elif relabel_type == "max":
        relabeled_action_dict = compute_max_relabelling(episode_list)
    else:
        raise ValueError

    for episode_idx, episode in enumerate(episode_list):
        relabeled_action = relabeled_action_dict[episode_idx]
        episode = episode_list[episode_idx]

        relabeled = False
        for transition in episode:
            if relabeled_action != transition["symbolic_action"]:
                relabeled = True
            transition["symbolic_action"] = relabeled_action
        episode.flags["soft_relabeled"] = relabeled

        # set 'done' flag
        episode[-1]["done"] = episode[-1]["done"]


def set_episode_reward(
    episode,
    relabeled_action,
    non_final_reward,
):
    """Set rewards in episode in-place, according to 'relabeled_action'.
    The final transition in 'episode' needs to have 'reward_all' set.
    """
    # set reward of all transitions in episode to 'non_final_reward', except final
    final_reward = episode[-1]["reward_all"][relabeled_action]
    set_final_reward(
        episode,
        final_reward,
        non_final_reward=non_final_reward,
    )

    episode[-1]["log_prob_k"] = episode[-1]["log_prob_k_all"][relabeled_action]
    for transition in episode[:-1]:
        # we need a "log_prob_k" field in every transition
        transition["log_prob_k"] = np.nan


def remove_temporary_episode_data(episode):
    for transition in episode:
        if "reward_all" in transition:
            del transition["reward_all"]
        if "log_prob_k_all" in transition:
            del transition["log_prob_k_all"]


def process_episode_batch(config, episode_batch, hybrid_agent):
    """
    Process a batch of episodes for relabelling

    Parameters
    ----------
    config: `dict`
        Config, keys: num_skills, n_z_samples, relabel
    episode_batch: list[`Episode`]
        Episode batch
    hybrid_agent: `HybridAgent`
        HybridAgent

    Returns
    -------
    processed_episode_list: list[`Episode`]
    """
    num_skills = config["num_skills"]
    device = config["device"]

    initial_state_t = (
        torch.from_numpy(np.stack([e[0]["initial_state"] for e in episode_batch]))
        .float()
        .to(device)
    )
    next_state_t = (
        torch.from_numpy(np.stack([e[-1]["next_state"] for e in episode_batch]))
        .float()
        .to(device)
    )
    symbolic_action_t = (
        torch.from_numpy(np.stack([e[0]["symbolic_action"] for e in episode_batch]))
        .long()
        .to(device)
    )

    (
        reward_k,
        reward_all,
        log_prob_k,
        log_prob_k_all,
        reward_min,
    ) = hybrid_agent.compute_reward(
        initial_state_t, next_state_t, symbolic_action_t, return_all=True
    )
    log_prob_k = log_prob_k.cpu().numpy()
    log_prob_k_all = log_prob_k_all.cpu().numpy()

    # clip log_prob_k, log_prob_k_all
    log_prob_k = np.clip(log_prob_k, a_min=-2 * math.log(num_skills), a_max=None)
    log_prob_k_all = np.clip(
        log_prob_k_all, a_min=-2 * math.log(num_skills), a_max=None
    )

    reward_all = reward_all.cpu().numpy()

    if hybrid_agent.can_predict:
        pred_match_k, pred_match_all = compute_binary_match(
            hybrid_agent.skill_model,
            initial_state_t,
            next_state_t,
            symbolic_action_t,
            return_all=True,
        )
        pred_match_k = pred_match_k.cpu().numpy()

    processed_episode_list = []
    for episode_idx, original_episode in enumerate(episode_batch):
        episode = deepcopy(original_episode)

        episode[-1]["reward_all"] = reward_all[episode_idx]
        episode[-1]["log_prob_k_all"] = log_prob_k_all[episode_idx]
        episode.flags["min_reward"] = reward_min

        if hybrid_agent.can_predict:
            is_only_skill = np.exp(log_prob_k[episode_idx]) > 0.9
            match = pred_match_k[episode_idx] == 1.0
            executed_unique_skill = is_only_skill
            correctly_predicted = match
            success = match & is_only_skill
        else:
            executed_unique_skill = False
            correctly_predicted = False
            success = False

        symbolic_state_changed = compute_symbolic_state_changed(
            hybrid_agent.map,
            initial_state_t[episode_idx],
            next_state_t[episode_idx],
        )
        episode.flags["symbolic_state_changed"] = symbolic_state_changed

        episode.flags["executed_unique_skill"] = executed_unique_skill
        episode.flags["correctly_predicted"] = correctly_predicted
        episode.flags["success"] = success

        processed_episode_list.append(episode)

    return processed_episode_list


def process_episode_list(config, episode_list, hybrid_agent):
    """Process episodes for relabelling batch-wise"""
    batchsize = config["relabel_episode_batchsize"]
    episode_batches = [
        episode_list[i : i + batchsize] for i in range(0, len(episode_list), batchsize)
    ]
    processed_episode_list = []
    with torch.no_grad():
        for chunk in episode_batches:
            r = process_episode_batch(config, chunk, hybrid_agent)
            processed_episode_list.extend(r)
    return processed_episode_list


def relabel_episodes(config, episode_list, hybrid_agent):
    num_skills = config["num_skills"]
    relabel = config["relabel"]
    evaluate_symbolic_state_change = config["evaluate_symbolic_state_change"]
    non_final_reward = config["non_final_reward"]
    relabel_type = config["relabel_type"]

    processed_episode_list = process_episode_list(config, episode_list, hybrid_agent)
    symbolic_state_changed = np.array(
        [e.flags["symbolic_state_changed"] for e in processed_episode_list]
    ).astype(bool)

    if relabel > 0:
        episodes_to_relabel = np.random.rand(len(processed_episode_list)) < relabel
        if evaluate_symbolic_state_change:
            # do not relabel episodes where symbolic state did not change
            # (they will be assigned a minimum reward in any case)
            episodes_to_relabel = episodes_to_relabel & symbolic_state_changed
        episodes_to_relabel_list = [
            ep for i, ep in enumerate(processed_episode_list) if episodes_to_relabel[i]
        ]
        if len(episodes_to_relabel_list) > 0:
            relabel_processed_episodes(
                episodes_to_relabel_list, num_skills, relabel_type
            )

    # set reward to min_final_reward for all 'k' where symbolic state did not change
    for abs_episode_idx, episode in enumerate(processed_episode_list):
        if not symbolic_state_changed[abs_episode_idx]:
            episode[-1]["reward_all"] = episode.flags["min_reward"] * np.ones(
                num_skills
            )

    for episode_idx, episode in enumerate(processed_episode_list):
        set_reward_kwargs = dict(
            non_final_reward=non_final_reward,
        )
        relabeled_action = episode[-1]["symbolic_action"]
        set_episode_reward(episode, relabeled_action, **set_reward_kwargs)
        remove_temporary_episode_data(episode)

    return processed_episode_list


def relabel_multi_episode_sources(
    relabel_specifier,
    rollout_datasource,
    relabel_ratio,
    buffer_episodes,
    recent_episodes,
    relabel_config,
    hybrid_agent,
):
    if relabel_specifier is None:
        if rollout_datasource == "both":
            relabel_input = "buffer+recent"
        elif rollout_datasource == "recent":
            relabel_input = "recent"
        elif rollout_datasource == "buffer":
            relabel_input = "buffer"
        else:
            raise ValueError
        relabel_specifier = [(relabel_input, relabel_ratio)]
    else:
        assert rollout_datasource is None
        assert relabel_ratio is None

    relabeled_episodes = []
    for datasource, relabel_ratio in relabel_specifier:
        input_episodes = []
        if "buffer" in datasource:
            input_episodes.extend(buffer_episodes)
        if "recent" in datasource:
            input_episodes.extend(recent_episodes)
        updated_relabel_config = dict(
            **relabel_config,
            relabel=relabel_ratio,
        )
        relabeled_episodes.extend(
            relabel_episodes(
                updated_relabel_config,
                input_episodes,
                hybrid_agent,
            )
        )

    return relabeled_episodes


def plot_histogram(df, field_name):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    all_ = df[field_name]
    s_relabeled = df[df["soft_relabeled"]][field_name]
    ax[0].hist(all_.to_numpy())
    ax[1].hist(s_relabeled.to_numpy())
    return fig


def log_episode_batch_statistics(
    env, num_skills, relabeled_episodes, logger, step, prefix
):
    # compute statistics
    stats_records = []
    for episode in relabeled_episodes:
        initial_symbolic_state = env.get_binary_symbolic_state(episode[0]["state"])
        final_symbolic_state = env.get_binary_symbolic_state(episode[-1]["next_state"])
        stats_records.append(
            {
                "episode_len": len(episode),
                "symbolic_action": episode[-1]["symbolic_action"],
                "reward": episode[-1]["reward"],
                "done": episode[-1]["done"],
                "soft_relabeled": episode.flags.get("soft_relabeled", False),
                "executed_unique_skill": episode.flags["executed_unique_skill"],
                "correctly_predicted": episode.flags["correctly_predicted"],
                "success": episode.flags["success"],
                "symbolic_state_changed": 1
                - np.array_equal(initial_symbolic_state, final_symbolic_state),
            }
        )
    df = pd.DataFrame(stats_records)

    w = logger.summary_writer
    add_figure = lambda name_, fig_: w.add_figure(prefix + name_, fig_, step)
    if VERBOSE_LOGGING:
        add_figure("/buf/hist_episode_len", plot_histogram(df, "episode_len"))
        add_figure("/buf/hist_reward", plot_histogram(df, "reward"))
    add_figure(
        "/buf/trajectories", env.plot_skill_episodes(relabeled_episodes, num_skills)
    )

    n_done_all = df["done"].sum()
    n_symbolic_state_changed = df["symbolic_state_changed"].sum()
    n_done_s_relabeled = (df["soft_relabeled"] & df["done"]).sum()
    n_done_n_relabeled = (~df["soft_relabeled"] & df["done"]).sum()
    n_s_relabeled = df["soft_relabeled"].sum()
    n_executed_unique_skill = df["executed_unique_skill"].sum()
    n_correctly_predicted = df["correctly_predicted"].sum()
    n_success = df["success"].sum()

    add_scalar = lambda name_, scalar_: logger.log_scalar(prefix + name_, scalar_, step)
    add_scalar("/buf/n_done_all", n_done_all)
    add_scalar("/buf/n_symbolic_state_changed", n_symbolic_state_changed)
    add_scalar("/buf/n_done_s_relabeled", n_done_s_relabeled)
    add_scalar("/buf/n_done_n_relabeled", n_done_n_relabeled)
    add_scalar("/buf/n_s_relabeled", n_s_relabeled)
    add_scalar("/buf/n_executed_unique_skill", n_executed_unique_skill)
    add_scalar("/buf/n_correctly_predicted", n_correctly_predicted)
    add_scalar("/buf/n_success", n_success)


def plot_episode_predicates(
    forward_model, device, env, episode_list, logger, step, prefix="val"
):
    s_initial_np = np.stack([e[-1]["initial_state"] for e in episode_list])
    s_initial_t = torch.from_numpy(s_initial_np).float().to(device)
    s_target_np = np.stack([e[-1]["next_state"] for e in episode_list])
    s_target_t = torch.from_numpy(s_target_np).float().to(device)
    symbolic_action_np = np.stack([e[-1]["symbolic_action"] for e in episode_list])
    action_t = torch.from_numpy(symbolic_action_np).long().to(device)

    z_true_initial = env.get_binary_symbolic_state(s_initial_t)
    z_true_target = env.get_binary_symbolic_state(s_target_t)

    fig = plot_predicates(
        forward_model,
        s_initial_t,
        z_true_initial,
        s_target_t,
        z_true_target,
        action_t,
    )

    logger.summary_writer.add_figure(prefix + "/predicates", fig, step)


# noinspection PyUnusedLocal,PyUnresolvedReferences
@experiment.on_init
def experiment_init(
    env_name,
    env_kwargs,
    ext_timelimit_obs,
    num_skills,
    skill_model_type,
    skill_model_kwargs,
    use_groundtruth_phi,
    train_sac,
    sac_args,
    max_env_interactions,
    n_new_episodes_per_epoch,
    n_validation_episodes,
    n_collect_workers,
    n_sampled_episodes_per_epoch,
    episode_buffer_capacity,
    rollout_datasource,
    relabel_sac,
    relabel_skm,
    relabel_specifier_skm,
    relabel_specifier_sac,
    evaluate_symbolic_state_change_sac,
    evaluate_symbolic_state_change_skm,
    new_episode_random_action_prob,
    new_episode_all_random_prob,
    relabel_episode_batchsize,
    relabel_type,
    relabel_opts,
    reward_type,
    reward_options,
    non_final_reward,
    batchsize_sac,
    n_steps_sac,
    batchsize_skm,
    n_steps_skm,
    n_steps_skm_outer,
    exclude_nop_episodes_skm,
    loss_n_z_samples,
    reward_n_z_samples,
    n_epochs_per_leap,
    log_every,
    evaluate_every,
    checkpoint_every,
    evaluate_when_finished,
    save_all_episodes,
    seed,
    _run,
):
    raise InitializationFinishedInterrupt


def load_last_checkpoint(checkpoint_dir):
    steps = []
    for file_ in checkpoint_dir.iterdir():
        if not file_.stem.startswith("step"):
            continue
        steps.append(int(file_.stem.split("_")[1]))
    steps = sorted(steps, reverse=True)

    last_checkpoint_step = last_checkpoint_file = last_checkpoint_data = None

    if len(steps) > 0:
        try_last = min(len(steps), 3)
        for trial in range(try_last):
            last_checkpoint_step = steps[trial]
            last_checkpoint_file = checkpoint_dir.joinpath(
                f"step_{last_checkpoint_step}.pkl"
            )
            # noinspection PyBroadException
            try:
                last_checkpoint_data = torch_load(last_checkpoint_file)
                return last_checkpoint_step, last_checkpoint_file, last_checkpoint_data
            except Exception:
                print(f"Failed loading checkpoint {last_checkpoint_file}")

    return last_checkpoint_step, last_checkpoint_file, last_checkpoint_data


def get_vector_env_type(env_name):
    vector_env_type = ShmemVectorEnv
    if ("Cursor" in env_name) or ("Real" in env_name):
        logging.warning("'Cursor'/'Real' envs do not work with parallel env processing")
        vector_env_type = DummyVectorEnv
    return vector_env_type


class EpisodeCollectorAsync:
    def __init__(
        self,
        env_name,
        env_kwargs,
        reset_split,
        num_skills,
        ext_timelimit_obs,
        n_collect_workers,
        base_seed,
        symbolic_action_rng,
        new_episode_random_action_prob,
        new_episode_all_random_prob,
    ):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.reset_split = reset_split
        self.num_skills = num_skills
        self.ext_timelimit_obs = ext_timelimit_obs
        self.n_collect_workers = n_collect_workers
        self.base_seed = base_seed
        self.symbolic_action_rng = symbolic_action_rng
        self.new_episode_random_action_prob = new_episode_random_action_prob
        self.new_episode_all_random_prob = new_episode_all_random_prob

        self.thread = Thread(target=self._worker)
        self.thread.start()
        self.collection_in_progress = False
        self.collection_requested = Event()
        self.termination_requested = Event()
        self.collection_finished = Event()
        self.worker_collect_kwargs = {}
        self.new_episodes = None

    def _initialize_env(self):
        # Initialize vector env
        vector_env_type = get_vector_env_type(self.env_name)
        vector_env = vector_env_type(
            [
                lambda: load_env(
                    self.env_name,
                    reset_split=self.reset_split,
                    env_kwargs=self.env_kwargs,
                    ext_timelimit_obs=self.ext_timelimit_obs,
                )
                for _ in range(self.n_collect_workers)
            ]
        )
        # seed the sub-envs as 'base_seed + 1' ... 'base_seed + n_collect_workers'
        vector_env.seed(self.base_seed)
        return vector_env

    def _worker(self):
        vector_env = self._initialize_env()
        while True:
            time.sleep(0.01)
            if self.collection_requested.is_set():
                self.collection_requested.clear()
                self.new_episodes = self._collect(
                    vector_env, **self.worker_collect_kwargs
                )
                self.collection_finished.set()
            if self.termination_requested.is_set():
                vector_env.close()
                break

    def collect_async(self, n_new_episodes_per_epoch, collect_kwargs, eval_mode):
        if self.collection_in_progress:
            raise RuntimeError("Fetch episodes first before calling collect_async")
        self.collection_in_progress = True
        self.worker_collect_kwargs = {
            "n_new_episodes_per_epoch": n_new_episodes_per_epoch,
            "collect_kwargs": collect_kwargs,
            "eval_mode": eval_mode,
        }
        self.collection_finished.clear()
        self.collection_requested.set()

    def _collect(self, vector_env, n_new_episodes_per_epoch, collect_kwargs, eval_mode):
        symbolic_actions = self.symbolic_action_rng.randint(
            0, self.num_skills, n_new_episodes_per_epoch
        )
        # Get prob of random actions in episode
        random_action_prob = np.zeros(n_new_episodes_per_epoch)
        if self.new_episode_random_action_prob > 0:
            random_action_prob[:] = self.new_episode_random_action_prob
        if self.new_episode_all_random_prob > 0:
            rand = self.symbolic_action_rng.rand(n_new_episodes_per_epoch)
            random_episodes = rand < self.new_episode_all_random_prob
            random_action_prob[random_episodes] = 1.0
        episodes = collect_episodes_parallel(
            vector_env=vector_env,
            symbolic_actions=symbolic_actions,
            evaluate=eval_mode,
            random_action_prob_list=list(random_action_prob),
            **collect_kwargs,
        )
        return episodes

    def wait(self):
        if not self.collection_in_progress:
            raise RuntimeError("Not in collection phase")
        self.collection_finished.wait()
        self.collection_in_progress = False
        return self.new_episodes

    def close(self):
        self.termination_requested.set()
        if self.thread.is_alive():
            self.thread.join()


# noinspection PyUnresolvedReferences
@experiment.on_continue
def experiment_continue(
    env_name,
    env_kwargs,
    ext_timelimit_obs,
    num_skills,
    skill_model_type,
    skill_model_kwargs,
    use_groundtruth_phi,
    train_sac,
    sac_args,
    max_env_interactions,
    n_new_episodes_per_epoch,
    n_validation_episodes,
    n_collect_workers,
    n_sampled_episodes_per_epoch,
    episode_buffer_capacity,
    rollout_datasource,
    relabel_sac,
    relabel_skm,
    relabel_specifier_skm,
    relabel_specifier_sac,
    evaluate_symbolic_state_change_sac,
    evaluate_symbolic_state_change_skm,
    new_episode_random_action_prob,
    new_episode_all_random_prob,
    relabel_episode_batchsize,
    relabel_type,
    relabel_opts,
    reward_type,
    reward_options,
    non_final_reward,
    batchsize_sac,
    n_steps_sac,
    batchsize_skm,
    n_steps_skm,
    n_steps_skm_outer,
    exclude_nop_episodes_skm,
    loss_n_z_samples,
    reward_n_z_samples,
    n_epochs_per_leap,
    log_every,
    evaluate_every,
    checkpoint_every,
    evaluate_when_finished,
    save_all_episodes,
    seed,
    _run,
):
    root_run_directory = experiment.parent_run_directory

    sac_args = dict(**sac_args)

    if not torch.cuda.is_available():
        sac_args["cuda"] = False

    device = "cuda" if sac_args["cuda"] else "cpu"

    if n_sampled_episodes_per_epoch == "all":
        n_sampled_episodes_per_epoch = episode_buffer_capacity

    # Argument checks
    if episode_buffer_capacity < n_sampled_episodes_per_epoch:
        raise ValueError(
            "'episode_buffer_capacity' must not be smaller than "
            "'n_sampled_episodes_per_epoch'"
        )

    # Checkpointing
    ckpt_dir = root_run_directory.joinpath("checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # Load most recent available checkpoint
    first_epoch_idx, last_checkpoint_file, checkpoint_data = load_last_checkpoint(
        ckpt_dir
    )
    if last_checkpoint_file is None:
        first_epoch_idx = 0

    # seeding
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    symbolic_action_seed = 3242 + seed + first_epoch_idx * 100
    base_seed_vec_train = 6242 + seed + first_epoch_idx * 100
    base_seed_vec_val = 7242 + seed + first_epoch_idx * 100

    # Generate envs for parallel data collection
    reset_split = "train"
    prototype_env = load_env(
        env_name,
        reset_split,
        env_kwargs,
        time_limit=True,
        ext_timelimit_obs=ext_timelimit_obs,
        prototype=True,
    )

    # use different envs for unseen boards
    if evaluate_every > 0 or evaluate_when_finished:
        vector_env_val = get_vector_env_type(env_name)(
            [
                lambda: load_env(
                    env_name,
                    reset_split="test",
                    env_kwargs=env_kwargs,
                    ext_timelimit_obs=ext_timelimit_obs,
                )
                for _ in range(n_collect_workers)
            ]
        )
        vector_env_val.seed(base_seed_vec_val)
    else:
        vector_env_val = None

    # Hybrid agent
    hybrid_agent = HybridAgent(
        prototype_env,
        num_skills,
        skill_model_type,
        use_groundtruth_phi,
        sac_args,
        train_sac,
        loss_n_z_samples,
        reward_n_z_samples,
        reward_type,
        reward_options,
        device,
        skill_model_kwargs,
    )

    memory = EpisodeMemory(capacity=episode_buffer_capacity, seed=1)

    # Load checkpoint
    if checkpoint_data is not None:
        env_interactions_lpfm = checkpoint_data["env_interactions_lpfm"]
        env_interactions_sac = checkpoint_data["env_interactions_sac"]
        hybrid_agent.load_checkpoint(checkpoint_data, with_optim=True)
        memory.load_dict(checkpoint_data["memory"])
        recent_episodes = checkpoint_data["recent_episodes"]
        print(f"Loaded checkpoint from {last_checkpoint_file}")

        if save_all_episodes:
            with open(root_run_directory.joinpath(f"all_episodes.pkl"), "rb") as handle:
                all_episodes = pkl.load(handle)
        else:
            all_episodes = []

    else:
        all_episodes = []
        recent_episodes = []
        env_interactions_lpfm = 0
        env_interactions_sac = 0

    env_interactions_total = env_interactions_sac + env_interactions_lpfm

    print(
        f"Starting with epoch {first_epoch_idx}, "
        f"interactions SAC: {env_interactions_sac}, "
        f"interactions skll model: {env_interactions_lpfm}"
    )

    # Logging
    logger = Logger(_run, root_run_directory, log_every=log_every)

    base_config = {
        "num_skills": num_skills,
        "device": device,
    }
    relabel_base_config = {
        "relabel_episode_batchsize": relabel_episode_batchsize,
        "non_final_reward": non_final_reward,
        "relabel_type": relabel_type,
        "relabel_opts": relabel_opts,
    }

    # full relabelling of data from replay buffer for skill model
    relabel_config_skm = dict(
        **base_config,
        **relabel_base_config,
        evaluate_symbolic_state_change=evaluate_symbolic_state_change_skm,
    )

    # help SAC training by partially relabelling data (to retain bad examples)
    relabel_config_sac = dict(
        **base_config,
        **relabel_base_config,
        evaluate_symbolic_state_change=evaluate_symbolic_state_change_sac,
    )

    # no relabelling for validation
    relabel_config_val = dict(
        **base_config,
        **relabel_base_config,
        relabel=0,
        evaluate_symbolic_state_change=False,
    )
    collect_kwargs = dict(
        hybrid_agent=hybrid_agent,
    )

    symbolic_action_rng = np.random.RandomState(symbolic_action_seed)
    train_reset_split = "train"
    train_episode_collector = EpisodeCollectorAsync(
        env_name,
        env_kwargs,
        train_reset_split,
        num_skills,
        ext_timelimit_obs,
        n_collect_workers,
        base_seed_vec_train,
        symbolic_action_rng,
        new_episode_random_action_prob,
        new_episode_all_random_prob,
    )

    last_epoch_idx = first_epoch_idx + n_epochs_per_leap

    def _cleanup():
        if vector_env_val:
            vector_env_val.close()
        train_episode_collector.close()

    if env_interactions_total >= max_env_interactions:
        _cleanup()
        return

    def validate(step):
        validation_buffer = EpisodeMemory(capacity=n_validation_episodes, seed=1)
        symbolic_actions_val = symbolic_action_rng.randint(
            0, num_skills, n_validation_episodes
        )
        episodes_val = collect_episodes_parallel(
            vector_env=vector_env_val,
            symbolic_actions=symbolic_actions_val,
            evaluate=True,
            random_action_prob_list=[
                None,
            ]
            * len(symbolic_actions_val),
            **collect_kwargs,
        )
        [validation_buffer.add_episode(e) for e in episodes_val]
        relabeled_episodes_val = relabel_episodes(
            relabel_config_val,
            validation_buffer.buffer,
            hybrid_agent,
        )
        log_episode_batch_statistics(
            prototype_env,
            num_skills,
            relabeled_episodes_val,
            logger,
            step,
            prefix="val",
        )
        log_episode_batch_statistics(
            prototype_env,
            num_skills,
            relabeled_episodes_val,
            logger,
            env_interactions_lpfm + env_interactions_sac,
            prefix="val_interactions",
        )
        if hybrid_agent.can_predict:
            plot_episode_predicates(
                hybrid_agent.skill_model,
                device,
                prototype_env,
                relabeled_episodes_val[:10],
                logger,
                step,
                prefix="val",
            )

    for epoch_idx in tqdm(range(first_epoch_idx, last_epoch_idx)):

        # Collect validation data
        # Log validation performance
        if evaluate_every > 0 and epoch_idx % evaluate_every == 0:
            validate(epoch_idx)

        # Start asynchronous episode collection
        train_episode_collector.collect_async(
            n_new_episodes_per_epoch, collect_kwargs, eval_mode=False
        )

        skm_step_config = dict(
            **base_config,
            n_steps_skm=n_steps_skm,
            batchsize_skm=batchsize_skm,
            exclude_nop_episodes_skm=exclude_nop_episodes_skm,
        )

        # Update skill model
        if (
            hybrid_agent.train_skill_model or hybrid_agent.train_sac
        ) and memory.size >= n_sampled_episodes_per_epoch:
            for _ in range(n_steps_skm_outer):
                # Sample from buffer with symbolic action relabelling
                if n_sampled_episodes_per_epoch < memory.size:
                    buffer_episodes = memory.sample(n_sampled_episodes_per_epoch)
                else:
                    buffer_episodes = memory.buffer

                relabeled_episodes_skm = relabel_multi_episode_sources(
                    relabel_specifier_skm,
                    rollout_datasource,
                    relabel_skm,
                    buffer_episodes,
                    recent_episodes,
                    relabel_config_skm,
                    hybrid_agent,
                )

                if hybrid_agent.train_skill_model:
                    hybrid_agent.step_skill_model(
                        skm_step_config, relabeled_episodes_skm, logger, epoch_idx
                    )

            if hybrid_agent.train_sac:
                if n_sampled_episodes_per_epoch < memory.size:
                    buffer_episodes = memory.sample(n_sampled_episodes_per_epoch)
                else:
                    buffer_episodes = memory.buffer

                relabeled_episodes_sac = relabel_multi_episode_sources(
                    relabel_specifier_sac,
                    rollout_datasource,
                    relabel_sac,
                    buffer_episodes,
                    recent_episodes,
                    relabel_config_sac,
                    hybrid_agent,
                )

                sac_step_config = dict(
                    n_steps_sac=n_steps_sac, batchsize_sac=batchsize_sac
                )
                hybrid_agent.step_sac(
                    sac_step_config,
                    relabeled_episodes_sac,
                    logger,
                    global_step=epoch_idx,
                )

            if epoch_idx % log_every == 0:
                log_episode_batch_statistics(
                    prototype_env,
                    num_skills,
                    relabeled_episodes_sac,
                    logger,
                    epoch_idx,
                    prefix="train",
                )
                log_episode_batch_statistics(
                    prototype_env,
                    num_skills,
                    relabeled_episodes_sac,
                    logger,
                    env_interactions_lpfm + env_interactions_sac,
                    prefix="train_interactions",
                )

        # Log number of environment interactions
        env_interactions_total = env_interactions_sac + env_interactions_lpfm
        finished = env_interactions_total >= max_env_interactions

        # Wait for new episodes to be collected
        new_episodes = train_episode_collector.wait()

        recent_episodes.extend(new_episodes)
        if save_all_episodes:
            all_episodes.extend(new_episodes)
        recent_episodes = recent_episodes[-n_sampled_episodes_per_epoch:]
        [memory.add_episode(e) for e in new_episodes]
        env_interactions_lpfm += sum(len(e) for e in new_episodes)

        # Evaluate again when finished
        if finished and evaluate_when_finished:
            validate(epoch_idx + 1)

        # Checkpointing
        if finished or ((epoch_idx + 1) % checkpoint_every == 0):
            print(f"Epoch {epoch_idx+1}, checkpointing")
            filename = ckpt_dir.joinpath(f"step_{epoch_idx+1}.pkl")
            checkpoint_data = hybrid_agent.save_checkpoint()
            checkpoint_data["epoch_idx"] = epoch_idx
            checkpoint_data["env_interactions_lpfm"] = env_interactions_lpfm
            checkpoint_data["env_interactions_sac"] = env_interactions_sac
            checkpoint_data["recent_episodes"] = recent_episodes
            checkpoint_data["memory"] = memory.save_dict()
            torch.save(checkpoint_data, filename)
            if save_all_episodes:
                with open(
                    root_run_directory.joinpath(f"all_episodes.pkl"), "wb"
                ) as handle:
                    pkl.dump(all_episodes, handle)

        logger.maybe_log_scalar(
            "env/env_interactions_lpfm", env_interactions_lpfm, epoch_idx + 1
        )
        logger.maybe_log_scalar(
            "env/env_interactions_sac", env_interactions_sac, epoch_idx + 1
        )
        logger.maybe_log_scalar(
            "env/env_interactions_total",
            env_interactions_total,
            epoch_idx + 1,
        )

        if finished:
            break

    _cleanup()

    finished = env_interactions_total >= max_env_interactions
    if not finished:
        raise WaitingForRestartInterrupt
    else:
        return


if __name__ == "__main__":
    experiment.run_commandline()
