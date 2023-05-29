"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch

from seads.common import Logger, ReplayMemory, to_onehot
from seads.modelfree.sac.sac import SAC
from seads.skill_model.forward_model import ForwardModelComponent
from seads.skill_model.skill_discriminator import SkillDiscriminatorComponent
from seads.utils import ArgumentDict, torch_load


def fill_buffer_from_episodes(relabeled_episodes):
    state_buffer = ReplayMemory(capacity=1_000_000, seed=1)
    for e in relabeled_episodes:
        for transition in e:
            state_buffer.add_transition(transition)
    return state_buffer


class DiracDistribution:
    def __init__(self, value):
        self._value = value

    @property
    def mode(self):
        return self._value

    def rsample(self, sample_shape):
        slice_ = (None,) * len(sample_shape) + (Ellipsis,)
        sample = self._value[slice_].expand(*sample_shape, *self._value.shape)
        return sample

    def sample(self, sample_shape):
        return self.rsample(sample_shape).detach()


class HybridAgent:
    def __init__(
        self,
        env,
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
        skill_model_kwargs=None,
    ):
        self._action_space = env.action_space
        self._num_skills = num_skills
        self._observation_shape = env.observation_space.shape
        self._train_sac = train_sac
        self._loss_n_z_samples = loss_n_z_samples
        self._reward_n_z_samples = reward_n_z_samples
        self._reward_type = reward_type
        self._reward_options = reward_options
        self._device = device
        self._num_parameters = {}

        if not use_groundtruth_phi:
            raise NotImplementedError

        self.phi_fcn = lambda s: env.get_binary_symbolic_state(s).float()
        symbolic_shape = env.binary_symbolic_shape

        # Init skill model
        self._skill_model_type = skill_model_type
        if skill_model_type == "forward_model":
            skill_model_class = ForwardModelComponent
            self.can_predict = True
        elif skill_model_type == "discriminator":
            skill_model_class = SkillDiscriminatorComponent
            self.can_predict = False
        else:
            raise ValueError

        self.skill_model = skill_model_class(
            map_fcn=self.map,
            num_skills=self.num_skills,
            device=device,
            observation_shape=self._observation_shape,
            symbolic_shape=symbolic_shape,
            loss_n_z_samples=loss_n_z_samples,
            **skill_model_kwargs,
        )

        # Init SAC
        sac_input_size = int(np.prod(self._observation_shape)) + self._num_skills
        sac_args = dict(**sac_args)
        self._sac_agent = SAC(sac_input_size, env.action_space, ArgumentDict(sac_args))

    def map(self, observation):
        mapping = self.phi_fcn(observation)
        dist = DiracDistribution(value=mapping)
        return dist

    def _get_sac_input(self, state, symbolic_action):
        action_onehot = to_onehot(symbolic_action, self._num_skills)
        sac_input = np.concatenate((state, action_onehot), axis=-1)
        return sac_input

    def sac_input_fcn(self, batch, is_next):
        if is_next:
            state = batch["next_state"]
        else:
            state = batch["state"]
        sac_input = self._get_sac_input(state, batch["symbolic_action"])
        return sac_input

    @property
    def num_skills(self):
        return self._num_skills

    @property
    def train_sac(self):
        return self._train_sac

    @property
    def train_skill_model(self):
        return self.skill_model.train_any

    def select_action(self, state, symbolic_action, evaluate, random_action_prob=None):
        sac_input = self._get_sac_input(state, symbolic_action)
        if evaluate:
            action = self._sac_agent.select_action(sac_input, evaluate=True)
        else:
            action = self._sac_agent.select_action(sac_input, evaluate=False)
            if random_action_prob is not None:
                for input_idx in range(sac_input.shape[0]):
                    if hasattr(random_action_prob, "__len__"):
                        prob = random_action_prob[input_idx]
                    else:
                        prob = random_action_prob
                    if prob is not None and prob > 0:
                        replace_by_random = np.random.rand() < prob
                        if replace_by_random:
                            action[input_idx] = self._action_space.sample()
        return action

    def compute_reward(
        self, initial_state, next_state, symbolic_action, return_all=False
    ):
        device = self._device
        if isinstance(initial_state, np.ndarray):
            initial_state = np.atleast_2d(initial_state)
            initial_state = torch.from_numpy(initial_state).float().to(device)
        if isinstance(next_state, np.ndarray):
            next_state = np.atleast_2d(next_state)
            next_state = torch.from_numpy(next_state).float().to(device)
        if isinstance(symbolic_action, np.ndarray) or isinstance(symbolic_action, int):
            symbolic_action = np.atleast_1d(symbolic_action)
            symbolic_action = torch.from_numpy(symbolic_action).long().to(device)
        reward_dict = self.skill_model.compute_intrinsic_reward(
            initial_state,
            next_state,
            symbolic_action,
            self._reward_n_z_samples,
            self._reward_options,
        )
        reward_k = reward_dict["reward"]
        reward_all = reward_dict["reward_all"]
        min_reward = reward_dict["reward_min"]
        log_prob_k = reward_dict["log_q_k_gvn_z_zp"]
        log_prob_k_all = reward_dict["log_q_k_gvn_z_zp_all"]
        if return_all:
            return reward_k, reward_all, log_prob_k, log_prob_k_all, min_reward
        else:
            return reward_k, min_reward

    def step_sac(self, sac_step_config, episode_list, logger, global_step):
        if not self.train_sac:
            raise RuntimeError
        state_buffer = fill_buffer_from_episodes(episode_list)
        sac_step(
            sac_step_config,
            self._sac_agent,
            self.sac_input_fcn,
            state_buffer,
            logger,
            global_step,
        )

    def step_skill_model(self, config, episode_list, logger, global_step):
        skill_model_step(
            self.skill_model,
            config,
            episode_list,
            logger,
            global_step,
        )

    def save_checkpoint(self, filename=None):
        checkpoint_data = {
            "sac": self._sac_agent.state_dict(),
            "skill_model": self.skill_model.get_state_dict(with_optim=True),
        }
        if filename is None:
            return checkpoint_data
        else:
            torch.save(checkpoint_data, filename)

    def load_checkpoint(self, filename_or_dict, with_optim=False):
        if not isinstance(filename_or_dict, dict):
            checkpoint_data = torch_load(filename_or_dict)
        else:
            checkpoint_data = filename_or_dict
        self._sac_agent.load_state_dict(checkpoint_data["sac"])
        # legacy checkpoints
        if "lpfm" in checkpoint_data:
            skill_model_state_dict = checkpoint_data["lpfm"]
        elif "skill_discriminator" in checkpoint_data:
            skill_model_state_dict = checkpoint_data["skill_discriminator"]
        else:
            skill_model_state_dict = checkpoint_data["skill_model"]
        self.skill_model.load_state_dict(skill_model_state_dict, with_optim)


def sac_step(config, agent, sac_input_fcn, buffer, logger, global_step):
    """
    Run update steps on agent

    Parameters
    ----------
    config: dict
        Config containing ["n_steps_sac", "batchsize_sac"]
    agent: `SAC`
        SAC agent
    sac_input_fcn: function
        Function (batch, is_next) -> np.ndarray
    buffer: `ReplayMemory`
        Replay buffer
    logger: `Logger`
        Logger
    global_step: int
        Global step
    """
    n_steps = config["n_steps_sac"]
    batchsize_sac = config["batchsize_sac"]

    def process_batch_fcn(batch):
        state = sac_input_fcn(batch, is_next=False)
        action = batch["action"]
        reward = batch["reward"]
        next_state = sac_input_fcn(batch, is_next=True)
        done = batch["done"]
        return state, action, reward, next_state, done

    buffer.batch_process_fcn = process_batch_fcn

    if not hasattr(sac_step, "updates"):
        # static counter for number of total updates
        sac_step.updates = 0

    for i in range(n_steps):
        # Update parameters of all the networks
        (
            critic_1_loss,
            critic_2_loss,
            policy_loss,
            ent_loss,
            alpha,
        ) = agent.update_parameters(buffer, batchsize_sac, sac_step.updates)
        sac_step.updates += 1

    logger.maybe_log_scalar("loss/critic_1", critic_1_loss, global_step)
    logger.maybe_log_scalar("loss/critic_2", critic_2_loss, global_step)
    logger.maybe_log_scalar("loss/policy", policy_loss, global_step)
    logger.maybe_log_scalar("loss/entropy_loss", ent_loss, global_step)
    logger.maybe_log_array("entropy_temperature/alpha", alpha, global_step)


def skill_model_step(skill_model, config, relabeled_episodes, logger, global_step):
    device = config["device"]
    batchsize = config["batchsize_skm"]
    n_steps = config["n_steps_skm"]

    final_state_buffer = ReplayMemory(capacity=1_000_000, seed=1)

    if config["exclude_nop_episodes_skm"]:
        filtered_episodes = []
        for e in relabeled_episodes:
            if e.flags["symbolic_state_changed"]:
                filtered_episodes.append(e)
    else:
        filtered_episodes = relabeled_episodes

    for e in filtered_episodes:
        final_state_buffer.add_transition(e[-1])

    for step_idx in range(n_steps):
        batch = final_state_buffer.sample(batchsize)
        initial_state_t = torch.from_numpy(batch["initial_state"]).float().to(device)
        next_state_t = torch.from_numpy(batch["next_state"]).float().to(device)
        action_t = torch.from_numpy(batch["symbolic_action"]).long().to(device)

        losses, metrics = skill_model.train_step(
            initial_state_t,
            next_state_t,
            action_t,
            additional_metrics=None,
        )

    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            logger.maybe_log_scalar(f"losses/{k}", v.mean().item(), global_step)

    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            logger.maybe_log_scalar(f"metrics/{k}", v.mean().item(), global_step)
