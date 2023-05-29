"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical

from seads.skill_model.common import compute_intrinsic_reward, sample_at_dim


class MLPDiscriminatorModel(nn.Module):
    def __init__(self, num_skills, symbolic_shape, hidden_dim, input_type):
        """
        Skill discriminator q(k | z_0, z_T)

        Parameters
        ----------
        num_skills: int
            Number of skills
        symbolic_shape: tuple
            Shape of symbolic state
        hidden_dim: int
            Hidden dimension of the MLP
        input_type: str
            Input representation for the MLP, can be 'concat' or 'concatxor'
        """
        super(MLPDiscriminatorModel, self).__init__()
        self._num_skills = num_skills
        self._symbolic_shape = symbolic_shape
        symbolic_flat_dim = np.prod(np.array(symbolic_shape))
        self._input_type = input_type
        input_dim = {
            "concat": 2 * symbolic_flat_dim,
            "concatxor": 3 * symbolic_flat_dim,
        }[input_type]
        self.logit_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_skills),
        )

    def forward(self, z_initial, z_target):
        bs_initial = z_initial.shape[: -len(self._symbolic_shape)]
        bs_target = z_target.shape[: -len(self._symbolic_shape)]
        assert bs_initial == bs_target
        z_initial_flat = z_initial.view(*bs_initial, -1)
        z_target_flat = z_target.view(*bs_target, -1)
        if self._input_type == "concat":
            mlp_input = torch.cat((z_initial_flat, z_target_flat), dim=-1)
        elif self._input_type == "concatxor":
            xor = (z_initial_flat - z_target_flat).abs()
            mlp_input = torch.cat((z_initial_flat, z_target_flat, xor), dim=-1)
        else:
            raise ValueError
        logits = self.logit_mlp(mlp_input)
        distribution = Categorical(logits=logits)
        return distribution

    def compute_logprob(self, z_initial, z_target, symbolic_action):
        dist = self(z_initial, z_target)
        log_prob = dist.log_prob(symbolic_action)
        return log_prob


class SkillDiscriminatorComponent:
    def __init__(
        self,
        map_fcn,
        num_skills,
        device,
        observation_shape,
        symbolic_shape,
        loss_n_z_samples,
        **kwargs
    ):
        """
        Skill model q(k | z_T, z_0), implemented with skill discriminator

        Parameters
        ----------
        map_fcn: function
            Mapping function
        num_skills: int
            Number of skills
        device: str
            PyTorch device
        observation_shape: tuple
            Shape of full observation
        symbolic_shape: tuple
            Shape of symbolic state
        loss_n_z_samples: int
            Number of samples (from the mapping function) to use for computing the training loss
        kwargs: dict
            Further arguments
        """
        self._map_fcn = map_fcn
        self.num_skills = num_skills
        self._device = device
        self._loss_n_z_samples = loss_n_z_samples

        self.discriminator = MLPDiscriminatorModel(
            num_skills, symbolic_shape, kwargs["hidden_dim"], kwargs["input_type"]
        ).to(device)
        if kwargs["train_discriminator"]:
            self.optim = getattr(optim, kwargs["optim_classname"])(
                self.discriminator.parameters(), **kwargs["optim_kwargs"]
            )
            self.train_any = True
        else:
            self.train_any = False

    def map(self, observation):
        return self._map_fcn(observation)

    def train_step(
        self,
        s_noisy_initial,
        s_noisy_target,
        symbolic_action,
        additional_metrics,
    ):
        z_init_dist = self.map(s_noisy_initial)
        z_target_dist = self.map(s_noisy_target)
        z_init_sample = sample_at_dim(z_init_dist, 1, self._loss_n_z_samples)
        z_target_sample = sample_at_dim(z_target_dist, 1, self._loss_n_z_samples)

        self.optim.zero_grad()
        logprob = self.discriminator.compute_logprob(
            z_init_sample, z_target_sample, symbolic_action[:, None]
        ).mean()
        loss = -1.0 * logprob
        loss.backward()
        self.optim.step()
        metrics = {"discriminator_logprob": logprob}
        losses = {"discriminator_loss": loss}
        return losses, metrics

    def get_state_dict(self, with_optim=False):
        state_dict = {
            "discriminator": self.discriminator.state_dict(),
        }
        if with_optim:
            state_dict["optim"] = self.optim.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, with_optim=False):
        self.discriminator.load_state_dict(state_dict["discriminator"])
        if with_optim:
            self.optim.load_state_dict(state_dict["optim"])

    def compute_log_prob_k(
        self, s_noisy_initial, s_noisy_target, symbolic_action, n_z_samples
    ):
        z_init_dist = self.map(s_noisy_initial)
        z_target_dist = self.map(s_noisy_target)
        z_init_sample = sample_at_dim(z_init_dist, 1, n_z_samples)
        z_target_sample = sample_at_dim(z_target_dist, 1, n_z_samples)
        skill_dist = self.discriminator(z_init_sample, z_target_sample)
        log_prob_k = skill_dist.log_prob(symbolic_action[:, None])
        log_prob_k_all = skill_dist.logits
        # average over n_z_samples
        log_prob_k = log_prob_k.mean(dim=1)
        log_prob_k_all = log_prob_k_all.mean(dim=1)
        return {
            "log_q_k_gvn_z_zp_all": log_prob_k_all,
            "log_q_k_gvn_z_zp": log_prob_k,
        }

    def compute_intrinsic_reward(
        self,
        s_noisy_initial,
        s_noisy_target,
        symbolic_action,
        n_z_samples,
        reward_options,
    ):
        log_prob_dict = self.compute_log_prob_k(
            s_noisy_initial, s_noisy_target, symbolic_action, n_z_samples
        )
        reward_dict = compute_intrinsic_reward(
            log_q_k=log_prob_dict["log_q_k_gvn_z_zp_all"],
            log_q_z=None,
            symbolic_action=symbolic_action,
            num_skills=self.num_skills,
            reward_components=reward_options["reward_components"],
        )
        return dict(**reward_dict, **log_prob_dict)
