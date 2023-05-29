"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import distributions, nn, optim

from seads.common import to_onehot
from seads.skill_model.common import (
    compute_intrinsic_reward,
    gather_k,
    introduce_dim,
    sample_at_dim,
)


class XORMLPProbabilisticForwardModel(nn.Module):
    def __init__(self, symbolic_shape, num_skills, hidden_dim):
        """
        Forward model q(z_T | z_0, k), parametrizing the probability of a bit in the symbolic state to flip

        Parameters
        ----------
        symbolic_shape: tuple
            Shape of symbolic state
        num_skills: int
            Number of skills
        hidden_dim: int
            Hidden dimension of the MLP
        """
        super(XORMLPProbabilisticForwardModel, self).__init__()
        self._symbolic_shape = symbolic_shape
        self._num_skills = num_skills

        symbolic_size = int(np.prod(symbolic_shape))
        self._flip_prob_mlp = nn.Sequential(
            nn.Linear(num_skills + symbolic_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, symbolic_size),
            nn.Sigmoid(),
        )

    @property
    def symbolic_shape(self):
        return self._symbolic_shape

    @property
    def num_skills(self):
        return self._num_skills

    def predict(self, symbolic_state, symbolic_action):
        symbolic_action_onehot = to_onehot(symbolic_action, self.num_skills)
        bs = symbolic_state.shape[: -len(self.symbolic_shape)]
        symbolic_flat = symbolic_state.view(*bs, -1)
        flip_prob = self._flip_prob_mlp(
            torch.cat((symbolic_flat, symbolic_action_onehot), dim=-1)
        ).view(*bs, *self.symbolic_shape)
        symbolic_state = symbolic_state.float()
        prob_true = (1 - symbolic_state) * flip_prob + symbolic_state * (1 - flip_prob)
        pred_dist = distributions.Bernoulli(probs=prob_true)
        return pred_dist

    def compute_logprob(self, z_initial, z_target, symbolic_action):
        dist = self.predict(z_initial, symbolic_action)
        log_prob = dist.log_prob(z_target)
        return log_prob


class ForwardModelComponent:
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
        Skill model q(k | z_T, z_0), implemented with forward model

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
        super(ForwardModelComponent, self).__init__()

        self._map_fcn = map_fcn
        self.num_skills = num_skills
        self._device = device
        self.symbolic_shape = symbolic_shape
        self._loss_n_z_samples = loss_n_z_samples

        self.pfm = XORMLPProbabilisticForwardModel(
            symbolic_shape, num_skills, kwargs["pfm_kwargs"]["hidden_dim"]
        ).to(device)
        if kwargs["train_pfm"]:
            self.optim = getattr(optim, kwargs["optim_pfm_classname"])(
                self.pfm.parameters(), **kwargs["optim_pfm_kwargs"]
            )
            self.train_any = True
        else:
            self.train_any = False

    def to(self, device):
        self.pfm = self.pfm.to(device)
        return self

    def map(self, observation):
        return self._map_fcn(observation)

    def predict(self, symbolic_state, symbolic_action):
        bernoulli_pred = self.pfm.predict(symbolic_state, symbolic_action)
        return bernoulli_pred

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
        logprob = self.pfm.compute_logprob(
            z_init_sample, z_target_sample, symbolic_action[:, None]
        ).mean()
        loss = -1.0 * logprob
        loss.backward()
        self.optim.step()
        metrics = {"pfm_logprob": logprob}
        losses = {"pfm_loss": loss}
        return losses, metrics

    def compute_log_prob_k(
        self, s_noisy_initial, s_noisy_target, symbolic_action, n_z_samples
    ):
        bs = s_noisy_initial.shape[0]
        device = s_noisy_initial.device

        z_init_dist = self.map(s_noisy_initial)
        z_target_dist = self.map(s_noisy_target)
        z_init_sample = sample_at_dim(z_init_dist, 1, self._loss_n_z_samples)
        z_target_sample = sample_at_dim(z_target_dist, 1, self._loss_n_z_samples)

        # extend all inputs by another batch dimension to
        # compute forward predictions for *all* symbolic actions
        actions_all = torch.arange(self.num_skills).long().to(device)
        actions_all = introduce_dim(actions_all, 0, bs)
        # actions_all: [bs x num_skills]
        actions_all = introduce_dim(actions_all, 2, n_z_samples)
        # actions_all: [bs x num_skills x n_z_samples]
        z_init_sample_ext = introduce_dim(z_init_sample, 1, self.num_skills)
        z_target_sample_ext = introduce_dim(z_target_sample, 1, self.num_skills)
        # z_*_sample_ext: [bs x num_skills x n_z_samples x <z_dim>]

        pred_distribution = self.predict(z_init_sample_ext, actions_all)

        log_prob = pred_distribution.log_prob(z_target_sample_ext)

        # sum over <z_dim>
        log_prob = log_prob.sum(dim=tuple(range(3, log_prob.dim())))
        # log_prob: [bs, num_skills, n_z_samples]
        log_prob_sum = torch.logsumexp(log_prob, dim=1, keepdim=True)
        # log_prob_sum: [bs, 1, n_z_samples]

        log_prob_ratio = log_prob - log_prob_sum
        # log_prob_ratio: [bs, num_skills, n_z_samples]

        index_ = symbolic_action[:, None]
        # index_: [bs, 1]

        # log  [ p(z' | k, z) ]
        log_q_zp_gvn_k_z_all = log_prob.mean(dim=-1)
        log_q_zp_gvn_k_z = torch.gather(
            log_q_zp_gvn_k_z_all, dim=1, index=index_
        ).squeeze(1)

        log_prob_ratio_last_action = log_prob_ratio.permute(0, 2, 1)
        log_prob_ratio_k = gather_k(log_prob_ratio_last_action, symbolic_action)
        log_q_k_gvn_z_zp = log_prob_ratio_k.mean(dim=-1)
        log_q_k_gvn_z_zp_all = log_prob_ratio.mean(dim=-1)

        return {
            "log_q_zp_gvn_k_z_all": log_q_zp_gvn_k_z_all,
            "log_q_zp_gvn_k_z": log_q_zp_gvn_k_z,
            "log_q_k_gvn_z_zp_all": log_q_k_gvn_z_zp_all,
            "log_q_k_gvn_z_zp": log_q_k_gvn_z_zp,
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
            log_q_z=log_prob_dict["log_q_zp_gvn_k_z_all"],
            symbolic_action=symbolic_action,
            num_skills=self.num_skills,
            reward_components=reward_options["reward_components"],
        )
        return dict(**reward_dict, **log_prob_dict)

    def get_state_dict(self, with_optim=False):
        state_dict = {
            "pfm": self.pfm.state_dict(),
        }
        if with_optim:
            state_dict["optim_pfm"] = self.optim.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, with_optim=False):
        self.pfm.load_state_dict(state_dict["pfm"])
        if with_optim:
            self.optim.load_state_dict(state_dict["optim_pfm"])


def compute_binary_match(
    forward_model,
    s_noisy_initial,
    s_noisy_target,
    symbolic_action,
    return_all,
):
    """
    Compute binary match between target and predicted predicates

    Parameters
    ----------
    forward_model: `ForwardModelComponent`
        Forward model
    s_noisy_initial: `torch.Tensor`, shape [`bs` x <obs_dim>]
        Initial noisy observation (s_0)
    s_noisy_target: `torch.Tensor`, shape [`bs` x <obs_dim>]
        Target noisy observation (s_N)
    symbolic_action: `torch.Tensor`, shape [`bs`]
        Symbolic action (k)
    return_all: bool
        Return binary match for all symbolic actions

    Returns
    -------
    match_k: `torch.Tensor`, shape [bs]
        Binary match (in [0, 1])
    match_all: `torch.Tensor`, shape [`bs`, `num_skills`]
        Binary match (in [0, 1]) (all symb. actions)
    """
    bs = s_noisy_initial.shape[0]
    device = s_noisy_initial.device
    num_skills = forward_model.num_skills
    z_init_dist = forward_model.map(s_noisy_initial)
    z_target_dist = forward_model.map(s_noisy_target)
    z_init_bin = z_init_dist.mode
    z_target_bin = z_target_dist.mode
    z_init_bin = introduce_dim(z_init_bin, 1, num_skills)
    z_target_bin = introduce_dim(z_target_bin, 1, num_skills)
    # z_init_bin/z_target_bin: [bs x num_skills x <z_dim>]

    # extend all inputs by another batch dimension to
    # compute forward predictions for *all* symbolic actions
    actions_all = torch.arange(num_skills).long().to(device)
    actions_all = introduce_dim(actions_all, 0, bs)

    pred_distribution = forward_model.predict(z_init_bin, actions_all)
    # noinspection PyUnresolvedReferences
    # 'mode' attribute is patched into 'Bernoulli'
    pred_bin = pred_distribution.mode

    match = pred_bin == z_target_bin
    # match: [bs x num_skills x <z_dim>]
    match_rel_all = match.sum(dim=tuple(range(2, match.dim()))) / np.prod(
        match.shape[2:]
    )

    index_ = symbolic_action[:, None]
    # index_: [bs, 1]
    match_rel_k = torch.gather(match_rel_all, dim=1, index=index_).squeeze(1)
    # match_k: [bs]

    if return_all:
        return match_rel_k, match_rel_all
    else:
        return match_rel_k


def plot_predicates(
    forward_model,
    s_noisy_initial,
    z_true_initial,
    s_noisy_target,
    z_true_target,
    symbolic_action,
):
    z_mapped_initial = forward_model.map(s_noisy_initial).mode
    z_mapped_target = forward_model.map(s_noisy_target).mode
    prediction = forward_model.predict(z_mapped_initial, symbolic_action).mode

    n_display_samples = len(s_noisy_initial)
    fig, ax = plt.subplots(
        nrows=n_display_samples, ncols=5, figsize=(5 * 2, n_display_samples * 2)
    )

    def show_img(ax_, tensor_):
        if tensor_.dim() == 1:
            halfsize = int(np.ceil(np.sqrt(tensor_.shape[0])))
            n_missing = halfsize ** 2 - tensor_.shape[0]
            arr = tensor_.cpu().detach().numpy().astype(float)
            ar_padded = np.pad(arr, (0, n_missing))
            ar_padded = ar_padded.reshape(halfsize, halfsize)
            ax_.imshow(ar_padded, vmin=0, vmax=1)
            ax_.set_xticks([])
            ax_.set_yticks([])
        else:
            ax_.imshow(tensor_.cpu().detach().numpy(), vmin=0, vmax=1)
            ax_.set_xticks([])
            ax_.set_yticks([])

    for row_idx in range(n_display_samples):
        show_img(ax[row_idx, 0], z_true_initial[row_idx])
        show_img(ax[row_idx, 1], z_mapped_initial[row_idx])
        show_img(ax[row_idx, 2], z_true_target[row_idx])
        show_img(ax[row_idx, 3], z_mapped_target[row_idx])
        show_img(ax[row_idx, 4], prediction[row_idx])

    ax[0, 0].set_title("s_true_initial")
    ax[0, 1].set_title("z_mapped_initial")
    ax[0, 2].set_title("s_true_target")
    ax[0, 3].set_title("z_mapped_target")
    ax[0, 4].set_title("prediction")
    return fig
