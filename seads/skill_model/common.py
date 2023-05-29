"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

import math
from warnings import warn

import torch


def maybe_rsample(distribution, sample_shape):
    """
    Sample using .rsample() (if implemented), fallback to .sample() otherwise

    Parameters
    ----------
    distribution: `torch.distribution.Distribution`
        Distribution object
    sample_shape: tuple
        Sample shape

    Returns
    -------
    sample: torch.Tensor
        Samples
    """
    try:
        sample = distribution.rsample(sample_shape)
    except NotImplementedError:
        warn(
            f"{type(distribution).__name__} has no .rsample(), falling back to .sample())"
        )
        sample = distribution.sample(sample_shape)
    return sample


def sample_at_dim(distribution, sample_dim, n_samples):
    """
    Sample from `distribution` and put samples at particular dimension

    Parameters
    ----------
    distribution: `torch.distribution.Distribution`
        Distribution object
    sample_dim: int
        Sample dimension
    n_samples: int
        Number of samples

    Returns
    -------
    samples: `torch.Tensor`
        Tensor of samples where `samples.shape[sample_dim] == n_samples`.
    """
    samples = maybe_rsample(distribution, sample_shape=(n_samples,))
    if sample_dim != 0:
        # move samples to destination dim
        target_dims = (
            tuple(range(1, sample_dim + 1))
            + (0,)
            + tuple(range(sample_dim + 1, samples.dim()))
        )
        samples = samples.permute(*target_dims)
    # samples: [<>, n_samples, <>]
    return samples


def gather_k(tensor, last_dim_elems):
    """
    Gather items from last dimension of tensor, given `last_dim_elems`

    Parameters
    ----------
    tensor: `torch.Tensor`, shape [bs x <any_dim> x N]
        Tensor to gather elements from
    last_dim_elems: `torch.LongTensor`, shape [bs,]
        Index tensor for last dimension, in [0, ..., N-1]

    Returns
    -------
    tensor_k: `torch.Tensor`, shape [bs x <any_dim>]
        Selected elements
    """
    # noinspection PyTypeChecker
    # Type checker does not like mixing 'slice' with 'None'
    slice_ = (slice(None),) + (None,) * (tensor.dim() - 1)  # :, None, None, ...
    index_ = last_dim_elems[slice_].expand(*tensor.shape[:-1], 1)
    tensor_k = torch.gather(tensor, dim=-1, index=index_).squeeze(-1)
    return tensor_k


def introduce_dim(tensor, dim, size):
    """
    Introduce dimension of size `size` at `dim` of `tensor`.


    Parameters
    ----------
    tensor: `torch.Tensor`
        Input tensor
    dim: int
        Dimension to introduce
    size: int
        Size of dimension to introduce

    Returns
    -------
    expanded_tensor: `torch.Tensor`
        Expanded tensor. It holds that `expanded_tensor.shape[dim] == size` and
        `expanded_tensor[..., k, ...] == tensor, for all `k`, where k indices the `dim`-th dimension.
    """
    orig_shape = tensor.shape
    expanded_tensor = tensor.unsqueeze(dim)
    new_shape = orig_shape[:dim] + (size,) + orig_shape[dim:]
    expanded_tensor = expanded_tensor.expand(*new_shape)
    return expanded_tensor


def compute_symbolic_state_changed(map_fcn, s_noisy_initial, s_noisy_target):
    """
    Check if symbolic state has changed

    Parameters
    ----------
    map_fcn: function
        Mapping function, mapping full observation to symbolic observation
    s_noisy_initial: `torch.Tensor`
        Initial (full) observation
    s_noisy_target: `torch.Tensor`
        Target (full) observation

    Returns
    -------
    symbolic_state_changed: bool
        True if symbolic state has changed
    """
    z_init_dist = map_fcn(s_noisy_initial)
    z_target_dist = map_fcn(s_noisy_target)
    z_init_bin = z_init_dist.mode
    z_target_bin = z_target_dist.mode
    symbolic_state_changed = not torch.equal(z_init_bin, z_target_bin)
    return symbolic_state_changed


def compute_intrinsic_reward(
    log_q_k,
    log_q_z,
    symbolic_action,
    num_skills,
    reward_components,
):
    """
    Compute intrinsic reward

    Parameters
    ----------
    log_q_k: `torch.Tensor`, shape [<bs> x num_skills]
        log q(k | z_1, z_T)
    log_q_z: `torch.Tensor`, shape [<bs> x num_skills], optional
        log q(z_T | z_1, k), can be 'None' (e.g. for skill discriminator)
    symbolic_action: `torch.LongTensor`, shape [<bs>]
        Symbolic actions, given as integers
    num_skills: int
        Number of symbolic actions
    reward_components: iterable
        Set of (additive) reward components, with elements in
        * "log_q_k" -> log q(k | z_1, z_T)
        * "neg_log_q_k_baseline" -> - top2_k log q(k | z_1, z_T)  (second-best normalization)
        * "log_K" -> log K
        * "log_novb" -> novelty bonus

    Returns
    -------
    return_dict: dict
        reward: `torch.Tensor`, shape [<bs>]
            Intrinsic reward for symbolic action in `symbolic_action`
        reward_all: `torch.Tensor`, shape [<bs>, `num_skills`]
            Intrinsic reward for all symbolic actions
        reward_min: float
            Minimal attainable reward
    """
    assert log_q_k.dim() >= 2
    assert symbolic_action.shape == log_q_k.shape[:-1]
    assert log_q_k.shape[-1] == num_skills

    if len(reward_components) != len(set(reward_components)):
        raise ValueError("'reward_components' must not include duplicate elements.")

    valid_reward_components = ["log_q_k", "neg_log_q_k_baseline", "log_K", "log_novb"]

    for component in reward_components:
        if component not in valid_reward_components:
            raise ValueError(f"Invalid reward component '{component}'")

    if log_q_z is not None:
        assert log_q_k.shape == log_q_z.shape

    base_reward_min = -2.0 * math.log(num_skills)

    # clip log_q_k
    log_q_k = torch.clip(log_q_k, min=-2.0 * math.log(num_skills))

    # novelty bonus
    if log_q_z is not None:
        max_log_prob = torch.max(log_q_z, dim=-1)[0]
        if not torch.all(log_q_z <= 0):
            raise ValueError("Asserting log_q_z to be non-positive")
    else:
        max_log_prob = None

    reward_all: torch.Tensor = 0

    if len(set(reward_components)) != len(reward_components):
        raise ValueError("reward components must be unique!")

    if "log_q_k" in reward_components:
        reward_all += log_q_k

    if "neg_log_q_k_baseline" in reward_components:
        # baseline (second-highest value in log_q_k (over k))
        baseline = torch.topk(log_q_k, 2, largest=True, sorted=True).values[..., 1]
        reward_all += -1.0 * baseline.unsqueeze(-1)

    if "log_K" in reward_components:
        reward_all += math.log(num_skills)

    if "log_novb" in reward_components:
        # reward is only increased, minimum value of this term is '0'
        if max_log_prob is None:
            raise ValueError("log_q_z was not provided")
        max_log_prob = torch.clip(max_log_prob, min=-1.0 * math.log(num_skills))
        reward_all += (-1.0 * max_log_prob).unsqueeze(-1)

    reward_k = gather_k(reward_all, symbolic_action)

    return_dict = {
        "reward": reward_k,
        "reward_all": reward_all,
        "reward_min": float(base_reward_min),
    }
    return return_dict
