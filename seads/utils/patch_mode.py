"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

from warnings import warn

from torch.distributions import (
    Bernoulli,
    Categorical,
    ContinuousBernoulli,
    RelaxedBernoulli,
    TransformedDistribution,
)


def patch_mode():
    # patch 'mode' into distributions.Bernoulli
    if hasattr(Bernoulli, "mode"):
        warn("'Bernoulli' class already has 'mode' attribute")
    Bernoulli.mode = property(bernoulli_mode)

    # patch 'mode' into distributions.ContinuousBernoulli
    if hasattr(ContinuousBernoulli, "mode"):
        warn("'ContinuousBernoulli' class already has 'mode' attribute")
    ContinuousBernoulli.mode = property(bernoulli_mode)

    # patch 'mode' into distributions.RelaxedBernoulli
    if hasattr(RelaxedBernoulli, "mode"):
        warn("'RelaxedBernoulli' class already has 'mode' attribute")
    RelaxedBernoulli.mode = property(bernoulli_mode)

    # patch 'mode' into distributions.Categorical
    if hasattr(Categorical, "mode"):
        warn("'Categorical' class already has 'mode' attribute")
    Categorical.mode = property(categorical_mode)

    # patch 'mode' into distributions.TransformedDistribution
    if hasattr(TransformedDistribution, "mode"):
        warn("'TransformedDistribution' class already has 'mode' attribute")
    TransformedDistribution.mode = property(transformed_distribution_mode)


def bernoulli_mode(obj):
    return obj.probs > 0.5


def categorical_mode(obj):
    return obj.logits.argmax(dim=-1)


def transformed_distribution_mode(obj):
    x = obj.base_dist.mode
    for transform in obj.transforms:
        x = transform(x)
    return x
