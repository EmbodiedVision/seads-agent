import torch
import torch.nn as nn
from torch.distributions import Normal, RelaxedBernoulli

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def build_net(input_dim, output_dim, hidden_dim):
    """
    Build a MLP with a linear output layer and relu-squashed hidden layers,
    with len(hidden_dim) hidden layers
    """
    # Code author: Jan Achterhold
    if isinstance(hidden_dim, int):
        hidden_dim = [
            hidden_dim,
        ]
    layers = [nn.Linear(input_dim, hidden_dim[0]), nn.ReLU()]
    for idx in range(1, len(hidden_dim)):
        layers.append(nn.Linear(hidden_dim[idx - 1], hidden_dim[idx]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim[-1], output_dim))
    net = nn.Sequential(*layers)
    return net


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim, hidden_dim]

        self.q1 = build_net(num_inputs + num_actions, 1, hidden_dim)
        self.q2 = build_net(num_inputs + num_actions, 1, hidden_dim)
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = self.q1(xu)
        x2 = self.q2(xu)
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        action_space=None,
        legacy_policy_size=False,
    ):
        super(GaussianPolicy, self).__init__()

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim, hidden_dim]
        net_hidden_dim = hidden_dim if legacy_policy_size else hidden_dim[:-1]
        self.net = build_net(num_inputs, hidden_dim[-1], net_hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim[-1], num_actions)
        self.log_std_linear = nn.Linear(hidden_dim[-1], num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, state):
        x = torch.relu(self.net(state))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class RelaxedBernoulliPolicy(nn.Module):
    # Code author: Jan Achterhold

    def __init__(self, num_inputs, num_actions, hidden_dim, relaxed_bernoulli_temp):
        super(RelaxedBernoulliPolicy, self).__init__()

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim, hidden_dim]
        self.net = build_net(num_inputs, num_actions, hidden_dim)
        self.temperature = nn.Parameter(
            torch.Tensor([relaxed_bernoulli_temp]), requires_grad=False
        )
        self.LOGIT_MIN = -5
        self.LOGIT_MAX = 5

        self.apply(weights_init_)

    def forward(self, state):
        x = self.net(state)
        bernoulli_logits = torch.clamp(x, min=self.LOGIT_MIN, max=self.LOGIT_MAX)
        return bernoulli_logits

    def sample(self, state):
        bernoulli_logits = self.forward(state)
        dist = RelaxedBernoulli(self.temperature, logits=bernoulli_logits)
        mean = (dist.probs > 0.5).float()
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()

        self.mean = build_net(num_inputs, num_actions, hidden_dim)
        self.noise = torch.Tensor(num_actions)
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, state):
        mean = torch.tanh(self.mean(state)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0.0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.0), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
