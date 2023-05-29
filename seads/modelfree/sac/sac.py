import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .model import DeterministicPolicy, GaussianPolicy, QNetwork, RelaxedBernoulliPolicy
from .utils import hard_update, soft_update


class SAC(nn.Module):
    def __init__(self, num_inputs, action_space, args):
        super(SAC, self).__init__()

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(
            num_inputs,
            action_space.shape[0],
            args.hidden_size,
        ).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(
            num_inputs,
            action_space.shape[0],
            args.hidden_size,
        ).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type in ["Gaussian", "RelaxedBernoulli"]:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(
                    torch.Tensor(action_space.shape).to(self.device)
                ).item()
                self.log_alpha = torch.tensor(
                    [math.log(args.alpha)], device=self.device
                )
                self.log_alpha.requires_grad = True
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            if self.policy_type == "Gaussian":
                legacy_policy_size = getattr(args, "legacy_policy_size", False)
                self.policy = GaussianPolicy(
                    num_inputs,
                    action_space.shape[0],
                    args.hidden_size,
                    action_space,
                    legacy_policy_size,
                ).to(self.device)
            elif self.policy_type == "RelaxedBernoulli":
                self.policy = RelaxedBernoulliPolicy(
                    num_inputs,
                    action_space.shape[0],
                    args.hidden_size,
                    args.relaxed_bernoulli_temp,
                ).to(self.device)
            else:
                raise ValueError
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        elif self.policy_type == "Deterministic":
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            raise ValueError

    def select_action(self, state, evaluate=False):
        squeeze = False
        if state.ndim == 1:
            state = state[None, :]
            squeeze = True
        state = torch.FloatTensor(state).to(self.device)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        action = action.detach().cpu().numpy()
        if squeeze:
            action = action[0]
        return action

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = memory.sample(batch_size=batch_size)

        return self.update_parameters_batch(
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            done_batch,
            updates,
        )

    def update_parameters_batch(
        self,
        state_batch,
        action_batch,
        next_state_batch,
        reward_batch,
        done_batch,
        updates,
    ):
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * (
                min_qf_next_target
            )

        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (
            (self.alpha * log_pi) - min_qf_pi
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
        )

    def state_dict(self):
        # Code author: Jan Achterhold

        state_dict = {
            "policy": self.policy.state_dict(),
            "policy_optim": self.policy_optim.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }
        if self.automatic_entropy_tuning:
            state_dict["log_alpha"] = self.log_alpha.item()
            state_dict["alpha_optim"] = self.alpha_optim.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, with_optim=True):
        # Code author: Jan Achterhold

        self.policy.load_state_dict(state_dict["policy"])
        if with_optim:
            self.policy_optim.load_state_dict(state_dict["policy_optim"])
        self.critic.load_state_dict(state_dict["critic"])
        if state_dict["critic_target"] is not None:
            self.critic_target.load_state_dict(state_dict["critic_target"])
        else:
            hard_update(self.critic_target, self.critic)
        if with_optim:
            self.critic_optim.load_state_dict(state_dict["critic_optim"])

        if self.automatic_entropy_tuning:
            if state_dict["log_alpha"] is not None:
                self.log_alpha.data.copy_(
                    torch.tensor([state_dict["log_alpha"]], device=self.device)
                )
            if state_dict["alpha_optim"] is not None:
                self.alpha_optim.load_state_dict(state_dict["alpha_optim"])
