from typing import List
import numpy as np
import torch
import dataclasses
from tqdm import tqdm
from src.model.Discrete.LSTM.LSTMCritic import LSTMCritic
from src.model.Discrete.LSTM.LSTMActor import LSTMActor
from src.model.Discrete.MLP.MLPActor import MLPActor
from src.model.Discrete.MLP.MLPCritic import MLPCritic
from src.model.Discrete.CNN.CNN import CNN
from src.PPOs.AbstractPPO import AbstractPPO


def get_model_flattened_params(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])


@dataclasses.dataclass
class DiscretePPO(AbstractPPO):
    """Discrete Proximal Policy Optimization (PPO) agent."""

    def __post_init__(self) -> None:
        """Perform post initialization checks and setup any additional attributes"""
        self.continuous_action_space = False
        super().__post_init__()

        self.action_size = self.env.action_space.n
        if self.recurrent:
            self.actor = LSTMActor(state_size=self.state_size, action_size=self.action_size,
                                   hidden_size=self.actor_hidden_size).to(self.device)
            self.critic = LSTMCritic(
                state_size=self.state_size, hidden_size=self.critic_hidden_size).to(self.device)
        else:
            self.actor = MLPActor(state_size=512, action_size=self.action_size,
                                    hidden_size=self.actor_hidden_size).to(self.device)
            self.critic = MLPCritic(state_size=512,hidden_size=self.critic_hidden_size).to(self.device)
        self.CNN = CNN(channels=1).to(self.device)

        """else:
            self.actor = MLPActor(state_size=self.state_size, action_size=self.action_size,
                                  hidden_size=self.actor_hidden_size).to(self.device)
            self.critic = MLPCritic(
                state_size=self.state_size, hidden_size=self.critic_hidden_size).to(self.device)"""

        print('Initializing discrete PPO agent')
        self.initialize_optimizer()
        # write the hyperparameters

    def choose_action(self, state: np.ndarray) -> (int, torch.Tensor):
        """Choose an action based on the current state

        Arguments
        --------
        state: np.ndarray
            The current state of the environment

        Returns
        -------
        action: int
            The action to take
        log_prob: torch.Tensor
            The log probability of the action
        """
        with torch.no_grad():
            state = torch.tensor(
                state, device=self.device, dtype=torch.float32)
            if self.recurrent:
                state = state.unsqueeze(0)
            state_cnned = self.CNN(state)
            action_probs = self.actor(state_cnned)
            # Compute the mask
            mask = self.get_mask(self.env.action_space.n)

            # Mask the action probabilities
            action_probs = action_probs * mask

            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self):
        """Update the policy and value parameters using the PPO algorithm"""
        torch.autograd.set_detect_anomaly(True)

        for _ in tqdm(range(self.epochs)):
            num_samples = len(self.buffer.rewards) - 1
            indices = torch.randperm(num_samples)

            for i in range(0, num_samples, self.minibatch_size):
                batch_indices = indices[i:i + self.minibatch_size]
                states, actions, old_log_probs, advantages, discounted_rewards = self.buffer.get_minibatch(
                    batch_indices)

                states = torch.stack(states)
                if self.recurrent:
                    states = states.unsqueeze(1)
                state_cnned = self.CNN(states)
                values = self.critic(state_cnned)
                values = values.squeeze().squeeze() if self.recurrent else values.squeeze()

                action_probs = self.actor(state_cnned)
                # Compute the mask
                masks_list = [self.get_mask(
                    self.env.action_space.n) for state in states]
                masks = torch.stack(masks_list)

                action_probs = action_probs * masks

                dist = torch.distributions.Categorical(action_probs)
                entropy = dist.entropy()
                discounted_rewards = torch.stack(discounted_rewards)
                discounted_rewards = discounted_rewards.squeeze().squeeze()
                actions = torch.stack(actions)
                actions = actions.squeeze()

                new_log_probs = dist.log_prob(actions)
                advantages = torch.stack(advantages)
                advantages = torch.squeeze(advantages)
                old_log_probs = torch.stack(old_log_probs).squeeze()

                ratio = torch.exp(new_log_probs - old_log_probs.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip,
                                    1 + self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2)

                critic_loss = self.critic_loss(values, discounted_rewards)
                loss = actor_loss + self.value_loss_coef * \
                    critic_loss - self.entropy_coef * entropy
                self.writer.add_scalar(
                    "Value Loss", critic_loss.mean(), self.total_updates_counter)
                self.writer.add_scalar(
                    "MLPActor Loss", actor_loss.mean(), self.total_updates_counter)
                self.writer.add_scalar("Entropy", entropy.mean(
                ) * self.entropy_coef, self.total_updates_counter)
                self.total_updates_counter += 1
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.mean().backward()
                # After the backward call

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # Update steps here...

        self.decay_learning_rate()
        self.buffer.clean_buffer()
