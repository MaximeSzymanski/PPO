from typing import List
import numpy as np
import torch
import dataclasses
from tqdm import tqdm
from src.model.Discrete.LSTM.LSTMCritic import LSTMCritic
from src.model.Discrete.LSTM.LSTMActor import LSTMActor
from src.model.Discrete.MLP.MLPActor import MLPActor
from src.model.Discrete.MLP.MLPCritic import MLPCritic
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
            self.actor = MLPActor(state_size=self.state_size, action_size=self.action_size,
                                  hidden_size=self.actor_hidden_size).to(self.device)
            self.critic = MLPCritic(
                state_size=self.state_size, hidden_size=self.critic_hidden_size).to(self.device)

        print('Initializing discrete PPO agent')
        self.initialize_optimizer()
        # write the hyperparameters

    def get_mask(self, action_size: int,state: torch.tensor) -> torch.Tensor:
        """Get a mask for the action probabilities

        Arguments
        ---------
        action_size: int
            The number of actions

        Returns
        -------
        mask: torch.Tensor
            The mask for the action probabilities
        """
        # Scale values to match the scale of calculations
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        number_stocks_in_portfolio = state[-1][-1][-1].item()
        price_of_stock = state[-1][-1][1].item()
        portfolio_value = state[-1][-1][-2].item()

        number_stocks_in_portfolio *= 10
        portfolio_value *= 10000
        price_of_stock *= 10

        # Initialize mask
        action_mask = np.zeros(3)

        # Check for invalid values
        if np.isinf(portfolio_value) or np.isnan(portfolio_value) or np.isinf(price_of_stock) or np.isnan(
                price_of_stock):
            print('Portfolio value or stock price is either infinite or NaN.')
            return torch.tensor(action_mask, dtype=torch.float32,device=self.device)  # All actions are not possible

        # Always able to hold
        action_mask[0] = 1

        # Able to buy if there's enough money, considering 10 stocks at a time
        if portfolio_value >= 10 * price_of_stock:
            action_mask[1] = 1

        # Able to sell if there are enough stocks in the portfolio
        if number_stocks_in_portfolio >= 10:
            action_mask[2] = 1

        return torch.tensor(action_mask, dtype=torch.float32, device=self.device)
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

            action_probs = self.actor(state)
            # Compute the mask
            mask = self.get_mask(self.env.action_space.n,state)

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

                values = self.critic(states)
                values = values.squeeze().squeeze() if self.recurrent else values.squeeze()
                action_probs = self.actor(states)
                # Compute the mask
                masks_list = [self.get_mask(
                    self.env.action_space.n,state) for state in states]
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
