from typing import List
import numpy as np
import torch
import dataclasses
import gymnasium as gym
from torch import nn as nn
from tqdm import tqdm

from RolloutBuffer import RolloutBuffer
from AbstractPPO import AbstractPPO

def get_model_flattened_params(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])


def get_action_mask(number_stocks_in_portfolio, price_of_stock, portfolio_value,device):
    """
    Compute the mask for actions the agent can take (buy, sell, hold), based on current state.

    Args:
        number_stocks_in_portfolio: current number of stocks in portfolio.
        price_of_stock: current price of the stock.
        portfolio_value: current value of the portfolio.

    Returns:
        torch.Tensor: A mask tensor where each element indicates if the action is possible (1) or not (0).
    """

    # Scale values to match the scale of calculations
    number_stocks_in_portfolio *= 10
    portfolio_value *= 10000
    price_of_stock *= 10

    # Initialize mask
    action_mask = np.zeros(3)

    # Check for invalid values
    if np.isinf(portfolio_value) or np.isnan(portfolio_value) or np.isinf(price_of_stock) or np.isnan(price_of_stock):
        print('Portfolio value or stock price is either infinite or NaN.')
        return torch.tensor(action_mask, dtype=torch.float32)  # All actions are not possible

    # Always able to hold
    action_mask[0] = 1

    # Able to buy if there's enough money, considering 10 stocks at a time
    if portfolio_value >= 10 * price_of_stock:
        action_mask[1] = 1

    # Able to sell if there are enough stocks in the portfolio
    if number_stocks_in_portfolio >= 10:
        action_mask[2] = 1

    return torch.tensor(action_mask, dtype=torch.float32,device=device)


class MLPActor(nn.Module):

    def __init__(self, state_size: int = 0, action_size: int = 0, hidden_size: int = 0) -> None:
        super(MLPActor, self).__init__()
        self.Dense = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Dense(x)
        x = nn.Softmax(dim=-1)(x)
        return x

    def _init_weights(self,module) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.1)

class MLPCritic(nn.Module):

    def __init__(self, state_size: int = 0,  hidden_size: int = 0) -> None:
        super(MLPCritic, self).__init__()
        self.Dense = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Dense(x)


        return x

    def _init_weights(self,module) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
class extract_LSTM_features(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tensor, _ = x
        shape = tensor.size()
        if len(shape) == 2:
            reshaped_tensor = torch.flatten(tensor)
        elif len(shape) == 3:
            reshaped_tensor = torch.flatten(tensor, start_dim=1)
        else:
            raise ValueError("Unsupported tensor shape")
        return reshaped_tensor
class LSTMActor(nn.Module):
    def __init__(self, state_size: int = 0, action_size: int = 0, hidden_size: int = 0) -> None:
        super(LSTMActor, self).__init__()
        self.LSTM = nn.Sequential(
            nn.LSTM(input_size=state_size,hidden_size= 64, num_layers=1, batch_first=True),
            extract_LSTM_features(),
            nn.Tanh(),
            nn.Linear(64*50, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.LSTM(x)

        x = nn.Softmax(dim=-1)(x)
        return x

    def _init_weights(self,module) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.1)



class LSTMCritic(nn.Module):

    def __init__(self, state_size: int = 0, hidden_size=0) -> None:
        super(LSTMCritic, self).__init__()
        self.LSTM = nn.Sequential(
            nn.LSTM(input_size=state_size, hidden_size=64, num_layers=1, batch_first=True),
            extract_LSTM_features(),
            nn.Tanh(),
            nn.Linear(64*50, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.LSTM(x)
        return x

    def _init_weights(self, module) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.1)



@dataclasses.dataclass
class DiscretePPO(AbstractPPO):
    """
    Proximal Policy Optimization

    :arguments:


        minibatch_size (int): Number of samples per minibatch
        epochs (int): Number of epochs to train
        timestep_per_episode (int): Maximum number of timesteps per episode
        timestep_per_update (int): Number of timesteps per update
        hidden_size (int): Number of hidden units in the network
        lr (float): Learning rate
        eps_clip (float): Clipping parameter for DiscretePPO
        entropy_coef (float): Entropy coefficient
        value_loss_coef (float): Value loss coefficient
        gae_lambda (float): Lambda coefficient for Generalized Advantage Estimation
        gamma (float): Discount factor
        decay_rate (float): Decay rate for the Adam optimizer. Pourcentage of the learning rate that will be decayed each update
        env_worker (int): Number of parallel environments
        env_name (str): Name of the environment

    :returns: DiscretePPO agent
    """


    def __post_init__(self) -> None:
        print("Initializing DiscretePPO")
        window_size = 50
        self.env = gym.make(self.env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        if self.recurrent:
            self.actor = LSTMActor(state_size=self.state_size, action_size=self.action_size, hidden_size=self.hidden_size).to(self.device)
            self.critic = LSTMCritic(state_size=self.state_size, hidden_size= self.hidden_size).to(self.device)

        else:
            self.actor = MLPActor(state_size=self.state_size, action_size=self.action_size, hidden_size=self.hidden_size).to(self.device)
            self.critic = MLPCritic(state_size=self.state_size, hidden_size=self.hidden_size).to(self.device)

        self.buffer = RolloutBuffer(minibatch_size=self.minibatch_size, gamma=self.gamma, gae_lambda=self.gae_lambda)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, state: np.ndarray) -> List:
        with torch.no_grad():
            state = torch.tensor(state,device=self.device,dtype=torch.float32)
            action_probs = self.actor(state)
            # Compute the mask
            """
            mask = get_action_mask(state[-1][-1].item(), state[-1][1].item(), state[-1][-2].item(),self.device)
            # Mask the action probabilities
            action_probs = action_probs * mask
            """

            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def update(self):
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
                action_probs = self.actor(states)
                # Compute the mask
                """
                 masks_list = [get_action_mask(state[-1][-1].item(), state[-1][1].item(), state[-1][-2].item(),self.device) for state in states]
                 masks = torch.stack(masks_list)
                 action_probs = action_probs * masks
                """

                dist = torch.distributions.Categorical(action_probs)
                entropy = dist.entropy()
                discounted_rewards = torch.stack(discounted_rewards)
                actions = torch.stack(actions)
                actions = actions.squeeze()

                new_log_probs = dist.log_prob(actions)
                advantages = torch.stack(advantages)
                advantages = torch.squeeze(advantages)
                old_log_probs = torch.stack(old_log_probs)

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
