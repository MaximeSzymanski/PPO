import copy
import time
from typing import List
import numpy as np
import torch
import torch.nn as nn
import dataclasses
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from PIL import Image

from STOCKSENV import StockEnv
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





@dataclasses.dataclass
class RolloutBuffer():
    rewards: List[torch.Tensor] = dataclasses.field(
        init=False,  default= None)
    values: List[torch.Tensor] = dataclasses.field(
        init=False,  default= None)
    log_probs: List[torch.Tensor] = dataclasses.field(
        init=False,  default= None)
    actions: List[torch.Tensor] = dataclasses.field(
        init=False,  default= None)
    dones: List[torch.Tensor] = dataclasses.field(
        init=False,  default= None)
    states: List[torch.Tensor] = dataclasses.field(
        init=False,  default= None)
    advantages: List[torch.Tensor] = dataclasses.field(
        init=False, default= None)
    masks: List[torch.Tensor] = dataclasses.field(
        init=False,  default= None)
    returns: List[torch.Tensor] = dataclasses.field(
        init=False,  default= None)
    minibatch_size: int = dataclasses.field(init=True, default=64)
    gamma: float = dataclasses.field(init=True, default=0.99)
    gae_lambda: float = dataclasses.field(init=True, default=0.95)

    def __post_init__(self):
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.actions = []
        self.dones = []
        self.states = []
        self.advantages = []
        self.masks = []
        self.returns = []



    def add_step_to_buffer(self, reward: torch.Tensor, value: torch.Tensor, log_prob: torch.Tensor, action: torch.Tensor, done: torch.Tensor, state: list[torch.Tensor], mask: torch.Tensor) -> None:

        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.actions.append(action)
        self.dones.append(done)
        self.states.append(state)
        self.masks.append(mask)

    def compute_advantages(self) -> None:
        gae = 0
        returns = []
        self.values = torch.stack(self.values).detach()
        for i in reversed(range(len(self.rewards) - 1)):
            delta = self.rewards[i] + self.gamma * \
                self.values[i + 1] * (1 - self.dones[i]) - self.values[i]
            gae = delta + self.gamma * \
                self.gae_lambda * (1 - self.dones[i]) * gae
            returns.insert(0, gae + self.values[i])

        returns = torch.stack(returns).detach()
        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # reverse the advantages

        self.advantages = (returns) - (self.values[:-1])

        # normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()
                           ) / (self.advantages.std() + 1e-8)
        self.returns = (returns)

    def clean_buffer(self) -> None:
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.actions = []
        self.dones = []
        self.states = []
        self.advantages = []
        self.masks = []
        self.returns = []

    def get_minibatch(self, batch_indices):
        states = [self.states[i] for i in batch_indices]
        actions = [self.actions[i] for i in batch_indices]
        log_probs = [self.log_probs[i] for i in batch_indices]
        advantages = [self.advantages[i] for i in batch_indices]
        returns = [self.returns[i] for i in batch_indices]

        # Compute discounted rewards

        return states, actions, log_probs, advantages, returns


class MLPActor(nn.Module):

    def __init__(self, state_size: int = 0, action_size: int = 0, hidden_size: int = 0) -> None:
        super(MLPActor, self).__init__()
        self.Dense = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, action_size),
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

    def __init__(self, state_size: int = 0, action_size: int = 0, hidden_size: int = 0) -> None:
        super(MLPCritic, self).__init__()
        self.Dense = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
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
            # For shape [50, 64]
            reshaped_tensor = torch.flatten(tensor)
        elif len(shape) == 3:
            # For shape [63, 50, 64]
            reshaped_tensor = torch.flatten(tensor, start_dim=1)
        else:
            raise ValueError("Unsupported tensor shape")
        return reshaped_tensor
class LSTMActor(nn.Module):
    def __init__(self, state_size: int = 0, action_size: int = 0, hidden_size: int = 0) -> None:
        super(LSTMActor, self).__init__()
        self.LSTM = nn.Sequential(
            nn.LSTM(input_size=4,hidden_size= 64, num_layers=1, batch_first=True),
            extract_LSTM_features(),
            nn.Tanh(),
            nn.Linear(64*50, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_size),
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
            nn.LSTM(input_size=4, hidden_size=64, num_layers=1, batch_first=True),
            extract_LSTM_features(),
            nn.Tanh(),
            nn.Linear(64*50, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
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
class DiscretePPO():
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

    # Hyperparameters
    minibatch_size: int = dataclasses.field(init=True, default=64)
    epochs: int = dataclasses.field(init=True, default=10)
    timestep_per_episode: int = dataclasses.field(init=True, default=512)
    timestep_per_update: int = dataclasses.field(init=True, default=2048)
    hidden_size: int = dataclasses.field(init=False, default=256)
    lr: float = dataclasses.field(init=True, default=3e-4)
    eps_clip: float = dataclasses.field(init=True, default=0.2)
    entropy_coef: float = dataclasses.field(init=True, default=0.01)
    value_loss_coef: float = dataclasses.field(init=True, default=0.5)
    gae_lambda: float = dataclasses.field(init=True, default=0.95)
    gamma: float = dataclasses.field(init=True, default=0.99)
    total_timesteps_counter: int = dataclasses.field(init=False, default=0)
    total_updates_counter: int = dataclasses.field(init=False, default=0)
    current_episode: int = dataclasses.field(init=False, default=0)
    decay_rate: float = dataclasses.field(init=True, default=0.99)
    env_worker: int = dataclasses.field(init=True, default=4)
    env_name: str = dataclasses.field(init=True, default="CartPole-v1")

    # Environment
    env: gym.Env = dataclasses.field(init=False)
    state_size: int = dataclasses.field(init=False, default=0)

    action_size: int = dataclasses.field(init=False, default=0)

    # MLPActor and MLPCritic Networks
    #actor: MLPActor = MLPActor(state_size, action_size, hidden_size)
    actor: nn.Module = dataclasses.field(default=None, init=False)
    critic: nn.Module = dataclasses.field(default= None, init=False)
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Buffer
    buffer: RolloutBuffer = dataclasses.field(default_factory=RolloutBuffer)

    # Optimizers
    actor_optimizer: torch.optim.Adam = dataclasses.field(init=False)
    critic_optimizer: torch.optim.Adam = dataclasses.field(init=False)

    # Losses
    critic_loss: nn.MSELoss = nn.MSELoss()

    # Tensorboard
    writer: SummaryWriter = SummaryWriter()


    def __post_init__(self) -> None:
        print("Initializing DiscretePPO")
        window_size = 50
        self.env = StockEnv(window_size=window_size)
        self.state_size = 6
        self.action_size = 3


        self.actor = LSTMActor(state_size=self.state_size, action_size=self.action_size, hidden_size=self.hidden_size).to(self.device)
        self.critic = LSTMCritic(state_size=self.state_size, hidden_size= self.hidden_size).to(self.device)
        self.buffer = RolloutBuffer(minibatch_size=self.minibatch_size, gamma=self.gamma, gae_lambda=self.gae_lambda)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, state: np.ndarray) -> List:
        with torch.no_grad():
            state = torch.tensor(state,device=self.device,dtype=torch.float32)
            action_probs = self.actor(state)
            # Compute the mask

            mask = get_action_mask(state[-1][-1].item(), state[-1][1].item(), state[-1][-2].item(),self.device)
            # Mask the action probabilities
            action_probs = action_probs * mask
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob


    def decay_learning_rate(self) -> None:
        # decay critic learning rate
        self.writer.add_scalar(
            "Learning Rate", self.critic_optimizer.param_groups[0]['lr'], self.total_updates_counter)
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] *= self.decay_rate
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] *= self.decay_rate

    def rollout_episodes(self) -> float:
        number_of_step = 0
        reward_sum = 0
        number_episode = 0
        average_reward = 0
        best_reward = -np.inf
        while number_of_step < self.timestep_per_update:

            state, info = self.env.reset()
            self.current_episode += 1
            ep_reward = 0
            done = False
            for _ in range(self.timestep_per_episode):

                action, log_prob = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.total_timesteps_counter += 1
                reward_sum += reward
                ep_reward += reward
                self.writer.add_scalar(
                    "Reward total timestep", reward, self.total_timesteps_counter)
                value = self.critic(torch.tensor(state,device=self.device,dtype=torch.float32))
                reward = torch.tensor([reward],device=self.device,dtype=torch.float32)
                mask = torch.tensor([not done],device=self.device,dtype=torch.float32)
                done = torch.tensor([done],device=self.device,dtype=torch.float32)
                state = torch.tensor(state,device=self.device,dtype=torch.float32)
                action = torch.tensor([action],device=self.device,dtype=torch.float32)
                self.buffer.add_step_to_buffer(
                    reward, value, log_prob, action, done, state, mask)
                state = next_state
                number_of_step += 1
                if done or number_of_step == self.timestep_per_update:
                    number_episode += 1
                    average_reward += ep_reward
                    self.writer.add_scalar(
                        "Reward", ep_reward, self.current_episode)
                    if ep_reward > best_reward:
                        best_reward = ep_reward
                        self.writer.add_scalar(
                            "Best reward", best_reward, self.current_episode)

                    break

        self.buffer.compute_advantages()
        return best_reward, average_reward/number_episode

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
                masks_list = [get_action_mask(state[-1][-1].item(), state[-1][1].item(), state[-1][-2].item(),self.device) for state in states]
                masks = torch.stack(masks_list)
                action_probs = action_probs * masks
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

    def save_model(self, path: str = 'models/') -> None:
        # check if the path exists
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), f"{path}actor.pth")
        torch.save(self.critic.state_dict(), f"{path}critic.pth")

    def load_model(self, path: str = 'models/') -> None:
        print("Loading model")

        self.actor.load_state_dict(torch.load(f"{path}modelactor.pth",map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{path}modelcritic.pth",map_location=self.device))
    def evaluate(self):
        state, info= self.env.reset()
        output_file = 'results/gif/render.gif'
        frames = []
        done = False
        tot_reward = 0
        portfolio_value = []
        while not done:
            action, _ = self.choose_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            portfolio_value.append(self.env.portfolio_after)
            tot_reward += reward
            # next sate is [[value]], we need to convert it to [value]
            state = next_state
            #frame = self.env.render()
            #frame = Image.fromarray(frame)
            #frames.append(frame)
        # create a gif using PIL
        """frames[0].save(output_file, format='GIF',
                          append_images=frames[1:],
                            save_all=True,
                            duration=300, loop=0)"""
        print("Reward: ", tot_reward)
        # plot portfolio value over time
        plt.plot(portfolio_value)
        plt.show()

        #self.env.close()
