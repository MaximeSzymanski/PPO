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

def get_model_flattened_params(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])


@dataclasses.dataclass
class RolloutBuffer():
    rewards: List[torch.Tensor] = dataclasses.field(
        init=False, default_factory=list)
    values: List[torch.Tensor] = dataclasses.field(
        init=False, default_factory=list)
    log_probs: List[torch.Tensor] = dataclasses.field(
        init=False, default_factory=list)
    actions: List[torch.Tensor] = dataclasses.field(
        init=False, default_factory=list)
    dones: List[torch.Tensor] = dataclasses.field(
        init=False, default_factory=list)
    states: List[torch.Tensor] = dataclasses.field(
        init=False, default_factory=list)
    advantages: List[torch.Tensor] = dataclasses.field(
        init=False, default_factory=list)
    masks: List[torch.Tensor] = dataclasses.field(
        init=False, default_factory=list)
    returns: List[torch.Tensor] = dataclasses.field(
        init=False, default_factory=list)
    minibatch_size: int = dataclasses.field(init=True, default=64)
    gamma: float = dataclasses.field(init=True, default=0.99)
    gae_lambda: float = dataclasses.field(init=True, default=0.95)



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

        returns = torch.stack(returns).squeeze(-1).squeeze(-1)
        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # reverse the advantages
        self.values = self.values.squeeze(-1)
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


class Actor(nn.Module):

    def __init__(self, state_size: int = 0, action_size: int = 1, hidden_size: int = 0) -> None:
        super(Actor, self).__init__()
        self.Dense = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size*2),
            nn.Tanh()
        )
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Dense(x)
        means, log_stds = [], []

        if x.shape[0] == 2:
            means.append(x[0])
            log_stds.append(x[1])
        else:
            for mean, log_std in x:
                means.append(mean)
                log_stds.append(log_std)

        means = torch.stack(means)
        log_stds = torch.stack(log_stds)
        stds = torch.exp(log_stds).clamp(min=-20, max=2)
        return means, stds

    def _init_weights(self,module) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.1)


class Critic(nn.Module):

    def __init__(self, state_size: int = 0, hidden_size=0) -> None:
        super(Critic, self).__init__()
        self.Dense = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.Dense(x)
        return x

    def _init_weights(self, module) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.1)


@dataclasses.dataclass
class ContinuousPPO():
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
    actor: Actor = dataclasses.field(default_factory=Actor, init=False)
    critic: Critic = dataclasses.field(default_factory=Critic, init=False)

    # Buffer
    buffer: RolloutBuffer = dataclasses.field(default_factory=RolloutBuffer)

    # Optimizers
    actor_optimizer: torch.optim.Adam = dataclasses.field(init=False)
    critic_optimizer: torch.optim.Adam = dataclasses.field(init=False)

    # Losses
    critic_loss: nn.MSELoss = nn.MSELoss()

    # Tensorboard
    writer: SummaryWriter = SummaryWriter()

    # Path to save the model
    path: str = dataclasses.field(init=True, default="models/")
    def __post_init__(self) -> None:
        print("Initializing ContinousPPO")
        print('env_name: ', self.env_name)
        self.env = gym.make(self.env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = len(self.env.action_space.sample())
        print("State size: ", self.state_size)
        print("Action size: ", self.action_size)


        self.actor = Actor(state_size=self.state_size, action_size=self.action_size, hidden_size=self.hidden_size)
        self.critic = Critic(state_size=self.state_size,hidden_size= self.hidden_size)
        self.buffer = RolloutBuffer(minibatch_size=self.minibatch_size, gamma=self.gamma, gae_lambda=self.gae_lambda)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, state: np.ndarray) -> List:
        with torch.no_grad():

            state = torch.FloatTensor(state)
            # if the state is 3,1, we remove the first dimension

            mean, std = self.actor(state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        action = action.numpy()
        # add a dimension to the action
        action = np.expand_dims(action, axis=0)


        return action, log_prob


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
        best_reward = 0
        while number_of_step < self.timestep_per_update:

            state = self.env.reset()

            state = state[0]
            self.current_episode += 1
            ep_reward = 0
            done = False
            for _ in range(self.timestep_per_episode):

                action, log_prob = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                # reward is a numpy array, we need to convert it to a float
                # same for action
                action = action[0]
                reward = reward[0]

                next_state = next_state.reshape(-1)
                self.total_timesteps_counter += 1
                reward_sum += reward
                ep_reward += reward
                self.writer.add_scalar(
                    "Reward total timestep", reward, self.total_timesteps_counter)
                state = torch.FloatTensor(state)

                value = self.critic(state)
                reward = torch.FloatTensor([reward])
                mask = torch.FloatTensor([not done])
                done = torch.FloatTensor([done])
                state = torch.FloatTensor(state)
                action = torch.FloatTensor([action])
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
                mean, std = self.actor(states)
                values = values.squeeze()

                dist = torch.distributions.Normal(mean, std)

                entropy = dist.entropy()
                discounted_rewards = torch.stack(discounted_rewards)
                actions = torch.stack(actions)
                actions = actions.squeeze()

                new_log_probs = dist.log_prob(actions)
                advantages = torch.stack(advantages)
                advantages = torch.squeeze(advantages)
                old_log_probs = torch.stack(old_log_probs)
                old_log_probs = torch.squeeze(old_log_probs)
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

    def save_model(self, path = "models/"):
        print("Saving model")
        # create the folder if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), f"{path}actor.pth")
        torch.save(self.critic.state_dict(), f"{path}critic.pth")

    def load_model(self, path = "models/"):
        print("Loading model")
        self.actor.load_state_dict(torch.load(f"{path}actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}critic.pth"))

    def evaluate(self):
        state, info= self.env.reset()

        done = False
        while not done:
            action, _ = self.choose_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            # next sate is [[value]], we need to convert it to [value]
            state  = [item for sublist in next_state for item in sublist]
        self.env.close()