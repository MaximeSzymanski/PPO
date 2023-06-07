

import dataclasses
from abc import ABCMeta, abstractmethod
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import gymnasium as gym
from src.utils.RolloutBuffer import RolloutBuffer
import numpy as np
import matplotlib.pyplot as plt
import os


@dataclasses.dataclass
class AbstractPPO(metaclass=ABCMeta):
    """
    Abstract class for PPO
    """
    critic_loss: nn.MSELoss = nn.MSELoss()
    critic_optimizer: torch.optim.Adam = dataclasses.field(init=False)
    actor_optimizer: torch.optim.Adam = dataclasses.field(init=False)
    buffer: RolloutBuffer = dataclasses.field(default_factory=RolloutBuffer)
    recurrent: bool = dataclasses.field(init=True, default=False)
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    critic: nn.Module = dataclasses.field(default=None, init=False)
    actor: nn.Module = dataclasses.field(default=None, init=False)
    action_size: int = dataclasses.field(init=False, default=0)
    state_size: int = dataclasses.field(init=False, default=0)
    env: gym.Env = dataclasses.field(init=False)
    env_name: str = dataclasses.field(init=True, default="CartPole-v1")
    env_worker: int = dataclasses.field(init=True, default=4)
    decay_rate: float = dataclasses.field(init=True, default=0.99)
    current_episode: int = dataclasses.field(init=False, default=0)
    total_updates_counter: int = dataclasses.field(init=False, default=0)
    total_timesteps_counter: int = dataclasses.field(init=False, default=0)
    gamma: float = dataclasses.field(init=True, default=0.99)
    gae_lambda: float = dataclasses.field(init=True, default=0.95)
    value_loss_coef: float = dataclasses.field(init=True, default=0.5)
    entropy_coef: float = dataclasses.field(init=True, default=0.01)
    eps_clip: float = dataclasses.field(init=True, default=0.2)
    lr: float = dataclasses.field(init=True, default=3e-4)
    actor_hidden_size: dict = dataclasses.field(init=True,default_factory=lambda: {"layer": [64], "activ": "tanh"})
    critic_hidden_size: dict = dataclasses.field(init=True,default_factory=lambda: {"layer": [64], "activ": "tanh"})
    timestep_per_update: int = dataclasses.field(init=True, default=2048)
    timestep_per_episode: int = dataclasses.field(init=True, default=512)
    epochs: int = dataclasses.field(init=True, default=10)
    minibatch_size: int = dataclasses.field(init=True, default=64)
    continuous_action_space: bool = dataclasses.field(init=True, default=True)
    render: bool = dataclasses.field(init=True, default=False)
    writer: SummaryWriter = dataclasses.field(init=False,default=None)




    @abstractmethod
    def choose_action(self, state):
        pass

    def __post_init__(self):
        tensorboard_path = f'tensorboard_logs/{self.env_name}'
        self.writer = SummaryWriter(tensorboard_path)

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
                state = torch.tensor(
                    state, device=self.device, dtype=torch.float32)
                """if self.recurrent:
                    state = state.unsqueeze(1)"""
                value = self.critic(state)
                reward = torch.tensor(
                    [reward], device=self.device, dtype=torch.float32)
                mask = torch.tensor(
                    [not done], device=self.device, dtype=torch.float32)
                done = torch.tensor(
                    [done], device=self.device, dtype=torch.float32)

                action = torch.tensor(
                    [action], device=self.device, dtype=torch.float32)
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

    @abstractmethod
    def update(self):
        pass

    def save_model(self, path: str = 'models/') -> None:
        # check if the path exists
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), f"{path}actor.pth")
        torch.save(self.critic.state_dict(), f"{path}critic.pth")
        # create a txt file with the hyperparameters and the architecture
        with open(f"{path}hyperparameters.txt", "w") as f:
            f.write(f"Hyperparameters and architectures: \n{self.__dict__}")


    def load_model(self, path: str = 'models/') -> None:
        print("Loading model")
        if not os.path.exists(path):
            os.makedirs(path)
        self.actor.load_state_dict(torch.load(
            f"{path}actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(
            f"{path}critic.pth", map_location=self.device))

    def evaluate(self):
        state, info = self.env.reset()
        output_file = 'results/gif/render.gif'
        frames = []
        done = False
        tot_reward = 0
        while not done:
            action, _ = self.choose_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            tot_reward += reward
            # next sate is [[value]], we need to convert it to [value]
            state = next_state
            # frame = self.env.render()
            # frame = Image.fromarray(frame)
            # frames.append(frame)
        # create a gif using PIL
        """frames[0].save(output_file, format='GIF',
                          append_images=frames[1:],
                            save_all=True,
                            duration=300, loop=0)"""
        print("Reward: ", tot_reward)
        # plot portfolio value over time
        plt.show()

        # self.env.close()
