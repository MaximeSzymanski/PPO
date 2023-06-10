

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
import shap

@dataclasses.dataclass
class AbstractPPO(metaclass=ABCMeta):
    """Abstract class for PPO.

    Attributes
    ----------
    critic_loss : nn.MSELoss
        Mean squared error loss.
    critic_optimizer : torch.optim.Adam
        Adam optimizer for critic.
    actor_optimizer : torch.optim.Adam
        Adam optimizer for actor.
    buffer : RolloutBuffer
        Rollout buffer.
    recurrent : bool
        Whether the model is recurrent.
    device : torch.device
        Device to use.
    critic : nn.Module
        Critic network.
    actor : nn.Module
        Actor network.
    action_size : int
        Action size.
    state_size : int
        State size.
    env : gym.Env
        Environment.
    env_name : str
        Environment name.
    env_worker : int
        Number of environment workers.
    decay_rate : float
        Decay rate of the learning rate in percentage per update.
    current_episode : int
        Current episode.
    total_updates_counter : int
        Total number of updates.
    total_timesteps_counter : int
        Total number of timesteps.
    gamma : float
        Discount factor.
    gae_lambda : float
        Lambda coefficient for Generalized Advantage Estimation.
    value_loss_coef : float
        Value loss coefficient.
    entropy_coef : float
        Entropy coefficient.
    eps_clip : float
        Clipping parameter for PPO.
    lr : float
        Learning rate.
    actor_hidden_size : dict
        Dictionary containing the hidden layer sizes and activation functions.  {"layer": [l1_size...,l2_size...,ln_size], "activ": "a1_size...,a2_size...,an_size..."}, hidden layer size and activation function.
    critic_hidden_size : dict
        Dictionary containing the hidden layer sizes and activation functions.  {"layer": [l1_size...,l2_size...,ln_size], "activ": "a1_size...,a2_size...,an_size..."}, hidden layer size and activation function.

    timestep_per_update : int
        Number of timesteps before updating.
    timestep_per_episode : int
        Max number of timesteps per episode.
    epochs : int
        Number of epochs per update.
    minibatch_size : int
        Minibatch size.
    continuous_action_space : bool
        Whether the action space is continuous.
    render : bool
        Whether to render the environment.
    writer : SummaryWriter
        Tensorboard writer.
    shapley_value : bool
        Whether to compute shapley value.
    class_name : list[str]
        Name of the class.
    features_name : list[str]
        List of the features names.



    Methods
    -------
    choose_action(state)
        Choose an action given a state.
    __post_init__()
        Initialize tensorboard writer.
    decay_learning_rate()
        Decay the learning rate.
    rollout_episode()
        Rollout some episodes
    update()
        Update the model.
    save_model()
        Save the model.
    load_model()
        Load the model.
    evaluate()
        Evaluate the model.

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
    shapley_value: bool = dataclasses.field(init=True, default=False)
    class_name: list[str] = dataclasses.field(init=True, default_factory=[])
    features_name: list[str] = dataclasses.field(init=True, default_factory=[])





    @abstractmethod
    def choose_action(self, state):
        """Choose an action given a state.

        Parameters
        ----------
        state : torch.Tensor
            State."""
        pass

    def __post_init__(self):
        """Initialize tensorboard writer."""

        tensorboard_path = f'tensorboard_logs/{self.env_name}'
        self.writer = SummaryWriter(tensorboard_path)

    def decay_learning_rate(self) -> None:
        """Decay the learning rate."""
        # decay critic learning rate
        self.writer.add_scalar(
            "Learning Rate", self.critic_optimizer.param_groups[0]['lr'], self.total_updates_counter)
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] *= self.decay_rate
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] *= self.decay_rate

    def rollout_episodes(self) -> float:
        """Rollout some episodes.

        Returns
        -------
        best_reward : float
            Best reward of the episodes.
        average_reward : float
            Average reward of the episodes.
            """
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

        return  best_reward, average_reward/number_episode
    def get_mask(self,state: torch.Tensor,action_space_size : int) -> torch.Tensor:
        """Get the mask of the action space.

        Parameters
        ----------
        state : torch.Tensor
            State.
        action_space_size : int
            Size of the action space.

        Returns
        -------
        mask : torch.Tensor
            Mask of the action space.
        """
        # mask is a list of 1 of size state
        mask = torch.ones(action_space_size, device=self.device)
        return mask

    @abstractmethod
    def update(self):
        """Update the actor and the critic."""
        pass

    def save_model(self, path: str = 'src/saved_weights') -> None:
        """Save the model.

        Parameters
        ----------
        path : str, optional
            Path where to save the model, by default 'src/saved_weights'.
        """

        # check if the path exists
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), f"{path}actor.pth")
        torch.save(self.critic.state_dict(), f"{path}critic.pth")
        # create a txt file with the hyperparameters and the architecture
        with open(f"{path}hyperparameters.txt", "w") as f:
            f.write(f"Hyperparameters and architectures: \n{self.__dict__}")


    def load_model(self, path: str = 'src/saved_weights') -> None:
        """Load the model.

        Parameters
        ----------
        path : str, optional
            Path where to load the model, by default 'src/saved_weights/'.
        """
        print("Loading model")
        if not os.path.exists(path):
            os.makedirs(path)

        self.actor.load_state_dict(torch.load(
            f"{path}/{self.env_name}/actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(
            f"{path}/{self.env_name}/critic.pth", map_location=self.device))

    def evaluate(self):
        """Evaluate the model."""

        output_file = 'results/gif/render.gif'
        data_set = []
        iterations = 0
        tot_reward = 0
        done = False
        state, info = self.env.reset()
        data_set.append(state)

        while not done and iterations <= self.timestep_per_episode :
                action, _ = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                iterations += 1

                data_set.append(next_state)
                tot_reward += reward
                state = next_state


        print("Reward: ", tot_reward)
        if self.shapley_value:

            data_set = torch.tensor(data_set, device=self.device, dtype=torch.float32)
            explainer  = shap.DeepExplainer(self.actor, data_set)


            # compute shapley values
            shapeley_values = explainer.shap_values(data_set)
            feature_names = self.features_name
            class_names = self.class_name
            # plot shapley values

            shap.summary_plot(shapeley_values, data_set, feature_names=feature_names,class_names= class_names,show=True)

        # plot shapley values for each step

        # self.env.close()
