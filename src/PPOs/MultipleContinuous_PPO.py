import dataclasses
from typing import List

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from src.PPOs.AbstractPPO import AbstractPPO
from src.utils.RolloutBuffer import RolloutBuffer
from src.model.Continous.MLP.MLPActor import MLPActor
from src.model.Continous.MLP.MLPCritic import MLPCritic
from src.model.Continous.LSTM.LSTMActor import LSTMActor
from src.model.Continous.LSTM.LSTMCritic import LSTMCritic

def get_model_flattened_params(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])


@dataclasses.dataclass
class ContinuousPPO(AbstractPPO):
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

    # Path to save the model
    def __post_init__(self) -> None:
        super().__post_init__()
        print("Initializing ContinousPPO")
        print('env_name: ', self.env_name)

        if self.env_name== "LunarLander-v2":
            self.env = gym.make(self.env_name,continuous=True)
        else:
            if self.render:
                self.env = gym.make(self.env_name, render_mode='human')
            else:
                self.env = gym.make(self.env_name)

        self.state_size = self.env.observation_space.shape[0]

        self.action_size = len(self.env.action_space.sample())
        print("State size: ", self.state_size)
        print("Action size: ", self.action_size)
        self.episode_counter = 0
        if self.recurrent:

            self.actor = LSTMActor(state_size=self.state_size,
                              action_size=self.action_size, hidden_size=self.actor_hidden_size)
            self.critic = LSTMCritic(
                state_size=self.state_size, hidden_size=self.critic_hidden_size)
        else:
            self.actor = MLPActor(state_size=self.state_size,
                             action_size=self.action_size, hidden_size=self.actor_hidden_size)

            self.critic = MLPCritic(
                state_size=self.state_size, hidden_size=self.critic_hidden_size)
        self.buffer = RolloutBuffer(
            minibatch_size=self.minibatch_size, gamma=self.gamma, gae_lambda=self.gae_lambda)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr)

    def choose_action(self, state: np.ndarray) -> List:
        with torch.no_grad():
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device)
            # if the state is 3,1, we remove the first dimension
            if self.recurrent:
                state = state.unsqueeze(1)
            mean, std = self.actor(state)
            if self.recurrent:
                mean = mean.squeeze(1)
                std = std.squeeze(1)
            # this is a multivariate normal distribution
            square_std = std ** 2
            """ if len(square_std.shape) != 1:
                square_std= square_std.squeeze()
            if len(mean.shape) != 1:
                mean = mean.squeeze()"""

            covar_matrix = torch.diag(square_std)
            dist = torch.distributions.MultivariateNormal(mean, covar_matrix)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        action = action.numpy()
        # add a dimension to the action

        return action, log_prob

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
            ep_steps = 0
            done = False
            for _ in range(self.timestep_per_episode):

                action, log_prob = self.choose_action(state)

                next_state, reward, done, _, _ = self.env.step(action)
                # reward is a numpy array, we need to convert it to a float
                # same for action

                next_state = next_state.reshape(-1)
                self.total_timesteps_counter += 1
                reward_sum += reward
                ep_reward += reward
                self.writer.add_scalar(
                    "Reward total timestep", reward, self.total_timesteps_counter)
                state = torch.tensor(
                    state, dtype=torch.float32, device=self.device)
                if self.recurrent:
                    state = state.unsqueeze(1)
                value = self.critic(state)
                reward = torch.tensor(
                    [reward], dtype=torch.float32, device=self.device)
                mask = torch.tensor(
                    [not done], dtype=torch.float32, device=self.device)
                done = torch.tensor(
                    [done], dtype=torch.float32, device=self.device)

                action = torch.tensor(
                    action, dtype=torch.float32, device=self.device)
                self.buffer.add_step_to_buffer(
                    reward, value, log_prob, action, done, state, mask)
                state = next_state
                number_of_step += 1
                ep_steps += 1
                if done or number_of_step == self.timestep_per_update:

                    number_episode += 1
                    self.episode_counter += 1
                    average_reward += ep_reward
                    ep_reward = ep_reward
                    self.writer.add_scalar(
                        "Reward", ep_reward, self.current_episode)

                    if ep_reward > best_reward:
                        best_reward = ep_reward
                    break

        self.buffer.compute_advantages()
        self.writer.add_scalar(
            'Average reward', average_reward / number_episode, self.episode_counter)
        return best_reward, average_reward / number_episode

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
                if self.recurrent:
                    values = values.squeeze().squeeze()
                mean, std = self.actor(states)
                values = values.squeeze()

                square_std = std ** 2
                covar_matrix = torch.diag_embed(square_std)
                dist = torch.distributions.MultivariateNormal(
                    mean, covar_matrix)

                entropy = dist.entropy()
                discounted_rewards = torch.stack(discounted_rewards)
                discounted_rewards = torch.squeeze(discounted_rewards)
                actions = torch.stack(actions)

                #actions = actions.squeeze(1)
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

    def evaluate(self):
        state, info = self.env.reset()
        output_file = 'results/gif/render.gif'
        frames = []
        total_reward = 0
        done = False
        while not done:
            action, _ = self.choose_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            # next sate is [[value]], we need to convert it to [value]
            state = next_state
            frame = self.env.render()
            frame = Image.fromarray(frame)
            frames.append(frame)
            total_reward += reward
        # create a gif using PIL
        frames[0].save(output_file, format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=300, loop=0)
        print(total_reward)
        self.env.close()
