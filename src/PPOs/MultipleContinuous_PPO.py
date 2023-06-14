import dataclasses
import numpy as np
import torch
from tqdm import tqdm
from src.PPOs.AbstractPPO import AbstractPPO
from src.model.Continuous.MLP.MLPActor import MLPActor
from src.model.Continuous.MLP.MLPCritic import MLPCritic
from src.model.Continuous.LSTM.LSTMActor import LSTMActor
from src.model.Continuous.LSTM.LSTMCritic import LSTMCritic


def get_model_flattened_params(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])


@dataclasses.dataclass
class ContinuousPPO(AbstractPPO):
    """Continuous multi actions Proximal Policy Optimization (PPO) agent.
    """

    # Path to save the model
    def __post_init__(self) -> None:
        """Initialize the PPO agent.
        """
        self.continuous_action_space = True
        super().__post_init__()
        print("Initializing ContinousPPO")
        print('env_name: ', self.env_name)
        self.action_size = self.env.action_space.shape[0]

        self.episode_counter = 0
        if self.recurrent:

            self.actor = LSTMActor(state_size=self.state_size,
                                   action_size=self.action_size, hidden_size=self.actor_hidden_size).to(self.device)
            self.critic = LSTMCritic(
                state_size=self.state_size, hidden_size=self.critic_hidden_size).to(self.device)
        else:
            self.actor = MLPActor(state_size=self.state_size,
                                  action_size=self.action_size, hidden_size=self.actor_hidden_size).to(self.device)

            self.critic = MLPCritic(
                state_size=self.state_size, hidden_size=self.critic_hidden_size).to(self.device)

        self.initialize_optimizer()

    def choose_action(self, state: np.ndarray) -> (np.ndarray, torch.Tensor):
        """Choose an action from the action space.

        Arguments
        ---------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        action : np.ndarray
            The action chosen by the agent.
        log_prob : torch.Tensor
            The log probability of the action.
        """

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

    def rollout_episodes(self) -> (float, float):
        """Rollout the policy on the environment for a number of episodes.

        Returns
        -------
        best_reward : float
            The best reward for whole episode obtained during the rollout.
        average_reward : float
            The average reward for whole episode obtained during the rollout.
        """
        number_of_step = 0
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

                next_state = next_state.reshape(-1)
                self.total_timesteps_counter += 1
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
                if done or number_of_step == self.timestep_per_update:

                    number_episode += 1
                    self.episode_counter += 1
                    average_reward += ep_reward
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
        """Update the policy using the collected rollouts.
        """
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
