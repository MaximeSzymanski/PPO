from typing import List
import numpy as np
import torch
import dataclasses
from tqdm import tqdm
import torch.nn as nn
from src.model.Discrete.CNN.CNN import CNN_layer
from src.model.Discrete.LSTM.LSTMCritic import LSTMCritic
from src.model.Discrete.LSTM.LSTMActor import LSTMActor
from src.model.Discrete.MLP.MLPActor import MLPActor
from src.model.Discrete.MLP.MLPCritic import MLPCritic
from src.PPOs.AbstractPPO import AbstractPPO
from torch.utils.tensorboard import SummaryWriter
from src.utils.RolloutBuffer import RolloutBuffer

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
            self.actor = LSTMActor(state_size=512, action_size=self.action_size,
                                   hidden_size=self.actor_hidden_size).to(self.device)
            self.critic = LSTMCritic(
                state_size=self.state_size, hidden_size=self.critic_hidden_size).to(self.device)

        else:
            print(self.actor_hidden_size)
            print(self.state_size)
            print(self.action_size)
            self.actor = MLPActor(state_size=512, action_size=self.action_size,
                                  hidden_size=self.actor_hidden_size).to(self.device)
            self.critic = MLPCritic(
                state_size=512, hidden_size=self.critic_hidden_size).to(self.device)
            self.cnn = CNN_layer(input_channel=4, output_dim=512).to(self.device)
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

            state = np.array(state)

            state = torch.from_numpy(np.array(state,copy=False)).float().to(self.device)

            if self.recurrent:
                state = state.unsqueeze(0)
            # remove the last dimension

            state = self.cnn(state)
            action_probs = self.actor(state)
            # Compute the mask
            mask = self.get_mask(self.env.action_space.n)

            # Mask the action probabilities
            action_probs = action_probs * mask

            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return np.array(action.cpu()), log_prob

    def update(self, writer: SummaryWriter):
        """Update the policy and value parameters using the PPO algorithm"""
        torch.autograd.set_detect_anomaly(True)
        total_rollout_buffer = RolloutBuffer()
        for buffer in self.buffer_list:
            # add the buffer to the total buffer
            total_rollout_buffer.rewards.extend(buffer.rewards)
            total_rollout_buffer.values.extend(buffer.values)
            total_rollout_buffer.log_probs.extend(buffer.log_probs)
            total_rollout_buffer.actions.extend(buffer.actions)
            total_rollout_buffer.dones.extend(buffer.dones)
            total_rollout_buffer.states.extend(buffer.states)
            total_rollout_buffer.advantages.extend(buffer.advantages)
            total_rollout_buffer.masks.extend(buffer.masks)
            total_rollout_buffer.returns.extend(buffer.returns)

        print(f" Total number of samples in the buffer {len(total_rollout_buffer.rewards)}")

        for _ in tqdm(range(self.epochs)):
            num_samples = len(total_rollout_buffer.rewards) -1
            print(f"Number of samples in the buffer {num_samples}")
            print(f"number of advantages {len(total_rollout_buffer.advantages)}")
            print(f"number of returns {len(total_rollout_buffer.returns)}")
            print(f"Minibatch size {self.minibatch_size}")
            assert num_samples > self.minibatch_size, "The number of samples in the buffer is less than the minibatch size"
            assert (num_samples+1) % self.minibatch_size == 0, "The number of samples in the buffer is not a multiple of the minibatch size"
            # crate a list of indices from 0 to num_samples
            indices = np.arange(num_samples)
            # shuffle the indices
            np.random.shuffle(indices)

            for i in range(0, num_samples, self.minibatch_size):

                batch_indices = indices[i:i + self.minibatch_size]
                iter = 0

                states, actions, old_log_probs, advantages, discounted_rewards = total_rollout_buffer.get_minibatch(
                        batch_indices)

                states = torch.stack(states)
                if self.recurrent:
                    states = states.unsqueeze(1)

                states = states.squeeze()
                # add the batch dimension
                states = self.cnn(states)
                #states = self.cnn(states)
                values = self.critic(states)
                values = values.squeeze().squeeze() if self.recurrent else values.squeeze()
                action_probs = self.actor(states)
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

                iter += 1
                surr1 = -ratio * advantages
                surr2 = -torch.clamp(ratio, 1 - self.eps_clip,
                                        1 + self.eps_clip) * advantages
                actor_loss = torch.max(surr1, surr2).mean()
                crit_loss = self.critic_loss(values,discounted_rewards).mean()
                # prit one random value from the critic loss and the discounted rewards


                entropy_loss = entropy.mean()
                """print(f'Actor Loss: {actor_loss}')
                print(f'Critic Loss: {critic_loss}')
                print(f'Entropy Loss: {entropy_loss}')"""
                loss = actor_loss + self.value_loss_coef * \
                        crit_loss - self.entropy_coef * entropy_loss
                writer.add_scalar(
                        "Value Loss", crit_loss, self.total_updates_counter)
                writer.add_scalar(
                        "MLPActor Loss", actor_loss, self.total_updates_counter)

                writer.add_scalar("Entropy", entropy_loss , self.total_updates_counter)
                self.total_updates_counter += 1
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                self.cnn_optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.cnn.parameters(), 0.5)
                # After the backward call
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                self.cnn_optimizer.step()

                # Update steps here...

        self.decay_learning_rate()
        [buffer.clean_buffer() for buffer in self.buffer_list]
