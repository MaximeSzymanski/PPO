import dataclasses
from typing import List

import numpy as np
import torch


@dataclasses.dataclass
class RolloutBuffer():
    """Rollout Buffer for PPO. It has several lists to store the data from the environment

    Attributes
    ----------
    rewards : List[torch.Tensor]
        rewards from the environment
    values : List[torch.Tensor]
        values from the critic
    log_probs : List[torch.Tensor]
        log probabilities from the actor
    actions : List[torch.Tensor]
        actions from the actor
    dones : List[torch.Tensor]
        dones from the environment
    states : List[torch.Tensor]
        states from the environment
    advantages : List[torch.Tensor]
        advantages from the environment
    masks : List[torch.Tensor]
        masks to indicate the end of the episode. Used to calculate the returns
    returns : List[torch.Tensor]
        returns (i.e. discounted rewards) from the environment
    minibatch_size : int
        size of the minibatch
    gamma : float
        discount factor
    gae_lambda : float
        lambda factor for Generalized Advantage Estimation
    """

    rewards: List[torch.Tensor] = dataclasses.field(
        init=False,  default=List[torch.Tensor])
    values: List[torch.Tensor] = dataclasses.field(
        init=False,  default=List[torch.Tensor])
    log_probs: List[torch.Tensor] = dataclasses.field(
        init=False,  default=List[torch.Tensor])
    actions: List[torch.Tensor] = dataclasses.field(
        init=False,  default=List[torch.Tensor])
    dones: List[torch.Tensor] = dataclasses.field(
        init=False,  default=List[torch.Tensor])
    states: List[torch.Tensor] = dataclasses.field(
        init=False,  default=List[torch.Tensor])
    advantages: List[torch.Tensor] = dataclasses.field(
        init=False, default=List[torch.Tensor])
    masks: List[torch.Tensor] = dataclasses.field(
        init=False,  default=None)
    returns: List[torch.Tensor] = dataclasses.field(
        init=False,  default=None)
    minibatch_size: int = dataclasses.field(init=True, default=64)
    gamma: float = dataclasses.field(init=True, default=0.99)
    gae_lambda: float = dataclasses.field(init=True, default=0.95)

    def __post_init__(self):
        """Reset the buffer"""

        self.clean_buffer()

    def add_step_to_buffer(self, reward: torch.Tensor, value: torch.Tensor, log_prob: torch.Tensor, action: torch.Tensor, done: torch.Tensor, state: list[torch.Tensor], mask: torch.Tensor) -> None:
        """Add a step to the buffer.

        Parameters
        ----------
        reward : torch.Tensor
            reward from the environment
        value : torch.Tensor
            value from the critic
        log_prob : torch.Tensor
            log probability from the actor
        action : torch.Tensor
            action from the actor
        done : torch.Tensor
            done from the environment
        state : list[torch.Tensor]
            state from the environment
        mask : torch.Tensor
            mask to indicate the end of the episode. Used to calculate the returns

        """


        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.actions.append(action)
        self.dones.append(done)
        self.states.append(state)
        self.masks.append(mask)

    def compute_advantages(self) -> None:
        """
        Compute the advantages using Generalized Advantage Estimation
        Compute the returns (i.e. discounted rewards) using the values from the critic
        """
        with torch.no_grad():
            gae = 0
            returns = []
            self.values = torch.stack(self.values).detach()
            for i in reversed(range(len(self.rewards)-1)):
                delta = self.rewards[i] + self.gamma * \
                    self.values[i + 1] * (self.masks[i]) - self.values[i]
                gae = delta + self.gamma * \
                    self.gae_lambda * (self.masks[i]) * gae
                returns.insert(0, gae + self.values[i])

            adv = np.array(returns) - self.values[:-1].cpu().numpy()
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)
            self.advantages = torch.tensor(adv).float()
            self.returns = returns


    def clean_buffer(self) -> None:
        """Reset the buffer"""

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
        """Get a minibatch from the buffer

        Parameters
        ----------
        batch_indices : list[int]
            indices of the minibatch

        Returns
        -------
        states : list[torch.Tensor]
            states from the environment
        actions : list[torch.Tensor]
            actions from the actor
        log_probs : list[torch.Tensor]
            log probabilities from the actor
        advantages : list[torch.Tensor]
            advantages from the environment
        returns : list[torch.Tensor]
            returns (i.e. discounted rewards) from the environment
        """

        states = [self.states[i] for i in batch_indices]
        actions = [self.actions[i] for i in batch_indices]
        log_probs = [self.log_probs[i] for i in batch_indices]
        advantages = [self.advantages[i] for i in batch_indices]
        returns = [self.returns[i] for i in batch_indices]

        # Compute discounted rewards

        return states, actions, log_probs, advantages, returns