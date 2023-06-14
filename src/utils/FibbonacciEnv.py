import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FibonacciEnvironment(gym.Env):
    def __init__(self, sequence_length, is_discrete=True):
        super(FibonacciEnvironment, self).__init__()

        self.sequence_length = sequence_length
        self.is_discrete = is_discrete

        if self.is_discrete:
            # Discrete action space with 10 possible actions
            self.action_space = spaces.Discrete(10)
        else:
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32)  # Continuous action space

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(sequence_length,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = 0
        self.sequence = []

        # Pad the sequence with zeros
        self.sequence += [0] * (self.sequence_length)
        self.sequence[-2] = 0
        self.sequence[-1] = 1
        # put each element of sequnce in a list
        self.sequence = [self.sequence[i] for i in range(len(self.sequence))]
        return np.array(self.sequence, dtype=np.float32), None

    def step(self, action):
        self.current_step += 1

        # Append the next number in the Fibonacci sequence based on the action type

        next_number = self.sequence[-1] + self.sequence[-2]

        self.sequence.append(next_number)

        # Remove the first number in the sequence
        self.sequence = self.sequence[1:]

        # Pad the sequence with zeros
        self.sequence += [0] * (self.sequence_length - len(self.sequence))

        # Calculate the reward
        if self.current_step == self.sequence_length:
            done = True
        else:
            done = False
        reward = self.calculate_reward(action, next_number)

        return np.array(self.sequence, dtype=np.float32), reward, done, {}, None

    def calculate_reward(self, action, next_number):
        # The reward is the negative absolute difference between the predicted number and the actual next number
        return -abs(action - next_number)
