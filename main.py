# This is a sample Python script.
import torch

from Discrete_PPO import DiscretePPO as PPO
import gymnasium as gym
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

env = gym.make('LunarLander-v2')
obs_size = env.observation_space.shape[0]
n_actions = 1
hidden_size = 512
# hyperparams
lr = 0.00025
gamma = 0.99
K_epochs = 10
eps_clip = 0.2
entropy_coef = 0.1
value_loss_coef = 0.5
max_episodes = 1259
max_timesteps = 1259*4
gae_lambda = 0.95
env_name = 'LunarLander-v2'



PPO = PPO(
    lr=lr,
    gamma=gamma,
    eps_clip=eps_clip,
    epochs=K_epochs,
    entropy_coef=entropy_coef,
    value_loss_coef=value_loss_coef,
    timestep_per_episode =max_episodes,
    timestep_per_update=max_timesteps,
    env_name=env_name,
    minibatch_size=64,
    gae_lambda=gae_lambda,
)
# get logprob for action 0
PPO.load_model()

PPO.evaluate()




"""for i in range(1000):
    print('Iteration: ', i)
    best, avg = PPO.rollout_episodes()
    print("Current episode", PPO.current_episode)
    print('Best reward: ', best)
    print('Average reward: ', avg)
    PPO.update()
    if i % 10 == 0:
        PPO.save_model('model')
        print('Model saved')"""

