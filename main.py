# This is a sample Python script.
import torch
from STOCKSENV import StockEnv
from MultipleContinuous_PPO import ContinuousPPO as PPO
import gymnasium as gym
import os


if __name__ == '__main__':

    env = gym.make('LunarLander-v2')
    obs_size = env.observation_space.shape[0]
    n_actions = 1
    actor_hidden_size = dict(layer=[64,64,78], activ='tanh,relu,relu')
    critic_hidden_size = dict(layer=[64], activ='tanh')

    # hyperparams
    lr = 0.00025
    gamma = 0.99
    K_epochs = 10
    eps_clip = 0.2
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_episodes = 1024
    max_timesteps = 1024*4
    gae_lambda = 0.98
    recurrent = False
    env_name = 'BipedalWalker-v3'





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
        minibatch_size=128,
        gae_lambda=gae_lambda,
        recurrent=recurrent,
        actor_hidden_size=actor_hidden_size,
        critic_hidden_size=critic_hidden_size,
    )
    # get logprob for action 0






    for i in range(1000):
        print('Iteration: ', i)
        best, avg = PPO.rollout_episodes()
        print("Current episode", PPO.current_episode)
        print('Best reward: ', best)
        print('Average reward: ', avg)
        PPO.update()
        if i % 10 == 0:
            PPO.save_model('model')
            print('Model saved')

