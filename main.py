# This is a sample Python script.
import torch
from STOCKSENV import StockEnv
from MultipleContinuous_PPO import ContinuousPPO as PPO
import gymnasium as gym
import os
import argparse
import ast

if __name__ == '__main__':



    # Define the parser
    parser = argparse.ArgumentParser(description='PPO arguments')

    # Add arguments
    parser.add_argument('--actor_hidden_size', default="{'layer':[64,64,78], 'activ':'tanh,relu,relu'}",
                        help="Dictionary for the actor hidden layers and activation functions")
    parser.add_argument('--critic_hidden_size', default="{'layer':[64], 'activ':'tanh'}",
                        help="Dictionary for the critic hidden layers and activation functions")
    parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
    parser.add_argument('--K_epochs', type=int, default=10, help='Number of epochs'
                                                                  ' to update the policy for')
    parser.add_argument('--mini_batch_size', type=int, default=64, help='Size of the mini batch')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip parameter for PPO')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy term coefficient')
    parser.add_argument('--value_loss_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--max_timesteps_one_episode', type=int, default=10000, help='Maximum number of timestep in one episode')
    parser.add_argument('--timestep_per_update', type=int, default=10000, help='Number of timesteps per update')
    parser.add_argument('--gae_lambda', type=int, default=0.95, help='Lambda coefficient for Generalized Advantage Estimation')
    parser.add_argument('--env_name', type=str, default='BipedalWalker-v3', help='Name of the environment')
    parser.add_argument('--recurrent', type=bool, default=False, help='Whether to use a recurrent policy')
    # Continue to add other arguments...

    # Parse arguments
    args = parser.parse_args()

    # Convert string dictionaries to actual dictionaries
    actor_hidden_size = ast.literal_eval(args.actor_hidden_size)
    critic_hidden_size = ast.literal_eval(args.critic_hidden_size)
    lr = args.lr
    gamma = args.gamma
    K_epochs = args.K_epochs
    eps_clip = args.eps_clip
    entropy_coef = args.entropy_coef
    value_loss_coef = args.value_loss_coef
    gae_lambda = args.gae_lambda
    max_timesteps_one_episode = args.max_timesteps_one_episode
    timestep_per_update = args.timestep_per_update
    env_name = args.env_name
    recurrent = args.recurrent
    minibatch_size = args.mini_batch_size


    PPO = PPO(
        lr=lr,
        gamma=gamma,
        eps_clip=eps_clip,
        epochs=K_epochs,
        entropy_coef=entropy_coef,
        value_loss_coef=value_loss_coef,
        timestep_per_episode =max_timesteps_one_episode,
        timestep_per_update=timestep_per_update,
        env_name=env_name,
        minibatch_size=minibatch_size,
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

