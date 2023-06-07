# This is a sample Python script.
from src.utils.PPO_factory import PPOFactory
from src.utils.arg_parse import get_hyperparameters

def start_training():
    # Define the parser
    continuous, actor_hidden_size, critic_hidden_size, lr, gamma, K_epochs, eps_clip, entropy_coef, value_loss_coef, gae_lambda, max_timesteps_one_episode, timestep_per_update, env_name, recurrent, decay_rate, minibatch_size, render, save_freq = get_hyperparameters()
    PPO = PPOFactory.create_PPO(continuous=continuous, actor_hidden_size=actor_hidden_size,
                                critic_hidden_size=critic_hidden_size, lr=lr, gamma=gamma,
                                eps_clip=eps_clip, entropy_coef=entropy_coef, value_loss_coef=value_loss_coef,
                                gae_lambda=gae_lambda, timestep_per_update=timestep_per_update, env_name=env_name,
                                recurrent=recurrent,
                                decay_rate=decay_rate, minibatch_size=minibatch_size,
                                timestep_per_episode=max_timesteps_one_episode,
                                epochs=K_epochs, render=render)

    print(PPO)

    for i in range(1000):
        print('Iteration: ', i)
        best, avg = PPO.rollout_episodes()
        print("Current episode", PPO.current_episode)
        print('Best reward: ', best)
        print('Average reward: ', avg)
        PPO.update()
        if i % save_freq == 0:
            path = f'saved_weights/{str(PPO.env_name)}/{str(PPO.current_episode)}/'
            PPO.save_model(path)
            print('Model saved')

if __name__ == '__main__':
    start_training()



