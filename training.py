# This is a sample Python script.
from utils.PPO_factory import PPOFactory
from FibbonacciEnv import FibonacciEnvironment
from utils.arg_parse import get_hyperparameters
if __name__ == '__main__':



    # Define the parser
    continuous, actor_hidden_size, critic_hidden_size, lr, gamma, K_epochs, eps_clip, entropy_coef, value_loss_coef, gae_lambda, max_timesteps_one_episode, timestep_per_update, env_name, recurrent, decay_rate, minibatch_size,render= get_hyperparameters()
    PPO = PPOFactory.create_PPO(continuous=continuous, actor_hidden_size=actor_hidden_size,
                                critic_hidden_size=critic_hidden_size, lr=lr, gamma=gamma,
                                eps_clip=eps_clip, entropy_coef=entropy_coef, value_loss_coef=value_loss_coef,
                                gae_lambda=gae_lambda, timestep_per_update=timestep_per_update, env_name=env_name, recurrent=recurrent,
                                decay_rate=decay_rate, minibatch_size=minibatch_size,timestep_per_episode=max_timesteps_one_episode,
                                epochs=K_epochs,render=render)

    print(PPO)
    env = FibonacciEnvironment(sequence_length=5)
    print('action space shape',env.action_space.n)

    print('sample action',env.action_space.sample())
    print('sample observation',env.observation_space.sample())
    print(PPO.timestep_per_episode)
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

