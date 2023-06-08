
from src.utils.PPO_factory import PPOFactory
from src.utils.arg_parse import get_hyperparameters
def evaluate_model():
    continuous, actor_hidden_size, critic_hidden_size, lr, gamma, K_epochs, eps_clip, entropy_coef, value_loss_coef, gae_lambda, max_timesteps_one_episode, timestep_per_update, env_name, recurrent, decay_rate, minibatch_size, render,save_frequency= get_hyperparameters()

    PPO = PPOFactory.create_PPO(continuous=continuous, actor_hidden_size=actor_hidden_size,
                                critic_hidden_size=critic_hidden_size, lr=lr, gamma=gamma,
                                eps_clip=eps_clip, entropy_coef=entropy_coef, value_loss_coef=value_loss_coef,
                                gae_lambda=gae_lambda, timestep_per_update=timestep_per_update, env_name=env_name,
                                recurrent=recurrent,
                                decay_rate=decay_rate, minibatch_size=minibatch_size,
                                timestep_per_episode=max_timesteps_one_episode,
                                epochs=K_epochs, render=render)

    print(PPO)

    PPO.load_model()
    print('Model loaded')

    PPO.evaluate()
if __name__ == '__main__':
    evaluate_model()
