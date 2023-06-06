from PPOs.MultipleContinuous_PPO import ContinuousPPO
from PPOs.Discrete_PPO import DiscretePPO

class PPOFactory:
    @staticmethod
    def create_PPO(lr, gamma, eps_clip, epochs, entropy_coef, value_loss_coef, timestep_per_episode, timestep_per_update,
                env_name, minibatch_size, gae_lambda, recurrent, actor_hidden_size, critic_hidden_size, decay_rate,
                continuous):
        if continuous:
            return ContinuousPPO(lr=lr, gamma=gamma, eps_clip=eps_clip, epochs=epochs, entropy_coef=entropy_coef,
                                 value_loss_coef=value_loss_coef, timestep_per_episode=timestep_per_episode,
                                 timestep_per_update=timestep_per_update, env_name=env_name,
                                 minibatch_size=minibatch_size, gae_lambda=gae_lambda, recurrent=recurrent,
                                 actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                                 decay_rate=decay_rate)
        else:
            return DiscretePPO(lr=lr, gamma=gamma, eps_clip=eps_clip, epochs=epochs, entropy_coef=entropy_coef,
                               value_loss_coef=value_loss_coef, timestep_per_episode=timestep_per_episode,
                               timestep_per_update=timestep_per_update, env_name=env_name,
                               minibatch_size=minibatch_size, gae_lambda=gae_lambda, recurrent=recurrent,
                               actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                               decay_rate=decay_rate)