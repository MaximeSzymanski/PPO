from dataclasses import dataclass

config_args = {
    'Continuous_or_Discrete': 'Continuous',
    'actor_hidden_size': {'layer': [64, 64], 'activation': 'tanh'},
    'critic_hidden_size': {'layer': [64, 64], 'activation': 'tanh'},
    'lr': 0.0025,
    'gamma': 0.99,
    'K_epochs': 10,
    'mini_batch_size': 64,
    'eps_clip': 0.2,
    'entropy_coef': 0.01,
    'value_loss_coef': 0.5,
    'max_timesteps_one_episode': 50,
    'timestep_per_update': 50*10,
    'gae_lambda': 0.95,
    'env_name': 'BipedalWalker-v3',
    'recurrent': False,
    'decay_rate': 0.99,
    'render': False,
    'save_frequency': 10,
    'shapley_values': False,
    'class_name': [],
    'features_name': []
}


class ArgsHelper():
    Continuous_or_Discrete = 'Continuous'
    actor_hidden_size = "{'layer': [64, 64], 'activation': 'tanh'}"
    critic_hidden_size = "{'layer': [64, 64], 'activation': 'tanh'}"
    lr = 0.0025
    gamma = 0.99
    K_epochs = 10
    mini_batch_size = 64
    eps_clip = 0.2
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_timesteps_one_episode = 300
    timestep_per_update = 300 * 10
    gae_lambda = 0.95
    env_name = 'BipedalWalker-v3'
    recurrent = False
    decay_rate = 0.99
    render = False
    save_frequency = 10
    shapley_values = False
    class_name = []
    features_name = []

    def __init__(self):
        # read fields using config_args
        for key, value in config_args.items():
            setattr(self, key, value)
        # print each field of the class
        for field in dir(self):
            print(field, ': ', getattr(self, field),
                  " type: ", type(getattr(self, field)))
