import argparse
import ast


def generate_parser():
    parser = argparse.ArgumentParser(description='PPO arguments')
    parser.add_argument('--Continuous_or_Discrete', type=str, default='Continuous',
                        help='Continuous or Discrete. Default: Continuous')
    # Add arguments
    parser.add_argument('--actor_hidden_size', default="{'layer':[64,64,78], 'activ':'tanh,relu,relu'}",
                        help="Dictionary for the actor hidden layers and activation functions. Default: {'layer':[64,64,78], 'activ':'tanh,relu,relu'}")
    parser.add_argument('--critic_hidden_size', default="{'layer':[64], 'activ':'tanh'}",
                        help="Dictionary for the critic hidden layers and activation functions. Default: {'layer':[64], 'activ':'tanh'}")
    parser.add_argument('--lr', type=float, default=0.0025, help='Learning rate. Default: 0.00025')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards. Default: 0.99')
    parser.add_argument('--K_epochs', type=int, default=10, help='Number of epochs'
                                                                 ' to update the policy for. Default: 10')
    parser.add_argument('--mini_batch_size', type=int, default=50, help='Size of the mini batch. Default: 64')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip parameter for PPO .Default: 0.2')
    parser.add_argument('--entropy_coef', type=float, default=0.1, help='Entropy term coefficient. Default: 0.01')
    parser.add_argument('--value_loss_coef', type=float, default=0.5, help='Value loss coefficient. Default: 0.5')
    parser.add_argument('--max_timesteps_one_episode', type=int, default=50,
                        help='Maximum number of timestep in one episode. Default: 50')
    parser.add_argument('--timestep_per_update', type=int, default=50*10,
                        help='Number of timesteps per update. Default: 5000')
    parser.add_argument('--gae_lambda', type=int, default=0.95,
                        help='Lambda coefficient for Generalized Advantage Estimation. Default: 0.95')
    parser.add_argument('--env_name', type=str, default='BipedalWalker-v3',
                        help='Name of the environment. Default: BipedalWalker-v3')
    parser.add_argument('--recurrent', type=bool, default=False,
                        help='Whether to use a recurrent policy. Default: False')
    parser.add_argument('--decay_rate', type=int, default=0.99,
                        help='Decay rate for the Adam optimizer learning rate. Default: 0.99')
    parser.add_argument('--render', type=bool, default=False,help='Whether to render the environment. Default: False')
    return parser


def get_hyperparameters():
    args = generate_parser().parse_args()
    # Continue to add other arguments...
    # Parse arguments
    # check if the environment is continuous or discrete
    if args.Continuous_or_Discrete == 'Continuous':
        continuous = True
    elif args.Continuous_or_Discrete == 'Discrete':
        continuous = False
    else:
        raise ValueError('Continuous_or_Discrete should be either Continuous or Discrete')
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
    decay_rate = args.decay_rate
    minibatch_size = args.mini_batch_size
    render = args.render
    return continuous, actor_hidden_size, critic_hidden_size, lr, gamma, K_epochs, eps_clip, entropy_coef, value_loss_coef, gae_lambda, max_timesteps_one_episode, timestep_per_update, env_name, recurrent, decay_rate, minibatch_size, render
