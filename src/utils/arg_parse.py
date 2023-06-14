import argparse
import ast
import sys
from conf import ArgsHelper


def generate_parser():
    """Generate the parser for the arguments"""
    parser = argparse.ArgumentParser(description='PPO arguments')

    parser.add_argument('--Continuous_or_Discrete', type=str, default='Continuous',
                        help='Continuous or Discrete. Default: Continuous', choices=['Continuous', 'Discrete'])
    # Add arguments
    parser.add_argument('--actor_hidden_size', default="{'layer':[64,64,78], 'activ':'tanh,relu,relu'}",
                        help="Dictionary for the actor hidden layers and activation functions. Default: {'layer':[64,64,78], 'activ':'tanh,relu,relu'}")
    parser.add_argument('--critic_hidden_size', default="{'layer':[64], 'activ':'tanh'}",
                        help="Dictionary for the critic hidden layers and activation functions. Default: {'layer':[64], 'activ':'tanh'}")
    parser.add_argument('--lr', type=float, default=0.0025,
                        help='Learning rate. Default: 0.00025')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for rewards. Default: 0.99')
    parser.add_argument('--K_epochs', type=int, default=10, help='Number of epochs'
                                                                 ' to update the policy for. Default: 10')
    parser.add_argument('--mini_batch_size', type=int,
                        default=64, help='Size of the mini batch. Default: 64')
    parser.add_argument('--eps_clip', type=float, default=0.2,
                        help='Clip parameter for PPO .Default: 0.2')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy term coefficient. Default: 0.01')
    parser.add_argument('--value_loss_coef', type=float,
                        default=0.5, help='Value loss coefficient. Default: 0.5')
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
    parser.add_argument('--render', type=bool, default=False,
                        help='Whether to render the environment. Default: False')
    parser.add_argument('--save_frequency', type=int,
                        default=10, help='Save every N updates. Default: 10')
    parser.add_argument('--shapley_values', type=bool, default=False,
                        help='Whether to compute the Shapley values in evaluating mode. Default: False')
    parser.add_argument('--class_name', type=str, default=[
    ], help='Name of the classes (i.e actions) to compute the Shapley values. Default: []', nargs='+')
    parser.add_argument('--features_name', type=str, default=[
    ], help='Name of the features (i.e observations) to compute the Shapley values. Default: []', nargs='+')
    parser.add_argument('--record_video', type=bool, default=False,
                        help='Whether to record a video of the agent. Default: False')

    return parser


def get_hyperparameters(eval=False):
    """Get the hyperparameters from the parser"""

    args = generate_parser().parse_args()
    if len(sys.argv) == 1:
        print('=============================================================')
        print('No arguments provided, config.py will be used')
        print('=============================================================')

    else:
        actor_hidden_size = ast.literal_eval(args.actor_hidden_size)
        critic_hidden_size = ast.literal_eval(args.critic_hidden_size)
    print('=============================================================')
    print('Hyperparameters:')
    print('=============================================================')
    print('continuous action space', args.Continuous_or_Discrete)
    # Continue to add other arguments...
    # Parse arguments
    # check if the environment is continuous or discrete
    # remove spaces
    args.Continuous_or_Discrete = args.Continuous_or_Discrete.replace(" ", "")
    if args.Continuous_or_Discrete == 'Continuous':
        continuous = True
    elif args.Continuous_or_Discrete == 'Discrete':
        continuous = False

    if not eval:
        if args.record_video:
            raise ValueError(
                'Video can only be recorded in evaluation mode')
        if args.shapley_values:
            raise ValueError(
                'Shapley values can only be computed in evaluation mode')

    if eval and args.shapley_values:

        # check if class name and features name are list of str
        if isinstance(args.class_name, list) and isinstance(args.features_name, list):
            print('class_name: ', args.class_name)
            print('features_name: ', args.features_name)
            if not all(isinstance(x, str) for x in args.class_name) or not all(isinstance(x, str) for x in args.features_name):
                raise ValueError(
                    'class_name and features_name should be list of str')

    # Convert string dictionaries to actual dictionaries

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
    save_frequency = args.save_frequency
    shapley_values = args.shapley_values
    class_name = args.class_name
    features_name = args.features_name
    record_video = args.record_video

    return continuous, actor_hidden_size, critic_hidden_size, lr, gamma, K_epochs, eps_clip, entropy_coef, value_loss_coef, gae_lambda, max_timesteps_one_episode, timestep_per_update, env_name, recurrent, decay_rate, minibatch_size, render, save_frequency, shapley_values, class_name, features_name, record_video
