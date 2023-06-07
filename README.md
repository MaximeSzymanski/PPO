# PPO simple implementation

## What is this?

This is a simple implementation of PPO (Proximal Policy Optimization) algorithm.
The code is based on [OpenAI paper](https://arxiv.org/pdf/1707.06347.pdf).

## How to use

There are two different PPO : Continuous and Discrete.

### Continuous PPO

Continuous PPO is used for continuous action space (like BipedalWalker-v2 environement).
It handles MultiActions (like BipedalWalker-v2 environement).


### Discrete PPO

Discrete PPO is used for discrete action space (like LunarLander-v2 environement).
It doesn't handle MultiDiscrete (like LunarLander-v2 environement).

### Recurrent PPO

Recurrent PPO is used for environement with recurrent neural network (like Doom environement).
You have to use the LSTMActor and LSTMCritic classes.
Just specify it in the arguments when you launch the script.

For the moment, MultiDiscrete  is not implemented (I'm working on it).

## Install requirements

```bash
pip install -r requirements.txt
```

## Launch the training

To run the training:

```bash
python training.py --args
```

### Arguments
It takes several arguments:
- **Continous_or_Discrete**: "Continuous" or "Discrete" environement
- **recurrent**: "True" or "False" (if you want to use a recurrent neural network)
- **env_name**: "LunarLander-v2", "BipedalWalker-v2" or "DoomBasic-v0
- **actor_hidden_size**: size of the hidden layer of the actor, and the activation function.
- **critic_hidden_size**: size of the hidden layer of the critic, and the activation function.
- **lr** : learning rate
- **gamma**: discount factor
- **K_epochs**: number of epochs of gradient descent
- **eps_clip**: clip parameter for PPO
- **mini_batch_size**: size of the batch for gradient descent
- **entropy_coef**: entropy coefficient
- **value_loss_coef**: value loss coefficient
- **max_timesteps_one_episode** : maximum number of timesteps in one episode
- **timestep_per_update**: number of timesteps before updating the policy
- **decay_rate**: decay rate for the learning rate
- **render**: "True" or "False" (if you want to render the environement)


### Choose the architecture of the neural networks

You can customize the architecture of the neural networks using argements **actor_hidden_size** and **critic_hidden_size**.

You have to specify the size of the hidden layers and the activation function in a dictionnary, with the following format:
```python
{
    "layer" : [32,32,64,...,64],
    "activ" : ["relu","relu",...,"relu"]
}
```
If you choose only one activation function, it will be applied to all the hidden layers.
It works for both actor and critic, and for both continuous and discrete PPO (even for the recurrent PPO).


## Results



### LunarLander-v2 (Discrete)

Here are the results for LunarLander-v2 environement (reward per episode):

| Rewards                                              | Video                                     |
|------------------------------------------------------|-------------------------------------------|
| <img src=results/curves/lunar_lander.png width=100%> | <img src=results/gif/lunar.gif width=100%> |


### BipedalWalker-v2 (Continuous)

Here are the results for BipedalWalker-v2 environement (reward per episode):

| Rewards                                             | Video                                              |
|-----------------------------------------------------|----------------------------------------------------|
| <img src=results/curves/biped_walker.png width=100%> | <img src=results/gif/bipedal_walker.gif width=100%> |



### Fibonacci sequence (Discrete LSTM)

Here are the results for Fibonacci sequence (reward per episode):

| Rewards                                                    |                                           
|------------------------------------------------------------|
| <img src=results/curves/fibonacci_discrete.png width=100%> |

### Fibonacci sequence (Continuous LSTM)

Here are the results for Fibonacci sequence (reward per episode):

| Rewards                                                    |
|------------------------------------------------------------|
| <img src=results/curves/fibonacci_continuous.png width=100%> |



