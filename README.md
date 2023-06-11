# PPO simple implementation :rocket:

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
- **Continuous_or_Discrete** (str) : "Continuous" or "Discrete" environment. Default : **"Continuous"**
- **recurrent** (bool) : "True" or "False" (if you want to use a recurrent neural network). Default : **"False"**
- **env_name** (str) : "LunarLander-v2", "BipedalWalker-v2" or "DoomBasic-v0. Default : **"LunarLander-v2"**
- **actor_hidden_size** (dict) : size of the hidden layer of the actor, and the activation function. Default : **{"layer" : [32,32],"activ" : ["relu"]}**
- **critic_hidden_size** (dict) : size of the hidden layer of the critic, and the activation function. Default : **{"layer" : [32,32],"activ" : ["relu"]}**
- **lr** (float) : learning rate. Default : **0.0003**
- **gamma** (float) : discount factor. Default : **0.99**
- **K_epochs** (int) : number of epochs of gradient descent. Default : **4**
- **eps_clip** (float) : clip parameter for PPO. Default : **0.2**
- **mini_batch_size** (int) : size of the batch for gradient descent. Default : **64**
- **entropy_coef** (float) : entropy coefficient. Default : **0.01**
- **value_loss_coef** (float) : value loss coefficient. Default : **0.5**
- **max_timesteps_one_episode** (int) : maximum number of timesteps in one episode. Default : **2048**
- **timestep_per_update** (int) : number of timesteps before updating the policy. Default : **2048*4**
- **decay_rate** (float) : decay rate for the learning rate. Default : **0.99**
- **render** (bool) : "True" or "False" (if you want to render the environement). Default : **"False"**
- **save_frequency** (int) : number of updates before saving the model. Default : **10**
- **shapley_values** (bool) : "True" or "False" (if you want to compute the shapley values). Only works for discrete paradigm. Default : **"False"**
- **class_name** (list[str]) : Name of the classes (i.e actions) to compute the Shapley values. Default : **"[]"** (only works for discrete paradigm)
- **features_name** (list[str]) : Name of the features (i.e states) to compute the Shapley values. Default : **"[]"** (only works for discrete paradigm)
- **record_video** (bool) : "True" or "False" (if you want to record a video of the environement). Default : **"False"**


Example for training :
```bash
python train.py --Continous_or_Discrete "Discrete" --recurrent "False" --env_name "LunarLander-v2" --actor_hidden_size '{"layer" : [32,32],"activ" : ["relu"]}' --critic_hidden_size '{"layer" : [32,32],"activ" : ["relu"]}' --lr 0.0003 --gamma 0.99 --K_epochs 4 --eps_clip 0.2 --mini_batch_size 64 --entropy_coef 0.01 --value_loss_coef 0.5 --max_timesteps_one_episode 2048 --timestep_per_update 2048 --decay_rate 0.99 --render "False"
```
Then, for testing :
```bash
python eval.py --Continous_or_Discrete "Discrete" --recurrent "False" --env_name "LunarLander-v2" --actor_hidden_size '{"layer" : [32,32],"activ" : ["relu"]}' --critic_hidden_size '{"layer" : [32,32],"activ" : ["relu"]}' --lr 0.0003 --gamma 0.99 --K_epochs 4 --eps_clip 0.2 --mini_batch_size 64 --entropy_coef 0.01 --value_loss_coef 0.5 --max_timesteps_one_episode 2048 --timestep_per_update 2048 --decay_rate 0.99 --render "False"
```


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

| Rewards                                              | Video                                      |
|------------------------------------------------------|--------------------------------------------|
| <img src=results/curves/lunar_lander.png width=100%> | <img src=results/gif/lunar.gif width=100%> |


### BipedalWalker-v2 (Continuous)

Here are the results for BipedalWalker-v2 environement (reward per episode):

| Rewards                                              | Video                                               |
|------------------------------------------------------|-----------------------------------------------------|
| <img src=results/curves/biped_walker.png width=100%> | <img src=results/gif/bipedal_walker.gif width=100%> |


### Pendulum-v1 (Continuous)

Here are the results for Pendulum-v1 environement:

<img src="results/gif/pendulum.gif">

### Fibonacci sequence (Discrete LSTM)

Here are the results for Fibonacci sequence (reward per episode):

| Rewards                                                    |                                           
|------------------------------------------------------------|
| <img src=results/curves/fibonacci_discrete.png width=100%> |

### Fibonacci sequence (Continuous LSTM)

Here are the results for Fibonacci sequence (reward per episode):

| Rewards                                                      |
|--------------------------------------------------------------|
| <img src=results/curves/fibonacci_continuous.png width=100%> |



