from multiprocessing import Process
import torch
from torch.multiprocessing import Event, Value, Manager
import numpy as np
import os


class Worker(Process):
    def __init__(self, env, shared_buffer, timestep_per_update, PPO, signal, number_of_episode, episode_reward, episode_reward_list):
        super().__init__()
        self.env = env
        self.shared_buffer = shared_buffer
        self.timestep_per_update = timestep_per_update
        self.number_of_episode = number_of_episode
        self.PPO = PPO
        self.episode_reward = episode_reward
        self.signal = signal
        self.episode_reward_list = episode_reward_list

    def run(self):
        print(f"Worker {self.name} started, ticker: {self.env.company_ticker}")
        number_of_step = 0
        episode_reward = 0
        state, _ = self.env.reset()
        self.number_of_episode.value += 1
        while self.signal.value < self.timestep_per_update:
            #print(f"Worker {self.name} step {number_of_step}")
            number_of_step += 1
            self.signal.value += 1
            action, log_prob = self.PPO.choose_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            episode_reward += reward
            state = torch.tensor(state, device=self.PPO.device, dtype=torch.float32).detach()
            value = self.PPO.critic(state).detach()
            reward = torch.tensor([reward], device=self.PPO.device, dtype=torch.float32).detach()
            mask = torch.tensor([not done], device=self.PPO.device, dtype=torch.float32).detach()
            done = torch.tensor([done], device=self.PPO.device, dtype=torch.float32).detach()
            action = torch.tensor([action], device=self.PPO.device, dtype=torch.float32).detach()
            self.shared_buffer.put((reward, value, log_prob, action, done, state, mask))
            state = next_state

            if done:
                self.episode_reward_list.append(episode_reward)
                self.episode_reward.value += episode_reward
                episode_reward = 0
                self.number_of_episode.value += 1

                print(f"Worker {self.name} finished, ticker: {self.env.company_ticker}")
                break
                state, _ = self.env.reset()
            else:
                state = next_state

class WorkerManager:
    def __init__(self):
        self.workers_list = []
        self.number_of_step = 0
        self.average_reward = 0
        self.stop_event = Event()
        self.stop_signal = Value('i', 0)
        self.number_episode = Value('i', 0)
        self.episode_reward = Value('d', 0.0)
        self.episode_reward_list = Manager().list()

        best_reward = -np.inf
        # count number of file in data_train folder
        self.num_workers = 0
        for file in os.listdir("data_train"):
            if file.endswith(".csv"):
                self.num_workers += 1

        # Create a manager queue to handle shared memory across processes
        self.shared_buffer = Manager().Queue()
    def init_workers(self, PPO):
        print(f"Init worker : Number of workers: {self.num_workers}")
        self.workers_list = [Worker(PPO.env_pool[i], self.shared_buffer, PPO.timestep_per_update, PPO, self.stop_signal, self.number_episode, self.episode_reward, self.episode_reward_list) for i in range(self.num_workers)]

    def get_workers(self):
        return self.workers_list


class WorkerGet():
    worker_manager : WorkerManager= None

    @staticmethod
    def get_worker_manager(PPO):
        if WorkerGet.worker_manager is None:
            WorkerGet.worker_manager = WorkerManager()
            WorkerGet.worker_manager.init_workers(PPO)
        WorkerGet.worker_manager.init_workers(PPO)
        return WorkerGet.worker_manager

