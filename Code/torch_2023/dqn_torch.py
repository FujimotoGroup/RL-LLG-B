from copy import copy
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter

from modules import system_torch as s

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)
    
    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data])
        return state, action, reward, next_state, done

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        n1 = 128
        n2 = 128*128
        n3 = 128
        self.l1 = nn.Linear(6, n1)
        self.l2 = nn.Linear(n1, n2)
        self.l3 = nn.Linear(n2, n3)
        self.l4 = nn.Linear(n3, 3)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x

class Agent:
    def __init__(self,device):
        self.device = device
        self.gamma = 0.98
        self.lr = 0.0005
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 3

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet()
        self.qnet_target = QNet()
        self.optimizer = torch.optim.Adam(params=self.qnet.parameters(), lr=self.lr)
        self.qnet.to(self.device)

    def sync_qnet(self):
        self.qnet_target = deepcopy(self.qnet)

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.from_numpy(state[np.newaxis,:].astype(np.float32)).to(self.device)
            qs = self.qnet(state)
            return qs.argmax()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1)
        next_q.detach()
        target = reward + done * self.gamma * next_q

        loss_fn = nn.MSELoss(reduction='mean')
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss += loss.item()
        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()



def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    episodes = 100
    sync_interval = 20
    directory = "fig_dqn_torch"
    dt = 5e-13 # [s]
    alphaG = 0.01
    uniaxial_anisotropy = 540e0 # [Oe]
    action_size = 3
    field_max = 500e0 # [Oe]
    h0 = [h for h in np.linspace(0e0, field_max, action_size)] # [Oe]
    m0 = np.array([0e0, 0e0, 1e0])
    frequency = 10e9 # [Hz]
    step = int(1e0/(dt*frequency))
    print("step# = {}".format(step))

    agent = Agent(device)
    reward_history = []
    lr_history = []
    max_total_reward = 0e0

    for episode in range(episodes):
        print("episode:{:>5}".format(episode), end=":")
        dynamics = s.Dynamics(dt, alphaG, uniaxial_anisotropy, m0, limit=10*step)
        agent.loss = 0e0

        i = 0
        t = []
        m = []
        h = []
        total_reward = 0
        field = np.array([0e0, 0e0, 0e0])
        action = 0
        done = 1
        while (i < dynamics.limit):
            field_prev = np.array([h0[action], 0e0, 0e0])
            action = agent.get_action(dynamics.m, 0.1)
            dh = h0[action] - field_prev[0]

            S = dynamics.m

            for j in np.arange(step):
                field = field_prev + np.array([dh*(j%step)/step, 0e0, 0e0])
                dynamics.RungeKutta(field)
                if i % 10 == 0:
                    t.append(i*dt)
                    m.append(dynamics.m)
                    h.append(copy(field))
                i += 1

            reward = np.arccos(dynamics.m[2])/np.pi
            if i == dynamics.limit -1:
                done = 0

            agent.update(S, action, reward, dynamics.m, done)
            total_reward += reward

        if total_reward > max_total_reward:
            s.save_episode(episode, t, m, h, directory)
#            s.save_prob_value(episode, agent, directory)
            max_total_reward = total_reward

        print("reward = {:0>10.7f}".format(total_reward), end=", ")
#        print("loss_v = {:+>10.7f}".format(agent.loss_v), end=", ")

        if (episode != 0) & (episode % 10 == 0):
            s.save_episode(episode, t, m, h, directory)
#            s.save_prob_value(episode, agent, directory)
            s.save_history(reward_history, lr_history, directory)

        reward_history.append(total_reward)
        lr_history.append(agent.optimizer.param_groups[0]['lr'])
#        diff = np.abs(total_reward - np.array(reward_history[-20:-1]).mean())
#        print("diff = {:10.8f}".format(diff))
#        if diff < 1e-5:
#            s.save_episode(episode, t, m, h, directory)
#            s.save_prob_value(episode, agent, directory)
#            s.save_history(reward_history, lr_history, directory)
#            break

    s.save_history(reward_history, lr_history, directory)

if __name__ == '__main__':
    main()