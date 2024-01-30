from copy import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter

from modules import system_torch as s

class Policy(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        n1 = 128
        n2 = 128*128
        n3 = 128
        n4 = 32
        self.l1 = nn.Linear(3, n1)
        self.l2 = nn.Linear(n1, n2)
        self.l3 = nn.Linear(n2, n3)
        self.l4 = nn.Linear(n3, n4)
        self.pi = nn.Linear(n4, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.softmax(self.pi(x), dim=1)
        return x

class Value(nn.Module):
    def __init__(self):
        super().__init__()
        n1 = 128
        n2 = 128*128
        n3 = 128
        self.l1 = nn.Linear(3, n1)
        self.l2 = nn.Linear(n1, n2)
        self.l3 = nn.Linear(n2, n3)
        self.l4 = nn.Linear(n3, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x

class Agent:
    def __init__(self,device,action_size):
        self.device = device

        self.gamma = 0.1
#        self.epsilon = 0.05

        self.action_size = action_size
        self.pi = Policy(self.action_size)
#        self.lr_pi = 1e-3
#        self.optimizer_pi = torch.optim.SGD(params=self.pi.parameters(), lr=self.lr_pi, momentum=0.9)
        self.lr_pi = 2e-4
        self.optimizer_pi = torch.optim.Adam(params=self.pi.parameters(), lr=self.lr_pi)

#        self.scheduler_pi = torch.optim.lr_scheduler.CycleLR(self.optimizer_pi, self.lr_pi, 100)
#        self.scheduler_pi = torch.optim.lr_scheduler.StepLR(self.optimizer_pi, 50, 0.9)
        self.scheduler_pi = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_pi, patience=50, factor=0.2, verbose=True)

        self.v = Value()
        self.lr_v  = 1e-4
        self.optimizer_v = torch.optim.Adam(params=self.v.parameters(), lr=self.lr_v)

        self.pi.to(self.device)
        self.v.to(self.device)

    def get_action(self, S):
        S = torch.from_numpy(S[np.newaxis,:].astype(np.float32)).to(self.device)
        probs = self.pi(S)[0]
        cat = Categorical(probs)
        action = cat.sample().item()
#        if np.random.rand() < self.epsilon:
#            action = np.random.randint(0, self.action_size)
#        else:
#            action = probs.argmax()
        return action, probs[action]

    def update(self, m, action_prob, reward, next_m):
        m = torch.from_numpy(m[np.newaxis, :].astype(np.float32)).to(self.device)
        next_m = torch.from_numpy(next_m[np.newaxis, :].astype(np.float32)).to(self.device)

        target = reward + self.gamma * self.v(next_m)
        target.detach()
        v = self.v(m)
        loss_fn = nn.MSELoss(reduction='mean')
        loss_v = loss_fn(v, target)

        delta = target - v
        loss_pi = - torch.log(action_prob) * delta.item()

        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()

#        print("loss pi: {:0>10.7f}".format(loss_pi[0,0].data), end=", ")
#        print("loss v: {:0>10.7f}".format(loss_v.data))

        self.optimizer_v.step()
        self.optimizer_pi.step()

        self.loss_v += loss_v.item()
        self.loss_pi += loss_pi.item()

def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
#    writer = SummaryWriter(log_dir="log")

    episodes = 10000
    directory = "fig_torch"

    dt = 5e-13 # [s]
    alphaG = 0.01
    uniaxial_anisotropy = 540e0 # [Oe]

#    action_size = 1*2 + 1
    action_size = 5
    field_max = 500e0 # [Oe]
#    h0 = [h for h in np.linspace(-field_max, field_max, action_size)] # [Oe]
    h0 = [h for h in np.linspace(0e0, field_max, action_size)] # [Oe]
    m0 = np.array([0e0, 0e0, 1e0])

    frequency = 10e9 # [Hz]
    step = int(1e0/(dt*frequency))
    print("step# = {}".format(step))

    agent = Agent(device, action_size)
    reward_history = []
    lr_history = []

    max_total_reward = 0e0
    for episode in range(episodes):
        print("episode:{:>5}".format(episode), end=":")
        dynamics = s.Dynamics(dt, alphaG, uniaxial_anisotropy, m0, limit=10*step)
        agent.loss_pi = 0e0
        agent.loss_v = 0e0

        i = 0
        t = []
        m = []
        h = []
        total_reward = 0
        field = np.array([0e0, 0e0, 0e0])
#        action = int((action_size) / 2)
        action = 0
        while (i < dynamics.limit):
            field_prev = np.array([h0[action], 0e0, 0e0])
            action, action_prob = agent.get_action(dynamics.m)
            dh = h0[action] - field_prev[0]

            S = dynamics.m

            for j in np.arange(step):
                field = field_prev + np.array([dh*(j%step)/step, 0e0, 0e0])
#                field = np.array([h0[action], 0e0, 0e0])
                dynamics.RungeKutta(field)
                if i % 10 == 0:
#                    print(i)
                    t.append(i*dt)
                    m.append(dynamics.m)
                    h.append(copy(field))
                i += 1

            reward = np.arccos(dynamics.m[2])/np.pi
#            reward = (1e0 - dynamics.m[2])/2e0

            agent.update(S, action_prob, reward, dynamics.m)
            total_reward += reward

        if total_reward > max_total_reward:
            s.save_episode(episode, t, m, h, directory)
            s.save_prob_value(episode, agent, directory)
            max_total_reward = total_reward

        print("reward = {:0>10.7f}".format(total_reward), end=", ")
        print("loss_pi = {:+>10.7f}".format(agent.loss_pi), end=", ")
        print("loss_v = {:+>10.7f}".format(agent.loss_v), end=", ")

        if (episode != 0) & (episode % 10 == 0):
            s.save_episode(episode, t, m, h, directory)
            s.save_prob_value(episode, agent, directory)
            s.save_history(reward_history, lr_history, directory)

        reward_history.append(total_reward)
        lr_history.append(agent.optimizer_pi.param_groups[0]['lr'])

#        if agent.optimizer_pi.param_groups[0]['lr'] <= 1e-7:
#            break

#        agent.scheduler_pi.step(agent.loss_pi)

        diff = np.abs(total_reward - np.array(reward_history[-20:-1]).mean())
        print("diff = {:10.8f}".format(diff))
        if diff < 1e-5:
            s.save_episode(episode, t, m, h, directory)
            s.save_prob_value(episode, agent, directory)
            s.save_history(reward_history, lr_history, directory)
            break

#        writer.add_scalar("loss_v", agent.loss_v, episode)
#        writer.add_scalar("loss_pi", agent.loss_pi, episode)
#        writer.add_scalar("lr", agent.optimizer_pi.param_groups[0]['lr'], episode)
#        writer.add_scalar("total_reward", total_reward, episode)


    s.save_history(reward_history, lr_history, directory)

if __name__ == '__main__':
    main()
