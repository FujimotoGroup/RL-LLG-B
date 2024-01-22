from copy import copy
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from dezero import Model
from dezero import optimizers
from dezero import Variable
import dezero.functions as F
import dezero.layers as L
from collections import deque
import random

from modules import system as s
from modules import plot as p

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

class QNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(128)
        self.l4 = L.Linear(3)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x

class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 3

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet()
        self.qnet_target = QNet()
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)

    def sync_qnet(self):
        self.qnet_target = deepcopy(self.qnet)

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            qs = self.qnet(state)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1)
        next_q.unchain()
#        target = reward + done * self.gamma * next_q
        target = reward + self.gamma * next_q

        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        return loss.data


def main():
    episodes = 1000  # optional
    sync_interval = 20
    directory = "x_0.8t"  # optional

    dt = 5e-13 # [s]
    t_limit = 16e-1 # [ns]  # optional
    alphaG = 0.01
    anisotropy = np.array([0e0, 0e0, 540e0]) # [Oe]  # optional
    dh = 100 # [Oe]  # optional
    da = 1e-1 # [ns]  # optional
    m0 = np.array([0e0, 0e0, 1e0])

    agent = DQNAgent()
    reward_history = []
    high_reward = -50
    loss_history = []

    for episode in range(episodes):
        print("episode:{:>4}".format(episode), end=":")
        dynamics = s.Dynamics(dt, alphaG, anisotropy, m0, limit=t_limit*2e3+1)

        t = []
        m = []
        h = []

        epsilon = 0.1
        if episode > (episodes-50):
            epsilon = 0

        old_m = np.array([0e0, 0e0, 1e0])
        reward = 0
        total_reward = 0
        total_loss = 0
        cnt = 0
        done = 1
        field = np.array([0e0, 0e0, 0e0])
        old_state = np.concatenate([old_m, field])

        for i in np.arange(dynamics.limit):
            if i == 0:
                action = agent.get_action(old_state, epsilon)
                h0 = action - 1
                old_action = action                    

            field += np.array([dh*h0/(da*2000), 0e0, 0e0])      

            time = i*dt

            dynamics.RungeKutta(field)

            if i % 10 == 0:
#                reward += - dynamics.m[2] / (da/10)
                t.append(time)
                m.append(dynamics.m)
                h.append(copy(field))

            if i % (da*2000) == 0 and i != 0:
                state = np.concatenate([dynamics.m, field/1e4])
                action = agent.get_action(state, epsilon)
                h0 = action - 1

                reward = - dynamics.m[2]
                total_reward += reward

                if i == dynamics.limit - 1:
                    done = 0   
                             
                loss = agent.update(old_state, old_action, reward, state, done)

                if episode > sync_interval:
                    total_loss += loss
                    cnt += 1

                old_state = state
                old_action = action
                reward = 0

        if episode % sync_interval == 0:
            agent.sync_qnet()

        if total_reward > high_reward:
            s.save_episode(episode, t, m, h, directory)
            high_reward = total_reward
            m_max = m

        print("reward = {:.9f}".format(total_reward))

        reward_history.append(total_reward)

#        if episode > sync_interval:
#            ave_loss = total_loss / cnt
#            loss_history.append(ave_loss)
#            s.save_loss_history(loss_history, directory)

        s.save_reward_history(reward_history, directory)

    p.plot_energy(m_max, dynamics)
    p.plot_3d(m_max)
    np.savetxt('m.txt', m_max)


if __name__ == '__main__':
    main()