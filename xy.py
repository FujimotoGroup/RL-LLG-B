from copy import copy
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 18
import os
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
        self.l4 = L.Linear(9)

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
        self.action_size = 9

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
        target = reward + done * self.gamma * next_q
#        target = reward + self.gamma * next_q

        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        return loss.data


def main():
    episodes = 2000
    record = 50
    sync_interval = 20
    directory = "xy_Kz=100"
    os.mkdir(directory)

    t_limit = 2e-9 # [s]
    dt = t_limit / 1e3 # [s]
    limit = int(t_limit / dt)
    alphaG = 0.01
    anisotropy = np.array([0e0, 0e0, 100e0]) # [Oe]
    H_shape = np.array([0.012*10800*0, 0.98*10800*0, 0.008*10800*0]) # [Oe]
    dh = 100 # [Oe]
    da = 1e-10 # [s]
    m0 = np.array([0e0, 0e0, 1e0])
    b = 0

    agent = DQNAgent()
    reward_history = []
    best_reward = -500
    loss_history = []

    for episode in range(episodes):
        print("episode:{:>4}".format(episode+1), end=":")
        dynamics = s.Dynamics(dt, alphaG, anisotropy, H_shape, m0)

        t = []
        m = []
        h = []
        Hani = []
        Hshape = []

        epsilon = 0.1
#        if episode > episodes*0.9:
#            epsilon = 0

        old_m = np.array([0e0, 0e0, 1e0])
        old_mz = m0[2]
        max_slope = -0.01
        reward = 0
        total_reward = 0
        total_loss = 0
        cnt = 0
        done = 1
        field = np.array([0e0, 0e0, 0e0])
        t.append(0)
        m.append(old_m)
        h.append(copy(field))
        Hani.append(anisotropy*old_m)
        Hshape.append(H_shape*old_m)
        old_state = np.concatenate([old_m, field])

        for i in range(1, limit+1):
            if i == 1:
                action = agent.get_action(old_state, epsilon)
                if action == 0:
                    a = [-1, -1, 0]
                if action == 1:
                    a = [-1, 0, 0]
                if action == 2:
                    a = [-1, 1, 0]
                if action == 3:
                    a = [0, -1, 0]
                if action == 4:
                    a = [0, 0, 0]
                if action == 5:
                    a = [0, 1, 0]
                if action == 6:
                    a = [1, -1, 0]
                if action == 7:
                    a = [1, 0, 0]
                if action == 8:
                    a = [1, 1, 0]
                old_action = action                    

            field += dh*np.array(a)*dt/da   

            time = i*dt

            dynamics.RungeKutta(field)

#            if i % 10 == 0:
            t.append(time)
            m.append(dynamics.m)
            h.append(copy(field))
            Hani.append(anisotropy*dynamics.m)
            Hshape.append(H_shape*dynamics.m)
            slope = (dynamics.m[2]-old_mz) / dt
            old_mz = dynamics.m[2]
            if slope < max_slope:
                max_slope = slope
                b = dynamics.m[2] - slope*time

            if i % (da/dt) == 0:
                state = np.concatenate([dynamics.m, field/1e4])
                action = agent.get_action(state, epsilon)
                if action == 0:
                    a = [-1, -1, 0]
                if action == 1:
                    a = [-1, 0, 0]
                if action == 2:
                    a = [-1, 1, 0]
                if action == 3:
                    a = [0, -1, 0]
                if action == 4:
                    a = [0, 0, 0]
                if action == 5:
                    a = [0, 1, 0]
                if action == 6:
                    a = [1, -1, 0]
                if action == 7:
                    a = [1, 0, 0]
                if action == 8:
                    a = [1, 1, 0]

                reward = - dynamics.m[2]**3
                if field[0] == 0:
                    reward *= 1.06
                total_reward += reward

                if i == limit+1:
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

        if episode % record == record-1:
            s.save_episode(episode+1, t, m, h, directory)

        if total_reward > best_reward:
            best_episode = episode + 1
            best_reward = total_reward
            best_m = np.array(m)
            best_h = np.array(h)
            best_Hani = np.array(Hani)
            best_Hshape = np.array(Hshape)
            best_slope = max_slope
            best_b = b
            reversal_time = (-1-best_b)/best_slope

        print("reward = {:.9f}".format(total_reward))

        reward_history.append(total_reward)

#        if episode > sync_interval:
#            ave_loss = total_loss / cnt
#            loss_history.append(ave_loss)
#            s.save_loss_history(loss_history, directory)

        s.save_reward_history(reward_history, directory)

    s.save_episode(0, t, best_m, best_h, directory)

    x = np.linspace(0, t_limit, 1000)
    y = best_slope*x + best_b
    plt.figure(figsize=(6,6))
    plt.ylim(-1, 1)
    plt.xlabel('Time [s]')
    plt.ylabel('Magnetization')
    plt.plot(np.array(t), best_m[:,2], color='green', label='$m_z$')
    plt.plot(x, y, color='red', linestyle='dashed', label='tangent')
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(directory+"/reversal_time.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if best_h.max() <= best_Hani.max():
        y_max = best_Hani.max()
    if best_Hani.max() <= best_Hshape.max():
        y_max = best_Hshape.max()
    y_min = best_h.min()
    if best_h.min() >= best_Hani.min():
        y_min = best_Hani.min()
    if best_Hani.min() >= best_Hshape.min():
        y_min = best_Hshape.min()

    axes[0].set_ylim([y_min, y_max])
    axes[0].plot(t, best_h[:,0], label='$h_x$')
    axes[0].plot(t, best_h[:,1], label='$h_y$')
    axes[0].plot(t, best_h[:,2], label='$h_z$')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('$H_(ext)$ [Oe]')
    axes[0].legend()

    axes[1].set_ylim([y_min, y_max])
    axes[1].plot(t, best_Hani[:,0], label='$h_x$')
    axes[1].plot(t, best_Hani[:,1], label='$h_y$')
    axes[1].plot(t, best_Hani[:,2], label='$h_z$')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('$H_(ani)$ [Oe]')
    axes[1].legend()

    axes[2].set_ylim([y_min, y_max])
    axes[2].plot(t, best_Hshape[:,0], label='$h_x$')
    axes[2].plot(t, best_Hshape[:,1], label='$h_y$')
    axes[2].plot(t, best_Hshape[:,2], label='$h_z$')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('$H_(shape)$ [Oe]')
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(directory+"/field.png", dpi=200)
    plt.close()


#    p.plot_energy(m_max, dynamics)
#    p.plot_3d(best_m)
    np.savetxt(directory+"/m.txt", best_m)
    with open(directory+"/options.txt", mode='w') as f:
        f.write('alphaG = ')
        f.write(str(alphaG))
        f.write('\nanisotropy = ')
        f.write(str(anisotropy))
        f.write(' [Oe]\nH_shape = ')
        f.write(str(H_shape))
        f.write(' [Oe]\ndH = ')
        f.write(str(dh))
        f.write(' [Oe]\nda = ')
        f.write(str(da))
        f.write(' [s]\nm0 = ')
        f.write(str(m0))
        f.write('\n\nbest episode = ')
        f.write(str(best_episode))
        f.write('\nreversal time = ')
        f.write(str(reversal_time))
        f.write(' [s]\naverage reward = ')
        f.write(str(best_reward/(t_limit/da)))


if __name__ == '__main__':
    main()