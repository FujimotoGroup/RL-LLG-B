from copy import copy
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 18
import os
from datetime import datetime

from modules import system as s
#from modules import plot as p

#　経験再生
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
        state = torch.tensor(np.stack([x[0] for x in data])).cuda()
        action = torch.tensor(np.array([x[1] for x in data]).astype(int)).cuda()
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32)).cuda()
        next_state = torch.tensor(np.stack([x[3] for x in data])).cuda()
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32)).cuda()
        return state, action, reward, next_state, done

#　ニューラルネットワーク
class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(6, 128)          # 6:状態の数
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, action_size)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x

#　エージェント
class DQNAgent:
    def __init__(self):
        self.gamma = 0.98         # 割引率
        self.lr = 0.001           # 学習率
        self.lr_decay = 0.9999     # 学習率の減衰率
        self.buffer_size = 10000  # 経験再生のバッファサイズ
        self.batch_size = 32      # ミニバッチのサイズ
        self.action_size = 3      # 行動の選択肢

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size).cuda()
        self.qnet_target = QNet(self.action_size).cuda()
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state[np.newaxis, :]).cuda()
            qs = self.qnet(state)
            return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.replay_buffer.get_batch()

        state = state.cuda()
        action = action.cuda()
        reward = reward.cuda()
        next_state = next_state.cuda()
        done = done.cuda()

        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(1)[0]
        next_q.detach()
        target = reward + (1-done) * self.gamma * next_q
#        target = reward + self.gamma * next_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data


def main():
    start_time = datetime.now()  # 処理開始時刻
    episodes = 2000              # エピソード数
    record = episodes/50         # 記録間隔
    sync_interval = episodes/10  #　同期間隔
    directory = "H=x_dH=10_da=0.01_ani=(0,0,100)"   # ファイル名
    os.mkdir(directory)

    t_limit = 2e-9 # [s]         # 終了時間
    dt = t_limit / 1e3 # [s]
    limit = int(t_limit / dt)
    alphaG = 0.01                # ギルバート減衰定数
    anisotropy = np.array([0e0, 0e0, 100e0]) # [Oe]    # 異方性
    H_shape = np.array([0.012*10800*0, 0.98*10800*0, 0.008*10800*0])  # [Oe]   # 反磁場
    dh = 10 # [Oe]   # 行動間隔ごとの磁場変化
    da = 1e-11 # [s]   # 行動間隔  da<=1e-10
    m0 = np.array([0e0, 0e0, 1e0])  #  初期磁化
    b = 0

    agent = DQNAgent()
    reward_history = []
    best_reward = -500
#    loss_history = []

    for episode in range(episodes):
        print("episode:{:>4}".format(episode+1), end=":")
        dynamics = s.Dynamics(dt, alphaG, anisotropy, H_shape, m0)

        t = []
        m = []
        h = []
        Hani = []
        Hshape = []

#        epsilon = 0.1
        if episode < episodes/2:
            epsilon = (0.1-1)/(episodes/2-0)*(episode) + 1
        else:
            epsilon = 0.1

        old_m = np.array([0e0, 0e0, 1e0])
        old_mz = m0[2]
        max_slope = -0.01
        reward = 0
        total_reward = 0
        total_loss = 0
        cnt = 0
        done = 0
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
                h0 = action - 1
                old_action = action                    

            field += np.array([dh*h0*dt/da, 0e0, 0e0])

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
                h0 = action - 1

                reward = - dynamics.m[2]**3
                if field[0] == 0:
                    reward *= 1.08
                total_reward += reward

                if i == limit:
                    done = 1
                             
                loss = agent.update(old_state, old_action, reward, state, done)

                if episode > sync_interval:
                    total_loss += loss
                    cnt += 1

                old_state = state
                old_action = action
                reward = 0

        if episode % sync_interval == 0:
            agent.sync_qnet()

        agent.scheduler.step()

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

#        if episode == 100-1:
#            one_m = np.array(m)
#            one_h = np.array(h)

#        if episode == 200-1:
#            two_m = np.array(m)
#            two_h = np.array(h)

        print("reward = {:.9f}".format(total_reward))

        reward_history.append(total_reward)

#        if episode > sync_interval:
#            ave_loss = total_loss / cnt
#            loss_history.append(ave_loss)
#            s.save_loss_history(loss_history, directory)

        s.save_reward_history(reward_history, directory)

    s.save_episode(0, t, best_m, best_h, directory)

    end_time = datetime.now()           # 処理終了時刻
    duration = end_time - start_time    # 処理時間

    x = np.linspace(0, t_limit, 1000)
    y = best_slope*x + best_b
    plt.figure(figsize=(6,6))
    plt.ylim(-1, 1)
    plt.xlabel('Time [s]')
    plt.ylabel('Magnetization')
    plt.plot(np.array(t), best_m[:,0], label='$m_z$')
    plt.plot(np.array(t), best_m[:,1], label='$m_z$')
    plt.plot(np.array(t), best_m[:,2], label='$m_z$')
    plt.legend(fontsize=16)
    plt.plot(x, y, color='darkgreen', linestyle='dashed')
    plt.tight_layout()
    plt.savefig(directory+"/reversal_time.png", dpi=200)
    plt.close()


    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes = axes.flatten()

    y_max = best_h.max()
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
    axes[0].set_ylabel('$H_{\mathrm{ext}}$ [Oe]')
    axes[0].legend()

    axes[1].set_ylim([y_min, y_max])
    axes[1].plot(t, best_Hani[:,0], label='$h_x$')
    axes[1].plot(t, best_Hani[:,1], label='$h_y$')
    axes[1].plot(t, best_Hani[:,2], label='$h_z$')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('$H_{\mathrm{ani}}$ [Oe]')
    axes[1].legend()

    axes[2].set_ylim([y_min, y_max])
    axes[2].plot(t, best_Hshape[:,0], label='$h_x$')
    axes[2].plot(t, best_Hshape[:,1], label='$h_y$')
    axes[2].plot(t, best_Hshape[:,2], label='$h_z$')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('$H_{\mathrm{shape}}$ [Oe]')
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(directory+"/field.png", dpi=200)
    plt.close()


#    p.plot_energy(m_max, dynamics)
#    p.plot_3d(best_m)

    np.savetxt(directory+"/m.txt", best_m)
    np.savetxt(directory+"/h.txt", best_h)
    np.savetxt(directory+"/t.txt", t)
#    np.savetxt(directory+"/100th_m.txt", one_m)
#    np.savetxt(directory+"/100th_h.txt", one_h)
#    np.savetxt(directory+"/200th_m.txt", two_m)
#    np.savetxt(directory+"/200th_h.txt", two_h)
    np.savetxt(directory+"/reward history.txt", reward_history)

    with open(directory + "/options.txt", mode='w') as f:
        f.write(f"""
    alphaG = {alphaG}
    anisotropy = {anisotropy} [Oe]
    H_shape = {H_shape} [Oe]
    dH = {dh} [Oe]
    da = {da} [s]
    m0 = {m0}
    time limit = {t_limit} [s]

    best episode = {best_episode}
    reversal time = {reversal_time} [s]
    processing time = {duration}
    
    slope = {best_slope}
    segment = {best_b}
    average reward = {best_reward / (t_limit / da)}
    """)


if __name__ == '__main__':
    main()