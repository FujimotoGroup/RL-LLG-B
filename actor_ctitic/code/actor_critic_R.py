from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

from modules import system as s

class PolicyNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(3)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = F.softmax(x)
        return x
    
class ValueNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.pi = PolicyNet()
        self.v = ValueNet()
        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)

    def get_action(self, S):
        S = S[np.newaxis,:]
        probs = self.pi(S)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]

    def update(self, S, action_prob, reward, next_S, done):
        S = S[np.newaxis,:]
        next_S = next_S[np.newaxis,:]

        target = reward + self.gamma * self.v(next_S) * done
        target.unchain()
        v = self.v(S)
        loss_v = F.mean_squared_error(v, target)

        delta = target - v
        delta.unchain()
        loss_pi = -F.log(action_prob) * delta

        self.v.cleargrads()
        self.pi.cleargrads()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()

        return loss_v.data, loss_pi.data

def main():
    episodes = 1000

    dt = 5e-13 # [s]
    alphaG = 0.01
    uniaxial_anisotropy = 540e0 # [Oe]
    dh = 50e0 # [Oe]
    m0 = np.array([0e0, 0e0, 1e0])

    agent = Agent()
    reward_history = []
    high_reward = 0
    loss_v_history = []
    loss_pi_history = []

    for episode in range(episodes):
        print("episode:{:>4}".format(episode), end=":")
        dynamics = s.Dynamics(dt, alphaG, uniaxial_anisotropy, m0, limit=2001)

        t = []
        m = []
        h = []
        old_m = np.array([0e0, 0e0, 1e0])
        reward = 0
        total_reward = 0
        total_loss_v = 0
        total_loss_pi = 0
        cnt = 0
        done = 1
        field = np.array([0e0, 0e0, 0e0])

        for i in np.arange(dynamics.limit):
            if i % 400 == 0:
                action, prob = agent.get_action(dynamics.m)
                h0 = action - 1
                if i == 0:
                    old_prob = prob

            field += np.array([dh*h0*2e-2, 0e0, 0e0])      

            time = i*dt

            dynamics.RungeKutta(field)

            if i % 10 == 0:
                reward += - dynamics.m[2] / 40
                t.append(time)
                m.append(dynamics.m)
                h.append(copy(field))

            if i % 400 == 0 and i != 0:
                total_reward += reward

                if i == dynamics.limit -1:
                    done = 0

                loss_v, loss_pi = agent.update(old_m, old_prob, reward, dynamics.m, done)

                loss_pi = loss_pi[0]
                total_loss_v += loss_v
                total_loss_pi += loss_pi
                cnt += 1

                old_m = dynamics.m
                old_prob = prob
                reward = 0

        if total_reward > high_reward:
            s.save_episode(episode, t, m, h, "fig_actorcritic_R")
            high_reward = total_reward

        print("reward = {:.9f}".format(total_reward))

        reward_history.append(total_reward)

        ave_loss_v = total_loss_v / cnt
        ave_loss_pi = total_loss_pi / cnt
        loss_v_history.append(ave_loss_v)
        loss_pi_history.append(ave_loss_pi)

        s.save_reward_history(reward_history, "fig_actorcritic_R")

        s.save_loss_history(loss_v_history, "fig_actorcritic_R")
        s.save_loss_pi_history(loss_pi_history, "fig_actorcritic_R")


if __name__ == '__main__':
    main()