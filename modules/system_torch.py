import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch

class Magnetization:
    def __init__(self, m0:np.array):
        self.m = m0

class Dynamics(Magnetization):
    def __init__(self, dt:float, alphaG:float, uniaxial_anisotropy:float, m0:np.array, eps:float = 1e-9, limit:int = 10000):
        super().__init__(m0)

        self.t0 = 1e-10 # [s]
#        self.gamma = 1.76e11 # [rad / T s]
        self.gamma = 1.76e7 # [rad / Oe s]
        self.eps = eps

        self.dt = dt / self.t0
        self.alphaG = alphaG
        self.anisotrpy = uniaxial_anisotropy
        self.limit = limit

    def LLG(self, magnetization:np.array, field:np.array) -> np.array:
        H = np.array([0e0, 0e0, self.anisotrpy*self.m[2]]) + field
        g = self.gamma*self.t0 / (1e0 + self.alphaG**2)
        mxH = np.cross(magnetization, H)
        m = - g*mxH - g*self.alphaG * np.cross(magnetization, mxH)
        return m

    def RungeKutta(self, field:np.array):
        m1 = self.LLG(self.m,                  field)
        m2 = self.LLG(self.m + self.dt*m1/2e0, field)
        m3 = self.LLG(self.m + self.dt*m2/2e0, field)
        m4 = self.LLG(self.m + self.dt*m3,     field)
        self.m = self.m + self.dt/6e0 * (m1 + 2e0*m2 + 2e0*m3 + m4)

def plot(t:list, m:list, h:list, issave:bool=False):
    t = np.array(t)
    m = np.array(m)
    h = np.array(h)

    if not(issave):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_xlim([-1e0,1e0])
        ax.set_ylim([-1e0,1e0])
        ax.set_zlim([-1e0,1e0])
        ax.plot(m[:,0], m[:, 1], m[:,2])
        plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_ylim([-1e0,1e0])
    axes[0].plot(t, m[:,0], label='m_x')
    axes[0].plot(t, m[:,1], label='m_y')
    axes[0].plot(t, m[:,2], label='m_z')
    axes[0].legend()

    axes[1].plot(t, h[:,0], label='h_x')
    axes[1].plot(t, h[:,1], label='h_y')
    axes[1].plot(t, h[:,2], label='h_z')
    axes[1].legend()

    if not(issave):
        plt.show()
    else:
        return fig

def save_episode(episode:int, t:list, m:list, h:list, directory:str):
    fig = plot(t,m,h, issave=True)
    fig.tight_layout()
    fig.savefig(directory+"/dynamics_episode{:0=5}.png".format(episode), dpi=200)
    plt.close()

def save_prob_value(episode:int, agent, directory:str):
    n = agent.action_size
    mesh_num = 20
    theta_u = np.linspace(0e0, np.pi/2e0, mesh_num)
    phi = np.linspace(0e0, 2e0*np.pi, mesh_num)
#    theta, phi = np.meshgrid(theta, phi)
    probs = []
    values = []
    for t in theta_u:
        prob = []
        value = []
        for p in phi:
            x = np.sin(t)*np.cos(p)
            y = np.sin(t)*np.sin(p)
            z = np.cos(t)
            S = np.array([x, y, z])
            S = torch.from_numpy(S[np.newaxis,:].astype(np.float32)).to(agent.device)
            pi = agent.pi(S)[0].to(torch.device('cpu')).detach().numpy()
            v = agent.v(S)[0,0].to(torch.device('cpu')).detach().numpy()
            prob.append(pi)
            value.append(v)
        probs.append(prob)
        values.append(value)
    probs_u = np.array(probs)
    values_u = np.array(values)

    theta, phi = np.meshgrid(theta_u, phi)
    x_u = np.sin(theta)*np.cos(phi)
    y_u = np.sin(theta)*np.sin(phi)

    theta_d = np.linspace(np.pi/2e0, np.pi, mesh_num)
    phi = np.linspace(0e0, 2e0*np.pi, mesh_num)
    probs = []
    values = []
    for t in theta_d:
        prob = []
        value = []
        for p in phi:
            x = np.sin(t)*np.cos(p)
            y = np.sin(t)*np.sin(p)
            z = np.cos(t)
            S = np.array([x, y, z])
            S = torch.from_numpy(S[np.newaxis,:].astype(np.float32)).to(agent.device)
            pi = agent.pi(S)[0].to(torch.device('cpu')).detach().numpy()
            v = agent.v(S)[0,0].to(torch.device('cpu')).detach().numpy()
            prob.append(pi)
            value.append(v)
        probs.append(prob)
        values.append(value)
    probs_d = np.array(probs)
    values_d = np.array(values)

    theta, phi = np.meshgrid(theta_d, phi)
    x_d = np.sin(theta)*np.cos(phi)
    y_d = np.sin(theta)*np.sin(phi)

    fig, axes = plt.subplots(2, n, figsize=(5*n, 12))
    window = 2e0/n
    levels = np.linspace(0e0, window, 100)
    norm = mpl.colors.Normalize(vmin=0e0, vmax=window)
    for i in np.arange(n):
        axes[0][i].set_xlim([-1e0, 1e0])
        axes[0][i].set_ylim([-1e0, 1e0])
        axes[0][i].axis('equal')
        axes[0][i].contourf(x_u, y_u, probs_u[:,:,i], levels=levels, cmap="bwr", extend="max")
#        ax = axes[0][i].contourf(x_u, y_u, probs_u[:,:,i])
#        cax = make_axes_locatable(axes[0][i]).append_axes("right", size="3%", pad=0.1)
#        fig.colorbar(ax, cax=cax)

        axes[1][i].set_xlim([-1e0, 1e0])
        axes[1][i].set_ylim([-1e0, 1e0])
        axes[1][i].axis('equal')
        ax = axes[1][i].contourf(x_d, y_d, probs_d[:,:,i], levels=levels, cmap="bwr", extend="max")
#        ax = axes[1][i].contourf(x_d, y_d, probs_d[:,:,i])
#        cax = make_axes_locatable(axes[1][i]).append_axes("right", size="3%", pad=0.1)
#        fig.colorbar(ax, cax=cax)

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="bwr"),ax=axes.ravel().tolist(), extend="max")

#    fig.tight_layout()
    fig.savefig(directory+"/prob_episode{:0=5}.png".format(episode), dpi=200)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(6, 12))
    maximum = max(values_u.max(), values_d.max())
    levels = np.linspace(0e0, maximum, 200)
    norm = mpl.colors.Normalize(vmin=0e0, vmax=maximum)
    axes[0].set_xlim([-1e0, 1e0])
    axes[0].set_ylim([-1e0, 1e0])
    axes[0].axis('equal')
    axes[0].contourf(x_u, y_u, values_u, levels=levels, cmap="Blues", extend="max")
#    ax = axes[0][i].contourf(x_u, y_u, probs_u[:,:,i])
#    cax = make_axes_locatable(axes[0][i]).append_axes("right", size="3%", pad=0.1)
#    fig.colorbar(ax, cax=cax)

    axes[1].set_xlim([-1e0, 1e0])
    axes[1].set_ylim([-1e0, 1e0])
    axes[1].axis('equal')
    ax = axes[1].contourf(x_d, y_d, values_d, levels=levels, cmap="Blues", extend="max")
#    ax = axes[1][i].contourf(x_d, y_d, probs_d[:,:,i])
#    cax = make_axes_locatable(axes[1][i]).append_axes("right", size="3%", pad=0.1)
#    fig.colorbar(ax, cax=cax)

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="Blues"),ax=axes.ravel().tolist(), extend="max")

#    fig.tight_layout()
    fig.savefig(directory+"/value_episode{:0=5}.png".format(episode), dpi=200)
    plt.close()

def save_history(reward_history:list, lr_history:list, directory:str):
    reward_history = np.array(reward_history)
    lr_history = np.array(lr_history)
    episodes = np.arange(len(reward_history))

    slice_num = 10
    average = [ reward_history[i:i+slice_num].mean() for i in episodes[::slice_num]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].plot(episodes, reward_history)
    axes[0].plot(episodes[::slice_num], average)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('learning rate')
    axes[1].set_yscale('log')
    axes[1].plot(episodes, lr_history)
#    plt.show()
    fig.tight_layout()
    fig.savefig(directory+"/history.png", dpi=200)
    plt.close()
