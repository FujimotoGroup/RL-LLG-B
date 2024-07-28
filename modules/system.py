import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 18


class Magnetization:
    def __init__(self, m0:np.array):
        self.m = m0

class Dynamics(Magnetization):
    def __init__(self, dt:float, alphaG:float, anisotropy:np.array, H_shape, m0:np.array, eps:float = 1e-9):
        super().__init__(m0)

        self.gamma = 1.76e7 # [rad / Oe s]
        self.eps = eps
        self.dt = dt
        self.alphaG = alphaG
        self.anisotropy = anisotropy
        self.H_shape = H_shape

    def LLG(self, magnetization:np.array, field:np.array) -> np.array:
        H = self.anisotropy * magnetization + field - self.H_shape * magnetization
        g = self.gamma / (1e0 + self.alphaG**2)
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
    axes[0].plot(t, m[:,0], label='$m_x$')
    axes[0].plot(t, m[:,1], label='$m_y$')
    axes[0].plot(t, m[:,2], label='$m_z$')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Magnetization')
    axes[0].legend()

    axes[1].plot(t, h[:,0], label='$h_x$')
    axes[1].plot(t, h[:,1], label='$h_y$')
    axes[1].plot(t, h[:,2], label='$h_z$')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Magnetic Field [Oe]')
    axes[1].legend()

    if not(issave):
        plt.show()
    else:
        return fig

def save_episode(episode:int, t:list, m:list, h:list, directory:str):
    fig = plot(t,m,h, issave=True)
    fig.tight_layout()
    fig.savefig(directory+"/episode{:0=5}.png".format(episode), dpi=200)
    plt.close()

def save_reward_history(reward_history:list, directory:str):
    reward_history_array = np.array(reward_history)
    episodes = np.arange(len(reward_history))

    slice_num = 20
    average = [ reward_history_array[i:i+slice_num].mean() for i in episodes[::slice_num]]

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(reward_history)), reward_history, label='Reward of 1 Episode')
    plt.plot(episodes[::slice_num], average, label='Average Reward of 20 Episode')
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(directory+"/reward_history.png", dpi=200)
    plt.close()

def save_loss_history(loss_history:list, directory:str):
    plt.xlabel('Episode')
    plt.ylabel('loss')
    plt.plot(range(len(loss_history)), loss_history)
    plt.savefig(directory+"/loss_history.png", dpi=200)
    plt.close()

def save_loss_pi_history(loss_pi_history:list, directory:str):
    plt.xlabel('Episode')
    plt.ylabel('loss_pi')
    plt.plot(range(len(loss_pi_history)), loss_pi_history)
    plt.savefig(directory+"/loss_pi_history.png", dpi=200)
    plt.close()

def save_prob(episode:int, agent, directory:str):
    n = agent.action_size
    mesh_num = 20
    theta_u = np.linspace(0e0, np.pi/2e0, mesh_num)
    phi = np.linspace(0e0, 2e0*np.pi, mesh_num)
#    theta, phi = np.meshgrid(theta, phi)
    probs = []
    for t in theta_u:
        prob = []
        for p in phi:
            x = np.sin(t)*np.cos(p)
            y = np.sin(t)*np.sin(p)
            z = np.cos(t)
            S = np.array([x, y, z])
            S = S[np.newaxis,:]
            pi = agent.qnet(S)[0].data
            prob.append(pi)
        probs.append(prob)
    probs_u = np.array(probs)

    theta, phi = np.meshgrid(theta_u, phi)
    x_u = np.sin(theta)*np.cos(phi)
    y_u = np.sin(theta)*np.sin(phi)

    theta_d = np.linspace(np.pi/2e0, np.pi, mesh_num)
    phi = np.linspace(0e0, 2e0*np.pi, mesh_num)
    probs = []
    for t in theta_d:
        prob = []
        for p in phi:
            x = np.sin(t)*np.cos(p)
            y = np.sin(t)*np.sin(p)
            z = np.cos(t)
            S = np.array([x, y, z])
            S = S[np.newaxis,:]
            pi = agent.qnet(S)[0].data
            prob.append(pi)
        probs.append(prob)
    probs_d = np.array(probs)

    theta, phi = np.meshgrid(theta_d, phi)
    x_d = np.sin(theta)*np.cos(phi)
    y_d = np.sin(theta)*np.sin(phi)

    fig, axes = plt.subplots(2, n, figsize=(5*n, 12))
#    window = 2e0/n
#    levels = np.linspace(0e0, window, 100)
#    norm = mpl.colors.Normalize(vmin=0e0, vmax=window)
    for i in np.arange(n):
        axes[0][i].set_xlim([-1e0, 1e0])
        axes[0][i].set_ylim([-1e0, 1e0])
        axes[0][i].axis('equal')
#        axes[0][i].contourf(x_u, y_u, probs_u[:,:,i], levels=levels, cmap="bwr", extend="max")
        ax = axes[0][i].contourf(x_u, y_u, probs_u[:,:,i])
        cax = make_axes_locatable(axes[0][i]).append_axes("right", size="3%", pad=0.1)
        fig.colorbar(ax, cax=cax)

        axes[1][i].set_xlim([-1e0, 1e0])
        axes[1][i].set_ylim([-1e0, 1e0])
        axes[1][i].axis('equal')
#        ax = axes[1][i].contourf(x_d, y_d, probs_d[:,:,i], levels=levels, cmap="bwr", extend="max")
        ax = axes[1][i].contourf(x_d, y_d, probs_d[:,:,i])
        cax = make_axes_locatable(axes[1][i]).append_axes("right", size="3%", pad=0.1)
        fig.colorbar(ax, cax=cax)

#    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="bwr"),ax=axes.ravel().tolist(), extend="max")

#    fig.tight_layout()
    fig.savefig(directory+"/prob_episode{:0=5}.png".format(episode), dpi=200)
    plt.close()
