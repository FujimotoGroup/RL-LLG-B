import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 18

directory = "H=x_dH=10_da=0.01_ani=(0,-10000,100)"  # input

m_opt = np.loadtxt(directory+"/m.txt")
h_opt = np.loadtxt(directory+"/h.txt")
#m_one = np.loadtxt(directory+"/100th_m.txt")
#h_one = np.loadtxt(directory+"/100th_h.txt")
#m_two = np.loadtxt(directory+"/200th_m.txt")
#h_two = np.loadtxt(directory+"/200th_h.txt")
time = np.loadtxt(directory+"/t.txt")
history = np.loadtxt(directory+"/reward history.txt")
slope = -6783408045.165612  # input
segment = 2.8868502483333027  # input
t_limit = 2e-9

def fig_ep100_ep200(history,time, m_opt, h_opt, m_one, h_one, m_two, h_two):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    reward_history_array = np.array(history)
    episodes = np.arange(len(history))
    slice_num = 20
    average = [ reward_history_array[i:i+slice_num].mean() for i in episodes[::slice_num]]

    x = np.linspace(0, t_limit, 1000)
    y = slope*x + segment

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Rewards')
    axes[0].plot(episodes, history, label='Rewards for each Episode')
    axes[0].plot(episodes[::slice_num], average, label='Average Rewards for 20 Episodes')
    axes[0].legend(fontsize=14)

    axes[1].set_ylim([-1e0,1e0])
    l11, = axes[1].plot(time, m_opt[:,0], label='$m_x$')
    l12, = axes[1].plot(time, m_opt[:,1], label='$m_y$')
    l13, = axes[1].plot(time, m_opt[:,2], label='$m_z$')
    l14, = axes[1].plot(x, y, color='darkgreen', linestyle='dashed', label='tangent')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Magnetization')
    twin = axes[1].twinx()
    twin.set_ylim([-500,500])
    l15, = twin.plot(time, h_opt[:,0], label='$h_x$', color='deepskyblue')
    twin.set_ylabel('Magnetic Field [Oe]')
    axes[1].legend(handles=[l11, l12, l13, l15])
#    axes[1].legend(handles=[l13, l14, l15])

    axes[2].set_ylim([-1e0,1e0])
    l21, = axes[2].plot(time, m_one[:,0], label='$m_x$')
    l22, = axes[2].plot(time, m_one[:,1], label='$m_y$')
    l23, = axes[2].plot(time, m_one[:,2], label='$m_z$')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Magnetization')
    twin = axes[2].twinx()
    twin.set_ylim([-500,500])
    l24, = twin.plot(time, h_one[:,0], label='$h_x$', color='deepskyblue')
    twin.set_ylabel('Magnetic Field [Oe]')
    axes[2].legend(handles=[l21, l22, l23, l24])
#    axes[2].legend(handles=[l23, l24])

    axes[3].set_ylim([-1e0,1e0])
    l31, = axes[3].plot(time, m_two[:,0], label='$m_x$')
    l32, = axes[3].plot(time, m_two[:,1], label='$m_y$')
    l33, = axes[3].plot(time, m_two[:,2], label='$m_z$')
    axes[3].set_xlabel('Time [s]')
    axes[3].set_ylabel('Magnetization')
    twin = axes[3].twinx()
    twin.set_ylim([-500,500])
    l34, = twin.plot(time, h_two[:,0], label='$h_x$', color='deepskyblue')
    twin.set_ylabel('Magnetic Field [Oe]')
    axes[3].legend(handles=[l31, l32, l33, l34])
#    axes[3].legend(handles=[l33, l34])

    fig.tight_layout()
#    fig.savefig(directory+"/fig3.pdf", dpi=200)
    fig.savefig(directory+"/fig.png", dpi=200)
    plt.close()


def save_reward_history(history):
    reward_history_array = np.array(history)
    episodes = np.arange(len(history))
    slice_num = 20
    average = [ reward_history_array[i:i+slice_num].mean() for i in episodes[::slice_num]]

    plt.figure(figsize=(6,6))
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(episodes, history, label='Rewards for each Episode')
    plt.plot(episodes[::slice_num], average, label='Average Rewards for 20 Episodes')
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(directory+"/history.png", dpi=200)
    plt.close()


def fig(history,time, m_opt, h_opt):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes = axes.flatten()

    reward_history_array = np.array(history)
    episodes = np.arange(len(history))
    slice_num = 20
    average = [ reward_history_array[i:i+slice_num].mean() for i in episodes[::slice_num]]

    x = np.linspace(0, t_limit, 1000)
    y = slope*x + segment

    axes[0].plot(time, h_opt[:,0], label='$h_x$', color='deepskyblue')
    axes[0].plot(time, h_opt[:,1], label='$h_y$', color='goldenrod')
    axes[0].plot(time, h_opt[:,2], label='$h_z$', color='lawngreen')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Magnetic Field [Oe]')
    axes[0].legend()

    axes[1].set_ylim([-1e0,1e0])
    axes[1].plot(time, m_opt[:,0], label='$m_x$')
    axes[1].plot(time, m_opt[:,1], label='$m_y$')
    axes[1].plot(time, m_opt[:,2], label='$m_z$')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Magnetization')
    axes[1].legend()
    axes[1].plot(x, y, color='darkgreen', linestyle='dashed')

    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Rewards')
    axes[2].plot(episodes, history, label='Rewards for each Episode')
    axes[2].plot(episodes[::slice_num], average, label='Average Rewards for 20 Episodes')
    axes[2].legend(fontsize=14)

    fig.tight_layout()
#    fig.savefig(directory+"/fig3.pdf", dpi=200)
    fig.savefig(directory+"/FIG.png", dpi=200)
    plt.close()


#fig_ep100_ep200(history,time, m_opt, h_opt, m_one, h_one, m_two, h_two)
#save_reward_history(history)
fig(history,time, m_opt, h_opt)