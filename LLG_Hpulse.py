import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import copy
import os


gamma = 1.76e7


def LLG(S:np.array, B:np.array, alpha:float) -> np.array:
    S_norm = np.linalg.norm(S, ord=2)
    g = gamma / (1e0 - (alpha/S_norm)**2)
    BxS = np.cross(B, S)
    s = g*BxS - g*alpha/S_norm*np.cross(S, BxS)
    return s


def RungeKutta(S:np.array, B:np.array, alpha:float, dt:float) -> np.array:
    s1 = LLG(S, B, alpha)
    s2 = LLG(S + dt*s1/2, B, alpha)
    s3 = LLG(S + dt*s2/2, B, alpha)
    s4 = LLG(S + dt*s3, B, alpha)

    s = S + dt/6 * (s1 + 2*s2 + 2*s3 + s4)

    return s


def main():
    directory = "Pulse"
    os.mkdir(directory)

    t_pulse = 0.25e-9
    t_end = 1.5e-9
    dt = t_end / 1e3
    n = int(t_end / dt)
    n_pulse = int(t_pulse / dt)

    alpha = 0.008

    H_ext = np.array([80e0, 0e0, 0e0])
    H_ani = np.array([0e0, 0e0, 0e0])
    H_shape = np.array([4*np.pi*0.012*1000, 4*np.pi*0.98*1000, 4*np.pi*0.008*1000])

    S = np.array([0e0, 0e0, 1e0])
    H_ext0 = H_ext

    t = []
    s = []
    h = []

    t.append(0)
    s.append(S)
    h.append(np.array([0e0, 0e0, 0e0]))

    for i in np.arange(1,n):
        time = i*dt
        t.append(time)
        if i > n_pulse:
            H_ext = np.array([0e0, 0e0, 0e0])
        Heff = -H_ext -H_ani*S +H_shape*S
        S = RungeKutta(S, Heff, alpha, dt)
        s.append(S)
        h.append(copy(H_ext))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    t = np.array(t)
    s = np.array(s)
    h = np.array(h)    
    axes[0].set_ylim([-1, 1])
    axes[0].plot(t, s[:,0], label='m_x')
    axes[0].plot(t, s[:,1], label='m_y')
    axes[0].plot(t, s[:,2], label='m_z')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Magnetization')
    axes[0].legend()

    axes[1].plot(t, h[:,0], label='h_x')
    axes[1].plot(t, h[:,1], label='h_y')
    axes[1].plot(t, h[:,2], label='h_z')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Magnetic Field [Oe]')
    axes[1].legend()

    fig.savefig(directory+"/Pulse.png", dpi=200)
    plt.close()

    with open(directory + "/options.txt", mode='w') as f:
        f.write(f"""
    alphaG = {alpha}
    t_pulse = {t_pulse} [s]
    H_ext = {H_ext0} [Oe]
    H_ani = {H_ani} [Oe]
    H_shape = {H_shape} [Oe]
    """)

#    fig = plt.figure(figsize = (8, 8))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.set_xlabel("x", size = 14)
#    ax.set_ylabel("y", size = 14)
#    ax.set_zlabel("z", size = 14)
#    ax.plot(s[:,0], s[:,1], s[:,2], color = "red")
#    plt.show()


if __name__ == '__main__':
    main()
