import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


gamma = 1.76e7
alpha = 0.01
K = 100


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
    t0 = 0e0
    t1 = 5e-9
    n = 5000

    t, dt = np.linspace(t0, t1, n, retstep=True)
    s = np.empty((n,3))

    s[0] = np.array([0e0, 0e0, 1e0])
    H = np.array([500, 0e0, 0e0])
    B = np.array([0e0, 0e0, - K*s[0,2]]) - H

    for i in np.arange(1,n):
#        s[i] = RungeKutta(s[i-1], B, alpha, dt)
        s[i] = RungeKutta(s[i-1], -H, alpha, dt)

    t = np.array(t)
    fig, ax = plt.subplots()
    ax.plot(t, s[:,0], label='m_x')
    ax.plot(t, s[:,1], label='m_y')
    ax.plot(t, s[:,2], label='m_z')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Magnetization')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x", size = 14)
    ax.set_ylabel("y", size = 14)
    ax.set_zlabel("z", size = 14)
    ax.plot(s[:,0], s[:,1], s[:,2], color = "red")
    plt.show()


if __name__ == '__main__':
    main()
