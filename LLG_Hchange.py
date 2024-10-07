import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


gamma = 1.76e7
alpha = 0.01


def LLG(S:np.array, B:np.array, alpha:float) -> np.array:
    g = gamma / (1e0 - alpha**2)
    BxS = np.cross(B, S)
    s = - g*BxS - g*alpha*np.cross(S, BxS)
    return s


def LLGlinear(t:float, K:float, B0:float):
    g = gamma
    x = B0/K * (t-np.sin(g*K*t)/(g*K))
    y = B0/(K*K*g)*(1-np.cos(g*K*t))
    return np.array([x,y,np.sqrt(1-x**2-y**2)])


def RungeKutta(S:np.array, B:np.array, alpha:float, dt:float) -> np.array:
    s1 = LLG(S, B, alpha)
    s2 = LLG(S + dt*s1/2, B, alpha)
    s3 = LLG(S + dt*s2/2, B, alpha)
    s4 = LLG(S + dt*s3, B, alpha)

    s = S + dt/6 * (s1 + 2*s2 + 2*s3 + s4)

    return s


def main():
    t0 = 0e0
    t1 = 5e-10
    dH = 5000
    B0 = dH/t1
    dt = 5e-13
    n = int(t1/dt)
    K = 5400

    s = np.empty((n,3))
    B = np.empty((n,3))
    analytic = np.empty((n, 3))

    s[0] = np.array([0, 0, 1e0])
    analytic[0] = s[0]
    B[0] = np.array([0, 0, K])
    H = np.array([0e0, 0e0, 0e0])

    for i in np.arange(1,n):
        H += np.array([dH/n, 0e0, 0e0])
        B[i] = np.array([0e0, 0e0, K*s[i-1,2]]) + H
        s[i] = RungeKutta(s[i-1], B[i-1], alpha, dt)
        analytic[i] = LLGlinear(i*dt, K, B0)

    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("mx", size = 14)
    ax.set_ylabel("my", size = 14)
    ax.set_zlabel("mz", size = 14)
    ax.plot(s[:,0], s[:,1], s[:,2], color = "red", label='数値計算')
    ax.plot(analytic[:,0], analytic[:,1], analytic[:,2], color = "blue", label='線形近似式')
    ax.legend()
    plt.show()
    plt.close

    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("Hx", size = 14)
    ax.set_ylabel("Hy", size = 14)
    ax.set_zlabel("Hz", size = 14)
    ax.plot(B[:,0], B[:,1], B[:,2], color = "blue")
    plt.show()
    plt.close


if __name__ == '__main__':
    main()
