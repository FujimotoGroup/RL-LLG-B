import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


gamma = 1.76e11
alpha = 0.01
K = 1e-3


def LLG(S:np.array, B:np.array, alpha:float) -> np.array:
    S_norm = np.linalg.norm(S, ord=2)
    g = gamma / (1e0 - (alpha/S_norm)**2)
    BxS = np.cross(B, S)
    s = g*BxS - g*alpha/S_norm*np.cross(S, BxS)
    return s


def Line(S:float, B:np.array, n_x:float, n_y:float):
    g = gamma / (1e0 + alpha**2)
    dn_x = g * (-alpha*S*B[0] + S*B[1] + alpha*B[2]*n_x - B[2]*n_y)
    dn_y = g * (-S*B[0] - alpha*S*B[1] + B[2]*n_x + alpha*B[2]*n_y)
    return dn_x, dn_y


def RungeKutta1(S:np.array, B:np.array, alpha:float, dt:float) -> np.array:
    s1 = LLG(S, B, alpha)
    s2 = LLG(S + dt*s1/2, B, alpha)
    s3 = LLG(S + dt*s2/2, B, alpha)
    s4 = LLG(S + dt*s3, B, alpha)

    s = S + dt/6 * (s1 + 2*s2 + 2*s3 + s4)

    return s


def RungeKutta2(S:float, B:np.array, n_x:float, n_y:float, dt:float):
    dn_x1, dn_y1 = Line(S, B, n_x, n_y)
    dn_x2, dn_y2 = Line(S, B, n_x + dt*dn_x1/2, n_y + dt*dn_y1/2)
    dn_x3, dn_y3 = Line(S, B, n_x + dt*dn_x2/2, n_y + dt*dn_y2/2)
    dn_x4, dn_y4 = Line(S, B, n_x + dt*dn_x3, n_y + dt*dn_y3)

    dn_x = n_x + dt/6 * (dn_x1 + 2*dn_x2 + 2*dn_x3 + dn_x4)
    dn_y = n_y + dt/6 * (dn_y1 + 2*dn_y2 + 2*dn_y3 + dn_y4)

    return dn_x, dn_y


def main():
    t0 = 0e0
    t1 = 3e-6
    n = 5000

    t, dt = np.linspace(t0, t1, n, retstep=True)
    s = np.empty((n,3))
    n_x = np.empty(n)
    n_y = np.empty(n)

    s[0] = np.array([1e-3, 1e-3, 1e0])
    n_x[0] = s[0,0]
    n_y[0] = s[0,1]
    H = np.array([1e-3, 0e0, 0e0])
    B = np.array([0e0, 0e0, - K*s[0,2]]) - H

    for i in np.arange(1,n):
        s[i] = RungeKutta1(s[i-1], B, alpha, dt)
        n_x[i], n_y[i] = RungeKutta2(s[0,2], B, n_x[i-1], n_y[i-1], dt)


    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x", size = 14)
    ax.set_ylabel("y", size = 14)
    ax.set_zlabel("z", size = 14)
    ax.plot(s[:,0], s[:,1], s[:,2], color = "red")
    plt.show()


if __name__ == '__main__':
    main()
