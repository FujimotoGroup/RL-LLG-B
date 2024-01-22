import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_energy(m, dynamics):
    m = np.array(m)
    mx = np.arange(-1, 1, 0.01)
    my = np.arange(-1, 1, 0.01)
    X, Y = np.meshgrid(mx, my)
    X[X**2+Y**2 > 1] = np.nan
    Y[X**2+Y**2 > 1] = np.nan

    def energy(X, Y):
#        E = - dynamics.easy*(1-X**2-Y**2)**2 + dynamics.hard*Y**2
#        E = - dynamics.easy*(1-X**2-Y**2) + dynamics.hard*Y**2
        E = - (dynamics.anisotropy[0]*X**2 + dynamics.anisotropy[1]*Y**2 + dynamics.anisotropy[2]*(1-X**2-Y**2))
#        np.savetxt('energy.txt', E)
        return E

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
    ax.plot_surface(X, Y, energy(X, Y), alpha = 0.3)
    ax.plot(m[:, 0], m[:, 1], energy(m[:, 0], m[:, 1]))
#    print(energy(m[:, 0], m[:, 1]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_zlim(-500, 0)
    plt.show()
    plt.close()


def plot_3d(m):
    m = np.array(m)
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x", size = 14)
    ax.set_ylabel("y", size = 14)
    ax.set_zlabel("z", size = 14)
    ax.plot(m[:, 0], m[:, 1], m[:, 2], color = "red")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])
    plt.show() 
    plt.close()