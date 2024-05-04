import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable


m = np.loadtxt('m copy.txt')
anisotropy = np.array([0e0, 0e0, 540e0]) # optional

#energy
mx = np.arange(-1, 1, 0.01)
my = np.arange(-1, 1, 0.01)
X, Y = np.meshgrid(mx, my)
X[X**2+Y**2 > 1] = np.nan
Y[X**2+Y**2 > 1] = np.nan

def energy(X, Y):
    E = - (anisotropy[0]*X**2 + anisotropy[1]*Y**2 + anisotropy[2]*(1-X**2-Y**2))
    return E

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
ax.plot_surface(X, Y, energy(X, Y), alpha = 0.3)
ax.plot(m[:, 0], m[:, 1], energy(m[:, 0], m[:, 1]))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_zlim(-500, 0)
plt.show()
plt.close()

#m
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("x", size = 14)
ax.set_ylabel("y", size = 14)
ax.set_zlabel("z", size = 14)
ax.plot(m[:, 0], m[:, 1], m[:, 2], color = "blue")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
#ax.set_xticks([-1, -0.5, 0, 0.5, 1])
#ax.set_yticks([-1, -0.5, 0, 0.5, 1])
#ax.set_zticks([-1, -0.5, 0, 0.5, 1])
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_zticks([-1, 0, 1])
plt.show() 
plt.close()