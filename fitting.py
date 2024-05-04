import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

#with open('m.txt', 'r') as f:
#    m = f.read().split()
data = pd.read_csv("m.txt", header=None, delim_whitespace=True)

#m = np.array(m, dtype=float)
m = data.values
mz = m[:, 2]

#mz = []
#t = []
#i = 0

dt = 5e-12 # [s]

#for n in range(1203):
#    if n%3 == 2:
#        if m[n] < 0:
#            t.append(dt * i)
#            mz.append(m[n])
#            if dt*i > 0.8e-9:
#                break
#        i += 1
#
#    n += 1

t = np.linspace(0, 2e-9, mz.shape[0])
t_pred = np.linspace(np.min(t), 2e-9, 1000)

def plot(x, y, label):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Magnetization')
    ax1.set_xlim(0, 2e-9)
    ax1.set_ylim(-1, 0)
#    ax1.set_xscale('log')
#    ax1.set_yscale('log')

    for i in range(len(x)):
#        x = np.array(x[i])
#        y = np.array(y[i])
#        y = -y
#        x = x[i]
#        y = y[i]
        if i == 0:
            print(x[i].shape, y[i].shape)
            ax1.scatter(x[i], y[i], label=label[i])
        else:
            ax1.plot(x[i], y[i], label=label[i], c='red', linestyle='dashed', linewidth=2)
    ax1.legend()

    fig.tight_layout()
    plt.show()
    plt.close()
    return

#def Exp(t ,A , ti, t0):
def Exp(t ,A , ti):
#    t0 = 56e-11
#    return A * np.exp(-(t-t0)/ti) - 1
    return A * t-ti

#guess_exp = [1, 1e-10, 56e-11]
guess_exp = [1, 1e-10]
popt, pcov = curve_fit(Exp, t, mz, p0=guess_exp)
print('exp')
print('optimized parameters=', popt)
print('error=', np.sqrt(np.diag(pcov)))
m = Exp(t_pred, *popt)
plot([t, t_pred], [mz, m], ['data', 'curve-fit'])

def Pow(t, B, tr, a):
    t0 = 56e-11
    return B * pow(1/((t-t0)/tr+1), a) - 1
guess_pow = [1, 1e-10, 1]
popt, pcov = curve_fit(Pow, t, mz, p0=guess_pow)
print('pow')
print('optimized parameters=', popt)
print('error=', np.sqrt(np.diag(pcov)))
m = Pow(t_pred, *popt)
plot([t, t_pred], [mz, m], ['data', 'curve-fit'])
