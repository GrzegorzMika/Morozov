import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from Generator import LewisShedler
from EstimatorsDiscretize import TSVD, Landweber


def f(t):
    return t * (1 - t)


a = 5


def kernel(s, t):
    return np.exp(-a * np.abs(t - s))


size = 200000


def g(s):
    return (2 / a * s * (1 - s) + 1 / a ** 2 * (np.exp(-a * s) + np.exp(-a * (1 - s))) + 2 / a ** 3 * (
                np.exp(-a * s) + np.exp(-a * (1 - s)) - 2)) * size


gen = LewisShedler(g, 1, 0)
obs = gen.generate()

landweber = Landweber(kernel, 0, 1, 25000, obs, size, adjoint=True, tau=1)
landweber.estimate()

plt.rcParams['figure.figsize'] = 20, 10
plt.ylim(-0.1, 0.5)
plt.plot(f(cp.asnumpy(landweber.grid)), label='true')
plt.plot(cp.asnumpy(landweber.solution), label='solution')
plt.legend()
plt.show()