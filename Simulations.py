import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from Generator import LewisShedler
from EstimatorsDiscretize import TSVD, Landweber


ass = [1, 3, 5, 7, 10, 20, 50]
sizes = [100, 1000, 5000, 10000, 50000]
replicate = 5

def f(t):
    return t*(1-t)

for a in ass:
    for size in sizes:
        for i in range(replicate):

            def kernel(s, t):
                return np.exp(-a*np.abs(t-s))

            def g(s):
                return (2/a*s*(1-s)+1/a**2*(np.exp(-a*s) + np.exp(-a*(1-s))) + 2/a**3*(np.exp(-a*s) + np.exp(-a*(1-s)) -2))*size

            gen = LewisShedler(g, 1, 0)
            obs = gen.generate()

            tsvd = TSVD(kernel, 0, 1, 10000, obs, size, adjoint=True, tau=1)
            tsvd.estimate()

            plt.plot(f(cp.asnumpy(tsvd.grid)), label='true')
            plt.plot(cp.asnumpy(tsvd.solution), label='solution')
            plt.legend()
            plt.savefig('a{}_size{}_replication.png'.format(a, size, i))