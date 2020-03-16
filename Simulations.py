import numpy as np
import cupy as cp
import pandas as pd
from tqdm import tqdm
from EstimatorsDiscretize import Landweber, Tikhonov, TSVD
from Generator import LewisShedler

print('Simulation 3...')
n_size = 500


def lam(t):
    return np.divide(2, (t + 2) * (t ** 2 + 4 * t + 8)) * n_size


def kernel(x, y):
    return np.exp(-x * y)


def true(s):
    return np.exp(-2 * s) * np.square(np.sin(s))


generator = LewisShedler(lam, lower=0, upper=1)
observations = generator.generate()
tsvd = TSVD(kernel=kernel, lower=0, upper=1, grid_size=10000, observations=observations, sample_size=n_size,
            adjoint=False)

error = []
for i in tqdm(range(50)):
    tsvd.observations = generator.generate()
    tsvd.refresh()
    tsvd.estimate()
    error.append(tsvd.L2norm(tsvd.current, cp.asarray(true(tsvd.grid))))
results = pd.DataFrame({'error': error})
results.to_csv('simulation1_tsvd.csv')
del tsvd
cp._default_memory_pool.free_all_blocks()
print('Simulation 5...')
n_size = 500


def lam(t):
    return np.add(np.subtract(t / 3, t ** 3), 2 * t ** 4 / 3) * n_size


def kernel(x, y):
    return np.where(x >= y, 2 * y, 0)


def true(s):
    return np.multiply(s, 1 - s)


generator = LewisShedler(lam, lower=0, upper=1)
observations = generator.generate()
tsvd = TSVD(kernel=kernel, lower=0, upper=1, grid_size=10000, observations=observations, sample_size=n_size,
            adjoint=False)

error = []
for i in tqdm(range(50)):
    tsvd.observations = generator.generate()
    tsvd.refresh()
    tsvd.estimate()
    error.append(tsvd.L2norm(tsvd.current, cp.asarray(true(tsvd.grid))))
results = pd.DataFrame({'error': error})
results.to_csv('simulation2_tsvd.csv')
del tsvd
cp._default_memory_pool.free_all_blocks()
print('Simulation 6...')
n_size = 500


def lam(t):
    return np.exp(-t) * ((np.pi * (1 + np.e)) / (1 + np.pi ** 2)) * n_size


def kernel(x, y):
    return np.exp(y - x)


def true(s):
    return np.sin(np.pi * s)


generator = LewisShedler(lam, lower=0, upper=1)
observations = generator.generate()
tsvd = TSVD(kernel=kernel, lower=0, upper=1, grid_size=10000, observations=observations, sample_size=n_size,
            adjoint=False)

error = []
for i in tqdm(range(50)):
    tsvd.observations = generator.generate()
    tsvd.refresh()
    tsvd.estimate()
    error.append(tsvd.L2norm(tsvd.current, cp.asarray(true(tsvd.grid))))
results = pd.DataFrame({'error': error})
results.to_csv('simulation3_tsvd.csv')
del tsvd
cp._default_memory_pool.free_all_blocks()