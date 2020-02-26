import numpy as np
import pandas as pd
from tqdm import tqdm
from Estimators import Landweber, Tikhonov
from Generator import LewisShedler

print('Simulation 1...')
n_size = 500


def lam(t):
    return n_size * (1 + np.sin(20 * t))


def kernel(x, y):
    return np.where(x < y, (1 + np.sin(20 * y)), (1 + np.sin(20 * y)))


def true(s):
    return np.where(s > 1, 1, 1)


generator = LewisShedler(lam, lower=0, upper=1)
observations = generator.generate()
landweber = Tikhonov(kernel=kernel, lower=0, upper=1, grid_size=20000, observations=observations, sample_size=n_size,
                      adjoint=False, order=2)
solutions = []
error = []
steps = []
for i in tqdm(range(100)):
    landweber.observations = generator.generate()
    landweber.refresh()
    landweber.estimate()
    solutions.append(landweber.solution)
    error.append(landweber.L2norm(landweber.solution, true(landweber.grid)))
results = pd.DataFrame({'error': error, 'solutions': solutions})
results.to_csv('simulation1_t2.csv')

# print('Simulation 2...')
# n_size = 500
#
#
# def lam(t):
#     return n_size * (2 / 3 * (np.power(t + 1, 3 / 2) - np.power(t, 3 / 2)))
#
#
# def kernel(x, y):
#     return np.sqrt(x + y)
#
#
# def true(s):
#     return np.where(s > 1, 1, 1)
#
#
# generator = LewisShedler(lam, lower=0, upper=1)
# observations = generator.generate()
# landweber = Landweber(kernel=kernel, lower=0, upper=1, grid_size=20000, observations=observations, sample_size=n_size,
#                       adjoint=True, relaxation=5)
# solutions = []
# error = []
# steps = []
# for i in tqdm(range(1000)):
#     landweber.observations = generator.generate()
#     landweber.refresh()
#     landweber.estimate()
#     solutions.append(landweber.solution)
#     error.append(landweber.L2norm(landweber.solution, true(landweber.grid)))
# results = pd.DataFrame({'error': error, 'solutions': solutions})
# results.to_csv('simulation2.csv')
#
# print('Simulation 3...')
# n_size = 500
#
#
# def lam(t):
#     return np.divide(2, (t + 2) * (t ** 2 + 4 * t + 8)) * n_size
#
#
# def kernel(x, y):
#     return np.exp(-x * y)
#
#
# def true(s):
#     return np.exp(-2 * s) * np.square(np.sin(s))
#
#
# generator = LewisShedler(lam, lower=0, upper=1)
# observations = generator.generate()
# landweber = Landweber(kernel=kernel, lower=0, upper=1, grid_size=20000, observations=observations, sample_size=n_size,
#                       adjoint=True, relaxation=5)
# solutions = []
# error = []
# steps = []
# for i in tqdm(range(1000)):
#     landweber.observations = generator.generate()
#     landweber.refresh()
#     landweber.estimate()
#     solutions.append(landweber.solution)
#     error.append(landweber.L2norm(landweber.solution, true(landweber.grid)))
# results = pd.DataFrame({'error': error, 'solutions': solutions})
# results.to_csv('simulation3.csv')
#
# print('Simulation 4...')
# n_size = 500
#
#
# def lam(t):
#     return np.exp(-np.pi * np.square(t)) * n_size
#
#
# def kernel(x, y):
#     return 1 / np.sqrt(2 * np.pi) * (np.cos(x * y) + np.sin(x * y))
#
#
# def true(s):
#     return np.exp(-np.pi * np.square(s))
#
#
# generator = LewisShedler(lam, lower=0, upper=1)
# observations = generator.generate()
# landweber = Landweber(kernel=kernel, lower=0, upper=1, grid_size=20000, observations=observations, sample_size=n_size,
#                       adjoint=True, relaxation=5)
# solutions = []
# error = []
# steps = []
# for i in tqdm(range(1000)):
#     landweber.observations = generator.generate()
#     landweber.refresh()
#     landweber.estimate()
#     solutions.append(landweber.solution)
#     error.append(landweber.L2norm(landweber.solution, true(landweber.grid)))
# results = pd.DataFrame({'error': error, 'solutions': solutions})
# results.to_csv('simulation4.csv')
#
# print('Simulation 5...')
# n_size = 500
#
#
# def lam(t):
#     return np.add(np.subtract(t / 3, t ** 3), 2 * t ** 4 / 3) * n_size
#
#
# def kernel(x, y):
#     return np.where(x >= y, 2 * y, 0)
#
#
# def true(s):
#     return np.multiply(s, 1 - s)
#
#
# generator = LewisShedler(lam, lower=0, upper=1)
# observations = generator.generate()
# landweber = Landweber(kernel=kernel, lower=0, upper=1, grid_size=20000, observations=observations, sample_size=n_size,
#                       adjoint=False, relaxation=5)
# solutions = []
# error = []
# steps = []
# for i in tqdm(range(1000)):
#     landweber.observations = generator.generate()
#     landweber.refresh()
#     landweber.estimate()
#     solutions.append(landweber.solution)
#     error.append(landweber.L2norm(landweber.solution, true(landweber.grid)))
# results = pd.DataFrame({'error': error, 'solutions': solutions})
# results.to_csv('simulation5.csv')
#
# print('Simulation 6...')
# n_size = 500
#
#
# def lam(t):
#     return np.exp(-t) * ((np.pi * (1 + np.e)) / (1 + np.pi ** 2)) * n_size
#
#
# def kernel(x, y):
#     return np.exp(y - x)
#
#
# def true(s):
#     return np.sin(np.pi * s)
#
#
# generator = LewisShedler(lam, lower=0, upper=1)
# observations = generator.generate()
# landweber = Landweber(kernel=kernel, lower=0, upper=1, grid_size=20000, observations=observations, sample_size=n_size,
#                       adjoint=False, relaxation=5)
# solutions = []
# error = []
# steps = []
# for i in tqdm(range(1000)):
#     landweber.observations = generator.generate()
#     landweber.refresh()
#     landweber.estimate()
#     solutions.append(landweber.solution)
#     error.append(landweber.L2norm(landweber.solution, true(landweber.grid)))
# results = pd.DataFrame({'error': error, 'solutions': solutions})
# results.to_csv('simulation6.csv')
