import numpy as np
from Estimators import Landweber
from Generator import LewisShedler

n_size = 1000


def lam(t):
    return np.where(t < 0.5, n_size, n_size * (np.sin(t ** 2 * np.pi) + 1))


def kernel(x, y):
    return np.where(x < y, 1, 1)


generator = LewisShedler(lam, lower=0, upper=1)
observations = generator.generate()
landwerber = Landweber(kernel, lower=, upper=1, grid_size=10000, observations=observations,
                       sample_size=n_size, relaxation=0.05, max_iter=1000)

solutions = []
error = []

for i in range(100):
    landwerber.observations = generator.generate()
    landwerber.refresh()
    landwerber.estimate(compute=True)
    solutions.append(landwerber.solution)
    error.append(landwerber.L2norm(landwerber.solution, np.repeat([1], landwerber.solution.shape[0]), sqrt=True).compute())

with open('solutions.txt', 'w') as f:
    for item in solutions:
        f.write("%s\n" % item)

with open('errors.txt', 'w') as f:
    for item in error:
        f.write("%s\n" % item)