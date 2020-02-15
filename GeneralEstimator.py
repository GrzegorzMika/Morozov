from typing import Callable
import dask.array as da
import numpy as np
from Operator import Quadrature


class Estimator(Quadrature):
    def __init__(self, kernel, lower, upper, grid_size, observations, sample_size, quadrature):
        Quadrature.__init__(self, lower, upper, grid_size)
        try:
            kernel(np.array([1, 2]), np.array([1, 2]))
            self.kernel: Callable = kernel
        except ValueError:
            print('Force vectorization of kernel')
            self.kernel: Callable = np.vectorize(kernel)
        self.lower: float = float(lower)
        self.upper: float = float(upper)
        self.grid_size: int = grid_size
        self.quadrature: Callable = getattr(super(), quadrature)
        self.observations = observations
        self.sample_size: int = sample_size
        self.delta = None
        self.q_estimator = None

    def estimate_q(self, compute: bool = False):
        grid = da.linspace(self.lower, self.upper, self.grid_size)
        estimator = [da.sum(self.kernel(x, self.observations)) / self.sample_size for x in grid]
        estimator = da.stack(estimator, axis=0)
        if compute:
            estimator = estimator.compute()
        self.q_estimator = estimator
        return estimator

    def estimate_delta(self, compute: bool = False):
        grid = da.linspace(self.lower, self.upper, self.grid_size)
        v_function = [da.sum(self.quadrature(grid)*self.kernel(grid, y) ** 2) for y in self.observations]
        v_function = da.stack(v_function, axis=0)
        delta = da.sum(v_function) / self.sample_size ** 2
        if compute:
            delta = delta.compute()
        self.delta = delta
        return self.delta


