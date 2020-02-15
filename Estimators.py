from typing import Callable
import dask.array as da
from GeneralEstimator import Estimator
from Operator import Operator


class Landweber(Estimator, Operator):
    def __init__(self, kernel, lower, upper, grid_size, observations, sample_size, quadrature):
        Operator.__init__(self, kernel, lower, upper, grid_size, quadrature)
        Estimator.__init__(self, kernel, lower, upper, grid_size, observations, sample_size, quadrature)
        self.kernel: Callable = kernel
        self.lower = lower
        self.upper = upper
        self.grid_size = grid_size
        self.observations = observations
        self.sample_size = sample_size
        Operator.approximate(self)
