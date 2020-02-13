import numpy as np
from typing import Union
from decorators import vectorize


class Estimator:

    def __init__(self, kernel, v_function, observations, sample_size):
        self.kernel = np.vectorize(kernel)
        self.v_function = np.vectorize(v_function)
        self.observations = observations
        self.sample_size = sample_size

    def noise_level(self) -> float:
        return np.sum(self.v_function(self.observations)) / (self.sample_size ** 2)

    @vectorize(signature="(),()->()", otypes=Union[float, np.ndarray])
    def q_estimator(self, x: float) -> [float, np.ndarray]:
        return np.sum(self.kernel(x, self.observations)) / (self.sample_size ** 2)

    def quadrature_weights(self, grid, type='rectangle'):
        return         