import numpy as np

from decorators import vectorize


class Estimator:
    def __init__(self, kernel, v_function, observations, sample_size):
        self.kernel = np.vectorize(kernel)
        self.v_function = np.vectorize(v_function)
        self.observations = observations
        self.sample_size = sample_size

    def noise_level(self):
        return np.sum(self.v_function(self.observations)) / (self.sample_size ** 2)

    @vectorize(signature="(),()->()")
    def q_estimator(self, x):
        return np.sum(self.kernel(x, self.observations)) / (self.sample_size ** 2)
