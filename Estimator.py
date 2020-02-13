from functools import wraps

import numpy as np


def vectorize(otypes=None, signature=None):
    """Numpy vectorization wrapper that works with instance methods."""

    def decorator(fn):
        vectorized = np.vectorize(fn, otypes=otypes, signature=signature)

        @wraps(fn)
        def wrapper(*args):
            return vectorized(*args)

        return wrapper

    return decorator


class Estimator:
    def __init__(self, kernel, v_function, observations, sample_size):
        self.kernel = kernel
        self.v_function = v_function
        self.observations = observations
        self.sample_size = sample_size

    def noise_level(self):
        return np.sum(self.v_function(self.observations)) / (self.sample_size ** 2)

    @vectorize(signature="(),()->()")
    def q_estimator(self, x):
        return np.sum(self.kernel(x, self.observations)) / (self.sample_size ** 2)
