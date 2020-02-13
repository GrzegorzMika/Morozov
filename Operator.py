import numpy as np
from multiprocessing import Pool, cpu_count
from decorators import vectorize


class Quadrature:
    def __init__(self, lower, upper, grid_size):
        self.lower = lower
        self.upper = upper
        self.grid_size = grid_size

    @vectorize(signature="(),()->()")
    def rectangle(self, t):
        assert self.upper >= t >= self.lower, 'Argument must belong to interval [{}, {}], but {} was given.'.format(
            self.lower, self.upper, t)
        return (self.upper - self.lower) / self.grid_size


class Operator(Quadrature):
    def __init__(self, kernel, lower, upper, grid_size, quadrature):
        super().__init__(lower, upper, grid_size)
        self.kernel = np.vectorize(kernel)
        self.lower = lower
        self.upper = upper
        self.grid_size = grid_size
        self.quadrature = quadrature

    def grid_list(self):
        return list(np.linspace(self.lower, self.upper, self.grid_size))

    def operator_column(self, t):
        return self.kernel(self.grid_list(), t) * getattr(super(), self.quadrature)(t)

    def approximate(self):
        grid_list = self.grid_list()
        columns = [self.operator_column(t) for t in grid_list]
        return np.stack(columns, axis=1)
