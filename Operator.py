import numpy as np
from decorators import vectorize
from warnings import warn
import ray


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
        self.quadrature = getattr(super(), quadrature)

    def grid_list(self):
        return list(np.linspace(self.lower, self.upper, self.grid_size))

    def operator_column(self, t):
        grid = np.linspace(self.lower, self.upper, self.grid_size)
        return self.kernel(grid, t) * self.quadrature(t)

    def approximate(self):
        if self.grid_size > 1000:
            warn("Class method is not parallelizable and may be extremely slow, use wrapper instead", RuntimeWarning)
        grid_list = self.grid_list()
        columns = [self.operator_column(t) for t in grid_list]
        return np.stack(columns, axis=1)


def approximate_operator(kernel, lower, upper, grid_size, quadrature):
    op = Operator(kernel, 0, 1, 50000, 'rectangle')
    @ray.remote
    def op_col(t):
        return op.operator_column(t)

    ray.init()
    col = [op_col.remote(t) for t in op.grid_list()]
    col = ray.get(col)
    ray.shutdown()
    col = np.stack(col, axis=1)
    # TODO Implement wrapper for parallel building of approximation
    pass
