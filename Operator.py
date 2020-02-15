import dask
import dask.array as da
import numpy as np
from dask.system import cpu_count
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

    @vectorize(signature="(),()->()")
    def dummy(self, t):
        return 1


class Operator(Quadrature):
    def __init__(self, kernel, lower, upper, grid_size, quadrature):
        super().__init__(lower, upper, grid_size)
        try:
            kernel(np.array([1, 2]), np.array([1, 2]));
            self.kernel = kernel
        except ValueError as err:
            print('Force vectorization of kernel')
            self.kernel = np.vectorize(kernel)
        self.lower = float(lower)
        self.upper = float(upper)
        self.grid_size = grid_size
        self.quadrature = getattr(super(), quadrature)

    def grid_list(self):
        return list(np.linspace(self.lower, self.upper, self.grid_size))

    @dask.delayed
    def operator_column(self, t):
        grid = da.linspace(self.lower, self.upper, self.grid_size)
        return self.kernel(grid, t) * self.quadrature(t)

    def approximate(self, compute: bool = False):
        operator = [da.from_delayed(self.operator_column(t), shape=(self.grid_size,), dtype=float) for t in
                    self.grid_list()]
        operator = da.stack(operator, axis=1)
        if compute:
            operator = operator.compute(num_workers=cpu_count())
        return operator
