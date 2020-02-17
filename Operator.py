from typing import Callable, Union, List
import numpy as np
from numba import jit
from decorators import vectorize, timer


class Quadrature:
    def __init__(self, lower: Union[int, float], upper: Union[int, float], grid_size: int):
        """
        Functionality to calculate weights in different quadrature schema.
        :param lower: Lower end of interval.
        :type lower: float
        :param upper: Upper end of interval.
        :type upper: float
        :param grid_size: Number of points in a grid.
        :type grid_size: int
        """
        self.lower: float = lower
        self.upper: float = upper
        self.grid_size: int = grid_size
        assert self.lower <= self.upper, "Wrong specification of interval"
        assert self.grid_size > 0, 'Grid has to have at least one point'
        assert isinstance(grid_size, int), 'Specify grid size as integer'

    @staticmethod
    @jit(nopython=True)
    def __rectangle_weight(a, b, c):
        return (b - a) / c

    def rectangle(self, t: float) -> float:
        """
        Calculate weight for rectangular quadrature with equal grid.
        :param t: Point in which the weight is calculated.
        :type t: float
        :return: Value of quadrature weight for point t.
        """
        return self.__rectangle_weight(self.lower, self.upper, self.grid_size)

    @vectorize(signature="(),()->()")
    def dummy(self, t: float) -> float:
        """
        Calculate weight for dummy quadrature. Weight is always equal to 1 (allows to get approximation of kernel, not operator).
        :param t: Point in which the weight is calculated.
        :type t: float
        :return: Value of quadrature weight for point t.
        """
        return 1.


class Operator(Quadrature):
    def __init__(self, kernel: Callable, lower: Union[float, int], upper: Union[float, int], grid_size: int,
                 quadrature: str = 'rectangle'):
        """
        Build an approximation of integral operator with given kernel on equal grid.
        :param kernel: Kernel of an operator being approximated.
        :type kernel: Callable
        :param lower: Lower end of an interval on which the operator is approximated.
        :type lower: float
        :param upper: Upper end of an interval on which the operator is approximated.
        :type upper: float
        :param grid_size: Size of the grid on which the operator is approximated.
        :type grid_size: int
        :param quadrature: Type of quadrature used to approximate the operator.
        :type quadrature: str
        """
        assert isinstance(lower, float) | isinstance(lower,
                                                     int), "Lower limit must be number, but was {} provided".format(
            lower)
        assert isinstance(upper, float) | isinstance(upper,
                                                     int), "Upper limit must be a number, but was {} provided".format(
            upper)
        assert isinstance(grid_size, int), 'Grid size must be an integer, but was {} provided'.format(grid_size)
        assert quadrature in ['rectangle', 'dummy'], 'This type of quadrature is not supported, currently only {} ' \
                                                     'are supported'.format(
            [method for method in dir(Quadrature) if not method.startswith('_')])
        assert callable(kernel), 'Kernel function must be callable'
        super().__init__(lower, upper, grid_size)
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
        self.__K: np.ndarray = np.zeros((self.grid_size, self.grid_size)).astype(np.float64)
        self.__KH: np.ndarray = np.zeros((self.grid_size, self.grid_size)).astype(np.float64)
        self.__grid: np.ndarray = np.linspace(self.lower, self.upper, self.grid_size)

    # noinspection PyPep8Naming
    @property
    def K(self):
        return self.__K

    # noinspection PyPep8Naming
    @property
    def KH(self):
        return self.__KH

    # noinspection PyPep8Naming
    @K.setter
    def K(self, K):
        self.__K = K

    # noinspection PyPep8Naming
    @KH.setter
    def KH(self, KH):
        self.__KH = KH

    def operator_column(self, t: np.ndarray) -> np.ndarray:
        """
        Function constructing nth column of an approximation. Its value is equal to the values of the operator with
        grid as first argument, value of grid points weighted by quadrature weight in nth grid point.
        :param t: grid, second argument to kernel function and argument to quadrature weight builder.
        :type t: Numpy ndarray
        :return: Numpy ndarray containing the nth column of the approximation.
        """
        return self.kernel(self.__grid, t) * self.quadrature(t)

    @timer
    def approximate(self) -> np.ndarray:
        """
        Build entire approximation of an operator as matrix of size grid size x grid size.
        :return: Numpy array containing the approximation of the operator on given grid.
        """
        print('Calculating operator approximation...')
        column_list: List[np.ndarray] = [self.operator_column(t) for t in self.__grid]
        np.stack(column_list, axis=1, out=self.__K)
        self.__KH = self.__K.transpose().conj()
        return self.__K
