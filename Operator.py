from typing import Callable, Union, List
from warnings import warn
import numpy as np
import cupy as cp
from numba import jit
from decorators import vectorize, timer


class Quadrature:
    def __init__(self, lower: Union[int, float], upper: Union[int, float], grid_size: int):
        """
        Functionality to calculate weights in different quadrature schema.
        Availiable quadratures: `
            - rectangle: rectangular quadrature with equl grid and function evaluated on thr left end of the interval
            - dummy: not a quadrature, just approximation of kernel function on a equal grid as in rectangle
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
        assert isinstance(grid_size, int), 'Specify grid size as integer'
        assert isinstance(lower, (int, float)), 'Specify lower limit as number'
        assert isinstance(upper, (int, float)), 'Specify upper limit as number'
        assert self.lower <= self.upper, "Wrong specification of interval"
        assert self.grid_size > 0, 'Grid has to have at least one point'

    @staticmethod
    @jit(nopython=True)
    def __rectangle_weight(a, b, c):
        return (b - a) / c

    def rectangle(self, t: float) -> float:
        """
        Calculate weight for rectangular quadrature based on the left end with equal grid.
        :param t: Point in which the weight is calculated.
        :type t: float
        :return: Value of quadrature weight for point t.
        """
        return self.__rectangle_weight(self.lower, self.upper, self.grid_size)

    def rectangle_grid(self) -> np.ndarray:
        """
        Return the grid on which the values of integrated function are evaluated.
        :return: Numpy array containing the points on which the function is evaluated.
        """
        return np.linspace(self.lower, self.upper, self.grid_size, endpoint=False)

    @vectorize(signature="(),()->()")
    def dummy(self, t: float) -> float:
        """
        Calculate weight for dummy quadrature. Weight is always equal to 1 (allows to get approximation of kernel, not
        operator).
        :param t: Point in which the weight is calculated.
        :type t: float
        :return: Value of quadrature weight for point t.
        """
        return 1.

    def dummy_grid(self) -> np.ndarray:
        """
        Equaly spaced grid for function evaluation.
        :return: Numpy array containing the points on which the function is evaluated.
        """
        return np.linspace(self.lower, self.upper, self.grid_size, endpoint=False)


class Operator(Quadrature):
    def __init__(self, kernel: Callable, lower: Union[float, int], upper: Union[float, int], grid_size: int,
                 adjoint: bool = False, quadrature: str = 'rectangle'):
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
        :param adjoint: Is the operator self_adjoint (True) or not (False)?
        :type adjoint: boolean
        :param quadrature: Type of quadrature used to approximate the operator.
        :type quadrature: str
        """
        assert isinstance(adjoint,
                          bool), 'Condition if operator is self-adjoint must be boolean, but was {} provided'.format(
            adjoint)
        assert quadrature in ['rectangle', 'dummy'], 'This type of quadrature is not supported, currently only {} ' \
                                                     'are supported'.format(
            [method for method in dir(Quadrature) if not method.startswith('_')])
        assert callable(kernel), 'Kernel function must be callable'
        assert lower <= upper, 'Wrong interval specified'
        super().__init__(lower, upper, grid_size)
        try:
            kernel(np.array([1, 2]), np.array([1, 2]))
            self.kernel: Callable = kernel
        except ValueError:
            warn('Force vectorization of kernel')
            self.kernel: Callable = np.vectorize(kernel)
        self.adjoint: bool = adjoint
        self.quadrature: Callable = getattr(super(), quadrature)
        self.__K: Union[np.ndarray, cp.ndarray] = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.__KH: Union[np.ndarray, cp.ndarray] = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.__grid: np.ndarray = getattr(super(), quadrature + '_grid')()

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

    def __operator_column(self, t: np.ndarray) -> np.ndarray:
        """
        Function constructing nth column of the operator approximation.
        :param t: grid, first argument to kernel function and argument to quadrature weight builder.
        :type t: Numpy ndarray
        :return: Numpy ndarray containing the nth column of the operator approximation.
        """
        return self.kernel(t, self.__grid) * self.quadrature(t)

    def __adjoint_operator_column(self, t: np.ndarray) -> np.ndarray:
        """
        Function constructing nth column of the adjoint operator approximation.
        :param t: grid, first argument to kernel function and argument to quadrature weight builder.
        :type t: Numpy ndarray
        :return: Numpy ndarray containing the nth column of the adjoint operator approximation.
        """
        kernel = self.kernel

        def adjoint_kernel(x, y):
            return kernel(y, x)

        return adjoint_kernel(t, self.__grid) * self.quadrature(t)

    @timer
    def approximate(self):
        """
        Build entire approximation of an operator as matrix of size grid size x grid size.
        :return: Numpy array containing the approximation of the operator on given grid.
        """
        print('Calculating operator approximation...')
        column_list: List[np.ndarray] = [self.__operator_column(t) for t in self.__grid]
        np.stack(column_list, axis=1, out=self.__K)
        print('Calculating adjoint operator approximation...')
        if self.adjoint:
            self.__KH = self.__K
        else:
            column_list: List[np.ndarray] = [self.__adjoint_operator_column(t) for t in self.__grid]
            np.stack(column_list, axis=1, out=self.__KH)
        self.__move_to_gpu()

    @timer
    def __move_to_gpu(self):
        self.__K = cp.asarray(self.__K, dtype=cp.float64)
        self.__KH = cp.asarray(self.__KH, dtype=cp.float64)
