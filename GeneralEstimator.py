from abc import abstractmethod
from typing import Callable, Union, Optional, List
import numpy as np
from numba import jit
from Operator import Quadrature
from decorators import timer
from warnings import warn


class Estimator(Quadrature):
    def __init__(self, kernel: Callable, lower: Union[float, int], upper: Union[float, int], grid_size: int,
                 observations: np.ndarray, sample_size: int, quadrature: str = 'rectangle'):
        Quadrature.__init__(self, lower, upper, grid_size)
        try:
            kernel(np.array([1, 2]), np.array([1, 2]))
            self.kernel: Callable = kernel
        except ValueError:
            warn('Force vectorization of kernel')
            self.kernel: Callable = np.vectorize(kernel)
        assert isinstance(lower, float) | isinstance(lower, int), "Lower limit must be number, but was {} " \
                                                                  "provided".format(lower)
        assert isinstance(upper, float) | isinstance(upper, int), "Upper limit must be a number, but was {} " \
                                                                  "provided".format(upper)
        assert isinstance(grid_size, int), 'Grid size must be an integer, but was {} provided'.format(grid_size)
        assert quadrature in ['rectangle', 'dummy'], 'This type of quadrature is not supported, currently only {} ' \
                                                     'are supported'.format(
            [method for method in dir(Quadrature) if not method.startswith('_')])
        assert callable(kernel), 'Kernel function must be callable'
        assert isinstance(observations, np.ndarray), 'Observations must be provided as numpy array, but {} was provided'.format(observations)
        assert isinstance(sample_size, int), 'Sample size must be an integer'
        self.lower: float = float(lower)
        self.upper: float = float(upper)
        self.grid_size: int = grid_size
        self.quadrature: Callable = getattr(super(), quadrature)
        self.__observations: np.ndarray = observations.astype(float)
        self.sample_size: int = sample_size
        self.__delta: Optional[float] = None
        self.__q_estimator: np.ndarray = np.zeros((self.grid_size,)).astype(float)
        self.__grid: np.ndarray = getattr(super(), quadrature + '_grid')()
        self.__weights: np.ndarray = self.quadrature(self.__grid)

    @property
    def delta(self) -> float:
        return self.__delta

    @delta.setter
    def delta(self, delta: float):
        self.__delta = delta

    @property
    def q_estimator(self) -> np.ndarray:
        return self.__q_estimator

    @q_estimator.setter
    def q_estimator(self, q_estimator: np.ndarray):
        self.__q_estimator = q_estimator

    @property
    def observations(self) -> np.ndarray:
        return self.__observations

    @observations.setter
    def observations(self, observations: np.ndarray):
        self.__observations = observations

    @timer
    def estimate_q(self) -> np.ndarray:
        """
        Estimate function q on given grid based on the observations.
        :return: Return numpy array containing estimated function q.
        """
        print('Estimating q function...')
        estimator_list: List[np.ndarray] = \
            [np.divide(np.sum(self.kernel(x, self.__observations)), self.sample_size) for x in self.__grid]
        estimator: np.ndarray = np.stack(estimator_list, axis=0).astype(np.float64)
        self.__q_estimator = estimator
        return estimator

    @timer
    def estimate_delta(self) -> float:
        """
        Estimate noise level based on the observations and approximation of function v.
        :return: Float indicating the estimated noise level.
        """
        print('Estimating noise level...')
        v_function_list: List[np.ndarray] = \
            [np.sum(np.multiply(self.__weights, np.square(self.kernel(self.__grid, y)))) for y in self.__observations]
        v_function: np.ndarray = np.stack(v_function_list, axis=0)
        delta: float = np.sqrt(np.divide(np.sum(v_function), np.square(self.sample_size)))
        self.__delta = delta
        print('Estimated noise level: {}'.format(delta))
        return delta

    @staticmethod
    @jit(nopython=True)
    def __L2norm(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum(np.multiply(np.square(np.subtract(x, y)), weights)))

    def L2norm(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the approximation of L2 norm of difference of two approximation of function.
        :param x: Approximation of function on given grid.
        :type x: np.ndarray
        :param y: Approximation of function on given grid.
        :type y: np.ndarray
        :return: Float representing the L2 norm of difference between given functions.
        """
        return self.__L2norm(x, y, self.__weights)

    @abstractmethod
    def estimate(self, *args, **kwargs):
        ...

    @abstractmethod
    def refresh(self, *args, **kwargs):
        ...
