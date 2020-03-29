from abc import abstractmethod, ABCMeta
from typing import Callable, Union, Optional, List
from warnings import warn

import cupy as cp
from warnings import warn

import numpy as np
from scipy.integrate import quad

from Operator import Quadrature
from decorators import timer


class EstimatorAbstract(metaclass=ABCMeta):
    @abstractmethod
    def estimate(self, *args, **kwargs):
        ...

    @abstractmethod
    def refresh(self, *args, **kwargs):
        ...

    @abstractmethod
    def estimate_q(self, *args, **kwargs):
        ...

    @abstractmethod
    def estimate_delta(self, *args, **kwargs):
        ...


class EstimatorDiscretize(EstimatorAbstract, Quadrature):
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
        assert isinstance(observations,
                          np.ndarray), 'Observations must be provided as numpy array, but {} was provided'.format(
            observations)
        assert isinstance(sample_size, int), 'Sample size must be an integer'
        self.lower: float = float(lower)
        self.upper: float = float(upper)
        self.grid_size: int = grid_size
        self.quadrature: Callable = getattr(super(), quadrature)
        self.__observations: np.ndarray = observations.astype(float)
        self.sample_size: int = sample_size
        self.__delta: float = 0.
        self.__q_estimator: cp.ndarray = cp.empty(self.grid_size, dtype=cp.float64)
        self.__grid: np.ndarray = getattr(super(), quadrature + '_grid')()
        self.__weights_np: np.ndarray = self.quadrature(self.__grid)
        self.__weights: cp.ndarray = cp.asarray(self.quadrature(self.__grid))

    @property
    def delta(self) -> float:
        return self.__delta

    @delta.setter
    def delta(self, delta: float):
        self.__delta = delta

    @property
    def q_estimator(self) -> cp.ndarray:
        return self.__q_estimator

    @q_estimator.setter
    def q_estimator(self, q_estimator: cp.ndarray):
        self.__q_estimator = q_estimator

    @property
    def observations(self) -> np.ndarray:
        return self.__observations

    @observations.setter
    def observations(self, observations: np.ndarray):
        self.__observations = observations

    @timer
    def estimate_q(self):
        """
        Estimate function q on given grid based on the observations.
        """
        print('Estimating q function...')
        estimator_list: List[np.ndarray] = \
            [np.divide(np.sum(self.kernel(x, self.__observations)), self.sample_size) for x in self.__grid]
        estimator: np.ndarray = np.stack(estimator_list, axis=0).astype(np.float64)
        self.__q_estimator = cp.asarray(estimator)

    @timer
    def estimate_delta(self):
        """
        Estimate noise level based on the observations and approximation of function w.
        """
        print('Estimating noise level...')
        w_function_list: List[np.ndarray] = \
            [np.sum(np.multiply(self.__weights_np, np.square(self.kernel(self.__grid, y)))) for y in
             self.__observations]
        w_function: np.ndarray = np.stack(w_function_list, axis=0)
        delta: float = np.sqrt(np.divide(np.sum(w_function), np.square(self.sample_size)))
        self.__delta = delta
        print('Estimated noise level: {}'.format(delta))

    def L2norm(self, x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        """
        Calculate the approximation of L2 norm of difference of two approximation of function.
        :param x: Approximation of function on given grid.
        :type x: np.ndarray
        :param y: Approximation of function on given grid.
        :type y: np.ndarray
        :return: Float representing the L2 norm of difference between given functions.
        """
        return cp.sqrt(cp.sum(cp.multiply(cp.square(cp.subtract(x, y)), self.__weights)))

    def estimate(self):
        raise NotImplementedError

    def refresh(self):
        raise NotImplementedError


class EstimatorSpectrum(EstimatorAbstract):
    def __init__(self, kernel: Callable, observations: np.ndarray, sample_size: int, transformed_measure: bool,
                 lower: Union[float, int] = 0, upper: Union[float, int] = 1):
        self.transformed_measure = transformed_measure
        assert isinstance(kernel, Callable), 'Kernel function must be callable'
        try:
            kernel(np.array([1, 2]), np.array([1, 2]))
            self.kernel: Callable = kernel
        except ValueError:
            warn('Force vectorization of kernel')
            self.kernel: Callable = np.vectorize(kernel)
        assert isinstance(lower, float) | isinstance(lower,
                                                     int), 'Lower bound for integration interval must be a number, but ' \
                                                           'was {} provided'.format(lower)
        self.lower: Union[float, int] = lower
        assert isinstance(upper, float) | isinstance(upper,
                                                     int), 'Upper bound for integration interval must be a number, but' \
                                                           ' was {} provided'.format(upper)
        self.upper: Union[float, int] = upper
        assert isinstance(observations, np.ndarray), 'Please provide the observations in a form of numpy array'
        self.__observations: np.ndarray = observations
        assert isinstance(sample_size, int), 'Sample size must be an integer, but was {} provided'.format(sample_size)
        self.sample_size: int = sample_size
        self.q_estimator: Optional[Callable] = None
        self.__w_function: Optional[Callable] = None
        self.delta: float = 0.

    @property
    def observations(self) -> np.ndarray:
        return self.__observations

    @observations.setter
    def observations(self, observations: np.ndarray):
        self.__observations = observations

    @timer
    def estimate_q(self) -> None:
        """
        Estimate function q based on the observations using the known kernel.
        """
        print('Estimating q function...')
        observations: np.ndarray = self.observations
        kernel: Callable = self.kernel
        sample_size: int = self.sample_size

        if self.transformed_measure:
            def __q_estimator(x: Union[float, int]) -> np.float64:
                x: np.ndarray = np.repeat(x, observations.shape[0])
                return np.divide(np.multiply(2, np.sum(np.less(observations, x))), sample_size)
        else:
            def __q_estimator(x: Union[float, int]) -> np.float64:
                x: np.ndarray = np.repeat(x, observations.shape[0])
                return np.divide(np.sum(kernel(x, observations)), sample_size)

        self.q_estimator = np.vectorize(__q_estimator)

    def __w_function_calculation(self) -> None:
        """
        Calculate the w function based on the known kernel to be used for the noise level estimation.
        """
        kernel: Callable = self.kernel

        def kernel_integrand(x: float, y: float) -> np.float64:
            return np.square(kernel(x, y))

        def w_function(y: float) -> float:
            return quad(kernel_integrand, self.lower, self.upper, args=y, limit=10000)[0]

        self.__w_function = np.vectorize(w_function)

    @timer
    def estimate_delta(self):
        """
        Estimate noise level based on the observations and known kernel (via w function).
        """
        print('Estimating noise level...')
        if self.transformed_measure:
            self.delta = np.sqrt(np.divide(np.sum(np.square(1-self.observations)), self.sample_size ** 2))
        else:
            self.__w_function_calculation()
            self.delta = np.sqrt(np.divide(np.sum(self.__w_function(self.observations)), self.sample_size ** 2))
        print('Estimated noise level: {}'.format(self.delta))

    def estimate(self, *args, **kwargs):
        raise NotImplementedError

    def refresh(self, *args, **kwargs):
        raise NotImplementedError
