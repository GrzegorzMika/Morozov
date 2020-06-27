from abc import abstractmethod, ABCMeta
from typing import Callable, Optional, Union, Generator
from warnings import warn

import numpy as np

from decorators import timer
from validate import validate_EstimatorSpectrum


# TODO: assumed form of a weight function rho_l(t) = (1+lt)^{-1/2}
class EstimatorAbstract(metaclass=ABCMeta):
    @abstractmethod
    def estimate(self, *args, **kwargs):
        ...

    @abstractmethod
    def estimate_q(self, *args, **kwargs):
        ...

    @abstractmethod
    def estimate_delta(self, *args, **kwargs):
        ...


class EstimatorSpectrum(EstimatorAbstract):
    def __init__(self, kernel: Callable, observations: np.ndarray, sample_size: int, rho: float,
                 transformed_measure: bool, singular_values: Generator, left_singular_functions: Generator,
                 right_singular_functions: Generator, max_size: int):

        validate_EstimatorSpectrum(kernel, observations, sample_size, rho, transformed_measure, singular_values,
                                   left_singular_functions, right_singular_functions, max_size)

        self.transformed_measure: bool = transformed_measure
        self.singular_values: Union[Generator, np.ndarray] = singular_values
        self.right_singular_functions: Union[Generator, np.ndarray] = right_singular_functions
        self.left_singular_functions: Union[Generator, np.ndarray] = left_singular_functions
        try:
            kernel(np.array([1, 2]), np.array([1, 2]))
            self.kernel: Callable = kernel
        except ValueError:
            warn('Force vectorization of kernel')
            self.kernel: Callable = np.vectorize(kernel)
        self.observations: np.ndarray = observations
        self.sample_size: int = sample_size
        self.rho: float = rho
        self.max_size: int = max_size
        self.q_estimator: Optional[Callable] = None
        self.__w_function: Optional[Callable] = None
        self.delta: float = 0.
        self.pi = np.empty(max_size)
        self.ui = np.empty(max_size)
        self.ui_square = np.empty(max_size)

        self.__precompute()

    def __precompute(self):
        sigmas = [next(self.singular_values) ** 2 for _ in range(self.max_size)]
        self.singular_values = np.array(sigmas)

        right_f = [next(self.right_singular_functions) for _ in range(self.max_size)]
        self.right_singular_functions = right_f

        left_f = [next(self.left_singular_functions) for _ in range(self.max_size)]
        self.left_singular_functions = left_f

        self.pi = np.divide(self.singular_values, 1 + self.rho * self.singular_values)

        observations = self.observations
        for i in range(self.max_size):
            tmp = np.sort(left_f[i](observations))
            self.ui[i] = np.sum(tmp)
            self.ui_square[i] = np.sum(np.square(tmp))

    @timer
    def estimate_q(self) -> None:
        """
        Estimate function q based on the observations using the known kernel.
        """
        print('Estimating q function...')
        observations = self.observations
        sample_size = self.sample_size

        if self.transformed_measure:
            def __q_estimator(x: float) -> np.float64:
                x: np.ndarray = np.repeat(x, observations.shape[0])
                return np.divide(np.multiply(2, np.sum(np.less_equal(observations, x))), sample_size)
        else:
            def __q_estimator(x: float) -> np.float64:
                x: np.ndarray = np.repeat(x, observations.shape[0])
                return np.divide(np.multiply(2, np.sum(
                    np.sort(np.multiply(observations, np.less_equal(observations, x))))), sample_size)

        self.q_estimator = np.vectorize(__q_estimator)

    @timer
    def estimate_delta(self):
        """
        Estimate noise level based on the observations and known kernel.
        """
        print('Estimating noise level...')

        self.delta = np.divide(np.sum(np.sort(np.multiply(self.pi, self.ui_square))), self.sample_size**2)

        print('Estimated noise level: {}'.format(self.delta))

    def estimate(self, *args, **kwargs):
        raise NotImplementedError
