import inspect
from abc import abstractmethod, ABCMeta
from typing import Callable, Optional
from warnings import warn

import numpy as np
from joblib import Memory

from decorators import timer
from validate import validate_EstimatorSpectrum

location = './cachedir'
memory = Memory(location, verbose=0, bytes_limit=1024 * 1024 * 1024)


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
    def __init__(self, kernel: Callable, observations: np.ndarray, sample_size: int, transformed_measure: bool,
                 rho: float, lower: float = 0, upper: float = 1):
        validate_EstimatorSpectrum(kernel, observations, sample_size, transformed_measure, rho, lower, upper)
        self.transformed_measure = transformed_measure
        try:
            kernel(np.array([1, 2]), np.array([1, 2]))
            self.kernel: Callable = kernel
        except ValueError:
            warn('Force vectorization of kernel')
            self.kernel: Callable = np.vectorize(kernel)
        self.lower: float = lower
        self.upper: float = upper
        self.observations: np.ndarray = observations
        self.sample_size: int = sample_size
        self.rho: float = rho
        self.q_estimator: Optional[Callable] = None
        self.__w_function: Optional[Callable] = None
        self.delta: float = 0.

    @timer
    def estimate_q(self) -> None:
        """
        Estimate function q based on the observations using the known kernel. # TODO: only applicable to LSW problem so far
        """
        print('Estimating q function...')
        observations: np.ndarray = self.observations
        sample_size: int = self.sample_size

        if self.transformed_measure:
            def __q_estimator(x: float) -> np.float64:
                x: np.ndarray = np.repeat(x, observations.shape[0])
                return np.divide(np.multiply(2, np.sum(np.less_equal(observations, x))), sample_size)
        else:
            def __q_estimator(x: float) -> np.float64:
                x: np.ndarray = np.repeat(x, observations.shape[0])
                return np.divide(np.multiply(2, np.sum(
                    np.sort(np.multiply(observations, np.less_equal(observations, x)), kind='heapsort'))), sample_size)

        self.q_estimator = np.vectorize(__q_estimator)

    @timer
    def estimate_delta(self):
        """
        Estimate noise level based on the observations and known kernel. # TODO: only applicable to LSW problem so far
        """
        print('Estimating noise level...')
        observations = self.observations
        rho = self.rho
        sample_size = self.sample_size
        kernel = self.kernel

        if self.transformed_measure:

            @memory.cache
            def delta_transformed(observations: np.ndarray, sample_size: int, rho: float, kernel_formula: str) -> float:
                constant = 4 * observations.shape[0] * (1 - rho * np.log(1 * rho)) / sample_size ** 2
                summand = np.sort(rho * np.log(1 + observations) - observations, kind='heapsort')
                return constant * 4 * np.divide(np.sum(summand), sample_size ** 2)

            self.delta = delta_transformed(observations, sample_size, rho, inspect.getsource(kernel))
        else:

            @memory.cache
            def delta_nontransformed(observations: np.ndarray, sample_size: int, rho: float,
                                     kernel_formula: str) -> float:
                square_observations = np.square(observations)
                log_observations = np.log(rho + observations)
                summand = np.sort(np.subtract(np.log(1 + rho) * square_observations,
                                              np.multiply(square_observations, log_observations)), kind='heapsort')
                return 4 * np.divide(np.sum(summand), sample_size ** 2)

            self.delta = delta_nontransformed(observations, sample_size, rho, inspect.getsource(kernel))

        print('Estimated noise level: {}'.format(self.delta))

    def estimate(self, *args, **kwargs):
        raise NotImplementedError
