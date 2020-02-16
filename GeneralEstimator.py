from abc import abstractmethod
from typing import Callable, Union, Optional
import numpy as np
from Operator import Quadrature
from decorators import timer


class Estimator(Quadrature):
    def __init__(self, kernel: Callable, lower: Union[float, int], upper: Union[float, int], grid_size: int,
                 observations: np.ndarray, sample_size: int, quadrature: str):
        Quadrature.__init__(self, lower, upper, grid_size)
        try:
            kernel(np.array([1, 2]), np.array([1, 2]))
            self.kernel: Callable = kernel
        except ValueError:
            print('Force vectorization of kernel')
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
        self.lower: float = float(lower)
        self.upper: float = float(upper)
        self.grid_size: int = grid_size
        self.quadrature: Callable = getattr(super(), quadrature)
        self.__observations = observations
        self.sample_size: int = sample_size
        self.__delta: Optional[float] = None
        self.__q_estimator: Optional[np.ndarray] = None

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
        print('Estimating q...')
        grid = np.linspace(self.lower, self.upper, self.grid_size)
        estimator = [np.sum(self.kernel(x, self.__observations)) / self.sample_size for x in grid]
        estimator = np.stack(estimator, axis=0)
        self.__q_estimator = estimator
        return estimator

    @timer
    def estimate_delta(self) -> Union[float, np.ndarray]:
        """
        Estimate noise level based on the observations and approximation of function v.
        :return: Float indicating the estimated noise level.
        """
        print('Estimating noise level...')
        grid = np.linspace(self.lower, self.upper, self.grid_size)
        v_function = [np.sum(self.quadrature(grid) * self.kernel(grid, y) ** 2) for y in self.__observations]
        v_function = np.stack(v_function, axis=0)
        delta = np.sum(v_function) / (self.sample_size ** 2)
        self.__delta = delta
        print('Estimated noise level: {}'.format(delta))
        return delta

    @abstractmethod
    def estimate(self, *args, **kwargs):
        ...

    @abstractmethod
    def refresh(self, *args, **kwargs):
        ...
