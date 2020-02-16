from abc import abstractmethod
from typing import Callable, Union, Optional, List
import dask.array as da
import numpy as np
from Operator import Quadrature


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
        self.__q_estimator: Optional[Union[np.ndarray, da.array]] = None

    @property
    def delta(self) -> float:
        return self.__delta

    @delta.setter
    def delta(self, delta: float):
        self.__delta = delta

    @property
    def q_estimator(self) -> Union[np.ndarray, da.array]:
        return self.__q_estimator

    @q_estimator.setter
    def q_estimator(self, q_estimator: Union[np.ndarray, da.array]):
        self.__q_estimator = q_estimator

    @property
    def observations(self) -> Union[np.ndarray, da.array]:
        return self.__observations

    @observations.setter
    def observations(self, observations: Union[np.ndarray, da.array]):
        self.__observations = observations

    def estimate_q(self, compute: bool = False) -> Union[np.ndarray, da.array]:
        """
        Estimate function q on given grid based on the observations.
        :param compute: Retrun estimated function as numpy array (True) or dask graph of computations (False).
        :type compute: boolean
        :return: Return numpy array containing estimated function q when compute is True or dask graph of computations
        when compute is False.
        """
        grid: da.array = da.linspace(self.lower, self.upper, self.grid_size)
        estimator: List[da.array] = [da.sum(self.kernel(x, self.__observations)) / self.sample_size for x in grid]
        estimator: da.array = da.stack(estimator, axis=0)
        if compute:
            # noinspection PyUnresolvedReferences
            estimator: np.ndarray = estimator.compute()
        self.__q_estimator = estimator
        return estimator

    def estimate_delta(self, compute: bool = False) -> Union[float, da.array]:
        """
        Estimate noise level based on the observations and approximation of function v.
        :param compute: Return estimated noise level as float (True) or dask graph of computations (False).
        :type compute: boolean
        :return: Float indicating the estimated noise level if compute is True or dask graph of computations if
        compute is False.
        """
        grid: da.array = da.linspace(self.lower, self.upper, self.grid_size)
        v_function: List[da.array] = [da.sum(self.quadrature(grid) * self.kernel(grid, y) ** 2) for y in self.__observations]
        v_function: da.array = da.stack(v_function, axis=0)
        delta: da.array = da.sum(v_function) / (self.sample_size ** 2)
        if compute:
            # noinspection PyUnresolvedReferences
            delta: float = delta.compute()
        self.__delta = delta
        return delta

    @abstractmethod
    def estimate(self, *args, **kwargs):
        ...

    @abstractmethod
    def refresh(self, *args, **kwargs):
        ...
