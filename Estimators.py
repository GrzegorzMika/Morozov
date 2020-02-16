from typing import Callable, Union
from warnings import warn
import dask.array as da
import dask
import numpy as np
from time import time
from dask.diagnostics import ProgressBar
from dask.system import cpu_count
from GeneralEstimator import Estimator
from Operator import Operator
from decorators import timer


class Landweber(Estimator, Operator):
    def __init__(self, kernel: Callable, lower: Union[float, int], upper: Union[float, int], grid_size: int,
                 observations: Union[da.array, np.ndarray], sample_size: int, quadrature: str = 'rectangle', **kwargs):
        Operator.__init__(self, kernel, lower, upper, grid_size, quadrature)
        Estimator.__init__(self, kernel, lower, upper, grid_size, observations, sample_size, quadrature)
        self.kernel: Callable = kernel
        self.lower: float = float(lower)
        self.upper: float = float(upper)
        self.grid_size: int = grid_size
        self.__observations: Union[da.array, np.ndarray] = observations
        self.sample_size: int = sample_size
        self.relaxation: float = kwargs.get('relaxation')
        if self.relaxation is None:
            self.relaxation = 0.01
        self.max_iter: int = kwargs.get('max_iter')
        if self.max_iter is None:
            self.max_iter = 100
        self.initial: Union[da.array, np.ndarray] = kwargs.get('initial_guess')
        if self.initial is None:
            self.initial = da.repeat(da.from_array(np.array([0])), self.grid_size)
        self.previous: Union[da.array, np.ndarray] = self.initial
        self.current: Union[da.array, np.ndarray] = self.initial
        Operator.approximate(self)
        self.KHK: Union[da.array, np.ndarray] = np.matmul(self.KH, self.K)
        Estimator.estimate_q(self)
        Estimator.estimate_delta(self)

    @property
    def solution(self) -> Union[np.ndarray, da.array]:
        return self.previous

    @solution.setter
    def solution(self, solution: Union[np.ndarray, da.array]):
        self.previous = solution

    def L2norm(self, x: Union[da.array, np.ndarray], y: Union[da.array, np.ndarray], sqrt: bool = False) -> da.array:
        """
        Calculate the approximation of L2 norm of difference of two approximation of function.
        :param x: Approximation of function on given grid.
        :type x: Union[da.array, np.ndarray]
        :param y: Approximation of function on given grid.
        :type y: Union[da.array, np.ndarray]
        :param sqrt: To return the norm after taking the square root (True) or not (False).
        :type sqrt: boolean (default: False)
        :return: Float representing the L2 norm of difference between given functions.

        """
        grid: da.array = da.linspace(self.lower, self.upper, self.grid_size)
        weights: np.ndarray = self.quadrature(grid)
        if sqrt:
            norm: da.array = da.sqrt(da.sum(((x - y) ** 2) * weights))
        else:
            norm: da.array = da.sum(((x - y) ** 2) * weights)
        return norm

    @timer
    def __iteration(self) -> Union[np.ndarray, da.array]:
        """
        One iteration of Landweber algorithm.
        :return: Union[np.ndarray, da.array] with the next solution from algorithm.
        """
        bracket: Union[da.array, np.ndarray] = self.q_estimator - np.matmul(self.K, self.previous)
        self.current = self.previous + self.relaxation * np.matmul(self.KH, bracket)
        return self.current

    def __stopping_rule(self) -> bool:
        """
        Implementation of Morozov discrepancy stopping rule. If the distance between solution and observations is smaller
        than estimated noise level, then the algorithm will stop.
        :return: boolean representing whether the stop condition is reached (False) or not (True).
        """
        norm: float = self.L2norm(np.matmul(self.KHK, self.current), self.q_estimator).compute()
        return norm > self.delta

    def __update_solution(self):
        self.previous = self.current

    @timer
    def __force_computations(self):
        """
        Force computations of all dask computation graphs.
        """
        # self.K, self.KH, self.KHK, self.delta, self.q_estimator, self.previous, self.current = dask.optimize(
        #     self.K, self.KH, self.KHK, self.delta, self.q_estimator, self.previous, self.current)
        with ProgressBar():
            self.K, self.KH, self.KHK, self.delta, self.q_estimator, self.previous, self.current = dask.compute(
                self.K, self.KH, self.KHK, self.delta, self.q_estimator, self.previous, self.current,
                num_workers=cpu_count())

    def estimate(self, compute: bool = False):
        """
        Implementation of Landweber algorithm for inverse problem with stopping rule based on Morozov discrepancy principle.
        The algorithm is prevented to take longer than max_iter iterations.
        :param compute: Provide computations in form of numpy arrays (True) or dask graphs (False).
        :type compute: boolean (default: False)
        """
        it: int = 1
        start: float = time()
        if compute:
            print('Force computations...')
            self.__force_computations()
        condition: bool = self.__stopping_rule()
        while condition:
            print('Iteration: {}'.format(it))
            it += 1
            self.__update_solution()
            self.__iteration()
            condition = self.__stopping_rule()
            if it > self.max_iter:
                warn('Maximum number of iterations reached!', RuntimeWarning)
                break
        print('Total elapsed time: {}'.format(time() - start))

    def refresh(self):
        """
        Allow to re-estimate the q function, noise level and the target using new observations without need to recalculate
        the approximation of operator. To be used in conjunction with observations.setter.
        """
        self.previous = self.initial
        self.current = self.initial
        Estimator.estimate_q(self)
        Estimator.estimate_delta(self)
