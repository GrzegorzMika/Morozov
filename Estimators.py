from time import time
from typing import Callable, Union
from warnings import warn
import numpy as np
from GeneralEstimator import Estimator
from Operator import Operator
from decorators import timer
from numba import jit
from scipy.linalg.blas import sgemm


class Landweber(Estimator, Operator):
    def __init__(self, kernel: Callable, lower: Union[float, int], upper: Union[float, int], grid_size: int,
                 observations: np.ndarray, sample_size: int, quadrature: str = 'rectangle', **kwargs):
        Operator.__init__(self, kernel, lower, upper, grid_size, quadrature)
        Estimator.__init__(self, kernel, lower, upper, grid_size, observations, sample_size, quadrature)
        self.kernel: Callable = kernel
        self.lower: float = float(lower)
        self.upper: float = float(upper)
        self.grid_size: int = grid_size
        self.__observations: np.ndarray = observations.astype(np.float64)
        self.sample_size: int = sample_size
        self.relaxation: float = kwargs.get('relaxation')
        if self.relaxation is None:
            self.relaxation = 0.01
        self.max_iter: int = kwargs.get('max_iter')
        if self.max_iter is None:
            self.max_iter = 100
        self.initial: np.ndarray = kwargs.get('initial_guess')
        if self.initial is None:
            self.initial = np.repeat(np.array([0]), self.grid_size).astype(np.float64)
        self.previous: np.ndarray = self.initial.astype(np.float64)
        self.current: np.ndarray = self.initial.astype(np.float64)
        Operator.approximate(self)
        self.__KHK: np.ndarray = self.__premultiplication()
        Estimator.estimate_q(self)
        Estimator.estimate_delta(self)
        self.__grid: np.ndarray = np.linspace(self.lower, self.upper, self.grid_size)
        self.__weights: np.ndarray = self.quadrature(self.__grid)
        if 1/np.square(np.linalg.norm(self.K)) > self.relaxation:
            warn("Relaxation parameter is probably too big, inverse of "
                 "estimated operator norm is equal to {}".format(1/np.square(np.linalg.norm(self.K))), RuntimeWarning)

    # noinspection PyPep8Naming
    @timer
    def __premultiplication(self) -> np.ndarray:
        """
        Perform a pre-multiplication of adjoint matrix and matrix
        @return: Numpy array with multiplication of adjoint operator and operator
        """
        # KHK: np.ndarray = np.zeros((self.grid_size, self.grid_size)).astype(np.float64)
        return sgemm(1.0, self.KH, self.K)

    # noinspection PyPep8Naming
    @property
    def KHK(self) -> np.ndarray:
        return self.__KHK

    # noinspection PyPep8Naming
    @KHK.setter
    def KHK(self, KHK: np.ndarray):
        self.__KHK = KHK

    @property
    def solution(self) -> np.ndarray:
        return self.previous

    @solution.setter
    def solution(self, solution: np.ndarray):
        self.previous = solution

    @staticmethod
    @jit(nopython=True)
    def __L2norm(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return np.sum(np.multiply(np.square(np.subtract(x, y)), weights))

    def L2norm(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the approximation of L2 norm of difference of two approximation of function (not the square root).
        :param x: Approximation of function on given grid.
        :type x: np.ndarray
        :param y: Approximation of function on given grid.
        :type y: np.ndarray
        :param sqrt: To return the norm after taking the square root (True) or not (False).
        :type sqrt: boolean (default: False)
        :return: Float representing the L2 norm of difference between given functions.

        """
        return self.__L2norm(x, y, self.__weights)

    @timer
    def __iteration(self) -> np.ndarray:
        """
        One iteration of Landweber algorithm.
        :return: Numpy ndarray with the next approximation of solution from algorithm.
        """
        np.add(self.previous, np.multiply(self.relaxation, np.matmul(self.KH,
                                np.subtract(self.q_estimator, np.matmul(self.K, self.previous)))), out=self.current)
        return self.current

    def __stopping_rule(self) -> bool:
        """
        Implementation of Morozov discrepancy stopping rule. If the distance between solution and observations is smaller
        than estimated noise level, then the algorithm will stop.
        :return: boolean representing whether the stop condition is reached (False) or not (True).
        """
        return self.L2norm(np.matmul(self.KHK, self.current), self.q_estimator) > self.delta

    def __update_solution(self):
        self.previous = self.current

    def estimate(self):
        """
        Implementation of Landweber algorithm for inverse problem with stopping rule based on Morozov discrepancy principle.
        The algorithm is prevented to take longer than max_iter iterations.
        """
        it: int = 1
        start: float = time()
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
