from time import time
from typing import Callable, Union
from warnings import warn
import numpy as np
from scipy.linalg.blas import sgemm
from GeneralEstimator import Estimator
from Operator import Operator
from decorators import timer


# TODO: implement tests
# TODO: what with higher precision? (compliance with BLAS spec)

class Landweber(Estimator, Operator):
    def __init__(self, kernel: Callable, lower: Union[float, int], upper: Union[float, int], grid_size: int,
                 observations: np.ndarray, sample_size: int, adjoint: bool = False, quadrature: str = 'rectangle',
                 **kwargs):
        Operator.__init__(self, kernel, lower, upper, grid_size, adjoint, quadrature)
        Estimator.__init__(self, kernel, lower, upper, grid_size, observations, sample_size, quadrature)
        self.kernel: Callable = kernel
        self.lower: float = float(lower)
        self.upper: float = float(upper)
        self.grid_size: int = grid_size
        self.__observations: np.ndarray = observations.astype(np.float64)
        self.sample_size: int = sample_size
        self.max_iter: int = kwargs.get('max_iter')
        if self.max_iter is None:
            self.max_iter = 100
        self.tau: float = kwargs.get('tau')
        if self.tau is None:
            self.tau = 1.
        self.initial: np.ndarray = kwargs.get('initial_guess')
        if self.initial is None:
            self.initial = np.repeat(np.array([0]), self.grid_size).astype(np.float64)
        self.previous: np.ndarray = np.copy(self.initial.astype(np.float64))
        self.current: np.ndarray = np.copy(self.initial.astype(np.float64))
        Operator.approximate(self)
        self.__KHK: np.ndarray = self.__premultiplication(self.KH, self.K)
        self.relaxation: float = kwargs.get('relaxation')
        if self.relaxation is None:
            self.relaxation = 1 / np.square(np.linalg.norm(self.KHK)) / 2
        else:
            self.relaxation = 1 / np.square(np.linalg.norm(self.KHK)) / self.relaxation
        if 1 / np.square(np.linalg.norm(self.KHK)) < self.relaxation:
            warn("Relaxation parameter is probably too big, inverse of "
                 "estimated operator norm is equal to {}".format(1 / np.square(np.linalg.norm(self.KHK))),
                 RuntimeWarning)
        Estimator.estimate_q(self)
        Estimator.estimate_delta(self)
        self.__grid: np.ndarray = getattr(super(), quadrature + '_grid')()
        self.__weights: np.ndarray = self.quadrature(self.__grid)

    # noinspection PyPep8Naming
    @staticmethod
    @timer
    def __premultiplication(A, B) -> np.ndarray:
        """
        Perform a pre-multiplication of adjoint matrix and matrix
        :return: Numpy array with result of multiplication
        """
        return sgemm(1.0, A, B)

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

    @property
    def grid(self) -> np.ndarray:
        return self.__grid

    @grid.setter
    def grid(self, grid: np.ndarray):
        self.__grid = grid

    @timer
    def __iteration(self) -> np.ndarray:
        """
        One iteration of Landweber algorithm.
        :return: Numpy ndarray with the next approximation of solution from algorithm.
        """
        self.current = np.copy(
            np.add(self.previous, np.multiply(self.relaxation, np.matmul(self.KHK,
                                                                         np.subtract(self.q_estimator,
                                                                                     np.matmul(self.KHK,
                                                                                               self.previous))))))
        return self.current

    def __stopping_rule(self) -> bool:
        """
        Implementation of Morozov discrepancy stopping rule. If the distance between solution and observations is smaller
        than estimated noise level, then the algorithm will stop.
        :return: boolean representing whether the stop condition is reached (False) or not (True).
        """
        return self.L2norm(np.matmul(self.KHK, self.current), self.q_estimator) > (self.tau * self.delta)

    def __update_solution(self):
        self.previous = np.copy(self.current)

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


class Tikhonov(Operator, Estimator):
    def __init__(self, kernel: Callable, lower: Union[float, int], upper: Union[float, int], grid_size: int,
                 observations: np.ndarray, sample_size: int, order: int = 1, adjoint: bool = False,
                 quadrature: str = 'rectangle',
                 **kwargs):
        Operator.__init__(self, kernel, lower, upper, grid_size, adjoint, quadrature)
        Estimator.__init__(self, kernel, lower, upper, grid_size, observations, sample_size, quadrature)
        self.kernel: Callable = kernel
        self.lower: float = float(lower)
        self.upper: float = float(upper)
        self.grid_size: int = grid_size
        self.__observations: np.ndarray = observations.astype(np.float64)
        self.sample_size: int = sample_size
        self.__order: int = order
        self.tau: float = kwargs.get('tau')
        if self.tau is None:
            self.tau = 1.
        self.__parameter_space_size = kwargs.get('parameter_space_size')
        if self.__parameter_space_size is None:
            self.__parameter_space_size = 100
        self.initial = np.repeat(np.array([0]), self.grid_size).astype(np.float64)
        self.previous: np.ndarray = np.copy(self.initial.astype(np.float64))
        self.current: np.ndarray = np.copy(self.initial.astype(np.float64))
        Operator.approximate(self)
        self.__KHK: np.ndarray = self.__premultiplication(self.KH, self.K)
        self.__KHKKHK: np.ndarray = self.__premultiplication(self.KHK, self.KHK)
        Estimator.estimate_q(self)
        Estimator.estimate_delta(self)
        self.smoothed_q_estimator = np.repeat(np.array([0]), self.grid_size).astype(np.float64)
        self.smoothed_q_estimator = np.matmul(self.KHK, self.q_estimator)
        self.__grid: np.ndarray = getattr(super(), quadrature + '_grid')()
        self.__weights: np.ndarray = self.quadrature(self.__grid)

    @property
    def order(self) -> int:
        return self.__order

    @order.setter
    def order(self, order: int):
        self.__order = order

    # noinspection PyPep8Naming
    @property
    def KHK(self) -> np.ndarray:
        return self.__KHK

    # noinspection PyPep8Naming
    @KHK.setter
    def KHK(self, KHK: np.ndarray):
        self.__KHK = KHK

    # noinspection PyPep8Naming
    @property
    def KHKKHK(self) -> np.ndarray:
        return self.__KHKKHK

    # noinspection PyPep8Naming
    @KHKKHK.setter
    def KHKKHK(self, KHKKHK: np.ndarray):
        self.__KHKKHK = KHKKHK

    @property
    def parameter_space_size(self) -> int:
        return self.__parameter_space_size

    @parameter_space_size.setter
    def parameter_space_size(self, parameter_space_size: int):
        self.__parameter_space_size = parameter_space_size

    @property
    def solution(self) -> np.ndarray:
        return self.previous

    @solution.setter
    def solution(self, solution: np.ndarray):
        self.previous = solution

    @property
    def grid(self) -> np.ndarray:
        return self.__grid

    @grid.setter
    def grid(self, grid: np.ndarray):
        self.__grid = grid

    # noinspection PyPep8Naming
    @staticmethod
    @timer
    def __premultiplication(A, B) -> np.ndarray:
        """
        Perform a pre-multiplication of adjoint matrix and matrix
        :return: Numpy array with result of multiplication
        """
        return sgemm(1.0, A, B)

    def __update_solution(self):
        self.previous = np.copy(self.current)