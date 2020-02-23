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
# TODO: be careful with data types
# TODO: if not convergent return 0

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
        self.__tau: float = kwargs.get('tau')
        if self.tau is None:
            self.__tau = 1.
        self.initial: np.ndarray = kwargs.get('initial_guess')
        if self.initial is None:
            self.initial = np.repeat(np.array([0]), self.grid_size).astype(np.float64)
        self.previous: np.ndarray = np.copy(self.initial).astype(np.float64)
        self.current: np.ndarray = np.copy(self.initial).astype(np.float64)
        Operator.approximate(self)
        self.__KHK: np.ndarray = self.__premultiplication(self.KH, self.K)
        self.__relaxation: float = kwargs.get('relaxation')
        if self.__relaxation is None:
            self.__relaxation = 1 / np.square(np.linalg.norm(self.KHK)) / 2
        else:
            self.__relaxation = 1 / np.square(np.linalg.norm(self.KHK)) / self.__relaxation
        Estimator.estimate_q(self)
        Estimator.estimate_delta(self)
        self.__grid: np.ndarray = getattr(super(), quadrature + '_grid')()
        self.__weights: np.ndarray = self.quadrature(self.__grid)

    # noinspection PyPep8Naming
    @staticmethod
    @timer
    def __premultiplication(A, B) -> np.ndarray:
        """
        Perform a pre-multiplication of two matrices
        :return: Numpy array with result of multiplication
        """
        return sgemm(1.0, A, B)

    @property
    def relaxation(self) -> np.float64:
        return self.__relaxation

    @relaxation.setter
    def relaxation(self, relaxation: np.float64):
        self.__relaxation = 1 / np.square(np.linalg.norm(self.KHK)) / relaxation

    @property
    def tau(self) -> float:
        return self.__tau

    @tau.setter
    def tau(self, tau: float):
        self.__tau = tau

    # noinspection PyPep8Naming
    @property
    def KHK(self) -> np.ndarray:
        return self.__KHK

    # noinspection PyPep8Naming
    @KHK.setter
    def KHK(self, KHK: np.ndarray):
        self.__KHK = KHK.astype(np.float64)

    @property
    def solution(self) -> np.ndarray:
        return self.previous

    @solution.setter
    def solution(self, solution: np.ndarray):
        self.previous = solution

    @property
    def grid(self) -> np.ndarray:
        return self.__grid

    # @grid.setter
    # def grid(self, grid: np.ndarray):
    #     self.__grid = grid.astype(np.float64)

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
                                                                                               self.previous)))))).astype(
            np.float64)
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
        The algorithm is prevented to take longer than max_iter iterations. If the algorithm did not converge, the initial
        solution is returned.
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
                self.previous = self.initial
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
        self.__parameter_space_size: int = kwargs.get('parameter_space_size')
        if self.__parameter_space_size is None:
            self.__parameter_space_size = 100
        self.parameter_grid: np.ndarray = np.flip(np.power(10, np.linspace(-15, 0, self.__parameter_space_size)))
        self.initial = np.repeat(np.array([0]), self.grid_size).astype(np.float64)
        self.previous: np.ndarray = np.copy(self.initial).astype(np.float64)
        self.current: np.ndarray = np.copy(self.initial).astype(np.float64)
        self.__solution: np.ndarray = np.copy(self.initial).astype(np.float64)
        self.__temporary_solution: np.ndarray = np.copy(self.initial).astype(np.float64)
        Operator.approximate(self)
        self.__KHK: np.ndarray = self.__premultiplication(self.KH, self.K)
        self.__KHKKHK: np.ndarray = self.__premultiplication(self.KHK, self.KHK)
        self.identity = np.identity(self.grid_size, dtype=np.float64)
        Estimator.estimate_q(self)
        Estimator.estimate_delta(self)
        self.smoothed_q_estimator = np.repeat(np.array([0]), self.grid_size).astype(np.float64)
        self.smoothed_q_estimator = np.matmul(self.KHK, self.q_estimator)
        self.__grid: np.ndarray = getattr(super(), quadrature + '_grid')()
        self.__weights: np.ndarray = self.quadrature(self.__grid)

    @property
    def tau(self) -> float:
        return self.__tau

    @tau.setter
    def tau(self, tau: float):
        self.__tau = tau

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
        self.__KHK = KHK.astype(np.float64)

    # noinspection PyPep8Naming
    @property
    def KHKKHK(self) -> np.ndarray:
        return self.__KHKKHK

    # noinspection PyPep8Naming
    @KHKKHK.setter
    def KHKKHK(self, KHKKHK: np.ndarray):
        self.__KHKKHK = KHKKHK.astype(np.float64)

    @property
    def parameter_space_size(self) -> int:
        return self.__parameter_space_size

    @parameter_space_size.setter
    def parameter_space_size(self, parameter_space_size: int):
        self.__parameter_space_size = parameter_space_size

    @property
    def solution(self) -> np.ndarray:
        return self.__solution

    @solution.setter
    def solution(self, solution: np.ndarray):
        self.__solution = solution.astype(np.float64)

    @property
    def grid(self) -> np.ndarray:
        return self.__grid

    # @grid.setter
    # def grid(self, grid: np.ndarray):
    #     self.__grid = grid

    # noinspection PyPep8Naming
    @staticmethod
    @timer
    def __premultiplication(A, B) -> np.ndarray:
        """
        Perform a pre-multiplication of two matrices
        :return: Numpy array with result of multiplication
        """
        return sgemm(1.0, A, B)

    def __update_solution(self):
        self.__temporary_solution = np.copy(self.current)

    def __stopping_rule(self) -> bool:
        """
        Implementation of Morozov discrepancy stopping rule. If the distance between solution and observations is smaller
        than estimated noise level, then the algorithm will stop.
        :return: boolean representing whether the stop condition is reached (False) or not (True).
        """
        return self.L2norm(np.matmul(self.KHK, self.__temporary_solution), self.q_estimator) > (self.tau * self.delta)

    def __iteration(self, gamma: np.float64) -> np.ndarray:
        try:
            self.current = np.linalg.solve(self.KHKKHK + gamma * self.identity,
                                           self.smoothed_q_estimator + gamma * self.previous)
        except np.linalg.LinAlgError:
            warn('Gamma parameter is too small!', RuntimeWarning)
        return self.current

    @timer
    def __estimate_one_step(self, gamma: np.float64):
        order = 1
        while order <= self.order:
            self.__iteration(gamma)
            self.previous = np.copy(self.current)
            order += 1
        self.__update_solution()
        self.previous = np.copy(self.initial)

    def estimate(self):
        start: float = time()
        step = 1
        for gamma in self.parameter_grid:
            print('Number of search steps done: {}'.format(step))
            step += 1
            self.__estimate_one_step(gamma)
            if not self.__stopping_rule():
                break
            self.__solution = np.copy(self.__temporary_solution)
        print('Total elapsed time: {}'.format(time() - start))

    def refresh(self):
        """
        Allow to re-estimate the q function, noise level and the target using new observations without need to recalculate
        the approximation of operator. To be used in conjunction with observations.setter.
        """
        self.previous = self.initial
        self.current = self.initial
        self.parameter_grid: np.ndarray = np.power(10, np.linspace(-15, 0, self.__parameter_space_size))
        Estimator.estimate_q(self)
        Estimator.estimate_delta(self)
