from time import time
from typing import Callable, Union
from warnings import warn
import numpy as np
import cupy as cp
from scipy.linalg.blas import sgemm
from GeneralEstimator import Estimator
from Operator import Operator
from decorators import timer
from cupyx.scipy import linalg


class Landweber(Estimator, Operator):
    def __init__(self, kernel: Callable, lower: Union[float, int], upper: Union[float, int], grid_size: int,
                 observations: np.ndarray, sample_size: int, adjoint: bool = False, quadrature: str = 'rectangle',
                 **kwargs):
        """
        Instance of Landweber solver for inverse problem in Poisson noise with integral operator.
        :param kernel: Kernel of the integral operator.
        :type kernel: Callable
        :param lower: Lower end of the interval on which the operator is defined.
        :type lower: float
        :param upper: Upper end of the interval on which the operator is defined.
        :type lower: float
        :param grid_size: Size pf grid used to approximate the operator.
        :type grid_size: int
        :param observations: Observations used for the estimation.
        :type observations: numpy.ndarray
        :param sample_size: Theoretical sample size (n).
        :type sample_size: int
        :param adjoint: Whether the operator is adjoint (True) or not (False).
        :type adjoint: boolean (default: False)
        :param quadrature: Type of quadrature used to approximate integrals.
        :type quadrature: str (default: recatngle)
        :param kwargs: Possible arguments:
            - max_iter: The maximum number of iterations of the algorithm (int, default: 100).
            - tau: Parameter used to rescale the obtained values of estimated noise level (float, default: 1).
            - initial: Initial guess for the solution (numpy.ndarray, default: 0).
            - relaxation: Parameter used in the iteration of the algorithm (step size, omega). This approximate square norm
             of an operator is divide by the value of relaxation parameter (float, default: 2).
        """
        Operator.__init__(self, kernel, lower, upper, grid_size, adjoint, quadrature)
        Estimator.__init__(self, kernel, lower, upper, grid_size, observations, sample_size, quadrature)
        self.kernel: Callable = kernel
        self.lower: float = float(lower)
        self.upper: float = float(upper)
        self.grid_size: int = grid_size
        self.__observations: np.ndarray = observations.astype(np.float64)
        self.sample_size: int = sample_size
        self.max_iter: int = kwargs.get('max_iter', 100)
        self.__tau: float = kwargs.get('tau', 1.)
        self.initial: cp.ndarray = kwargs.get('initial_guess',
                                              cp.repeat(cp.array([0]), self.grid_size).astype(cp.float64))
        self.previous: cp.ndarray = cp.copy(self.initial).astype(cp.float64)
        self.current: cp.ndarray = cp.copy(self.initial).astype(cp.float64)
        Operator.approximate(self)
        self.__KHK: cp.ndarray = self.__premultiplication(self.KH, self.K)
        self.__relaxation: float = kwargs.get('relaxation', 2)
        self.__relaxation = 1 / cp.square(cp.linalg.norm(self.KHK)) / self.__relaxation
        Estimator.estimate_q(self)
        Estimator.estimate_delta(self)
        self.__grid: np.ndarray = getattr(super(), quadrature + '_grid')()

    @staticmethod
    @timer
    def __premultiplication(A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
        return cp.matmul(A, B)

    @property
    def relaxation(self) -> np.float64:
        return self.__relaxation

    @relaxation.setter
    def relaxation(self, relaxation: np.float64):
        self.__relaxation = 1 / cp.square(cp.linalg.norm(self.KHK)) / relaxation

    @property
    def tau(self) -> float:
        return self.__tau

    @tau.setter
    def tau(self, tau: float):
        self.__tau = tau

    # noinspection PyPep8Naming
    @property
    def KHK(self) -> cp.ndarray:
        return self.__KHK

    # noinspection PyPep8Naming
    @KHK.setter
    def KHK(self, KHK: cp.ndarray):
        self.__KHK = KHK.astype(cp.float64)

    @property
    def solution(self) -> cp.ndarray:
        return self.previous

    @solution.setter
    def solution(self, solution: cp.ndarray):
        self.previous = solution

    @property
    def grid(self) -> np.ndarray:
        return self.__grid

    # @grid.setter
    # def grid(self, grid: np.ndarray):
    #     self.__grid = grid.astype(np.float64)

    # @timer
    # def __iteration(self) -> np.ndarray:
    #     """
    #     One iteration of Landweber algorithm.
    #     :return: Numpy ndarray with the next approximation of solution from algorithm.
    #     """
    #     self.current = np.copy(
    #         np.add(self.previous, np.multiply(self.relaxation, np.matmul(self.KHK,
    #                                                                      np.subtract(self.q_estimator,
    #                                                                                  np.matmul(self.KHK,
    #                                                                                            self.previous)))))).astype(
    #         np.float64)
    #     return self.current

    @timer
    def __iteration(self) -> cp.ndarray:
        """
        One iteration of Landweber algorithm.
        :return: Numpy ndarray with the next approximation of solution from algorithm.
        """
        self.current = cp.add(self.previous, cp.multiply(self.relaxation,
                                                         cp.matmul(self.KHK, cp.subtract(self.q_estimator,
                                                                                         cp.matmul(self.KHK,
                                                                                                   self.previous)))))
        return self.current

    def __stopping_rule(self) -> bool:
        """
        Implementation of Morozov discrepancy stopping rule. If the distance between solution and observations is smaller
        than estimated noise level, then the algorithm will stop.
        :return: boolean representing whether the stop condition is reached (False) or not (True).
        """
        return self.L2norm(cp.matmul(self.KHK, self.current), self.q_estimator) > (self.tau * self.delta)

    def __update_solution(self):
        self.previous = cp.copy(self.current)

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


class Tikhonov(Estimator, Operator):
    def __init__(self, kernel: Callable, lower: Union[float, int], upper: Union[float, int], grid_size: int,
                 observations: np.ndarray, sample_size: int, order: int = 1, adjoint: bool = False,
                 quadrature: str = 'rectangle', **kwargs):
        """
        Instance of iterated Tikhonov solver for inverse problem in Poisson noise with integral operator.
        :param kernel: Kernel of the integral operator.
        :type kernel: Callable
        :param lower: Lower end of the interval on which the operator is defined.
        :type lower: float
        :param upper: Upper end of the interval on which the operator is defined.
        :type lower: float
        :param grid_size: Size pf grid used to approximate the operator.
        :type grid_size: int
        :param observations: Observations used for the estimation.
        :type observations: numpy.ndarray
        :param sample_size: Theoretical sample size (n).
        :type sample_size: int
        :param order: Order of the iterated algorithm. Estimator for each regularization parameter is obtained after
        order iterations. Ordinary Tikhonov estimator is obtained for order = 1.
        :type order: int (default: 1)
        :param adjoint: Whether the operator is adjoint (True) or not (False).
        :type adjoint: boolean (default: False)
        :param quadrature: Type of quadrature used to approximate integrals.
        :type quadrature: str (default: recatngle)
        :param kwargs: Possible arguments:
            - tau: Parameter used to rescale the obtained values of estimated noise level (float, default: 1).
            - parameter_space_size: Number of possible values of regularization parameter calculated as values between
            10^(-15) and 1 with step dictated by the parameter_space_size (int, default: 100).
        """
        Operator.__init__(self, kernel, lower, upper, grid_size, adjoint, quadrature)
        Estimator.__init__(self, kernel, lower, upper, grid_size, observations, sample_size, quadrature)
        self.kernel: Callable = kernel
        self.lower: float = float(lower)
        self.upper: float = float(upper)
        self.grid_size: int = grid_size
        self.__observations: np.ndarray = observations.astype(np.float64)
        self.sample_size: int = sample_size
        self.__order: int = order
        self.__tau: float = kwargs.get('tau', 1.)
        self.__parameter_space_size: int = kwargs.get('parameter_space_size', 100)
        self.parameter_grid: np.ndarray = np.flip(np.power(10, np.linspace(-15, 0, self.__parameter_space_size)))
        self.initial = cp.repeat(cp.array([0]), self.grid_size).astype(cp.float64)
        self.previous: cp.ndarray = cp.copy(self.initial).astype(cp.float64)
        self.current: cp.ndarray = cp.copy(self.initial).astype(cp.float64)
        self.__solution: cp.ndarray = cp.copy(self.initial).astype(cp.float64)
        self.__temporary_solution: cp.ndarray = cp.copy(self.initial).astype(cp.float64)
        Operator.approximate(self)
        self.__KHK: cp.ndarray = self.__premultiplication(self.KH, self.K)
        self.__KHKKHK: cp.ndarray = self.__premultiplication(self.KHK, self.KHK)
        self.identity: cp.ndarray = cp.identity(self.grid_size, dtype=cp.float64)
        Estimator.estimate_q(self)
        Estimator.estimate_delta(self)
        self.smoothed_q_estimator = cp.repeat(cp.array([0]), self.grid_size).astype(cp.float64)
        self.smoothed_q_estimator = cp.matmul(self.KHK, self.q_estimator)
        self.__grid: np.ndarray = getattr(super(), quadrature + '_grid')()

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
    def KHK(self) -> cp.ndarray:
        return self.__KHK

    # noinspection PyPep8Naming
    @KHK.setter
    def KHK(self, KHK: cp.ndarray):
        self.__KHK = KHK.astype(cp.float64)

    # noinspection PyPep8Naming
    @property
    def KHKKHK(self) -> cp.ndarray:
        return self.__KHKKHK

    # noinspection PyPep8Naming
    @KHKKHK.setter
    def KHKKHK(self, KHKKHK: cp.ndarray):
        self.__KHKKHK = KHKKHK.astype(cp.float64)

    @property
    def parameter_space_size(self) -> int:
        return self.__parameter_space_size

    @parameter_space_size.setter
    def parameter_space_size(self, parameter_space_size: int):
        self.__parameter_space_size = parameter_space_size

    @property
    def solution(self) -> cp.ndarray:
        return self.__solution

    @solution.setter
    def solution(self, solution: cp.ndarray):
        self.__solution = solution.astype(cp.float64)

    @property
    def grid(self) -> np.ndarray:
        return self.__grid

    # @grid.setter
    # def grid(self, grid: np.ndarray):
    #     self.__grid = grid

    # noinspection PyPep8Naming
    @staticmethod
    @timer
    def __premultiplication(A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
        return cp.matmul(A, B)

    def __update_solution(self):
        self.__temporary_solution = np.copy(self.current)

    def __stopping_rule(self) -> bool:
        """
        Implementation of Morozov discrepancy stopping rule. If the distance between solution and observations is smaller
        than estimated noise level, then the algorithm will stop.
        :return: boolean representing whether the stop condition is reached (False) or not (True).
        """
        return self.L2norm(cp.matmul(self.KHK, self.__temporary_solution), self.q_estimator) > (self.tau * self.delta)

    def __iteration(self, gamma: cp.float64):
        """
        Iteration of the inner loop of the (iterated) Tikhonov method.
        :param gamma: Regularization parameter of the Tikhonov algorithm.
        :type gamma: float
        :return: Numpy array with the solution in given iteration.
        """
        LU, P = linalg.lu_factor(cp.add(self.KHKKHK, cp.multiply(gamma, self.identity)))
        self.current = linalg.lu_solve((LU, P), cp.add(self.smoothed_q_estimator, cp.multiply(gamma, self.previous)))
        # self.current = cp.linalg.solve(cp.add(self.KHKKHK, cp.multiply(gamma, self.identity)),
        #                                cp.add(self.smoothed_q_estimator, cp.multiply(gamma, self.previous)))

    @timer
    def __estimate_one_step(self, gamma: cp.float64):
        """
        Estimation routine for one given gamma parameter of (iterated) Tikhonov method.
        :param gamma: Regularization parameter.
        :type gamma: float
        """
        order = 1
        while order <= self.order:
            self.__iteration(gamma)
            self.previous = cp.copy(self.current)
            order += 1
        self.__update_solution()
        self.previous = cp.copy(self.initial)

    def estimate(self):
        """
        Implementation of iterated Tikhonov algorithm for inverse problem with stopping rule based on Morozov discrepancy principle.
        If the algorithm did not converge, the initial solution is returned.
        """
        start: float = time()
        step = 1
        for gamma in self.parameter_grid:
            print('Number of search steps done: {} from {}'.format(step, self.parameter_space_size))
            step += 1
            self.__estimate_one_step(gamma)
            if not self.__stopping_rule():
                break
            self.__solution = cp.copy(self.__temporary_solution)
        if (step == self.parameter_space_size) and (np.array_equal(self.__solution, self.__temporary_solution)):
            warn('Algorithm did not converge over given parameter space!', RuntimeWarning)
            self.__solution = cp.copy(self.initial)
        print('Total elapsed time: {}'.format(time() - start))

    def refresh(self):
        """
        Allow to re-estimate the q function, noise level and the target using new observations without need to recalculate
        the approximation of operator. To be used in conjunction with observations.setter.
        """
        self.previous = cp.copy(self.initial)
        self.current = cp.copy(self.initial)
        self.solution = cp.copy(self.initial)
        self.parameter_grid: np.ndarray = np.power(10, np.linspace(-15, 0, self.__parameter_space_size))
        Estimator.estimate_q(self)
        Estimator.estimate_delta(self)
