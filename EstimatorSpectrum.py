from typing import Callable, Optional, Generator, Union

import numpy as np
from numba import njit
from scipy.integrate import quad

from GeneralEstimator import EstimatorSpectrum
from decorators import timer
from validate import validate_TSVD, validate_Tikhonov, validate_Landweber


class TSVD(EstimatorSpectrum):
    def __init__(self,
                 kernel: Callable,
                 singular_values: Generator,
                 left_singular_functions: Generator,
                 right_singular_functions: Generator,
                 observations: np.ndarray,
                 sample_size: int,
                 transformed_measure: bool,
                 rho: Union[int, float],
                 lower: Union[int, float] = 0,
                 upper: Union[int, float] = 1,
                 tau: Union[int, float] = 1,
                 max_size: int = 1000):
        """
        Instance of TSVD solver for inverse problem in Poisson noise with known spectral decomposition.
        :param kernel: Kernel of the integral operator.
        :type kernel: Callable
        :param singular_values: Singular values of the operator.
        :type singular_values: Generator yielding floats
        :param left_singular_functions: Left singular functions of the operator.
        :type left_singular_functions: Generator yielding callables.
        :param right_singular_functions: Right singular functions of the operator.
        :type right_singular_functions: Generator yielding callables.
        :param observations: Observations used for the estimation.
        :type observations: numpy.ndarray
        :param sample_size: Theoretical sample size (n).
        :type sample_size: int
        :param transformed_measure: To performed the calculations with respect to the transformed measure xdx (True) or
        to stay with Lebesgue measure dx (False)
        :type transformed_measure: boolean
        :param rho: Weight strength in discrepancy equation.
        :type rho: Union[int, float]
        :param lower: Lower end of the interval on which the operator is defined.
        :type lower: Union[int, float]
        :param upper: Upper end of the interval on which the operator is defined.
        :type upper: Union[int, float]
        :param tau: Parameter used to rescale the obtained values of estimated noise level.
        :type tau: Union[int, float] (default 1)
        :param max_size: Maximum number of functions included in Fourier expansion.
        :type max_size: int (default 1000)
        """
        validate_TSVD(tau)
        EstimatorSpectrum.__init__(self, kernel, observations, sample_size, rho, transformed_measure, singular_values,
                                   left_singular_functions, right_singular_functions, max_size)

        self.lower: Union[int, float] = lower
        self.upper: Union[int, float] = upper
        self.tau: Union[int, float] = tau
        self.regularization_param: float = 0.
        self.oracle_param: Optional[float] = None
        self.oracle_loss: Optional[float] = None
        self.oracle_solution: Optional[np.ndarray] = None
        self.residual: Optional[float] = None
        self.solution: Optional[Callable] = None

    @staticmethod
    @njit
    def __regularization(lam: np.ndarray, alpha: float) -> np.ndarray:
        """
        Truncated singular value regularization
        :param lam: argument of regularizing function
        :param alpha: regularizing parameter
        :return: Result of applying regularization function to argument lambda.
        """
        return np.where(lam >= alpha, np.divide(1, lam), 0)

    @timer
    def estimate(self) -> None:
        """
        Implementation of truncated singular value decomposition algorithm for inverse problem with stopping rule
        based on Morozov discrepancy principle.
        """
        self.estimate_delta()

        weighted_ui = np.divide(np.multiply(self.pi, np.square(self.ui)), self.sample_size ** 2)

        for alpha in np.square(np.concatenate([[np.inf], self.singular_values])):
            regularization = np.square(
                np.subtract(np.multiply(self.__regularization(np.square(self.singular_values), alpha),
                                        np.square(self.singular_values)), 1))
            residual = np.sum(np.sort(np.multiply(weighted_ui, regularization)))
            self.regularization_param = alpha
            if residual <= self.tau * self.delta:
                self.residual = residual
                break

        solution_operator_part = \
            np.divide(
                np.multiply(
                    self.singular_values,
                    self.__regularization(np.square(self.singular_values), self.regularization_param)),
                self.sample_size)

        solution_scalar_part = np.multiply(solution_operator_part, self.ui)

        if self.transformed_measure:
            def solution(x: float) -> np.ndarray:
                summand = np.multiply(solution_scalar_part,
                                      np.array([fun(x) * x for fun in self.right_singular_functions]))
                return np.sum(np.sort(summand))
        else:
            def solution(x: float) -> np.ndarray:
                summand = np.multiply(solution_scalar_part, np.array([fun(x) for fun in self.right_singular_functions]))
                return np.sum(np.sort(summand))

        self.solution = np.vectorize(solution)

    @timer
    def oracle(self, true: Callable, patience: int = 5) -> None:
        """
        Find the oracle regularization parameter which minimizes the L2 norm and knowing the true solution.
        :param true: True solution.
        :param patience: Number of consecutive iterations to observe the loss behavior after the minimum was found to
        prevent to stack in local minimum (default: 5).
        """
        losses = []
        parameters = []
        best_loss = np.inf
        counter = 0

        def residual(solution):
            return lambda x: np.square(true(x) - solution(x))

        for alpha in np.square(self.singular_values):
            parameters.append(alpha)

            solution_operator_part = \
                np.divide(
                    np.multiply(
                        self.singular_values,
                        self.__regularization(np.square(self.singular_values), alpha)),
                    self.sample_size)

            solution_scalar_part = np.multiply(solution_operator_part, self.ui)

            if self.transformed_measure:
                def solution(x: float) -> np.ndarray:
                    summand = np.multiply(solution_scalar_part,
                                          np.array([fun(x) * x for fun in self.right_singular_functions]))
                    return np.sum(np.sort(summand))
            else:
                def solution(x: float) -> np.ndarray:
                    summand = np.multiply(solution_scalar_part,
                                          np.array([fun(x) for fun in self.right_singular_functions]))
                    return np.sum(np.sort(summand))

            solution = np.vectorize(solution)
            res = residual(solution)
            loss = quad(res, self.lower, self.upper, limit=1000)[0]
            losses.append(loss)
            if loss <= best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1
            if counter == patience:
                break

        res = residual(solution=self.solution)
        self.residual = quad(res, self.lower, self.upper, limit=10000)[0]
        self.oracle_param = parameters[losses.index(min(losses))]
        self.oracle_loss = min(losses)

        solution_operator_part = \
            np.divide(
                np.multiply(
                    self.singular_values,
                    self.__regularization(np.square(self.singular_values), self.oracle_param)),
                self.sample_size)

        solution_scalar_part = np.multiply(solution_operator_part, self.ui)

        if self.transformed_measure:
            def solution(x: float) -> np.ndarray:
                summand = np.multiply(solution_scalar_part,
                                      np.array([fun(x) * x for fun in self.right_singular_functions]))
                return np.sum(np.sort(summand))
        else:
            def solution(x: float) -> np.ndarray:
                summand = np.multiply(solution_scalar_part,
                                      np.array([fun(x) for fun in self.right_singular_functions]))
                return np.sum(np.sort(summand))
        solution = np.vectorize(solution)

        self.oracle_solution = solution(np.linspace(0, 1, 10000))


class Tikhonov(EstimatorSpectrum):
    def __init__(self,
                 kernel: Callable,
                 singular_values: Generator,
                 left_singular_functions: Generator,
                 right_singular_functions: Generator,
                 observations: np.ndarray,
                 sample_size: int,
                 transformed_measure: bool,
                 rho: Union[int, float],
                 order: int = 2,
                 lower: Union[int, float] = 0,
                 upper: Union[int, float] = 1,
                 tau: Union[int, float] = 1,
                 max_size: int = 100):
        """
        Instance of iterated Tikhonov solver for inverse problem in Poisson noise with known spectral decomposition.
        :param kernel: Kernel of the integral operator.
        :type kernel: Callable
        :param singular_values: Singular values of the operator.
        :type singular_values: Generator yielding floats
        :param left_singular_functions: Left singular functions of the operator.
        :type left_singular_functions: Generator yielding callables.
        :param right_singular_functions: Right singular functions of the operator.
        :type right_singular_functions: Generator yielding callables.
        :param observations: Observations used for the estimation.
        :type observations: numpy.ndarray
        :param sample_size: Theoretical sample size (n).
        :type sample_size: int
        :param transformed_measure: To performed the calculations with respect to the transformed measure xdx (True) or
        to stay with Lebesgue measure dx (False)
        :type transformed_measure: boolean
        :param rho: Weight strength in discrepancy equation.
        :type rho: Union[int, float]
        :param order: Order of the iterated algorithm. Estimator for each regularization parameter is obtained after
                    order iterations. Ordinary Tikhonov estimator is obtained for order = 1
        :type order: int
        :param lower: Lower end of the interval on which the operator is defined.
        :type lower: Union[int, float]
        :param upper: Upper end of the interval on which the operator is defined.
        :type upper: Union[int, float]
        :param tau: Parameter used to rescale the obtained values of estimated noise level.
        :type tau: Union[int, float] (default 1)
        :param max_size: Maximum number of functions included in Fourier expansion.
        :type max_size: int (default 100)
        """
        validate_Tikhonov(order, tau)

        EstimatorSpectrum.__init__(self, kernel, observations, sample_size, rho, transformed_measure, singular_values,
                                   left_singular_functions, right_singular_functions, max_size)

        self.lower: float = lower
        self.upper: float = upper
        self.tau: float = tau
        self.max_size: int = max_size
        self.order: int = order
        self.regularization_param: float = 0.
        self.grid: np.ndarray = np.flip(np.linspace(0, 1, 1000))
        self.oracle_param: Optional[float] = None
        self.oracle_loss: Optional[float] = None
        self.oracle_solution: Optional[np.ndarray] = None
        self.residual: Optional[float] = None
        self.solution: Optional[Callable] = None

    @staticmethod
    @njit
    def __regularization(lam: np.ndarray, alpha: float, order: int) -> np.ndarray:
        """
        Iterated Tikhonov regularization.
        :param lam: argument of regularizing function
        :param alpha: regularizing parameter
        :param order: number of iteration \
        :return: Result of applying regularization function to argument lambda.
        """
        return np.divide(np.power(lam + alpha, order) - np.power(alpha, order),
                         np.multiply(lam, np.power(lam + alpha, order)))

    @timer
    def estimate(self) -> None:
        """
        Implementation of iterated Tikhonv algorithm for inverse problem with stopping rule based on Morozov discrepancy principle.
        """
        self.estimate_delta()

        weighted_ui = np.divide(np.multiply(self.pi, np.square(self.ui)), self.sample_size ** 2)

        for alpha in self.grid:
            regularization = np.square(
                np.subtract(np.multiply(self.__regularization(np.square(self.singular_values), alpha, self.order),
                                        np.square(self.singular_values)), 1))
            residual = np.sum(np.sort(np.multiply(weighted_ui, regularization)))
            self.regularization_param = alpha
            if residual <= self.tau * self.delta:
                self.residual = residual
                break

        solution_operator_part = \
            np.divide(
                np.multiply(
                    self.singular_values,
                    self.__regularization(np.square(self.singular_values), self.regularization_param, self.order)),
                self.sample_size)

        solution_scalar_part = np.multiply(solution_operator_part, self.ui)

        if self.transformed_measure:
            def solution(x: float) -> np.ndarray:
                summand = np.multiply(solution_scalar_part,
                                      np.array([fun(x) * x for fun in self.right_singular_functions]))
                return np.sum(np.sort(summand))
        else:
            def solution(x: float) -> np.ndarray:
                summand = np.multiply(solution_scalar_part, np.array([fun(x) for fun in self.right_singular_functions]))
                return np.sum(np.sort(summand))

        self.solution = np.vectorize(solution)

    @timer
    def oracle(self, true: Callable, patience: int = 10) -> None:
        """
        Find the oracle regularization parameter which minimizes the L2 norm and knowing the true solution.
        :param true: True solution.
        :param patience: Number of consecutive iterations to observe the loss behavior after the minimum was found to
        prevent to stack in local minimum (default: 10).
        """
        losses = []
        parameters = []
        best_loss = np.inf
        counter = 0

        def residual(solution):
            return lambda x: np.square(true(x) - solution(x))

        for alpha in self.grid:
            parameters.append(alpha)

            solution_operator_part = \
                np.divide(
                    np.multiply(
                        self.singular_values,
                        self.__regularization(np.square(self.singular_values), alpha, self.order)),
                    self.sample_size)

            solution_scalar_part = np.multiply(solution_operator_part, self.ui)

            if self.transformed_measure:
                def solution(x: float) -> np.ndarray:
                    summand = np.multiply(solution_scalar_part,
                                          np.array([fun(x) * x for fun in self.right_singular_functions]))
                    return np.sum(np.sort(summand))
            else:
                def solution(x: float) -> np.ndarray:
                    summand = np.multiply(solution_scalar_part,
                                          np.array([fun(x) for fun in self.right_singular_functions]))
                    return np.sum(np.sort(summand))

            solution = np.vectorize(solution)
            res = residual(solution)
            loss = quad(res, self.lower, self.upper, limit=1000)[0]
            losses.append(loss)
            if loss <= best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1
            if counter == patience:
                break

        res = residual(solution=self.solution)
        self.residual = quad(res, self.lower, self.upper, limit=10000)[0]
        self.oracle_param = parameters[losses.index(min(losses))]
        self.oracle_loss = min(losses)

        solution_operator_part = \
            np.divide(
                np.multiply(
                    self.singular_values,
                    self.__regularization(np.square(self.singular_values), self.oracle_param, self.order)),
                self.sample_size)

        solution_scalar_part = np.multiply(solution_operator_part, self.ui)

        if self.transformed_measure:
            def solution(x: float) -> np.ndarray:
                summand = np.multiply(solution_scalar_part,
                                      np.array([fun(x) * x for fun in self.right_singular_functions]))
                return np.sum(np.sort(summand))
        else:
            def solution(x: float) -> np.ndarray:
                summand = np.multiply(solution_scalar_part,
                                      np.array([fun(x) for fun in self.right_singular_functions]))
                return np.sum(np.sort(summand))
        solution = np.vectorize(solution)

        self.oracle_solution = solution(np.linspace(0, 1, 10000))


class Landweber(EstimatorSpectrum):
    def __init__(self,
                 kernel: Callable,
                 singular_values: Generator,
                 left_singular_functions: Generator,
                 right_singular_functions: Generator,
                 observations: np.ndarray,
                 sample_size: int,
                 transformed_measure: bool,
                 rho: Union[int, float],
                 relaxation: Union[int, float] = 0.8,
                 max_iter: int = 1000,
                 lower: Union[int, float] = 0,
                 upper: Union[int, float] = 1,
                 tau: Union[int, float] = 1,
                 max_size: int = 100):
        """
        Instance of iterated Landweber solver for inverse problem in Poisson noise with known spectral decomposition.
        :param kernel: Kernel of the integral operator.
        :type kernel: Callable
        :param singular_values: Singular values of the operator.
        :type singular_values: Generator yielding floats
        :param left_singular_functions: Left singular functions of the operator.
        :type left_singular_functions: Generator yielding callables.
        :param right_singular_functions: Right singular functions of the operator.
        :type right_singular_functions: Generator yielding callables.
        :param observations: Observations used for the estimation.
        :type observations: numpy.ndarray
        :param sample_size: Theoretical sample size (n).
        :type sample_size: int
        :param transformed_measure: To performed the calculations with respect to the transformed measure xdx (True) or
        to stay with Lebesgue measure dx (False)
        :type transformed_measure: boolean
        :param rho: Weight strength in discrepancy equation.
        :type rho: Union[int, float]
        :param relaxation: Parameter used in the iteration of the algorithm (step size, omega). The square of the first
            singular value is scaled by this value
        :type relaxation: Union[int, float] (default: 0.8)
        :param max_iter: Maximum number of iterations of the algorithm
        :type max_iter: int (default: 100)
        :param lower: Lower end of the interval on which the operator is defined.
        :type lower: Union[int, float]
        :param upper: Upper end of the interval on which the operator is defined.
        :type upper: Union[int, float]
        :param tau: Parameter used to rescale the obtained values of estimated noise level.
        :type tau: Union[int, float] (default 1)
        :param max_size: Maximum number of functions included in Fourier expansion.
        :type max_size: int (default 1000)
        """
        validate_Landweber(relaxation, max_iter, tau)

        EstimatorSpectrum.__init__(self, kernel, observations, sample_size, rho, transformed_measure, singular_values,
                                   left_singular_functions, right_singular_functions, max_size)

        self.lower: float = lower
        self.upper: float = upper
        self.tau: float = tau
        self.max_iter: int = max_iter
        self.relaxation: float = relaxation
        self.regularization_param: int = 0
        self.oracle_param: Optional[float] = None
        self.oracle_loss: Optional[float] = None
        self.oracle_solution: Optional[np.ndarray] = None
        self.residual: Optional[float] = None
        self.solution: Optional[Callable] = None

    def __adjust_relaxation(self) -> None:
        """
        Collect a max_size number of singular values.
        """
        self.relaxation = self.relaxation / (self.singular_values[0] ** 2)

    @staticmethod
    def __regularization(lam: np.ndarray, k: int, beta: float) -> np.ndarray:
        """
        Landweber regularization with relaxation parameter beta.
        :param lam: argument of regularizing function
        :param k: regularizing parameter (number of iterations)
        :param beta: relaxation parameter
        :return: Result of applying regularization function to argument lambda.
        """
        if not k:
            return np.multiply(lam, 0)
        else:
            iterations = [np.power(np.subtract(1, np.multiply(beta, lam)), j) for j in range(k)]
            iterations = np.stack(iterations, axis=1)
            regularization = np.sum(iterations, axis=1) * beta
            return regularization

    @timer
    def estimate(self) -> None:
        """
        Implementation of Landweber algorithm for inverse problem with stopping rule based on Morozov discrepancy principle.
        """

        self.estimate_delta()
        self.__adjust_relaxation()

        weighted_ui = np.divide(np.multiply(self.pi, np.square(self.ui)), self.sample_size ** 2)

        for k in np.arange(0, self.max_iter):
            regularization = np.square(
                np.subtract(np.multiply(self.__regularization(np.square(self.singular_values), k, self.relaxation),
                                        np.square(self.singular_values)), 1))
            residual = np.sum(np.sort(np.multiply(weighted_ui, regularization)))
            self.regularization_param = k
            if residual <= self.tau * self.delta:
                self.residual = residual
                break

        solution_operator_part = \
            np.divide(
                np.multiply(
                    self.singular_values,
                    self.__regularization(np.square(self.singular_values), self.regularization_param, self.relaxation)),
                self.sample_size)

        solution_scalar_part = np.multiply(solution_operator_part, self.ui)

        if self.transformed_measure:
            def solution(x: float) -> np.ndarray:
                summand = np.multiply(solution_scalar_part,
                                      np.array([fun(x) * x for fun in self.right_singular_functions]))
                return np.sum(np.sort(summand))
        else:
            def solution(x: float) -> np.ndarray:
                summand = np.multiply(solution_scalar_part, np.array([fun(x) for fun in self.right_singular_functions]))
                return np.sum(np.sort(summand))

        self.solution = np.vectorize(solution)

    @timer
    def oracle(self, true: Callable, patience: int = 5) -> None:
        """
        Find the oracle regularization parameter which minimizes the L2 norm and knowing the true solution.
        :param true: True solution.
        :param patience: Number of consecutive iterations to observe the loss behavior after the minimum was found to
        prevent to stack in local minimum (default: 5).
        """
        losses = []
        parameters = []
        best_loss = np.inf
        counter = 0

        def residual(solution):
            return lambda x: np.square(true(x) - solution(x))

        start_k = self.regularization_param // 2 if self.regularization_param != self.max_iter else 0

        for k in np.arange(start_k, self.max_iter):
            parameters.append(k)

            solution_operator_part = \
                np.divide(
                    np.multiply(
                        self.singular_values,
                        self.__regularization(np.square(self.singular_values), k, self.relaxation)),
                    self.sample_size)

            solution_scalar_part = np.multiply(solution_operator_part, self.ui)

            if self.transformed_measure:
                def solution(x: float) -> np.ndarray:
                    summand = np.multiply(solution_scalar_part,
                                          np.array([fun(x) * x for fun in self.right_singular_functions]))
                    return np.sum(np.sort(summand))
            else:
                def solution(x: float) -> np.ndarray:
                    summand = np.multiply(solution_scalar_part,
                                          np.array([fun(x) for fun in self.right_singular_functions]))
                    return np.sum(np.sort(summand))

            solution = np.vectorize(solution)
            res = residual(solution)
            loss = quad(res, self.lower, self.upper, limit=1000)[0]
            losses.append(loss)
            if loss <= best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1
            if counter == patience:
                break

        res = residual(solution=self.solution)
        self.residual = quad(res, self.lower, self.upper, limit=10000)[0]
        self.oracle_param = parameters[losses.index(min(losses))]
        self.oracle_loss = min(losses)

        solution_operator_part = \
            np.divide(
                np.multiply(
                    self.singular_values,
                    self.__regularization(np.square(self.singular_values), self.oracle_param, self.relaxation)),
                self.sample_size)

        solution_scalar_part = np.multiply(solution_operator_part, self.ui)

        if self.transformed_measure:
            def solution(x: float) -> np.ndarray:
                summand = np.multiply(solution_scalar_part,
                                      np.array([fun(x) * x for fun in self.right_singular_functions]))
                return np.sum(np.sort(summand))
        else:
            def solution(x: float) -> np.ndarray:
                summand = np.multiply(solution_scalar_part,
                                      np.array([fun(x) for fun in self.right_singular_functions]))
                return np.sum(np.sort(summand))
        solution = np.vectorize(solution)

        self.oracle_solution = solution(np.linspace(0, 1, 10000))
