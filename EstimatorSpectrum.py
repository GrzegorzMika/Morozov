import inspect
from multiprocessing import cpu_count
from typing import Callable, Optional, Generator, Iterable, Union

import numpy as np
from dask.distributed import Client
from joblib import Memory
from numba import njit
from scipy.integrate import quad

from GeneralEstimator import EstimatorSpectrum
from decorators import timer
from validate import validate_TSVD, validate_Tikhonov, validate_Landweber

location = './cachedir'
memory = Memory(location, verbose=0, bytes_limit=1024 * 1024 * 1024)


class TSVD(EstimatorSpectrum):
    def __init__(self, kernel: Callable, singular_values: Generator, left_singular_functions: Generator,
                 right_singular_functions: Generator, observations: np.ndarray, sample_size: int,
                 transformed_measure: bool, rho: Union[int, float], lower: Union[int, float] = 0,
                 upper: Union[int, float] = 1, tau: Union[int, float] = 1, max_size: int = 100, njobs: Optional[int] = -1):
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
        :type max_size: int (default 100)
        :param njobs: Number of threds to be used to calculate Fourier expansion, negative means all available.
        :type njobs: int (default -1)
        """
        validate_TSVD(kernel, singular_values, left_singular_functions, right_singular_functions, observations,
                      sample_size, transformed_measure, rho, lower, upper, tau, max_size, njobs)
        EstimatorSpectrum.__init__(self, kernel, observations, sample_size, transformed_measure, rho, lower, upper)
        self.kernel: Callable = kernel
        self.singular_values: Generator = singular_values
        self.left_singular_functions: Generator = left_singular_functions
        self.right_singular_functions: Generator = right_singular_functions
        self.observations: np.ndarray = observations
        self.sample_size: int = sample_size
        self.lower: Union[int, float] = lower
        self.upper: Union[int, float] = upper
        self.tau: Union[int, float] = tau
        self.max_size: int = max_size
        self.q_fourier_coeffs: np.ndarray = np.repeat([0.], self.max_size)
        self.sigmas: np.ndarray = np.repeat([0.], self.max_size)
        self.regularization_param: float = 0.
        self.oracle_param: Optional[float] = None
        self.oracle_loss: Optional[float] = None
        self.oracle_solution: Optional[np.ndarray] = None
        self.residual: Optional[float] = None
        self.vs: list = []
        self.solution: Optional[Callable] = None
        if njobs is None or njobs < 0 or not isinstance(njobs, int):
            njobs = cpu_count()
        self.client = Client(threads_per_worker=1, n_workers=njobs)
        print('Dashboard available under: {}'.format(self.client.dashboard_link))

    @timer
    def __find_fourier_coeffs(self) -> None:
        """
        Calculate max_size Fourier coefficients of a q estimator with respect the the right singular functions of an operator.
        Coefficients are calculated in parallel using the dask backend, the progress is displayed on Dask dashbord running
        by default on localhost://8787.
        """
        self.estimate_q()
        print('Calculation of Fourier coefficients of q estimator...')
        q_estimator = self.q_estimator
        lower = self.lower
        upper = self.upper
        client = self.client

        if self.transformed_measure:
            def product(function: Callable) -> Callable:
                return lambda x: q_estimator(x) * function(x) * x
        else:
            def product(function: Callable) -> Callable:
                return lambda x: q_estimator(x) * function(x)

        def integrate(function: Callable) -> float:
            return quad(function, lower, upper, limit=10000)[0]

        products: Iterable = map(product, self.vs)

        @memory.cache
        def fourier_coeffs_helper_TSVD(observations: np.ndarray, kernel_function: str):
            futures = []
            for i, fun in enumerate(products):
                futures.append(client.submit(integrate, fun))
            return client.gather(futures)

        coeffs = fourier_coeffs_helper_TSVD(self.observations, inspect.getsource(self.kernel))
        self.q_fourier_coeffs = np.array(coeffs)

    def __singular_values(self) -> None:
        """
        Collect a max_size number of singular values.
        """
        sigma = [next(self.singular_values) for _ in range(self.max_size)]
        self.sigmas = np.array(sigma)

    def __singular_functions(self) -> None:
        """
        Collect a max_size right singular functions.
        """
        self.vs = [next(self.right_singular_functions) for _ in range(self.max_size)]

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
        self.__singular_functions()
        self.__singular_values()
        self.__find_fourier_coeffs()
        self.estimate_delta()

        for alpha in np.square(np.concatenate([[np.inf], self.sigmas])):
            regularization = np.square(np.subtract(np.multiply(self.__regularization(np.square(self.sigmas), alpha),
                                                               np.square(self.sigmas)), 1))
            weight = np.divide(1, np.square(self.sigmas) + self.rho)
            coeffs = np.square(self.q_fourier_coeffs)
            summand = np.sort(np.multiply(weight, np.multiply(regularization, coeffs)), kind='heapsort')
            residual = np.sqrt(np.sum(summand))
            self.regularization_param = alpha
            if residual <= np.sqrt(self.tau) * self.delta:
                break

        if self.transformed_measure:
            def solution(x: float) -> np.ndarray:
                return np.multiply(x, np.sum(np.multiply(
                    np.multiply(self.__regularization(np.square(self.sigmas), self.regularization_param),
                                self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs]))))
        else:
            def solution(x: float) -> np.ndarray:
                return np.sum(np.multiply(
                    np.multiply(self.__regularization(np.square(self.sigmas), self.regularization_param),
                                self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs])))

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
        oracle_solutions = []

        def residual(solution):
            return lambda x: np.square(true(x) - solution(x))

        for alpha in np.square(np.concatenate([[np.inf], self.sigmas])):
            parameters.append(alpha)

            if self.transformed_measure:
                def solution(x: float) -> np.ndarray:
                    return np.multiply(x, np.sum(np.multiply(
                        np.multiply(self.__regularization(np.square(self.sigmas), alpha), self.q_fourier_coeffs),
                        np.array([fun(x) for fun in self.vs]))))
            else:
                def solution(x: float) -> np.ndarray:
                    return np.sum(np.multiply(
                        np.multiply(self.__regularization(np.square(self.sigmas), alpha), self.q_fourier_coeffs),
                        np.array([fun(x) for fun in self.vs])))

            solution = np.vectorize(solution)
            oracle_solutions.append(solution(np.linspace(0, 1, 10000)))
            res = residual(solution)
            loss = quad(res, self.lower, self.upper, limit=10000)[0]
            losses.append(loss)
            if loss <= best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1
            if counter == patience:
                break
        res = residual(solution=self.solution)
        self.oracle_param = parameters[losses.index(min(losses))]
        self.oracle_loss = min(losses)
        self.oracle_solution = oracle_solutions[losses.index(min(losses))]
        self.residual = quad(res, self.lower, self.upper, limit=10000)[0]


class Tikhonov(EstimatorSpectrum):
    def __init__(self, kernel: Callable, singular_values: Generator, left_singular_functions: Generator,
                 right_singular_functions: Generator, observations: np.ndarray, sample_size: int,
                 transformed_measure: bool, rho: Union[int, float], order: int = 2, lower: Union[int, float] = 0,
                 upper: Union[int, float] = 1, tau: Union[int, float] = 1, max_size: int = 100, njobs: Optional[int] = -1):
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
        :param njobs: Number of threds to be used to calculate Fourier expansion, negative means all available.
        :type njobs: int (default -1)
        """
        validate_Tikhonov(kernel, singular_values, left_singular_functions, right_singular_functions,
                          observations, sample_size, transformed_measure, rho, order, lower, upper, tau,
                          max_size, njobs)
        EstimatorSpectrum.__init__(self, kernel, observations, sample_size, transformed_measure, rho, lower, upper)
        self.kernel: Callable = kernel
        self.singular_values: Generator = singular_values
        self.left_singular_functions: Generator = left_singular_functions
        self.right_singular_functions: Generator = right_singular_functions
        self.observations: np.ndarray = observations
        self.sample_size: int = sample_size
        self.lower: float = lower
        self.upper: float = upper
        self.tau: float = tau
        self.max_size: int = max_size
        self.order: int = order
        self.q_fourier_coeffs: np.ndarray = np.repeat([0.], self.max_size)
        self.sigmas: np.ndarray = np.repeat([0.], self.max_size)
        self.regularization_param: float = 0.
        self.oracle_param: Optional[float] = None
        self.oracle_loss: Optional[float] = None
        self.oracle_solution: Optional[np.ndarray] = None
        self.residual: Optional[float] = None
        self.vs: list = []
        self.solution: Optional[Callable] = None
        if njobs is None or njobs < 0 or not isinstance(njobs, int):
            njobs = cpu_count()
        self.client = Client(threads_per_worker=1, n_workers=njobs)
        print('Dashboard available under: {}'.format(self.client.dashboard_link))

    @timer
    def __find_fourier_coeffs(self) -> None:
        """
        Calculate max_size Fourier coefficients of a q estimator with respect the the right singular functions of an operator.
        Coefficients are calculated in parallel using the dask backend, the progress is displayed on Dask dashbord running
        by default on localhost://8787.
        """
        self.estimate_q()
        print('Calculation of Fourier coefficients of q estimator...')
        q_estimator: Callable = self.q_estimator
        lower: float = self.lower
        upper: float = self.upper
        client = self.client

        if self.transformed_measure:
            def product(function: Callable) -> Callable:
                return lambda x: q_estimator(x) * function(x) * x
        else:
            def product(function: Callable) -> Callable:
                return lambda x: q_estimator(x) * function(x)

        def integrate(function: Callable) -> float:
            return quad(function, lower, upper, limit=10000)[0]

        products: Iterable = map(product, self.vs)

        @memory.cache
        def fourier_coeffs_helper_Tikhonov(observations: np.ndarray, kernel_function: str):
            futures = []
            for i, fun in enumerate(products):
                futures.append(client.submit(integrate, fun))
            return client.gather(futures)

        coeffs = fourier_coeffs_helper_Tikhonov(self.observations, inspect.getsource(self.kernel))
        self.q_fourier_coeffs = np.array(coeffs)

    def __singular_values(self) -> None:
        """
        Collect a max_size number of singular values.
        """
        sigma: list = [next(self.singular_values) for _ in range(self.max_size)]
        self.sigmas = np.array(sigma)

    def __singular_functions(self) -> None:
        """
        Collect a max_size right singular functions.
        """
        self.vs = [next(self.right_singular_functions) for _ in range(self.max_size)]

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
        self.__singular_functions()
        self.__singular_values()
        self.__find_fourier_coeffs()
        self.estimate_delta()

        for alpha in np.flip(np.linspace(0, 3, 1000)):
            regularization = np.square(np.subtract(np.multiply(self.__regularization(np.square(self.sigmas), alpha, self.order),
                                                               np.square(self.sigmas)), 1))
            weight = np.divide(1, np.square(self.sigmas) + self.rho)
            coeffs = np.square(self.q_fourier_coeffs)
            summand = np.sort(np.multiply(weight, np.multiply(regularization, coeffs)), kind='heapsort')
            residual = np.sqrt(np.sum(summand))
            self.regularization_param = alpha
            if residual <= np.sqrt(self.tau) * self.delta:
                self.residual = residual
                break

        if self.transformed_measure:
            def solution(x: float) -> np.ndarray:
                return np.multiply(x, np.sum(np.multiply(
                    np.multiply(self.__regularization(np.square(self.sigmas), self.regularization_param, self.order),
                                self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs]))))
        else:
            def solution(x: float) -> np.ndarray:
                return np.sum(np.multiply(
                    np.multiply(self.__regularization(np.square(self.sigmas), self.regularization_param, self.order),
                                self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs])))

        self.solution = np.vectorize(solution)

    def refresh(self) -> None:
        pass

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
        oracle_solutions = []

        def residual(solution):
            return lambda x: np.square(true(x) - solution(x))

        for alpha in np.flip(np.linspace(0, 3, 1000)):
            parameters.append(alpha)

            if self.transformed_measure:
                def solution(x: float) -> np.ndarray:
                    return np.multiply(x, np.sum(np.multiply(np.multiply(
                        self.__regularization(np.square(self.sigmas), alpha, self.order),
                        self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs]))))
            else:
                def solution(x: float) -> np.ndarray:
                    return np.sum(np.multiply(np.multiply(
                        self.__regularization(np.square(self.sigmas), alpha, self.order),
                        self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs])))

            solution = np.vectorize(solution)
            oracle_solutions.append(solution(np.linspace(0, 1, 10000)))
            res = residual(solution)
            loss = quad(res, self.lower, self.upper, limit=10000)[0]
            losses.append(loss)
            if loss <= best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1
            if counter == patience:
                break
        res = residual(solution=self.solution)
        self.oracle_param = parameters[losses.index(min(losses))]
        self.oracle_loss = min(losses)
        self.oracle_solution = oracle_solutions[losses.index(min(losses))]
        self.residual = quad(res, self.lower, self.upper, limit=10000)[0]


class Landweber(EstimatorSpectrum):
    def __init__(self, kernel: Callable, singular_values: Generator, left_singular_functions: Generator,
                 right_singular_functions: Generator, observations: np.ndarray, sample_size: int,
                 transformed_measure: bool, rho: Union[int, float], relaxation: Union[int, float] = 0.8,
                 max_iter: int = 100, lower: Union[int, float] = 0, upper: Union[int, float] = 1,
                 tau: Union[int, float] = 1, max_size: int = 100, njobs: Optional[int] = -1):
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
        :type max_size: int (default 100)
        :param njobs: Number of threds to be used to calculate Fourier expansion, negative means all available.
        :type njobs: int (default -1)
        """
        validate_Landweber(kernel, singular_values, left_singular_functions, right_singular_functions,
                           observations, sample_size, transformed_measure, rho, relaxation, max_iter, lower, upper,
                           tau, max_size, njobs)
        EstimatorSpectrum.__init__(self, kernel, observations, sample_size, transformed_measure, rho, lower, upper)
        self.kernel: Callable = kernel
        self.singular_values: Generator = singular_values
        self.left_singular_functions: Generator = left_singular_functions
        self.right_singular_functions: Generator = right_singular_functions
        self.observations: np.ndarray = observations
        self.sample_size: int = sample_size
        self.lower: float = lower
        self.upper: float = upper
        self.tau: float = tau
        self.max_size: int = max_size
        self.max_iter: int = max_iter
        self.relaxation: float = relaxation
        self.q_fourier_coeffs: np.ndarray = np.repeat([0.], self.max_size)
        self.sigmas: np.ndarray = np.repeat([0.], self.max_size)
        self.regularization_param: int = 0
        self.oracle_param: Optional[float] = None
        self.oracle_loss: Optional[float] = None
        self.oracle_solution: Optional[np.ndarray] = None
        self.residual: Optional[float] = None
        self.vs: list = []
        self.solution: Optional[Callable] = None
        if njobs is None or njobs < 0 or not isinstance(njobs, int):
            njobs = cpu_count()
        self.client = Client(threads_per_worker=1, n_workers=njobs)
        print('Dashboard available under: {}'.format(self.client.dashboard_link))

    @timer
    def __find_fourier_coeffs(self) -> None:
        """
        Calculate max_size Fourier coefficients of a q estimator with respect the the right singular functions of an operator.
        Coefficients are calculated in parallel using the dask backend, the progress is displayed on Dask dashbord running
        by default on localhost://8787.
        """
        self.estimate_q()
        print('Calculation of Fourier coefficients of q estimator...')
        q_estimator: Callable = self.q_estimator
        lower: float = self.lower
        upper: float = self.upper
        client = self.client

        if self.transformed_measure:
            def product(function: Callable) -> Callable:
                return lambda x: q_estimator(x) * function(x) * x
        else:
            def product(function: Callable) -> Callable:
                return lambda x: q_estimator(x) * function(x)

        def integrate(function: Callable) -> float:
            return quad(function, lower, upper, limit=10000)[0]

        products: Iterable = map(product, self.vs)

        @memory.cache
        def fourier_coeffs_helper_Landweber(observations: np.ndarray, kernel_function: str):
            futures = []
            for i, fun in enumerate(products):
                futures.append(client.submit(integrate, fun))
            return client.gather(futures)

        coeffs = fourier_coeffs_helper_Landweber(self.observations, inspect.getsource(self.kernel))
        self.q_fourier_coeffs = np.array(coeffs)

    def __singular_values(self) -> None:
        """
        Collect a max_size number of singular values.
        """
        sigma: list = [next(self.singular_values) for _ in range(self.max_size)]
        self.sigmas = np.array(sigma)
        self.relaxation = 1 / (self.sigmas[0] ** 2) * self.relaxation

    def __singular_functions(self) -> None:
        """
        Collect a max_size right singular functions.
        """
        self.vs = [next(self.right_singular_functions) for _ in range(self.max_size)]

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
        self.__singular_functions()
        self.__singular_values()
        self.__find_fourier_coeffs()
        self.estimate_delta()

        for k in np.arange(0, self.max_iter):
            regularization = np.square(np.subtract(np.multiply(self.__regularization(np.square(self.sigmas), k, self.relaxation),
                                                               np.square(self.sigmas)), 1))
            weight = np.divide(1, np.square(self.sigmas) + self.rho)
            coeffs = np.square(self.q_fourier_coeffs)
            summand = np.sort(np.multiply(weight, np.multiply(regularization, coeffs)), kind='heapsort')
            residual = np.sqrt(np.sum(summand))
            self.regularization_param = k
            if residual <= np.sqrt(self.tau) * self.delta:
                self.residual = residual
                break

        if self.transformed_measure:
            def solution(x: float) -> np.ndarray:
                return np.multiply(x, np.sum(np.multiply(np.multiply(
                    self.__regularization(np.square(self.sigmas), self.regularization_param, self.relaxation),
                    self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs]))))
        else:
            def solution(x: float) -> np.ndarray:
                return np.sum(np.multiply(np.multiply(
                    self.__regularization(np.square(self.sigmas), self.regularization_param, self.relaxation),
                    self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs])))

        self.solution = np.vectorize(solution)

    def refresh(self) -> None:
        pass

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
        oracle_solutions = []

        def residual(solution):
            return lambda x: np.square(true(x) - solution(x))

        for k in np.arange(0, self.max_iter):
            parameters.append(k)

            if self.transformed_measure:
                def solution(x: float) -> np.ndarray:
                    return np.multiply(x, np.sum(np.multiply(
                        np.multiply(self.__regularization(np.square(self.sigmas), k, self.relaxation),
                                    self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs]))))
            else:
                def solution(x: float) -> np.ndarray:
                    return np.sum(np.multiply(
                        np.multiply(self.__regularization(np.square(self.sigmas), k, self.relaxation),
                                    self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs])))

            solution = np.vectorize(solution)
            oracle_solutions.append(solution(np.linspace(0, 1, 10000)))
            res = residual(solution)
            loss = quad(res, self.lower, self.upper, limit=10000)[0]
            losses.append(loss)
            if loss <= best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1
            if counter == patience:
                break
        res = residual(solution=self.solution)
        self.oracle_param = parameters[losses.index(min(losses))]
        self.oracle_loss = min(losses)
        self.oracle_solution = oracle_solutions[losses.index(min(losses))]
        self.residual = quad(res, self.lower, self.upper, limit=10000)[0]
