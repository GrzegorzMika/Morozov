from multiprocessing import cpu_count
from typing import Union, Callable, Optional, Generator, Iterable
import numpy as np
from dask.distributed import Client
from numba import njit
from scipy.integrate import quad
from GeneralEstimator import EstimatorSpectrum
from decorators import timer


class TSVD(EstimatorSpectrum):
    def __init__(self, kernel, singular_values, left_singular_functions, right_singular_functions, observations,
                 sample_size, lower: Union[float, int] = 0, upper: Union[float, int] = 1, **kwargs):
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
        :param lower: Lower end of the interval on which the operator is defined.
        :type lower: float
        :param upper: Upper end of the interval on which the operator is defined.
        :type lower: float
        :param kwargs: Possible arguments:
            - tau: Parameter used to rescale the obtained values of estimated noise level (float or int, default: 1).
            - max_size: Maximum number of functions included in Fourier expansion (int, default: 100).
        """
        EstimatorSpectrum.__init__(self, kernel, observations, sample_size, lower, upper)
        self.kernel: Callable = kernel
        self.singular_values: Generator = singular_values
        self.left_singular_functions: Generator = left_singular_functions
        self.right_singular_functions: Generator = right_singular_functions
        self.observations: np.ndarray = observations
        self.sample_size: int = sample_size
        self.lower: Union[float, int] = lower
        self.upper: Union[float, int] = upper
        self.tau: Union[float, int] = kwargs.get('tau', 1)
        self.max_size: int = kwargs.get('max_size', 100)
        self.q_fourier_coeffs: np.ndarray = np.repeat([0], self.max_size)
        self.sigmas: np.ndarray = np.repeat([0], self.max_size)
        self.regularization_param: float = 0.
        self.vs: list = []
        self.solution: Optional[Callable] = None
        self.client = Client(threads_per_worker=1, n_workers=cpu_count())

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
        lower: Union[float, int] = self.lower
        upper: Union[float, int] = self.upper

        def product(function: Callable) -> Callable:
            return lambda x: q_estimator(x) * function(x)

        def integrate(function: Callable) -> float:
            return quad(function, lower, upper, limit=5000)[0]

        products: Iterable = map(product, self.vs)

        futures = []
        for i, fun in enumerate(products):
            futures.append(self.client.submit(integrate, fun))
        coeffs = self.client.gather(futures)

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
    def __regularization(lam: np.ndarray, alpha: Union[float, int]) -> np.ndarray:
        """
        Truncated singular value regularization
        :param lam: argument of regularizing function
        :param alpha: regularizing parameter
        :return: Result of applying regularization function to argument lambda.
        """
        return np.where(lam >= alpha, np.divide(1, lam), 0)

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
            residual = np.sqrt(np.sum(np.multiply(np.square(
                np.subtract(np.multiply(self.__regularization(np.square(self.sigmas), alpha), np.square(self.sigmas)),
                            1)), np.square(self.q_fourier_coeffs))))
            self.regularization_param = alpha
            if residual <= np.sqrt(self.tau) * self.delta:
                break

        def solution(x: Union[float, int]) -> np.ndarray:
            return np.sum(
                np.multiply(np.multiply(self.__regularization(np.square(self.sigmas), self.regularization_param),
                                        self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs])))

        self.solution = np.vectorize(solution)

    def refresh(self) -> None:
        """
        Just to be consistent with general guidelines.
        """
        pass


class Tikhonov(EstimatorSpectrum):
    def __init__(self, kernel, singular_values, left_singular_functions, right_singular_functions, observations,
                 sample_size, lower: Union[float, int] = 0, upper: Union[float, int] = 1, **kwargs):
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
        :param lower: Lower end of the interval on which the operator is defined.
        :type lower: float
        :param upper: Upper end of the interval on which the operator is defined.
        :type lower: float
        :param kwargs: Possible arguments:
            - tau: Parameter used to rescale the obtained values of estimated noise level (float or int, default: 1).
            - max_size: Maximum number of functions included in Fourier expansion (int, default: 100).
            - order: Order of the iterated algorithm. Estimator for each regularization parameter is obtained after
                    order iterations. Ordinary Tikhonov estimator is obtained for order = 1 (int, default: 2).
        """
        EstimatorSpectrum.__init__(self, kernel, observations, sample_size, lower, upper)
        self.kernel: Callable = kernel
        self.singular_values: Generator = singular_values
        self.left_singular_functions: Generator = left_singular_functions
        self.right_singular_functions: Generator = right_singular_functions
        self.observations: np.ndarray = observations
        self.sample_size: int = sample_size
        self.lower: Union[float, int] = lower
        self.upper: Union[float, int] = upper
        self.tau: Union[float, int] = kwargs.get('tau', 1)
        self.max_size: int = kwargs.get('max_size', 100)
        self.__order: int = kwargs.get('order', 2)
        self.q_fourier_coeffs: np.ndarray = np.repeat([0], self.max_size)
        self.sigmas: np.ndarray = np.repeat([0], self.max_size)
        self.regularization_param: float = 0.
        self.vs: list = []
        self.solution: Optional[Callable] = None
        self.client = Client(threads_per_worker=1, n_workers=cpu_count())

    @property
    def order(self) -> int:
        return self.__order

    @order.setter
    def order(self, order: int) -> None:
        self.__order = order

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
        lower: Union[float, int] = self.lower
        upper: Union[float, int] = self.upper

        def product(function: Callable) -> Callable:
            return lambda x: q_estimator(x) * function(x)

        def integrate(function: Callable) -> float:
            return quad(function, lower, upper, limit=5000)[0]

        products: Iterable = map(product, self.vs)

        futures = []
        for i, fun in enumerate(products):
            futures.append(self.client.submit(integrate, fun))
        coeffs = self.client.gather(futures)

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
    def __regularization(lam: np.ndarray, alpha: Union[float, int], order: int) -> np.ndarray:
        """
        Iterated Tikhonov regularization.
        :param lam: argument of regularizing function
        :param alpha: regularizing parameter
        :param order: number of iteration \
        :return: Result of applying regularization function to argument lambda.
        """
        return np.divide(np.power(lam + alpha, order) - np.power(alpha, order),
                         np.multiply(lam, np.power(lam + alpha, order)))

    def estimate(self) -> None:
        """
        Implementation of iterated Tikhonv algorithm for inverse problem with stopping rule based on Morozov discrepancy principle.
        """
        self.__singular_functions()
        self.__singular_values()
        self.__find_fourier_coeffs()
        self.estimate_delta()

        for alpha in np.flip(np.linspace(0, 10, 1000)):  # warm start required
            residual = np.sqrt(np.sum(np.multiply(np.square(
                np.subtract(np.multiply(self.__regularization(np.square(self.sigmas), alpha, self.order),
                                        np.square(self.sigmas)),
                            1)), np.square(self.q_fourier_coeffs))))
            self.regularization_param = alpha
            if residual <= np.sqrt(self.tau) * self.delta:
                break

        def solution(x: Union[float, int]) -> np.ndarray:
            return np.sum(
                np.multiply(
                    np.multiply(self.__regularization(np.square(self.sigmas), self.regularization_param, self.order),
                                self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs])))

        self.solution = np.vectorize(solution)

    def refresh(self) -> None:
        """
        Just to be consistent with general guidelines.
        """
        pass


class Landweber(EstimatorSpectrum):
    def __init__(self, kernel, singular_values, left_singular_functions, right_singular_functions, observations,
                 sample_size, lower: Union[float, int] = 0, upper: Union[float, int] = 1, **kwargs):
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
        :param lower: Lower end of the interval on which the operator is defined.
        :type lower: float
        :param upper: Upper end of the interval on which the operator is defined.
        :type lower: float
        :param kwargs: Possible arguments:
            - tau: Parameter used to rescale the obtained values of estimated noise level (float or int, default: 1).
            - max_size: Maximum number of functions included in Fourier expansion (int, default: 100).
            - relaxation: Parameter used in the iteration of the algorithm (step size, omega). The square of the first
            singular value is scaled by this value(float, default: 0.8).
            - max_iter: Maximum number of iterations of the algorithm (int, default: 100)
        """
        EstimatorSpectrum.__init__(self, kernel, observations, sample_size, lower, upper)
        self.kernel: Callable = kernel
        self.singular_values: Generator = singular_values
        self.left_singular_functions: Generator = left_singular_functions
        self.right_singular_functions: Generator = right_singular_functions
        self.observations: np.ndarray = observations
        self.sample_size: int = sample_size
        self.lower: Union[float, int] = lower
        self.upper: Union[float, int] = upper
        self.tau: Union[float, int] = kwargs.get('tau', 1)
        self.max_size: int = kwargs.get('max_size', 100)
        self.max_iter: int = kwargs.get('max_iter', 100)
        self.__relaxation: float = kwargs.get('relaxation', 0.8)
        self.q_fourier_coeffs: np.ndarray = np.repeat([0], self.max_size)
        self.sigmas: np.ndarray = np.repeat([0], self.max_size)
        self.regularization_param: int = 0
        self.vs: list = []
        self.solution: Optional[Callable] = None
        self.client = Client(threads_per_worker=1, n_workers=cpu_count())

    @property
    def relaxation(self) -> float:
        return self.__relaxation

    @relaxation.setter
    def relaxation(self, relaxation: float) -> None:
        self.__relaxation = relaxation

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
        lower: Union[float, int] = self.lower
        upper: Union[float, int] = self.upper

        def product(function: Callable) -> Callable:
            return lambda x: q_estimator(x) * function(x)

        def integrate(function: Callable) -> float:
            return quad(function, lower, upper, limit=5000)[0]

        products: Iterable = map(product, self.vs)

        futures = []
        for i, fun in enumerate(products):
            futures.append(self.client.submit(integrate, fun))
        coeffs = self.client.gather(futures)

        self.q_fourier_coeffs = np.array(coeffs)

    def __singular_values(self) -> None:
        """
        Collect a max_size number of singular values.
        """
        sigma: list = [next(self.singular_values) for _ in range(self.max_size)]
        self.sigmas = np.array(sigma)
        self.relaxation = 1/(self.sigmas[0]**2)*self.relaxation

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
            regularization = np.sum(iterations, axis=1)*beta
            return regularization

    def estimate(self) -> None:
        """
        Implementation of Landweber algorithm for inverse problem with stopping rule based on Morozov discrepancy principle.
        """
        self.__singular_functions()
        self.__singular_values()
        self.__find_fourier_coeffs()
        self.estimate_delta()

        for k in np.arange(0, self.max_iter):
            residual = np.sqrt(np.sum(np.multiply(np.square(
                np.subtract(np.multiply(self.__regularization(np.square(self.sigmas), k, self.relaxation),
                                        np.square(self.sigmas)),
                            1)), np.square(self.q_fourier_coeffs))))
            self.regularization_param = k
            if residual <= np.sqrt(self.tau) * self.delta:
                break

        def solution(x: Union[float, int]) -> np.ndarray:
            return np.sum(
                np.multiply(
                    np.multiply(self.__regularization(np.square(self.sigmas), self.regularization_param, self.relaxation),
                                self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs])))

        self.solution = np.vectorize(solution)

    def refresh(self) -> None:
        """
        Just to be consistent with general guidelines.
        """
        pass
