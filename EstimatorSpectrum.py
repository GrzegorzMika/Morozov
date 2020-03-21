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
        self.estimate_q()
        print('Calculation Fourier coefficients of q estimator...')
        q_estimator: Callable = self.q_estimator
        lower: Union[float, int] = self.lower
        upper: Union[float, int] = self.upper

        def product(function: Callable) -> Callable:
            return lambda x: q_estimator(x) * function(x)

        def integrate(function: Callable) -> float:
            return quad(function, lower, upper, limit=1000)[0]  # TODO: higher precision

        products: Iterable = map(product, self.vs)

        futures = []
        for i, fun in enumerate(products):
            futures.append(self.client.submit(integrate, fun))
        coeffs = self.client.gather(futures)

        self.q_fourier_coeffs = np.array(coeffs)

    def __singular_values(self) -> None:
        sigma: list = [next(self.singular_values) for _ in range(self.max_size)]
        self.sigmas = np.array(sigma)

    def __singular_functions(self) -> None:
        self.vs = [next(self.right_singular_functions) for _ in range(self.max_size)]

    @staticmethod
    @njit
    def __regularization(lam: np.ndarray, alpha: Union[float, int]) -> np.ndarray:
        """
        Truncated singular value regularization
        :param lam: argument of regularizing function
        :param alpha: regularizing parameter
        :return:
        """
        return np.where(lam >= alpha, np.divide(1, lam), 0)

    def estimate(self) -> None:
        self.__singular_functions()
        self.__singular_values()
        self.__find_fourier_coeffs()
        self.estimate_delta()

        for alpha in np.square(np.concatenate([[np.inf], self.sigmas])):
            print(alpha)
            residual = np.sqrt(np.sum(np.multiply(np.square(
                np.subtract(np.multiply(self.__regularization(np.square(self.sigmas), alpha), np.square(self.sigmas)),
                            1)), np.square(self.q_fourier_coeffs))))
            print(residual)
            self.regularization_param = alpha
            if residual <= np.sqrt(self.tau) * self.delta:
                break

        def solution(x: Union[float, int]) -> np.ndarray:
            return np.sum(
                np.multiply(np.multiply(self.__regularization(np.square(self.sigmas), self.regularization_param),
                                        self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs])))

        self.solution = np.vectorize(solution)

    def refresh(self) -> None:
        pass




# class Tikhonov(EstimatorSpectrum):
#     def __init__(self, kernel, singular_values, left_singular_functions, right_singular_functions, observations,
#                  sample_size, lower: Union[float, int] = 0, upper: Union[float, int] = 1, **kwargs):
#         EstimatorSpectrum.__init__(self, kernel, observations, sample_size, lower, upper)
#         self.kernel: Callable = kernel
#         self.singular_values: Generator = singular_values
#         self.left_singular_functions: Generator = left_singular_functions
#         self.right_singular_functions: Generator = right_singular_functions
#         self.observations: np.ndarray = observations
#         self.sample_size: int = sample_size
#         self.lower: Union[float, int] = lower
#         self.upper: Union[float, int] = upper
#         self.tau: Union[float, int] = kwargs.get('tau', 1)
#         self.max_size: int = kwargs.get('max_size', 10000)
#         self.order: int = kwargs.get('order', 2)
#         self.q_fourier_coeffs: np.ndarray = np.repeat([0], self.max_size)
#         self.sigmas: np.ndarray = np.repeat([0], self.max_size)
#         self.regularization_param: float = 0.
#         self.vs: list = []
#         self.solution: Optional[Callable] = None
#         self.client = Client(threads_per_worker=1, n_workers=cpu_count())
#
#     @timer
#     def __find_fourier_coeffs(self) -> None:
#         self.estimate_q()
#         print('Calculation Fourier coefficients of q estimator...')
#         q_estimator: Callable = self.q_estimator
#         lower: Union[float, int] = self.lower
#         upper: Union[float, int] = self.upper
#
#         def product(function: Callable) -> Callable:
#             return lambda x: q_estimator(x) * function(x)
#
#         def integrate(function: Callable) -> float:
#             return quad(function, lower, upper, limit=1000)[0]  # TODO: higher precision
#
#         # singular_functions: list = [next(self.right_singular_functions) for _ in range(self.max_size)]
#         products: Iterable = map(product, self.vs)
#
#         futures = []
#         for i, fun in enumerate(products):
#             futures.append(self.client.submit(integrate, fun))
#         coeffs = self.client.gather(futures)
#
#         self.q_fourier_coeffs = np.array(coeffs)
#
#     def __singular_values(self) -> None:
#         sigma: list = [next(self.singular_values) for _ in range(self.max_size)]
#         self.sigmas = np.array(sigma)
#
#     def __singular_functions(self) -> None:
#         self.vs = [next(self.right_singular_functions) for _ in range(self.max_size)]
#
#     @staticmethod
#     @njit
#     def __regularization(lam: np.ndarray, alpha: Union[float, int], order) -> np.ndarray:
#         return np.divide(np.power(lam + alpha, order) -np.power(alpha, order), np.multiply(lam, np.power(lam + alpha, order)))
#
#     def estimate(self) -> None:
#         self.__singular_functions()
#         self.__singular_values()
#         self.__find_fourier_coeffs()
#         self.estimate_delta()
#
#         for alpha in np.flip(np.linspace(0, 10, 1000)):
#             print(alpha)
#             residual = np.sqrt(np.sum(np.multiply(np.square(
#                 np.subtract(np.multiply(self.__regularization(np.square(self.sigmas), alpha, self.order), np.square(self.sigmas)),
#                             1)), np.square(self.q_fourier_coeffs))))
#             print(residual)
#             self.regularization_param = alpha
#             if residual <= np.sqrt(self.tau) * self.delta:
#                 break
#
#         def solution(x: Union[float, int]) -> np.ndarray:
#             return np.sum(
#                 np.multiply(np.multiply(self.__regularization(np.square(self.sigmas), self.regularization_param, self.order),
#                                         self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs])))
#
#         self.solution = np.vectorize(solution)
#
#     # def tmp_solution(self, alpha):
#     #     alpha = alpha
#     #     def solution(x: Union[float, int]) -> np.ndarray:
#     #         return np.sum(
#     #             np.multiply(np.multiply(self.__regularization(np.square(self.sigmas), np.square(self.sigmas[alpha])),
#     #                                     self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs])))
#     #
#     #     return np.vectorize(solution)
#
#     def refresh(self) -> None:
#         pass
