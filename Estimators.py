from typing import Callable
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
    def __init__(self, kernel, lower, upper, grid_size, observations, sample_size, quadrature, **kwargs):
        Operator.__init__(self, kernel, lower, upper, grid_size, quadrature)
        Estimator.__init__(self, kernel, lower, upper, grid_size, observations, sample_size, quadrature)
        self.kernel: Callable = kernel
        self.lower = lower
        self.upper = upper
        self.grid_size = grid_size
        self.__observations = observations
        self.sample_size = sample_size
        self.relaxation = kwargs.get('relaxation')
        if self.relaxation is None:
            self.relaxation = 0.01
        self.max_iter = kwargs.get('max_iter')
        if self.max_iter is None:
            self.max_iter = 100
        self.initial = kwargs.get('initial_guess')
        if self.initial is None:
            self.initial = da.repeat(da.from_array(np.array([0])), self.grid_size)
        self.previous = self.initial
        self.current = self.initial
        Operator.approximate(self)
        self.KHK = np.matmul(self.KH, self.K)
        Estimator.estimate_q(self)
        Estimator.estimate_delta(self)

    @staticmethod

    @property
    def solution(self):
        return self.previous

    @solution.setter
    def solution(self, solution):
        self.previous = solution

    @staticmethod
    def L2norm(x, y):
        return da.sum((x - y) ** 2)

    @timer
    def __iteration(self):
        print('iteration')
        bracket = self.q_estimator - np.matmul(self.K, self.previous)
        self.current = self.previous + self.relaxation * np.matmul(self.KH, bracket)
        return self.current

    def __stopping_rule(self):
        norm = self.L2norm(np.matmul(self.KHK, self.current), self.q_estimator).compute()
        return norm > self.delta

    def __update_solution(self):
        self.previous = self.current


    @timer
    def __force_computations(self):
        self.K, self.KH, self.KHK, self.delta, self.q_estimator, self.previous, self.current = dask.optimize(
            self.K, self.KH, self.KHK, self.delta, self.q_estimator, self.previous, self.current)
        with ProgressBar():
            self.K, self.KH, self.KHK, self.delta, self.q_estimator, self.previous, self.current = dask.compute(
                self.K, self.KH, self.KHK, self.delta, self.q_estimator, self.previous, self.current, num_workers=cpu_count())

    def estimate(self):
        it = 1
        start = time()
        print('Force computations...')
        self.__force_computations()
        condition = self.__stopping_rule()
        print(condition)
        while condition:
            print('Iteration: {}'.format(it))
            it += 1
            self.__update_solution()
            self.__iteration()
            print(self.current)
            condition = self.__stopping_rule()
            if it > self.max_iter:
                warn('Maximum number of iterations reached!', RuntimeWarning)
                break
        print('Total elapsed time: {}'.format(time() - start))
