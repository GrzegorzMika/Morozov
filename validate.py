from typing import Callable, Optional, Generator

import numpy as np


def validate_EstimatorSpectrum(kernel: Callable, observations: np.ndarray, sample_size: int, transformed_measure: bool,
                               rho: float, lower: float, upper: float):
    assert isinstance(transformed_measure, bool), 'Please provide an information about measure transformation as ' \
                                                  'True or False'
    assert isinstance(kernel, Callable), 'Kernel function must be callable'
    assert isinstance(lower, float), 'Lower bound for integration interval must be a number, but ' \
                                     'was {} provided'.format(lower)
    assert isinstance(upper, float), 'Upper bound for integration interval must be a number, but' \
                                     ' was {} provided'.format(upper)
    assert isinstance(observations, np.ndarray), 'Please provide the observations in a form of numpy array'
    assert isinstance(sample_size, int), 'Sample size must be an integer, but was {} provided'.format(sample_size)
    assert (isinstance(rho, float)) & (rho > 0)


def validate_TSVD(kernel: Callable, singular_values: Generator, left_singular_functions: Generator,
                  right_singular_functions: Generator, observations: np.ndarray, sample_size: int,
                  transformed_measure: bool, rho: float, lower: float, upper: float, tau: float,
                  max_size: int, njobs: Optional[int]):
    assert isinstance(singular_values, Generator), 'Please provide the singular values as generator'
    assert isinstance(left_singular_functions, Generator), 'Please provide the left singular functions as generator'
    assert isinstance(right_singular_functions,
                      Generator), 'Please provide the right singular functions as generator'
    assert (isinstance(tau, float)) & (tau >= 1), 'Wrong tau has been specified'
    assert isinstance(max_size, int), 'Wrong max_size has been specified'
    assert isinstance(transformed_measure, bool), 'Please provide an information about measure transformation as ' \
                                                  'True or False'
    assert isinstance(kernel, Callable), 'Kernel function must be callable'
    assert isinstance(lower, float), 'Lower bound for integration interval must be a number, but ' \
                                     'was {} provided'.format(lower)
    assert isinstance(upper, float), 'Upper bound for integration interval must be a number, but' \
                                     ' was {} provided'.format(upper)
    assert isinstance(observations, np.ndarray), 'Please provide the observations in a form of numpy array'
    assert isinstance(sample_size, int), 'Sample size must be an integer, but was {} provided'.format(sample_size)
    assert (isinstance(rho, float)) & (rho > 0)
    assert njobs is None or isinstance(njobs, int)


def validate_Tikhonov(kernel: Callable, singular_values: Generator, left_singular_functions: Generator,
                      right_singular_functions: Generator, observations: np.ndarray, sample_size: int,
                      transformed_measure: bool, rho: float, order: int, lower: float, upper: float,
                      tau: float, max_size: int, njobs: Optional[int]):
    assert isinstance(singular_values, Generator), 'Please provide the singular values as generator'
    assert isinstance(left_singular_functions, Generator), 'Please provide the left singular functions as generator'
    assert isinstance(right_singular_functions,
                      Generator), 'Please provide the right singular functions as generator'
    assert (isinstance(tau, float)) & (tau >= 1), 'Wrong tau has been specified'
    assert isinstance(max_size, int), 'Wrong max_size has been specified'
    assert isinstance(transformed_measure, bool), 'Please provide an information about measure transformation as ' \
                                                  'True or False'
    assert isinstance(kernel, Callable), 'Kernel function must be callable'
    assert isinstance(lower, float), 'Lower bound for integration interval must be a number, but ' \
                                     'was {} provided'.format(lower)
    assert isinstance(upper, float), 'Upper bound for integration interval must be a number, but' \
                                     ' was {} provided'.format(upper)
    assert isinstance(observations, np.ndarray), 'Please provide the observations in a form of numpy array'
    assert isinstance(sample_size, int), 'Sample size must be an integer, but was {} provided'.format(sample_size)
    assert (isinstance(rho, float)) & (rho > 0)
    assert (isinstance(order, int)) & (order > 0)
    assert njobs is None or isinstance(njobs, int)


def validate_Landweber(kernel: Callable, singular_values: Generator, left_singular_functions: Generator,
                       right_singular_functions: Generator, observations: np.ndarray, sample_size: int,
                       transformed_measure: bool, rho: int, relaxation: float, max_iter: int, lower: float,
                       upper: float, tau: float, max_size: int, njobs: Optional[int]):
    assert isinstance(singular_values, Generator), 'Please provide the singular values as generator'
    assert isinstance(left_singular_functions, Generator), 'Please provide the left singular functions as generator'
    assert isinstance(right_singular_functions,
                      Generator), 'Please provide the right singular functions as generator'
    assert (isinstance(tau, float)) & (tau >= 1), 'Wrong tau has been specified'
    assert isinstance(max_size, int), 'Wrong max_size has been specified'
    assert isinstance(transformed_measure, bool), 'Please provide an information about measure transformation as ' \
                                                  'True or False'
    assert isinstance(kernel, Callable), 'Kernel function must be callable'
    assert isinstance(lower, float), 'Lower bound for integration interval must be a number, but ' \
                                     'was {} provided'.format(lower)
    assert isinstance(upper, float), 'Upper bound for integration interval must be a number, but' \
                                     ' was {} provided'.format(upper)
    assert isinstance(observations, np.ndarray), 'Please provide the observations in a form of numpy array'
    assert isinstance(sample_size, int), 'Sample size must be an integer, but was {} provided'.format(sample_size)
    assert (isinstance(rho, float)) & (rho > 0)
    assert (isinstance(relaxation, float)) & (relaxation > 0)
    assert (isinstance(max_iter, int)) & (max_iter > 0)
    assert njobs is None or isinstance(njobs, int)
