from typing import Callable, Optional, Generator, Union

import numpy as np


def validate_EstimatorSpectrum(kernel: Callable, observations: np.ndarray, sample_size: int, rho: float, transformed_measure: bool,
                               singular_values: Generator, left_singular_functions: Generator, right_singular_functions: Generator,
                               max_size: int):
    assert isinstance(transformed_measure, bool), 'Please provide an information about measure transformation as ' \
                                                  'True or False'
    assert isinstance(singular_values, Generator), 'Please provide the singular values as generator'
    assert isinstance(left_singular_functions, Generator), 'Please provide the left singular functions as generator'
    assert isinstance(right_singular_functions,
                      Generator), 'Please provide the right singular functions as generator'
    assert isinstance(kernel, Callable), 'Kernel function must be callable'
    assert isinstance(observations, np.ndarray), 'Please provide the observations in a form of numpy array'
    assert isinstance(sample_size, int), 'Sample size must be an integer, but was {} provided'.format(sample_size)
    assert (isinstance(max_size, int)) & (max_size > 0), 'Please provide the max size as an integer'
    assert (isinstance(rho, (float, int))) & (rho >= 0), 'Weight strength must be a non-negative float, but {} was provided'.format(rho)


def validate_TSVD(tau: Union[int, float]):
    assert (isinstance(tau, (int, float))) & (tau >= 1), 'Wrong tau has been specified'


def validate_Tikhonov(order: int, tau: Union[int, float]):
    assert (isinstance(tau, (int, float))) & (tau >= 1), 'Wrong tau has been specified'
    assert (isinstance(order, int)) & (order > 0)


def validate_Landweber(relaxation: Union[int, float], max_iter: int, tau: Union[int, float]):
    assert (isinstance(tau, (int, float))) & (tau >= 1), 'Wrong tau has been specified'
    assert (isinstance(relaxation, (int, float))) & (relaxation > 0)
    assert (isinstance(max_iter, int)) & (max_iter > 0)
