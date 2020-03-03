import os
import numpy as np
import cupy as cp
from numpy.testing import assert_equal, assert_almost_equal
from pytest import raises
from Estimators import TSVD
from Generator import LewisShedler


def identity(x, y):
    return np.where(x < y, 1, 1)


def lam(t):
    return 200 * (1 + np.sin(20 * t))


def kernel(x, y):
    return np.where(x < y, (1 + np.sin(20 * y)), (1 + np.sin(20 * y)))


generator = LewisShedler(intensity_function=lam, lower=0, upper=1, seed=1)
observations_random = generator.generate()

observations = np.repeat(0, 30)

estimator = TSVD(kernel=identity, lower=0, upper=1, grid_size=100, observations=observations, sample_size=50)


class TestAttributes:
    def test_kernel(self):
        assert hasattr(estimator, 'kernel')
        assert callable(estimator.kernel)

    def test_lower(self):
        assert hasattr(estimator, 'lower')
        assert estimator.lower == 0

    def test_upper(self):
        assert hasattr(estimator, 'upper')
        assert estimator.upper == 1

    def test_grid_size(self):
        assert hasattr(estimator, 'grid_size')
        assert estimator.grid_size == 100

    def test_observations(self):
        assert hasattr(estimator, 'observations')
        assert_equal(estimator.observations, observations)

    def test_sample_size(self):
        assert hasattr(estimator, 'sample_size')
        assert estimator.sample_size == 50

    def test_adjoint(self):
        assert hasattr(estimator, 'adjoint')
        assert not estimator.adjoint

    def test_tau(self):
        assert hasattr(estimator, 'tau')
        assert estimator.tau == 1.
        estimator_tmp = TSVD(kernel=identity, lower=0, upper=1, grid_size=100,
                                  observations=observations, sample_size=50, tau=1.7)
        assert estimator_tmp.tau == 1.7

    def test_initation(self):
        assert hasattr(estimator, 'previous')
        assert hasattr(estimator, 'current')
        assert_equal(cp.asnumpy(estimator.previous), np.repeat(np.array([0]), 100).astype(np.float64))
        assert_equal(cp.asnumpy(estimator.current), np.repeat(np.array([0]), 100).astype(np.float64))

    def test_delta(self):
        assert hasattr(estimator, 'delta')
        assert_almost_equal(estimator.delta, 0.10954451150103321, decimal=12)

    def test_q_estimator(self):
        assert hasattr(estimator, 'q_estimator')
        assert_almost_equal(cp.asnumpy(estimator.q_estimator), np.repeat(3 / 5, 100), decimal=12)

    def test_operator(self):
        assert hasattr(estimator, 'K')
        assert hasattr(estimator, 'KHK')
        assert_almost_equal(cp.asnumpy(estimator.K), np.ones((100, 100)) * 0.01)
        assert_almost_equal(cp.asnumpy(estimator.KHK), np.ones((100, 100)) * 0.01)

    def test_grid(self):
        assert hasattr(estimator, 'grid')
        assert_equal(estimator.grid, np.linspace(0, 1, 100, endpoint=False))


class TestInheritance:
    def test_rectangle(self):
        assert hasattr(estimator, 'rectangle')
        assert callable(estimator.rectangle)
        assert estimator.rectangle(1) == 0.01
        assert_equal(estimator.rectangle(np.array([1, 2])), np.array([0.01, 0.01]))

    def test_dummy_(self):
        assert hasattr(estimator, 'dummy')
        assert callable(estimator.dummy)
        assert estimator.dummy(1) == 1
        assert_equal(estimator.dummy(np.array([1, 2])), np.array([1, 1]))

    def test_rectangle_grid(self):
        assert hasattr(estimator, 'rectangle_grid')
        assert callable(estimator.rectangle_grid)
        assert_equal(estimator.rectangle_grid(), np.linspace(0, 1, 100, endpoint=False))

    def test_dummy_grid(self):
        assert hasattr(estimator, 'dummy_grid')
        assert callable(estimator.dummy_grid)
        assert_equal(estimator.dummy_grid(), np.linspace(0, 1, 100, endpoint=False))

    def test_approximate(self):
        assert hasattr(estimator, 'approximate')
        assert callable(estimator.approximate)


class TestFunctionalities:
    def test_estimate(self):
        estimator.estimate()
        assert_almost_equal(cp.asnumpy(estimator.solution), np.repeat([0.59999999686], 100), decimal=6)
        estimator_tmp = TSVD(kernel=identity, lower=0, upper=1, grid_size=1000, observations=observations_random,
                                  sample_size=200, relaxation=10)
        estimator_tmp.estimate()
        solution = np.load(os.path.join('test_files', 'solution_tsvd.npy'))
        assert_almost_equal(cp.asnumpy(estimator_tmp.solution), solution, decimal=12)

    def test_refresh(self):
        estimator.estimate()
        with raises(AssertionError):
            assert_equal(cp.asnumpy(estimator.solution), np.repeat([0], 100))
        estimator.observations = np.repeat([0], 40)
        estimator.refresh()
        assert_equal(cp.asnumpy(estimator.solution), np.repeat([0], 100))
        assert_almost_equal(cp.asnumpy(estimator.q_estimator), np.repeat(4 / 5, 100), decimal=12)


class TestException:
    pass
