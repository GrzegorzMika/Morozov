import numpy as np
import cupy as cp
from numpy.testing import assert_equal, assert_almost_equal
from pytest import raises
from GeneralEstimator import EstimatorDiscretize


def identity(x, y):
    return np.where(x < y, 1, 1)


observations = np.repeat(0, 30)

estimator = EstimatorDiscretize(kernel=identity, lower=0, upper=1, grid_size=100, observations=observations, sample_size=50)


class TestAttributes:
    def test_estimator(self):
        assert estimator is not None

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

    def test_quadrature(self):
        assert hasattr(estimator, 'quadrature')
        assert callable(estimator.quadrature)


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


class TestFunctionalities:
    def test_L2norm(self):
        assert estimator.L2norm(cp.asarray(np.repeat(1, 100)), cp.asarray(np.repeat(1, 100))) == 0
        assert_almost_equal(estimator.L2norm(cp.asarray(np.repeat(1, 100)), cp.asarray(np.repeat(0, 100))), 1, decimal=12)

    def test_q_estimator(self):
        estimator.estimate_q()
        assert_almost_equal(cp.asnumpy(estimator.q_estimator), np.repeat(3/5, 100), decimal=12)

    def test_delta(self):
        estimator.estimate_delta()
        assert_almost_equal(estimator.delta,  0.10954451150103321, decimal=12)


class TestException:
    def test_lower(self):
        with raises(AssertionError):
            EstimatorDiscretize(kernel=identity, lower='a', upper=1, grid_size=100, observations=observations, sample_size=50)

    def test_upper(self):
        with raises(AssertionError):
            EstimatorDiscretize(kernel=identity, lower=0, upper='a', grid_size=100, observations=observations, sample_size=50)

    def test_grid_size(self):
        with raises(AssertionError):
            EstimatorDiscretize(kernel=identity, lower=0, upper=1, grid_size='a', observations=observations, sample_size=50)
        with raises(AssertionError):
            EstimatorDiscretize(kernel=identity, lower=0, upper=1, grid_size=10., observations=observations, sample_size=50)

    def test_observations(self):
        with raises(AssertionError):
            EstimatorDiscretize(kernel=identity, lower=0, upper=1, grid_size=100, observations=1, sample_size=50)
        with raises(AssertionError):
            EstimatorDiscretize(kernel=identity, lower=0, upper=1, grid_size=100, observations=[1, 2], sample_size=50)

    def test_sample_size(self):
        with raises(AssertionError):
            EstimatorDiscretize(kernel=identity, lower=0, upper=1, grid_size=100, observations=observations, sample_size='a')
        with raises(AssertionError):
            EstimatorDiscretize(kernel=identity, lower=0, upper=1, grid_size=100, observations=observations, sample_size=50.)

    def test_not_implemented(self):
        with raises(NotImplementedError):
            estimator.estimate()
        with raises(NotImplementedError):
            estimator.refresh()
