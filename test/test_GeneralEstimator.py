import sys
import os
import numpy as np
import cupy as cp
from numpy.testing import assert_equal, assert_almost_equal
from pytest import raises

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GeneralEstimator import EstimatorDiscretize, EstimatorSpectrum


def identity(x, y):
    return np.where(x < y, 1, 1)


observations = np.repeat(0, 30)

estimator_d = EstimatorDiscretize(kernel=identity, lower=0, upper=1, grid_size=100, observations=observations,
                                  sample_size=50)


class TestAttributesDiscretize:
    def test_estimator(self):
        assert estimator_d is not None

    def test_kernel(self):
        assert hasattr(estimator_d, 'kernel')
        assert callable(estimator_d.kernel)

    def test_lower(self):
        assert hasattr(estimator_d, 'lower')
        assert estimator_d.lower == 0

    def test_upper(self):
        assert hasattr(estimator_d, 'upper')
        assert estimator_d.upper == 1

    def test_grid_size(self):
        assert hasattr(estimator_d, 'grid_size')
        assert estimator_d.grid_size == 100

    def test_observations(self):
        assert hasattr(estimator_d, 'observations')
        assert_equal(estimator_d.observations, observations)

    def test_sample_size(self):
        assert hasattr(estimator_d, 'sample_size')
        assert estimator_d.sample_size == 50

    def test_quadrature(self):
        assert hasattr(estimator_d, 'quadrature')
        assert callable(estimator_d.quadrature)


class TestInheritanceDiscretize:
    def test_rectangle(self):
        assert hasattr(estimator_d, 'rectangle')
        assert callable(estimator_d.rectangle)
        assert estimator_d.rectangle(1) == 0.01
        assert_equal(estimator_d.rectangle(np.array([1, 2])), np.array([0.01, 0.01]))

    def test_dummy_(self):
        assert hasattr(estimator_d, 'dummy')
        assert callable(estimator_d.dummy)
        assert estimator_d.dummy(1) == 1
        assert_equal(estimator_d.dummy(np.array([1, 2])), np.array([1, 1]))

    def test_rectangle_grid(self):
        assert hasattr(estimator_d, 'rectangle_grid')
        assert callable(estimator_d.rectangle_grid)
        assert_equal(estimator_d.rectangle_grid(), np.linspace(0, 1, 100, endpoint=False))

    def test_dummy_grid(self):
        assert hasattr(estimator_d, 'dummy_grid')
        assert callable(estimator_d.dummy_grid)
        assert_equal(estimator_d.dummy_grid(), np.linspace(0, 1, 100, endpoint=False))


class TestFunctionalitiesDiscretize:
    def test_L2norm(self):
        assert estimator_d.L2norm(cp.asarray(np.repeat(1, 100)), cp.asarray(np.repeat(1, 100))) == 0
        assert_almost_equal(estimator_d.L2norm(cp.asarray(np.repeat(1, 100)), cp.asarray(np.repeat(0, 100))), 1,
                            decimal=12)

    def test_q_estimator(self):
        estimator_d.estimate_q()
        assert_almost_equal(cp.asnumpy(estimator_d.q_estimator), np.repeat(3 / 5, 100), decimal=12)

    def test_delta(self):
        estimator_d.estimate_delta()
        assert_almost_equal(estimator_d.delta, 0.10954451150103321, decimal=12)


class TestExceptionDiscretize:
    def test_lower(self):
        with raises(AssertionError):
            EstimatorDiscretize(kernel=identity, lower='a', upper=1, grid_size=100, observations=observations,
                                sample_size=50)

    def test_upper(self):
        with raises(AssertionError):
            EstimatorDiscretize(kernel=identity, lower=0, upper='a', grid_size=100, observations=observations,
                                sample_size=50)

    def test_grid_size(self):
        with raises(AssertionError):
            EstimatorDiscretize(kernel=identity, lower=0, upper=1, grid_size='a', observations=observations,
                                sample_size=50)
        with raises(AssertionError):
            EstimatorDiscretize(kernel=identity, lower=0, upper=1, grid_size=10., observations=observations,
                                sample_size=50)

    def test_observations(self):
        with raises(AssertionError):
            EstimatorDiscretize(kernel=identity, lower=0, upper=1, grid_size=100, observations=1, sample_size=50)
        with raises(AssertionError):
            EstimatorDiscretize(kernel=identity, lower=0, upper=1, grid_size=100, observations=[1, 2], sample_size=50)

    def test_sample_size(self):
        with raises(AssertionError):
            EstimatorDiscretize(kernel=identity, lower=0, upper=1, grid_size=100, observations=observations,
                                sample_size='a')
        with raises(AssertionError):
            EstimatorDiscretize(kernel=identity, lower=0, upper=1, grid_size=100, observations=observations,
                                sample_size=50.)

    def test_not_implemented(self):
        with raises(NotImplementedError):
            estimator_d.estimate()
        with raises(NotImplementedError):
            estimator_d.refresh()


estimator_s = EstimatorSpectrum(kernel=identity, observations=observations, sample_size=50, transformed_measure=True,
                                lower=0, upper=1)


class TestAttributesSpectrum:
    def test_estimator(self):
        assert estimator_s is not None

    def test_kernel(self):
        assert hasattr(estimator_s, 'kernel')
        assert callable(estimator_s.kernel)

    def test_lower(self):
        assert hasattr(estimator_s, 'lower')
        assert estimator_s.lower == 0

    def test_upper(self):
        assert hasattr(estimator_s, 'upper')
        assert estimator_s.upper == 1

    def test_observations(self):
        assert hasattr(estimator_s, 'observations')
        assert_equal(estimator_s.observations, observations)

    def test_sample_size(self):
        assert hasattr(estimator_s, 'sample_size')
        assert estimator_s.sample_size == 50

    def test_transformed_measure(self):
        assert hasattr(estimator_s, 'transformed_measure')
        assert isinstance(estimator_s.transformed_measure, bool)
        assert estimator_s.transformed_measure


class TestFunctionalitiesSpectrum:
    def test_q_estimator(self):
        estimator_s.estimate_q()
        assert callable(estimator_s.q_estimator)
        assert_almost_equal(estimator_s.q_estimator(0.5), np.array([6 / 5]))

    def test_delta(self):
        estimator_s.estimate_delta()
        assert_almost_equal(estimator_s.delta, 0.10954451150103321, decimal=12)


class TestExceptionSpectrum:
    def test_lower(self):
        with raises(AssertionError):
            EstimatorSpectrum(kernel=identity, lower='a', upper=1, observations=observations, transformed_measure=True,
                              sample_size=50)

    def test_upper(self):
        with raises(AssertionError):
            EstimatorSpectrum(kernel=identity, lower=0, upper='a', observations=observations, transformed_measure=True,
                              sample_size=50)

    def test_transformed_measure(self):
        with raises(AssertionError):
            EstimatorSpectrum(kernel=identity, lower=0, upper=1, observations=observations, transformed_measure=1,
                              sample_size=50)
        with raises(AssertionError):
            EstimatorSpectrum(kernel=identity, lower=0, upper=1, observations=observations, transformed_measure='True',
                              sample_size=50)

    def test_observations(self):
        with raises(AssertionError):
            EstimatorSpectrum(kernel=identity, lower=0, upper=1, observations=1, sample_size=50,
                              transformed_measure=True, )
        with raises(AssertionError):
            EstimatorSpectrum(kernel=identity, lower=0, upper=1, observations=[1, 2], sample_size=50,
                              transformed_measure=True, )

    def test_sample_size(self):
        with raises(AssertionError):
            EstimatorSpectrum(kernel=identity, lower=0, upper=1, observations=observations, transformed_measure=True,
                              sample_size='a')
        with raises(AssertionError):
            EstimatorSpectrum(kernel=identity, lower=0, upper=1, observations=observations, transformed_measure=True,
                              sample_size=50.)

    def test_not_implemented(self):
        with raises(NotImplementedError):
            estimator_s.estimate()
        with raises(NotImplementedError):
            estimator_s.refresh()
