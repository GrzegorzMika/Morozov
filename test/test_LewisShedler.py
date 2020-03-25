import os
import sys
from pytest import raises
import numpy as np
from numpy.testing import assert_equal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Generator


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def lam(t):
    return np.where(t > 0.5, 100, 0)


def lam_negative(t):
    return np.where(t > 0.5, -100, 0)


generator = Generator.LewisShedler(intensity_function=lam, lower=0, upper=1, seed=1, lambda_hat=100)


class TestGeneratorInstance:
    def test_instance(self):
        assert generator is not None

    def test_lower(self):
        assert generator.lower == 0

    def test_upper(self):
        assert generator.upper == 1

    def test_max_size(self):
        assert generator.max_size == 500

    def test_lambda_hat(self):
        assert generator.lambda_hat == 100.

    def test_generation(self):
        assert hasattr(generator, 'generate')

    def test_visualize(self):
        assert hasattr(generator, 'visualize')

    def test_generation_method(self):
        assert callable(generator.generate)

    def test_visualize_method(self):
        assert callable(generator.visualize)


observations = generator.generate()
observations_test = np.load(find('test_result.npy', '/home'))


class TestObservationns:
    def test_type(self):
        assert isinstance(observations, np.ndarray)

    def test_type_eacher(self):
        assert observations.dtype == np.float64

    def test_shape(self):
        assert observations.shape == (52,)

    def test_result(self):
        assert_equal(observations, observations_test)


class TestException:
    def test_callable(self):
        with raises(AssertionError):
            Generator.LewisShedler(intensity_function=1, lower=0, upper=1, seed=1)

    def test_lower(self):
        with raises(AssertionError):
            Generator.LewisShedler(intensity_function=lam, lower='a', upper=1, seed=1)

    def test_upper(self):
        with raises(AssertionError):
            Generator.LewisShedler(intensity_function=lam, lower=0, upper='a', seed=1)

    def test_seed(self):
        with raises(AssertionError):
            Generator.LewisShedler(intensity_function=lam, lower=0, upper=1, seed='a')

    def test_order(self):
        with raises(ValueError):
            Generator.LewisShedler(intensity_function=lam, lower=1, upper=0, seed=1)

    def test_lambda_hat(self):
        with raises(ValueError):
            Generator.LewisShedler(intensity_function=lam, lower=0, upper=1, seed=1, lambda_hat=-1)

    def test_intensity_function(self):
        with raises(ValueError):
            Generator.LewisShedler(intensity_function=lam_negative, lower=0, upper=1, seed=1, lambda_hat=100)


def test_lambda_hat_calculation():
    generator = Generator.LewisShedler(intensity_function=lam, lower=0, upper=1, seed=1)
    assert generator.lambda_hat == 100.
