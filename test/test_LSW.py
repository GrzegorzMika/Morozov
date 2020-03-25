import os
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from pytest import raises, warns
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Generator


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def pdf(x):
    return np.where(x < 0.5, 1, x) / 0.875


def pdf_nonnormalized_small(x):
    return np.exp(-x ** 2)


def pdf_nonnormalized_big(x):
    return np.exp(-x ** 2) * 100


generator = Generator.LSW(pdf='beta', sample_size=100, seed=1, a=1, b=1)


class TestGeneratorString:
    def test_instance(self):
        assert generator is not None

    def test_pdf(self):
        assert hasattr(generator, 'pdf')
        assert generator.pdf == 'beta'

    def test_sample_size(self):
        assert hasattr(generator, 'sample_size')
        assert generator.sample_size == 98

    def test_inverse_transformation(self):
        assert hasattr(generator, 'inverse_transformation')
        assert generator.inverse_transformation

    def test_r_sample(self):
        assert hasattr(generator, 'r_sample')
        assert generator.r_sample is None

    def test_z_sample(self):
        assert hasattr(generator, 'z_sample')
        assert generator.z_sample is None

    def test_kwargs(self):
        assert hasattr(generator, 'kwargs')
        assert generator.kwargs == {'a': 1, 'b': 1}

    def test_generation(self):
        assert hasattr(generator, 'generate')

    def test_visualize(self):
        assert hasattr(generator, 'visualize')

    def test_generation_method(self):
        assert callable(generator.generate)

    def test_visualize_method(self):
        assert callable(generator.visualize)

    def test_cdf(self):
        assert hasattr(generator, 'cdf')
        assert callable(generator.cdf)

    def test_solve(self):
        assert hasattr(generator, '_LSW__solve')
        assert callable(generator._LSW__solve)
        assert generator._LSW__solve(lambda x: x - 1) == 1


class TestObservationnsString:
    def test_sample_r(self):
        assert hasattr(generator, 'sample_r')
        assert callable(generator.sample_r)
        generator.sample_r()
        assert_equal(generator.r_sample, np.load(find('r_sample.npy', '/home')))

    def test_sample_z(self):
        assert hasattr(generator, 'sample_z')
        assert callable(generator.sample_z)
        generator.sample_z()
        assert_equal(generator.z_sample, np.load(find('z_sample.npy', '/home')))

    def test_observations(self):
        observations = generator.generate()
        assert_equal(observations, np.load(find('lsw_sample.npy', '/home')))


generator_custom = Generator.LSW(pdf=pdf, sample_size=100, seed=1)


class TestGeneratorCustom:
    def test_instance(self):
        assert generator_custom is not None

    def test_pdf(self):
        assert hasattr(generator_custom, 'pdf')
        assert generator_custom.pdf == pdf

    def test_sample_size(self):
        assert hasattr(generator_custom, 'sample_size')
        assert generator_custom.sample_size == 98

    def test_inverse_transformation(self):
        assert hasattr(generator_custom, 'inverse_transformation')
        assert not generator_custom.inverse_transformation

    def test_kwargs(self):
        assert hasattr(generator_custom, 'kwargs')
        assert generator_custom.kwargs == {}

    def test_generation(self):
        assert hasattr(generator_custom, 'generate')

    def test_visualize(self):
        assert hasattr(generator_custom, 'visualize')

    def test_generation_method(self):
        assert callable(generator_custom.generate)

    def test_visualize_method(self):
        assert callable(generator_custom.visualize)

    def test_cdf(self):
        assert hasattr(generator_custom, 'cdf')
        assert callable(generator_custom.cdf)
        assert_almost_equal(generator_custom.cdf(1), np.array([1]), decimal=8)

    def test_solve(self):
        assert hasattr(generator_custom, '_LSW__solve')
        assert callable(generator_custom._LSW__solve)
        assert generator_custom._LSW__solve(lambda x: x - 1) == 1


class TestObservationsCustom:
    def test_sample_r(self):
        assert hasattr(generator_custom, 'sample_r')
        assert callable(generator_custom.sample_r)
        generator_custom.sample_r()
        assert_equal(generator_custom.r_sample, np.load(find('r_sample_custom.npy', '/home')))

    def test_sample_z(self):
        assert hasattr(generator_custom, 'sample_z')
        assert callable(generator_custom.sample_z)
        generator_custom.sample_z()
        assert_equal(generator_custom.z_sample, np.load(find('z_sample_custom.npy', '/home')))

    def test_observations(self):
        observations = generator_custom.generate()
        assert_equal(observations, np.load(find('lsw_sample_custom.npy', '/home')))


class TestExceptions:
    def test_pdf(self):
        with raises(AssertionError):
            Generator.LSW(pdf=1, sample_size=100, seed=1)
        with raises(AssertionError):
            Generator.LSW(pdf=True, sample_size=100, seed=1)

    def test_sample_size(self):
        with raises(AssertionError):
            Generator.LSW(pdf=pdf, sample_size='a', seed=1)
        with raises(AssertionError):
            Generator.LSW(pdf=pdf, sample_size=100., seed=1)

    def test_normalization(self):
        with warns(RuntimeWarning):
            Generator.LSW(pdf=pdf_nonnormalized_big, sample_size=100, seed=1)
        with warns(RuntimeWarning):
            Generator.LSW(pdf=pdf_nonnormalized_small, sample_size=100, seed=1)
