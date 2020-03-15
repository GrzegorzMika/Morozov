from numpy.testing import assert_array_equal

from SVD import SpectrumGenerator, LordWillisSpektor
import numpy as np


class TestSpectrumGenerator:
    def test_instance(self):
        assert SpectrumGenerator is not None

    def test_singular_values(self):
        assert hasattr(SpectrumGenerator, 'singular_values')

    def test_left_singular_functions(self):
        assert hasattr(SpectrumGenerator, 'left_functions')

    def test_right_singular_functions(self):
        assert hasattr(SpectrumGenerator, 'right_functions')

    def test_abstract_method(self):
        assert SpectrumGenerator.__abstractmethods__ == {'left_functions', 'right_functions', 'singular_values'}


LordWillisSpektor = LordWillisSpektor()


class TestLWS:
    def test_singular_values(self):
        sigma = LordWillisSpektor.singular_values
        sigma_values = [next(sigma) for _ in range(10)]
        assert sigma_values == [2 / np.pi, 2 / (3 * np.pi), 2 / (5 * np.pi), 2 / (7 * np.pi), 2 / (9 * np.pi),
                                2 / (11 * np.pi), 2 / (13 * np.pi), 2 / (15 * np.pi), 2 / (17 * np.pi),
                                2 / (19 * np.pi)]

    def test_right_singular_functions(self):
        grid = np.linspace(0, 5, 1000)
        rf = LordWillisSpektor.right_functions
        rfun = [next(rf) for _ in range(5)]
        assert_array_equal(rfun[0](grid), 2 * np.sin(np.pi * np.square(grid) / 2))
        assert_array_equal(rfun[1](grid), 2 * np.sin(3 * np.pi * np.square(grid) / 2))
        assert_array_equal(rfun[2](grid), 2 * np.sin(5 * np.pi * np.square(grid) / 2))
        assert_array_equal(rfun[3](grid), 2 * np.sin(7 * np.pi * np.square(grid) / 2))
        assert_array_equal(rfun[4](grid), 2 * np.sin(9 * np.pi * np.square(grid) / 2))

    def test_left_singular_functions(self):
        grid = np.linspace(0, 5, 1000)
        lf = LordWillisSpektor.left_functions
        lfun = [next(lf) for _ in range(5)]
        assert_array_equal(lfun[0](grid), 2 * np.cos(np.pi * np.square(grid) / 2))
        assert_array_equal(lfun[1](grid), 2 * np.cos(3 * np.pi * np.square(grid) / 2))
        assert_array_equal(lfun[2](grid), 2 * np.cos(5 * np.pi * np.square(grid) / 2))
        assert_array_equal(lfun[3](grid), 2 * np.cos(7 * np.pi * np.square(grid) / 2))
        assert_array_equal(lfun[4](grid), 2 * np.cos(9 * np.pi * np.square(grid) / 2))
