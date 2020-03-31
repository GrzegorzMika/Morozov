import sys
import os
import numpy as np
from numpy.testing import assert_equal
from pytest import raises

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Operator import Quadrature

quadrature = Quadrature(lower=0, upper=1, grid_size=10)


class TestQuadrature:
    def test_quadrature(self):
        assert quadrature is not None

    def test_lower(self):
        assert quadrature.lower == 0

    def test_upper(self):
        assert quadrature.upper == 1

    def test_grid_size(self):
        assert quadrature.grid_size == 10

    def test_rectangle(self):
        assert hasattr(quadrature, 'rectangle')
        assert callable(quadrature.rectangle)
        assert quadrature.rectangle(1) == 0.1
        assert_equal(quadrature.rectangle(np.array([1, 2])), np.array([0.1, 0.1]))

    def test_dummy_(self):
        assert hasattr(quadrature, 'dummy')
        assert callable(quadrature.dummy)
        assert quadrature.dummy(1) == 1
        assert_equal(quadrature.dummy(np.array([1, 2])), np.array([1, 1]))

    def test_rectangle_grid(self):
        assert hasattr(quadrature, 'rectangle_grid')
        assert callable(quadrature.rectangle_grid)
        assert_equal(quadrature.rectangle_grid(), np.linspace(0, 1, 10, endpoint=False))

    def test_dummy_grid(self):
        assert hasattr(quadrature, 'dummy_grid')
        assert callable(quadrature.dummy_grid)
        assert_equal(quadrature.dummy_grid(), np.linspace(0, 1, 10, endpoint=False))


class TestQuadratureException:
    def test_order(self):
        with raises(AssertionError):
            Quadrature(lower=1, upper=0, grid_size=10)

    def test_grid_size(self):
        with raises(AssertionError):
            Quadrature(lower=0, upper=1, grid_size=0)
        with raises(AssertionError):
            Quadrature(lower=0, upper=1, grid_size=-1)
        with raises(AssertionError):
            Quadrature(lower=0, upper=1, grid_size='a')
        with raises(AssertionError):
            Quadrature(lower=0, upper=1, grid_size=1.)

    def test_lower(self):
        with raises(AssertionError):
            Quadrature(lower='a', upper=1, grid_size=10)

    def test_upper(self):
        with raises(AssertionError):
            Quadrature(lower=0, upper='a', grid_size=10)
