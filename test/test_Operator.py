import numpy as np
from numpy.testing import assert_equal

from Operator import Quadrature, Operator
from pytest import raises


def identity(x, y):
    return np.where(x < y, 1, 1)


operator = Operator(kernel=identity, lower=0, upper=1, grid_size=100)


class TestOperatorAttributes:
    def test_operator(self):
        assert operator is not None

    def test_kernel(self):
        assert hasattr(operator, 'kernel')
        assert callable(operator.kernel)

    def test_lower(self):
        assert operator.lower == 0

    def test_upper(self):
        assert operator.upper == 1

    def test_grid_size(self):
        assert operator.grid_size == 100

    def test_adjoint(self):
        assert not operator.adjoint

    def test_quadrature(self):
        assert hasattr(operator, 'rectangle')
        assert callable(operator.rectangle)

    def test_matrixes(self):
        assert_equal(operator.K, np.zeros((100, 100)).astype(np.float64))
        assert_equal(operator.KH, np.zeros((100, 100)).astype(np.float64))

    def test_grid(self):
        assert_equal(operator._Operator__grid, np.linspace(0, 1, 100, endpoint=False))

    def test_approximate(self):
        assert hasattr(operator, 'approximate')
        assert callable(operator.approximate)

    def test_columns_builder(self):
        assert hasattr(operator, '_Operator__operator_column')
        assert callable(operator._Operator__operator_column)

    def test_adjoint_builder(self):
        assert hasattr(operator, '_Operator__adjoint_operator_column')
        assert callable(operator._Operator__adjoint_operator_column)

    def test_inheritance(self):
        assert hasattr(operator, 'rectangle')
        assert hasattr(operator, 'dummy')
        assert hasattr(operator, 'rectangle_grid')
        assert hasattr(operator, 'dummy_grid')
        assert callable(operator.rectangle)
        assert callable(operator.dummy)
        assert callable(operator.rectangle_grid)
        assert callable(operator.dummy_grid)


class OperatorFunctionalities:
    pass


class OperatorException:
    pass
