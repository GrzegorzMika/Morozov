import numpy as np
from numpy.testing import assert_equal

from Operator import Quadrature, Operator
from pytest import raises


def identity(x, y):
    return np.where(x < y, 1, 1)


def kernel_nonvector(x, y):
    if x < y:
        return 1
    else:
        return 0


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


def diagonal(x, y):
    return np.where(x == y, 1, 0)

operator_nonadj = Operator(kernel=diagonal, lower=0, upper=1, grid_size=100)
operator_adj = Operator(kernel=diagonal, lower=0, upper=1, grid_size=100, adjoint=True)


class OperatorFunctionalities:
    def test_approximate_diagonal(self):
        operator_nonadj.approximate()
        assert_equal(operator_nonadj.K, np.diag(np.repeat(0.01, 100)))
        assert_equal(operator_nonadj.KH, np.diag(np.repeat(0.01, 100)))
        operator_adj.approximate()
        assert_equal(operator_adj.K, np.diag(np.repeat(0.01, 100)))
        assert_equal(operator_adj.KH, np.diag(np.repeat(0.01, 100)))
        assert_equal(operator_adj.K, operator_nonadj.KH)
        assert_equal(operator_adj.KH, operator_nonadj.KH)
        assert_equal(operator_adj.KH, operator_nonadj.K)
        assert_equal(operator_adj.K, operator_nonadj.K)
    def test_approximate_


class OperatorException:
    def test_lower(self):
        with raises(AssertionError):
            Operator(kernel=identity, lower='a', upper=1, grid_size=10)

    def test_upper(self):
        with raises(AssertionError):
            Operator(kernel=identity, lower=0, upper='a', grid_size=10)

    def test_grid_size(self):
        with raises(AssertionError):
            Operator(kernel=identity, lower=0, upper=1, grid_size='a')

    def test_adjoint(self):
        with raises(AssertionError):
            Operator(kernel=identity, lower=0, upper=1, grid_size=10, adjoint='a')
        with raises(AssertionError):
            Operator(kernel=identity, lower=0, upper=1, grid_size=10, adjoint=1)

    def test_quadrature(self):
        with raises(AssertionError):
            Operator(kernel=identity, lower=0, upper=1, grid_size=10, quadrature=1)
        with raises(AssertionError):
            Operator(kernel=identity, lower=0, upper=1, grid_size=10, quadrature='lol')

    def test_order(self):
        with raises(AssertionError):
            Operator(kernel=identity, lower=1, upper=0, grid_size=10)

    def test_kernel(self):
        with raises(AssertionError):
            Operator(kernel=1, lower=0, upper=1, grid_size=10)
        with raises(AssertionError):
            Operator(kernel='a', lower=0, upper=1, grid_size=10)

    def test_kernel_vectorization(self):
        operator = Operator(kernel=kernel_nonvector, lower=0, upper=1, grid_size=10)
        assert_equal(operator.kernel(np.array([0, 1]), np.array([0, 1])), np.array([0, 0], [0, 0]))
