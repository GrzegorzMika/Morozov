from typing import Callable, Union, List, Optional
import dask
import dask.array as da
import numpy as np
from dask.system import cpu_count
from decorators import vectorize, timer


class Quadrature:
    def __init__(self, lower: Union[int, float], upper: Union[int, float], grid_size: int):
        """
        Functionality to calculate weights in different quadrature schema.
        :param lower: Lower end of interval.
        :type lower: float
        :param upper: Upper end of interval.
        :type upper: float
        :param grid_size: Number of points in a grid.
        :type grid_size: int
        """
        self.lower: float = lower
        self.upper: float = upper
        self.grid_size: int = grid_size
        assert self.lower <= self.upper, "Wrong specification of interval"
        assert self.grid_size > 0, 'Grid has to have at least one point'
        assert isinstance(grid_size, int), 'Specify grid size as integer'

    @vectorize(signature="(),()->()")
    def rectangle(self, t: float) -> float:
        """
        Calculate weight for rectangular quadrature with equal grid.
        :param t: Point in which the weight is calculated.
        :type t: float
        :return: Value of quadrature weight for point t.
        """
        assert self.upper >= t >= self.lower, 'Argument must belong to interval [{}, {}], but {} was given.'.format(
            self.lower, self.upper, t)
        return (self.upper - self.lower) / self.grid_size

    @vectorize(signature="(),()->()")
    def dummy(self, t: float) -> float:
        """
        Calculate weight for dummy quadrature. Weight is always equal to 1 (allows to get approximation of kernel, not operator).
        :param t: Point in which the weight is calculated.
        :type t: float
        :return: Value of quadrature weight for point t.
        """
        return 1.


class Operator(Quadrature):
    def __init__(self, kernel: Callable, lower: Union[float, int], upper: Union[float, int], grid_size: int,
                 quadrature: str = 'rectangle'):
        """
        Build an approximation of integral operator with given kernel on equal grid.
        :param kernel: Kernel of an operator being approximated.
        :type kernel: Callable
        :param lower: Lower end of an interval on which the operator is approximated.
        :type lower: float
        :param upper: Upper end of an interval on which the operator is approximated.
        :type upper: float
        :param grid_size: Size of the grid on which the operator is approximated.
        :type grid_size: int
        :param quadrature: Type of quadrature used to approximate the operator.
        :type quadrature: str
        """
        assert isinstance(lower, float) | isinstance(lower,
                                                     int), "Lower limit must be number, but was {} provided".format(
            lower)
        assert isinstance(upper, float) | isinstance(upper,
                                                     int), "Upper limit must be a number, but was {} provided".format(
            upper)
        assert isinstance(grid_size, int), 'Grid size must be an integer, but was {} provided'.format(grid_size)
        assert quadrature in ['rectangle', 'dummy'], 'This type of quadrature is not supported, currently only {} ' \
                                                     'are supported'.format(
            [method for method in dir(Quadrature) if not method.startswith('_')])
        assert callable(kernel), 'Kernel function must be callable'
        super().__init__(lower, upper, grid_size)
        try:
            kernel(np.array([1, 2]), np.array([1, 2]))
            self.kernel: Callable = kernel
        except ValueError:
            print('Force vectorization of kernel')
            self.kernel: Callable = np.vectorize(kernel)
        self.lower: float = float(lower)
        self.upper: float = float(upper)
        self.grid_size: int = grid_size
        self.quadrature: Callable = getattr(super(), quadrature)
        self.__K: Optional[Union[np.ndarray, da.array]] = None
        self.__KH: Optional[Union[np.ndarray, da.array]] = None

    @property
    def K(self):
        return self.__K

    @property
    def KH(self):
        return self.__KH

    @K.setter
    def K(self, K):
        self.__K = K

    @KH.setter
    def KH(self, KH):
        self.__KH = KH

    @property
    def __grid_list(self) -> List[float]:
        return list(np.linspace(self.lower, self.upper, self.grid_size))

    @dask.delayed
    def operator_column(self, t: float) -> da.array:
        """
        Function constructing nth column of an approximation. Its value is equal to the values of the operator with
        grid as first argument, value of nth grid point weighted by quadrature weight in nth grid point.
        :param t: nth grid point, second argument to kernel function and argument to quadrature weight builder.
        :type t: float
        :return: Dask array containing the nth column of the approximation.
        """
        grid: da.array = da.linspace(self.lower, self.upper, self.grid_size)
        return self.kernel(grid, t) * self.quadrature(t)

    @timer
    def approximate(self, compute: bool = False) -> Union[da.array, np.ndarray]:
        """
        Build entire approximation of an operator as matrix of size grid size x grid size.
        :param compute: Return approximation as dask array (False) or for computations to numpy array (True).
        :type compute: boolean
        :return: Numpy array containing the approximation of the operator on given grid if compute is True, otherwise
        dask array containing the graph of computations of an approximation.
        """
        column_list: List[da.array] = [da.from_delayed(self.operator_column(t), shape=(self.grid_size,), dtype=float)
                                       for t in self.__grid_list]
        operator: da.array = da.stack(column_list, axis=1)
        if compute:
            operator: np.ndarray = operator.compute(num_workers=cpu_count())
        self.K = operator
        self.KH = operator.transpose().conj()
        return operator
