import warnings
from abc import abstractmethod, ABCMeta
from typing import Callable, Union, List, Any
import numpy as np


class Generator(metaclass=ABCMeta):
    def __init__(self):
        ...

    @abstractmethod
    def generate(self):
        ...

    @abstractmethod
    def generate_parallel(self, compute: bool = False):
        ...

    @abstractmethod
    def visualize(self):
        ...


class LewisShedler(Generator):

    def __init__(self, intensity_function: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
                 upper: float, lower: float = 0, seed: float = None, lambda_hat: float = None):
        """
        Generator of observations from inhomogeneous Poisson process using Lewis-Shedler algorithm.
        :param intensity_function: intensity function of a simulated inhomogeneous Poisson process
        :type intensity_function: Callable
        :param upper: upper limit of an interval on which process is simulated
        :type upper: float
        :param lower: lower limit of an interval on which process is simulated
        :type lower: float (default: 0.)
        :param seed: seed value for replicability
        :type seed: float (default: None)
        :param lambda_hat: maximum of intensity function on a given interval, if None then the value is approximated
        in algorithm (default: None)
        :type lambda_hat: float (default: None)
        """
        super().__init__()

        assert callable(intensity_function), "intensity_function must be a callable!"
        try:
            intensity_function(np.array([1, 2]))
            self.intensity_function = intensity_function
        except ValueError:
            print('Force vectorization of intensity function')
            self.intensity_function = np.vectorize(intensity_function)
        assert isinstance(upper, float) | isinstance(upper, int), "Wrong type of upper limit!"
        assert isinstance(lower, float) | isinstance(lower, int), "Wrong type of lower limit!"
        if lambda_hat is not None:
            assert isinstance(lambda_hat, float) | isinstance(lambda_hat, int), "Wrong type of lambda_hat!"
        if seed is not None:
            assert isinstance(seed, float) | isinstance(seed, int), "Wrong type of seed!"
        if np.sum(self.intensity_function(np.random.uniform(lower, upper, int(1e5))) < 0) > 0:
            raise ValueError("Intensity function must be greater than or equal to 0!")
        if lower >= upper:
            raise ValueError("Wrong interval is specified! (lower {} >= upper {})".format(lower, upper))
        if lambda_hat is not None and lambda_hat < 0:
            raise ValueError(
                "Maximum of intensity function must be greater than or equal to 0, found {}".format(lambda_hat))

        self.upper = upper
        self.lower = lower
        if lambda_hat is not None:
            self.lambda_hat = float(lambda_hat)
        else:
            self.lambda_hat = np.max(self.intensity_function(np.linspace(self.lower, self.upper, int(1e7))))
        self.max_size = int(5 * int(np.ceil((self.upper - self.lower) * self.lambda_hat)))

        print('Maximum of the intensity function: {}'.format(self.lambda_hat))
        np.random.seed(seed)

    def generate(self) -> np.ndarray:
        """
        Simulation of an Inhomogeneous Poisson process with bounded intensity function λ(t), on [lower, upper] using
        algorithm from "Simulation of nonhomogeneous Poisson processes by thinning." Naval Res. Logistics Quart, 26:403–
        413, 1973. Naming conventions follows "Thinning Algorithms for Simulating Point Processes" by Yuanda Chen.
        Optimized implementation for speed.
        :return: numpy array containing the simulated values of inhomogeneous process.
        """
        u: np.ndarray = np.random.uniform(0, 1, self.max_size)
        w: np.ndarray = np.concatenate((0, -np.log(u) / self.lambda_hat), axis=None)
        s: np.ndarray = np.cumsum(w)
        s = s[s < self.upper]
        d: np.ndarray = np.random.uniform(0, 1, len(s))
        t: np.ndarray = self.intensity_function(s) / self.lambda_hat
        t = s[(d <= t) & (t <= self.upper)]
        return t

    def generate_slow(self) -> np.ndarray:
        """
        Simulation of an Inhomogeneous Poisson process with bounded intensity function λ(t), on [lower, upper] using
        algorithm from "Simulation of nonhomogeneous Poisson processes by thinning." Naval Res. Logistics Quart, 26:403–
        413, 1973. Naming conventions follows "Thinning Algorithms for Simulating Point Processes" by Yuanda Chen.
        Original implementation of an algorithm, not optimized.
        :return: numpy array containing the simulated values of inhomogeneous process.
        """
        warnings.warn('You are using not optimized version of algorithm', RuntimeWarning)
        t: List[Union[Union[int, float], Any]] = []
        s: List[Union[Union[int, float], Any]] = []
        t.append(0)
        s.append(0)
        while s[-1] < (self.upper - self.lower):
            u: float = np.random.uniform(0, 1, 1)[0]
            w: float = -np.log(u) / self.lambda_hat
            s.append(s[-1] + w)
            d: float = np.random.uniform(0, 1, 1)[0]
            if d <= self.intensity_function(s[-1]) / self.lambda_hat:
                t.append(s[-1])
        return np.array(t)

    def visualize(self, save=False):
        """
        Auxiliary function to visualize and save the visualizations of an intensity function,
        exemplary trajectory and location of points
        :param save: to save (True) or not to save (False) the visualizations
        :type save: boolean (default: False)
        """
        import matplotlib.pyplot as plt
        import inspect

        plt.style.use('seaborn-whitegrid')
        plt.rcParams['figure.figsize'] = [10, 5]

        grid = np.linspace(self.lower, self.upper, 10000)
        func = self.intensity_function(np.linspace(self.lower, self.upper, 10000))
        try:
            plt.plot(grid, func)
        except:
            plt.plot(grid, np.repeat(func, 10000))
        plt.title('Intensity function')
        plt.xlabel('time')
        plt.ylabel('value')
        if save:
            try:
                plt.savefig('intensity_function_' + inspect.getsource(self.intensity_function).split('return')[
                    1].strip() + '.png')
                print('Saved as ' + 'intensity_function_' + inspect.getsource(self.intensity_function).split('return')[
                    1].strip() + '.png')
            except:
                warnings.warn("Saving intensity function failed!")
        plt.show()
        plt.clf()

        t = self.generate()
        plt.step(t, list(range(0, len(t))))
        plt.title('Simulated trajectory')
        plt.xlabel('time')
        plt.ylabel('value')
        if save:
            try:
                plt.savefig(
                    'trajectory_' + inspect.getsource(self.intensity_function).split('return')[1].strip() + '.png')
                print('Saved as ' + 'trajectory_' + inspect.getsource(self.intensity_function).split('return')[
                    1].strip() + '.png')
            except:
                warnings.warn("Saving trajectory failed!")
        plt.show()
        plt.clf()

        plt.plot(t, list(np.repeat(0, len(t))), '.')
        plt.title('Simulated points')
        plt.xlabel('time')
        if save:
            try:
                plt.savefig('points_' + inspect.getsource(self.intensity_function).split('return')[1].strip() + '.png')
                print('Saved as ' + 'points_' + inspect.getsource(self.intensity_function).split('return')[
                    1].strip() + '.png')
            except:
                warnings.warn("Saving points failed!")
        plt.show()
        plt.clf()
