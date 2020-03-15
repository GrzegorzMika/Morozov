from abc import ABCMeta, abstractmethod
from typing import Callable

import numpy as np


class SpectrumGenerator(metaclass=ABCMeta):
    @abstractmethod
    def singular_values(self, *args, **kwargs):
        ...

    @abstractmethod
    def right_functions(self, *args, **kwargs):
        ...

    @abstractmethod
    def left_functions(self, *args, **kwargs):
        ...


class LordWillisSpektor(SpectrumGenerator):
    @staticmethod
    def __right(nu: int) -> Callable:
        return lambda x: 2 * np.sin((2 * nu + 1) * np.pi * np.square(x) / 2)

    @staticmethod
    def __left(nu: int) -> Callable:
        return lambda y: 2 * np.cos((2 * nu + 1) * np.pi * np.square(y) / 2)

    @property
    def singular_values(self) -> float:
        """
        Calculate explicitly the singular values for the operator in Lord-Willis-Spektor problem according to Z. Szkutnik,
        "A note on minimax rates of convergence in the Spektor-Lord-Willis problem", Opuscula Mathematica, Vol. 30, No. 2, 2010.
        It provides a generator the obtain the consecutive values.
        """
        nu = 0
        while True:
            yield 2 / (np.pi * (2 * nu + 1))
            nu += 1

    @property
    def right_functions(self) -> Callable:
        """
        Calculate explicitly the right singular functions for the operator in Lord-Willis-Spektor problem according to
        Z. Szkutnik, "A note on minimax rates of convergence in the Spektor-Lord-Willis problem", Opuscula Mathematica,
        Vol. 30, No. 2, 2010. It provides a generator the obtain the consecutive functions.
        """
        nu = 0
        while True:
            yield self.__right(nu)
            nu += 1

    @property
    def left_functions(self) -> Callable:
        """
        Calculate explicitly the left singular functions for the operator in Lord-Willis-Spektor problem according to
        Z. Szkutnik, "A note on minimax rates of convergence in the Spektor-Lord-Willis problem", Opuscula Mathematica,
        Vol. 30, No. 2, 2010. It provides a generator the obtain the consecutive functions.
        """
        nu = 0
        while True:
            yield self.__left(nu)
            nu += 1
