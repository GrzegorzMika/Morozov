import numpy as np


def BETA(x):
    return 20 * x ** 3 * (1 - x)


def NM(x):
    return (0.7 / (np.sqrt(2 * np.pi) * 0.08) * np.exp(-np.power(x - 0.7, 2) / (2 * 0.08 ** 2)) + 0.3 // (
            np.sqrt(2 * np.pi) * 0.08) * np.exp(-np.power(x - 0.35, 2) / (2 * 0.08 ** 2))) / 0.9004671523265059


def SMLA(x):
    return np.where(x <= 0.5, 4 * x ** 2, 2 - 4 * (1 - x) ** 2)


def SMLB(x):
    return (np.where((x > 0.00000000001), 1.241 * np.multiply(np.power(2 * x - x ** 2, -1.5),
                                                              np.exp(1.21 * (1 - np.power(2 * x - x ** 2, -1)))),
                     0)) / 0.9998251040790366


def kernel(x, y):
    return np.where(x >= y, 2 * y, 0)


def kernel_transformed(x, y):
    return np.where(x >= y, 2, 0)
