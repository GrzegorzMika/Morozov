import numpy as np


def BETA(x):
    """
    Beta(4, 2) probability distribution function.
    """
    return 20 * x ** 3 * (1 - x)


def NM(x):
    """
    Mixture of two gaussion distributions N(0.7, 0.08) and N(0.35, 0.08) restricted to interval [0, 1] and scaled to
    form a proper probability distribution function.
    """
    return (0.7 / (np.sqrt(2 * np.pi) * 0.08) * np.exp(-np.power(x - 0.7, 2) / (2 * 0.08 ** 2)) + 0.3 // (
            np.sqrt(2 * np.pi) * 0.08) * np.exp(-np.power(x - 0.35, 2) / (2 * 0.08 ** 2))) / 0.9004671523265059


def SMLA(x):
    """
    Taken from Minerbo, G. N. and Levy, M. E., "Inversion of Abel’s integral equation by means of orthogonal polynomials.",
    SIAM J. Numer. Anal. 6, 598-616 and swapped to satisfy SMLA(0) = 0.
    """
    return np.where(x <= 0.5, 4 * x ** 2, 2 - 4 * (1 - x) ** 2)


def SMLB(x):
    """
    Taken from Minerbo, G. N. and Levy, M. E., "Inversion of Abel’s integral equation by means of orthogonal polynomials.",
    SIAM J. Numer. Anal. 6, 598-616 and swapped to satisfy SMLB(0) = 0.
    """
    return (np.where((x > 0.00000000001), 1.241 * np.multiply(np.power(2 * x - x ** 2, -1.5),
                                                              np.exp(1.21 * (1 - np.power(2 * x - x ** 2, -1)))),
                     0)) / 0.9998251040790366


def BIMODAL(x):
    """
    B. Ćmiel, "Poisson intensity estimation for the Spektor–Lord–Willis problem using a wavelet shrinkage approach",
    Journal of Multivariate Analysis, 112 (2012) 194–206
    """
    return np.add(np.where(x <= 0.8, 28125 / 512 * x ** 2 * (0.8 - x) ** 2, 0),
                  np.where(x >= 0.6, 9375 / 8 * (0.6 - x) ** 2 * (1 - x) ** 2, 0))


def STEP(x):
    """
    A. Dudek, Z. Szkutnik, "Minimax unfolding spheres size distribution from linear sections", Statistica Sinica 18 (2008) 1063–1080
    """
    return np.where(x <= 1 / 3, 0.6, np.where(x <= 3 / 4, 0.9, 1.7))


def kernel(x, y):
    return np.where(x >= y, 2 * y, 0)


def kernel_transformed(x, y):
    return np.where(x >= y, 2, 0)
