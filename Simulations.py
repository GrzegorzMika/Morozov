import numpy as np
import pandas as pd
from tqdm import tqdm

from EstimatorSpectrum import TSVD, Landweber
from Generator import LewisShedler
from SVD import LordWillisSpektor


def true(x):
    return np.multiply(x ** 3, 1 - x) * 20


size = 2000


def kernel(x, y):
    return np.where(x >= y, 2 * y, 0)


if __name__ == '__main__':
    parameter = []
    oracle = []
    oracle_error = []
    solutions = []
    residual = []
    for _ in tqdm(range(1000)):
        lsw = LordWillisSpektor(transformed_measure=False)
        right = lsw.right_functions

        try:
            size_n = np.random.poisson(lam=size, size=None)


            def g(x):
                return 2 * size_n * (4 * np.power(x, 6) - 5 * np.power(x, 5) + x)


            generator = LewisShedler(intensity_function=g, upper=1, lower=0)
            obs = generator.generate()
        except ValueError:
            size_n = np.random.poisson(lam=size, size=None)


            def g(x):
                return 2 * size_n * (4 * np.power(x, 6) - 5 * np.power(x, 5) + x)


            generator = LewisShedler(intensity_function=g, upper=1, lower=0)
            obs = generator.generate()

        tsvd = TSVD(kernel=kernel, singular_values=lsw.singular_values,
                    left_singular_functions=lsw.left_functions, right_singular_functions=lsw.right_functions,
                    observations=obs, sample_size=size, max_size=100, tau=1)

        tsvd.estimate()
        tsvd.oracle(true)
        solution = list(tsvd.solution(np.linspace(0, 1, 10000)))
        parameter.append(tsvd.regularization_param)
        oracle.append(tsvd.oracle_param)
        oracle_error.append(tsvd.oracle_loss)
        solutions.append(solution)
        residual.append(tsvd.residual)
        tsvd.refresh()

    results = pd.DataFrame({'Parameter': parameter, 'Oracle': oracle, 'Oracle_loss': oracle_error,
                            'Residual': residual, 'Solution': solutions})
    results.to_csv('Simulation1.csv')
