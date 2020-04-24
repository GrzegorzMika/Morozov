import sys
from os import path

import numpy as np
import pandas as pd
from test_functions import kernel, BETA, NM, SMLA, SMLB

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from EstimatorSpectrum import Tikhonov
from Generator import LSW
from SVD import LordWillisSpektor

replications = 10
size = [10000]
max_size = 50
max_iter = 30
orders = [2]
taus = [1, 1.5]
taus_name = ['10', '15']
functions = [BETA, NM, SMLA, SMLB]
functions_name = ['BETA', 'NM', 'SMLA', 'SMLB']

if __name__ == '__main__':
    for s in size:
        for i, fun in enumerate(functions):
            for order in orders:
                for j, tau in enumerate(taus):
                    generator = LSW(pdf=fun, sample_size=s, seed=123)
                    results = {'selected_param': [], 'oracle_param': [], 'oracle_loss': [], 'loss': [], 'solution': [],
                               'oracle_solution': []}
                    for _ in range(replications):
                        try:
                            spectrum = LordWillisSpektor(transformed_measure=False)
                            obs = generator.generate()
                            tikhonov = Tikhonov(kernel=kernel, singular_values=spectrum.singular_values,
                                                left_singular_functions=spectrum.left_functions,
                                                right_singular_functions=spectrum.right_functions,
                                                observations=obs, sample_size=s, transformed_measure=False,
                                                max_size=max_size, order=order, njobs=-1)
                            tikhonov.estimate()
                            tikhonov.oracle(fun)
                            solution = list(tikhonov.solution(np.linspace(0, 1, 10000)))
                            results['selected_param'].append(tikhonov.regularization_param)
                            results['oracle_param'].append(tikhonov.oracle_param)
                            results['oracle_loss'].append(tikhonov.oracle_loss)
                            results['loss'].append(tikhonov.residual)
                            results['solution'].append(solution)
                            results['oracle_solution'].append(list(tikhonov.oracle_solution))
                            tikhonov.client.close()
                        except:
                            pass
                    pd.DataFrame(results).to_csv('Tikhonov_non_{}_{}_{}_{}.csv'.format(functions_name[i], s, order, taus_name[j]))
