import sys
from os import path

import numpy as np
import pandas as pd
from test_functions import kernel, BETA, NM, SMLA, SMLB

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from EstimatorSpectrum import TSVD
from Generator import LSW
from SVD import LordWillisSpektor

replications = 10
size = [1000000]
max_size = 10
functions = [BETA, NM, SMLA, SMLB]
functions_name = ['BETA', 'NM', 'SMLA', 'SMLB']

if __name__ == '__main__':
    for s in size:
        for i, fun in enumerate(functions):
            generator = LSW(pdf=fun, sample_size=s, seed=123)
            results = {'selected_param': [], 'oracle_param': [], 'oracle_loss': [], 'loss': [], 'solution': [],
                       'oracle_solution': []}
            for _ in range(replications):
                try:
                    spectrum = LordWillisSpektor(transformed_measure=True)
                    obs = generator.generate()
                    tsvd = TSVD(kernel=kernel, singular_values=spectrum.singular_values,
                                left_singular_functions=spectrum.left_functions,
                                right_singular_functions=spectrum.right_functions, transformed_measure=True,
                                observations=obs, sample_size=s, max_size=max_size, njobs=-1)
                    tsvd.estimate()
                    tsvd.oracle(fun)
                    solution = list(tsvd.solution(np.linspace(0, 1, 10000)))
                    results['selected_param'].append(tsvd.regularization_param)
                    results['oracle_param'].append(tsvd.oracle_param)
                    results['oracle_loss'].append(tsvd.oracle_loss)
                    results['loss'].append(tsvd.residual)
                    results['solution'].append(solution)
                    results['oracle_solution'].append(tsvd.oracle_solution)
                    tsvd.client.close()
                except:
                    pass
            pd.DataFrame(results).to_csv('Test1_TSVD_{}_{}.csv'.format(functions_name[i], s))
