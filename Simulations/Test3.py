import sys
from os import path

import numpy as np
import pandas as pd
from test_functions import kernel, BETA, NM, SMLA, SMLB

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from EstimatorSpectrum import TSVD
from Generator import LSW
from SVD import LordWillisSpektor

replications = 1
size = [2000, 10000]
max_size = 10
taus = [1., 1.1, 1.2, 1.5, 2.]
functions = [BETA, NM, SMLA, SMLB]
functions_name = ['BETA', 'NM', 'SMLA', 'SMLB']

if __name__ == '__main__':
    for s in size:
        for i, fun in enumerate(functions):
            generator = LSW(pdf=fun, sample_size=s, seed=123)
            results = {'selected_param': [], 'oracle_param': [], 'oracle_loss': [], 'loss': [], 'solution': [],
                       'oracle_solution': [], 'tau': []}
            for tau in taus:
                for _ in range(replications):
                    try:
                        spectrum = LordWillisSpektor(transformed_measure=False)
                        obs = generator.generate()
                        tsvd = TSVD(kernel=kernel, singular_values=spectrum.singular_values,
                                    left_singular_functions=spectrum.left_functions,
                                    right_singular_functions=spectrum.right_functions,
                                    observations=obs, sample_size=s, max_size=max_size, tau=tau)

                        tsvd.estimate()
                        tsvd.oracle(fun)
                        solution = list(tsvd.solution(np.linspace(0, 1, 10000)))
                        results['selected_param'].append(tsvd.regularization_param)
                        results['oracle_param'].append(tsvd.oracle_param)
                        results['oracle_loss'].append(tsvd.oracle_loss)
                        results['loss'].append(tsvd.residual)
                        results['solution'].append(solution)
                        results['oracle_solution'].append(tsvd.oracle_solution)
                        results['tau'].append(tau)
                        tsvd.client.close()
                    except:
                        pass
            pd.DataFrame(results).to_csv('Test3_TSVD_{}_{}.csv'.format(functions_name[i], s))
