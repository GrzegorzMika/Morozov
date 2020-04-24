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
size = [2000]
max_size = 10
# taus = [1, 1.1, 1.2, 1.5, 1.8, 2, 2.2]
# taus_name = ['10', '11', '12', '15', '18', '20', '22']
# functions = [BETA, NM, SMLA, SMLB]
# functions_name = ['BETA', 'NM', 'SMLA', 'SMLB']
taus = [1.5]
taus_name = ['15']
functions = [NM]
functions_name = ['NM']
if __name__ == '__main__':
    for s in size:
        for i, fun in enumerate(functions):
            for j, tau in enumerate(taus):
                generator = LSW(pdf=fun, sample_size=s, seed=123)
                results = {'selected_param': [], 'oracle_param': [], 'oracle_loss': [], 'loss': [], 'solution': [],
                           'oracle_solution': []}
                for _ in range(replications):
                    try:
                        spectrum = LordWillisSpektor(transformed_measure=True)
                        obs = generator.generate()
                        tsvd = TSVD(kernel=kernel, singular_values=spectrum.singular_values,
                                    left_singular_functions=spectrum.left_functions,
                                    right_singular_functions=spectrum.right_functions,
                                    observations=obs, sample_size=s, transformed_measure=True,
                                    max_size=max_size, njobs=-1, tau=tau)
                        tsvd.estimate()
                        tsvd.oracle(fun)
                        solution = list(tsvd.solution(np.linspace(0, 1, 10000)))
                        results['selected_param'].append(tsvd.regularization_param)
                        results['oracle_param'].append(tsvd.oracle_param)
                        results['oracle_loss'].append(tsvd.oracle_loss)
                        results['loss'].append(tsvd.residual)
                        results['solution'].append(solution)
                        results['oracle_solution'].append(list(tsvd.oracle_solution))
                        tsvd.client.close()
                    except:
                        pass
                pd.DataFrame(results).to_csv('TSVD2_{}_{}.csv'.format(functions_name[i], taus_name[j]))
