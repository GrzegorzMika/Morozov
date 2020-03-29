import sys
from os import path

import numpy as np
import pandas as pd
from test_functions import kernel_transformed, BETA, NM, SMLA, SMLB

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from EstimatorSpectrum import Landweber
from Generator import LSW
from SVD import LordWillisSpektor

replications = 10
size = 10000
max_size = 50
max_iter = 30
taus = [1.1, 1.2]
functions = [BETA, NM, SMLA, SMLB]
functions_name = ['BETA', 'NM', 'SMLA', 'SMLB']

if __name__ == '__main__':
    for t in taus:
        for i, fun in enumerate(functions):
            generator = LSW(pdf=fun, sample_size=size, seed=123)
            results = {'selected_param': [], 'oracle_param': [], 'oracle_loss': [], 'loss': [], 'solution': [],
                       'oracle_solution': []}
            for _ in range(replications):
                try:
                    spectrum = LordWillisSpektor(transformed_measure=True)
                    obs = generator.generate()
                    landweber = Landweber(kernel=kernel_transformed, singular_values=spectrum.singular_values,
                                          left_singular_functions=spectrum.left_functions,
                                          right_singular_functions=spectrum.right_functions,
                                          observations=obs, sample_size=size, max_size=max_size, tau=t,
                                          max_iter=max_iter, transformed_measure=True, njobs=3)
                    landweber.estimate()
                    landweber.oracle(fun)
                    solution = list(landweber.solution(np.linspace(0, 1, 10000)))
                    results['selected_param'].append(landweber.regularization_param)
                    results['oracle_param'].append(landweber.oracle_param)
                    results['oracle_loss'].append(landweber.oracle_loss)
                    results['loss'].append(landweber.residual)
                    results['solution'].append(solution)
                    results['oracle_solution'].append(landweber.oracle_solution)
                    landweber.client.close()
                except:
                    pass
            pd.DataFrame(results).to_csv('TransformedMeasure_Landweber_{}_{}.csv'.format(functions_name[i], t))
