import numpy as np
import pandas as pd

from EstimatorSpectrum import Tikhonov
from Generator import LSW
from SVD import LordWillisSpektor
from test_functions import kernel_transformed, BIMODAL, BETA, SMLA, SMLB

replications = 10
size = [2000, 10000, 1000000]
max_size = 100
order = 2
functions = [BIMODAL]
functions_name = ['BIMODAL']
taus = [1]
taus_name = ['10']
rhos = [750, 1000, 2000, 3000, 5000, 6000, 7000, 8000, 9000, 10000, 50000, 100000]
rhos_name = ['750', '1000', '2000', '3000', '5000', '6000', '7000', '8000', '9000', '10000', '50000', '100000']

if __name__ == '__main__':
    for s in size:
        for i, fun in enumerate(functions):
            for j, tau in enumerate(taus):
                for k, rho in enumerate(rhos):
                    generator = LSW(pdf=fun, sample_size=s, seed=914)
                    results = {'selected_param': [], 'oracle_param': [], 'oracle_loss': [], 'loss': [], 'solution': [],
                               'oracle_solution': []}
                    for _ in range(replications):
                        spectrum = LordWillisSpektor(transformed_measure=True)
                        obs = generator.generate()
                        tikhonov = Tikhonov(kernel=kernel_transformed, singular_values=spectrum.singular_values,
                                            left_singular_functions=spectrum.left_functions,
                                            right_singular_functions=spectrum.right_functions,
                                            observations=obs, sample_size=s, max_size=max_size, tau=tau,
                                            transformed_measure=True, rho=rho, order=order)
                        tikhonov.estimate()
                        tikhonov.oracle(fun, patience=10)
                        solution = list(tikhonov.solution(np.linspace(0, 1, 10000)))
                        results['selected_param'].append(tikhonov.regularization_param)
                        results['oracle_param'].append(tikhonov.oracle_param)
                        results['oracle_loss'].append(tikhonov.oracle_loss)
                        results['loss'].append(tikhonov.residual)
                        results['solution'].append(solution)
                        results['oracle_solution'].append(list(tikhonov.oracle_solution))
                        pd.DataFrame(results).to_csv(
                            'Tikhonov_rho_{}_tau_{}_size_{}_fun_{}.csv'.format(rhos_name[k], taus_name[j], s,
                                                                            functions_name[i]))
