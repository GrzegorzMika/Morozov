import sys
from os import path

from test_functions import *

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from Generator import LSW

# verify the coverage of points for different functions used for tests

functions = [BETA, BIMODAL, SMLA, SMLB, NM, BM1, BM2, INCREASING, STEP, TRIANGULAR]
functions_name = ['BETA', 'BIMODAL', 'SMLA', 'SMLB', 'NM', 'BM1', 'BM2', 'INCREASING', 'STEP', 'TRIANGULAR']

size = 1000000
replications = 100

with open('./output/coverage.txt', 'w+') as f:
    f.write('Function, Coverage\n')

for i, fun in enumerate(functions):
    generator = LSW(pdf=fun, sample_size=size, seed=914)
    sample = []
    for _ in range(replications):
        obs = generator.generate()
        sample.append(obs.shape[0])
    cover = np.mean(sample)/size
    with open('./output/coverage.txt', 'a+') as f:
        f.write('{}, {}\n'.format(functions_name[i], cover))