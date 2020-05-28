import sys
from os import path
import numpy as np
from test_functions import BETA, BIMODAL, SMLA, SMLB

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from Generator import LSW

functions = [BETA, BIMODAL, SMLA, SMLB]
functions_name = ['BETA', 'BIMODAL', 'SMLA', 'SMLB']

size = 1000000
replications = 100

with open('coverage.txt', 'w+') as f:
    f.write('Function, Coverage\n')

for i, fun in enumerate(functions):
    generator = LSW(pdf=fun, sample_size=size, seed=914)
    sample = []
    for _ in range(replications):
        obs = generator.generate()
        sample.append(obs.shape[0])
    cover = np.mean(sample)/size
    with open('coverage.txt', 'a+') as f:
        f.write('{}, {}\n'.format(functions_name[i], cover))