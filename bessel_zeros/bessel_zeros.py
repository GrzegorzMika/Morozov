import numpy as np
import os

zeros = np.loadtxt('./bessel_zeros.txt')
np.save('bessel_zeros', zeros)

if os.path.exists('./bessel_zeros.txt'):
    os.remove('./bessel_zeros.txt')
