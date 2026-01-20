import numpy as np

roots, weights = np.polynomial.legendre.leggauss(4)
print(type(roots))
print(weights)