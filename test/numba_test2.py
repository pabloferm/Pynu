from numba import float64, guvectorize
import numpy as np
import time


@guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->(n)")
def nb_f(x, y, res):
    for l, (i, j) in enumerate(zip(x, y)):
        res[l] = i + j


def np_f(x, y):
    return x + y


nen = 1000000
x = np.random.random(nen)
y = np.random.random(nen)


start = time.time()
nb_f(x, y)
end = time.time()
print("Elapsed (numba with compilation) = %s" % (end - start))

start = time.time()
nb_f(x, y)
end = time.time()
print("Elapsed (numba compiled) = %s" % (end - start))

start = time.time()
np_f(x, y)
end = time.time()
print("Elapsed (numpy) = %s" % (end - start))
