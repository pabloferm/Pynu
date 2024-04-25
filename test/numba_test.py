from numba import njit
import numpy as np
import time

nen = 1000000
x = np.random.random(nen)
y = np.random.random(nen)


@njit(parallel=True)
def go_fast(a, b):  # Function is compiled and runs in machine code
    return a * np.log(b)


# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
go_fast(x, y)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
go_fast(x, y)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
