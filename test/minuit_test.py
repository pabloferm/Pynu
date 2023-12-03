from iminuit import Minuit
import numpy as np
from scipy.optimize import minimize
import time

def cost_function(x, y, z):
    return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

start = time.time()

m = Minuit(cost_function, x=0, y=0, z=0)

m.migrad()  # run optimiser
end = time.time()
print("Elapsed (numba with compilation) = %s" % (end - start))

# m.hesse()   # run covariance estimator

print(m.values)  # x: 2, y: 3, z: 4
print(m.errors)  # x: 1, y: 1, z: 1

start = time.time()

x0 = np.zeros(3)
print(type(x0))

res = minimize(cost_function, x0, method='L-BFGS-B')

end = time.time()
print("Elapsed (numba with compilation) = %s" % (end - start))
print(res.x)