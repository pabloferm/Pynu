from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("mcmc_cython", ["mcmc_cython.pyx"])
]

setup(
    ext_modules=cythonize(extensions),
)
