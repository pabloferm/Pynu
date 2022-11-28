from setuptools import setup
import os

os.environ["PYNU"] = os.path.dirname(os.path.abspath(__name__))

setup(
    name='pynu',
    version='1.1.0',    
    description='A Python-based package for neutrino analyses',
    url='https://github.com/pabloferm/Pynu',
    author='Pablo',
    author_email='pablo.fernandez@dipc.org',
    packages=['pynu'],
    install_requires=['pandas',
                      'numpy',
                      'nuSQuIDS',
                      'h5py',
                      'scipy',
                      'nuflux'                     
                      ],
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux', 
        'Programming Language :: Python :: 3.5',
    ],
)
