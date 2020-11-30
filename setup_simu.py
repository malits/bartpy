
from distutils.core import setup, Extension
import os
import re
import sys

import numpy as np

#os.environ['CC'] = 'gcc-mp-6'

BART_PATH = os.environ['TOOLBOX_PATH']

omp = 'gomp'
if sys.platform == 'darwin':
       omp = 'omp'

module = Extension('_simu',
                     extra_compile_args=['-fopenmp'],
                     extra_link_args=[f'-l{omp}'],
                     include_dirs=[f'{BART_PATH}/src/', '/opt/local/include/', '/opt/local/lib/',
                                   np.get_include()],
                     sources=[f'{BART_PATH}/src/simu/phantom.c',
                            'simu_wrap.c'],
                     libraries=['box', 'calib', 'dfwavelet', 'geom',
                                   'grecon', 'iter', 'linops', 'lowrank', 
                                   'misc', 'moba', 'nlops', 'noir', 'noncart',
                                   'num', 'sake', 'sense', 'simu', 'wavelet',
                                   'openblas', 'fftw3f', 'fftw3', 'fftw3f_threads',],
                     library_dirs=[f'{BART_PATH}/lib/', '/opt/local/include/', '/opt/local/lib/'],
                     )

setup (name = 'simu',
       version = '0.1',
       author      = "BART",
       description = 'Numerical Simulations in BART',
       ext_modules = [module],
       py_modules = ["simu"]
       )
       