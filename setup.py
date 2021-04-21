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

simu = Extension('_simu_swig',
                     extra_compile_args=['-fopenmp'],
                     extra_link_args=[f'-l{omp}'],
                     include_dirs=[f'{BART_PATH}/src/', '/opt/local/include/', '/opt/local/lib/',
                                   np.get_include()],
                     sources=[f'{BART_PATH}/src/simu/phantom.c',
                            'bartpy/simu/simu_wrap.c'],
                     libraries=['box', 'calib', 'dfwavelet', 'geom',
                                   'grecon', 'iter', 'linops', 'lowrank', 
                                   'misc', 'moba', 'nlops', 'noir', 'noncart',
                                   'num', 'sake', 'sense', 'simu', 'wavelet',
                                   'openblas', 'fftw3f', 'fftw3', 'fftw3f_threads',],
                     library_dirs=[f'{BART_PATH}/lib/', '/opt/local/include/', '/opt/local/lib/'],
                     )

fft = Extension('_fft_swig',
                     extra_compile_args=['-fopenmp'],
                     extra_link_args=[f'-l{omp}'],
                     include_dirs=[f'{BART_PATH}/src/', '/opt/local/include/', '/opt/local/lib/',
                                   np.get_include()],
                     sources=[f'{BART_PATH}/src/num/fft.c',
                            'bartpy/num/fft_wrap.c'],
                     libraries=['box', 'calib', 'dfwavelet', 'geom',
                                   'grecon', 'iter', 'linops', 'lowrank', 
                                   'misc', 'moba', 'nlops', 'noir', 'noncart',
                                   'num', 'sake', 'sense', 'simu', 'wavelet',
                                   'openblas', 'fftw3f', 'fftw3', 'fftw3f_threads',],
                     library_dirs=[f'{BART_PATH}/lib/', '/opt/local/include/', '/opt/local/lib/'],
                     )

linops = Extension('_linop_swig',
                     extra_compile_args=['-fopenmp'],
                     extra_link_args=[f'-l{omp}'],
                     include_dirs=[f'{BART_PATH}/src/', '/opt/local/include/', '/opt/local/lib/',
                                   np.get_include()],
                     sources=[f'{BART_PATH}/src/linops/someops.c', f'{BART_PATH}/src/linops/linop.c',
                            'bartpy/linops/linop_wrap.c'],
                     libraries=['box', 'calib', 'dfwavelet', 'geom',
                                   'grecon', 'iter', 'linops', 'lowrank', 
                                   'misc', 'moba', 'nlops', 'noir', 'noncart',
                                   'num', 'sake', 'sense', 'simu', 'wavelet',
                                   'openblas', 'fftw3f', 'fftw3', 'fftw3f_threads',],
                     library_dirs=[f'{BART_PATH}/lib/', '/opt/local/include/', '/opt/local/lib/'],)

iter_module = Extension('_italgos',
                     extra_compile_args=['-fopenmp'],
                     extra_link_args=[f'-l{omp}'],
                     include_dirs=[f'{BART_PATH}/src/', '/opt/local/include/', '/opt/local/lib/',
                                   np.get_include()],
                     sources=[f'{BART_PATH}/src/iter/italgos.c', 'bartpy/italgos/iter_wrap.c'],
                     libraries=['box', 'calib', 'dfwavelet', 'geom',
                                   'grecon', 'iter', 'linops', 'lowrank', 
                                   'misc', 'moba', 'nlops', 'noir', 'noncart',
                                   'num', 'sake', 'sense', 'simu', 'wavelet',
                                   'openblas', 'fftw3f', 'fftw3', 'fftw3f_threads',],
                     library_dirs=[f'{BART_PATH}/lib/', '/opt/local/include/', '/opt/local/lib/'],)

setup(
    name="bartpy",
    version="0.0",
    author="mrirecon",
    author_email="mrirecon@lists.eecs.berkeley.edu",
    description="Python interface for BART: the Berkeley Advanced Reconstruction Toolbox",
    url="https://github.com/malits/bart-swig",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License",
        "Operating System :: OS X",
        "Operating System :: Linux"
    ],
    ext_modules = [simu, fft, linops, iter_module],
    package_dir = {},
    packages = ["bartpy", "bartpy.utils", "bartpy.simu", "bartpy.num", "bartpy.linops", "bartpy.tools"],
)