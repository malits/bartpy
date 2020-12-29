#experimental Mac OS X fix for parallel issue
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import numpy as np

from .fft_swig import *
from .fft_swig import fft as _fft
from .fft_swig import ifft as _ifft

def fft(src, flags=0, unitary=False, centered=True):
    """
    Perform a Fast Fourier Transform

    :param src: Input data
    :param flags: flags
    :param unitary: Boolean values that toggles unitary FFT
    :param centered: Boolean value that toggles centered and uncentered FFT

    :returns: Fourier-transformed data
    """

    dims = list(src.shape)
    singelton_dims = (16 - len(dims))

    dims = dims + [1] * singelton_dims

    for i in range(singelton_dims):
        src = src[..., np.newaxis]

    if centered:
        result = _fft(dims, flags, np.asfortranarray(src))
    else:
        raise NameError("Not yet implemented")

    return result.squeeze()

def ifft(src, flags=0, unitary=False, centered=True):
    """
    Perform an inverse FFT

    :param src: Input data
    :param flags: indicate active dimensions
    :param unitary: Boolean value that toggles unitary FFT
    :param centered: Boolean value that toggles centering

    :returns: Inverse-FFT'd data.
    """

    dims = list(src.shape)
    singelton_dims = (16 - len(dims))

    dims = dims + [1] * singelton_dims

    for i in range(singelton_dims):
        src = src[..., np.newaxis]

    if centered:
        result = _ifft(dims, flags, np.asfortranarray(src))
    else:
        raise NameError("Not yet implemented")

    return result.squeeze()