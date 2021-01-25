from ..utils.md_utils import *

from .linop_swig import cdiag_create, rdiag_create, identity_create

import numpy as np

def diag(shape, diag, flags, dtype="r"):
    """
    Create real-valued diagonal operator

    :param diag: diagonal elements
    :param flags:
    TODO: change this parameter
    :param dtype: "r" or "c" to indicate real or complex
    """
    diag_arr = np.array(diag, dtype="complex64")

    if len(shape) > 16:
        raise ValueError("Cannot exceed 16 dimensions")

    dims = expand_dims(shape)

    if dtype == "c":
        return cdiag_create(dims, flags, diag_arr)
    elif dtype == "r":
        return rdiag_create(dims, flags, diag_arr)
    else:
        raise TypeError(f"\"{dtype}\" is not a valid datatype.")

def identity(dims):
    """
    Create an identity linear operator

    :param dims: array of dimensions
    """
    dims = expand_dims(dims)

    return identity_create(dims)