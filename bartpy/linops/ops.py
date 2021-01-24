from .linop_swig import cdiag_create, rdiag_create, identity_create

import numpy as np

def diag(diag, flags, dtype="r"):
    """
    Create real-valued diagonal operator

    :param diag: diagonal elements
    :param flags:
    TODO: change this parameter
    :param dtype: "r" or "c" to indicate real or complex
    """
    diag_arr = np.array(diag, dtype="complex64")
    dims = list(diag_arr.shape) + [1] * (16 - len(diag_arr))

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
    dims = list(dims) + [1] * (16 - len(dims))

    return identity_create(dims)