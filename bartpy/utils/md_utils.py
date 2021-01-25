# Utilities for manipulating multidimensional (md) arrays

import numpy as np

def expand_dims(dims):
    """
    Add singleton dims for easier array processing
    """
    ndims = len(dims)
    if ndims < 16:
        return list(dims) + [1] * (16 - ndims)
    return ndims

def expand_array(array):
    """
    Add singleton dimensions to array
    """
    array = np.asfortranarray(array, dtype=np.complex64)
    ndims = array.ndim
    
    if ndims < 16: 
        array = np.expand_dims(array, axis=list(range(ndims, 16)))
        
    return array