import numpy as np

from .linop_swig import forward as _forward

def forward(op, dest_dims, src_arr):
    """
    TODO: Figure out a way to access operator out dimensions

    Apply linear operator
    :param op: Linear operator
    :src_arr: source_numpy_array
    """
    dst_dim = dest_dims + [1] * (16 - len(dest_dims))
    src = np.asfortranarray(src_arr, dtype=np.complex64)
    ndims = len(dest_dims)
    
    if ndims < 16: 
        src = np.expand_dims(src, axis=list(range(ndims, 16)))

    return _forward(op, dst_dim, src).squeeze()