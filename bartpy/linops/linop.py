import numpy as np

from .linop_swig import forward as _forward
from .linop_swig import plus as _plus

from ..utils.md_utils import expand_array, expand_dims

def forward(op, dest_dims, src_arr):
    """
    TODO: Figure out a way to access operator out dimensions

    Apply linear operator
    :param op: Linear operator
    :src_arr: source_numpy_array
    """
    dst_dim = expand_dims(dest_dims)
    src = expand_array(src_arr)

    return _forward(op, dst_dim, src).squeeze()

def plus(op_a, op_b):
    """
    Add two linear operators

    :param op_a: First operator
    :param op_b: Second operator

    :returns the sum of the two operators:
    """
    return _plus(op_a, op_b)