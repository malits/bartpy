import numpy as np

from .linop_swig import plus, chain, stack, domain, codomain
# FIXME: Clean up these imports, modify namespace in SWIG
from .linop_swig import forward as _forward
from .linop_swig import adjoint as _adjoint
from .linop_swig import normal as _normal
from .linop_swig import pseudo_inv as _pseudo_inv

from ..utils.md_utils import expand_array, expand_dims

def forward(op, dest_dims, src_arr):
    """
    TODO: Figure out a way to access operator out dimensions

    Apply linear operator
    :param op: Linear operator
    :dest_dims: Output dimensions
    :src_arr: source numpy array
    """
    dst_dim = expand_dims(dest_dims)
    src = expand_array(src_arr)

    return _forward(op, dst_dim, src).squeeze()

def adjoint(op, dest_dims, src_arr):
    """
    Apply adjoint
    :param op: Linear Operator
    :dest_dims: Output dimensions
    :src_arr: source numpy array
    """
    dst_dim = expand_dims(dest_dims)
    src = expand_array(src_arr)

    return _adjoint(op, dst_dim, src).squeeze()

def normal(op, dest_dims, src_arr):
    """
    :param op: Linear Operator
    :dest_dims: Output dimensions
    :src_arr: 
    """
    dst_dim = expand_dims(dest_dims)
    src = expand_array(src_arr)

    return _normal(op, dst_dim, src).squeeze()

def pseudo_inv(op, llambda, dest_dims, src_arr):
    """
    :param op: Linear Operator
    :dest_dims: Output dimensions
    :src_arr: 
    :llambda: lambda term
    """
    dst_dim = expand_dims(dest_dims)
    src = expand_array(src_arr)

    return _pseudo_inv(op, llambda, dst_dim, src)