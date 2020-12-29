import numpy as np

from .simu_swig import calc_bart, calc_phantom, calc_geo_phantom, calc_circ, calc_ring, calc_star

# TODO: Need to revise the object structure
def phantom(dims, ksp=False, d3=False, ptype="shepp"):
    """
    Calculate numerical phantom of specified dimensions

    :param dims: Array of dimensions. Cannot exceed 16.
    :param ksp: Toggles K-Space
    :param d3: Toggles 3D
    :param type: 'shepp', 'geo', 'circ', 'ring', 'star', 'bart'
    """

    # Holdover solution while I figure out GIL issue
    dims = list(dims)

    if len(dims) > 16 or len(dims) < 0:
        raise ValueError("Must be a 1-D Array of Dimensions of length between 0 and 16")

    dims += [1] * (16 - len(dims))

    if ptype == 'shepp':
        phantom = calc_phantom(dims, d3, ksp)

    elif ptype == 'geo':
        phantom = calc_geo_phantom(dims, d3, ksp)

    elif ptype == 'circ':
        phantom = calc_circ(dims, d3, ksp)

    elif ptype == 'ring':
        phantom = calc_ring(dims, ksp)

    elif ptype == 'star':
        phantom = calc_star(dims, ksp)

    elif ptype == 'bart':
        phantom = calc_bart(dims, ksp)

    return phantom.squeeze()

