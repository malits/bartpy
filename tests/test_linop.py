# These are the BART Linop tests recreated in
# bartpy / numpy
import numpy as np

from bartpy.linops.linop import forward, plus
from bartpy.linops.ops import cdiag


def test_linop_plus():
    N = 3
    dims = [8, 4, 2]

    diag1a = [3]
    diag1b = 