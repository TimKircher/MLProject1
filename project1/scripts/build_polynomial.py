# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree, add_degree_zero=False):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    if degree > 0:
        #mat = np.full((len(x),degree-1),x)
        b = np.repeat(x[:,np.newaxis], degree, 1)
        if add_degree_zero:
            b = np.column_stack((np.ones(len(x)),b) )
        b = np.cumprod(b, axis=1)
        return b
    else:
        return np.ones((len(x),1))
