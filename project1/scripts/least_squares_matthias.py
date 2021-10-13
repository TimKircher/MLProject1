# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)   #system of type A*x = b solve(A,b)
    e = (y - tx @ w)
    mse = 1/(2*len(e)) * e.T @ e
    return mse, w
