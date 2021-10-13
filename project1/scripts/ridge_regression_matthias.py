# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    w = np.linalg.solve(tx.T @ tx + (lambda_*2*len(y)) * np.identity(tx.shape[1]), tx.T @ y)   #system of type A*x = b solve(A,b)
    e = (y - tx @ w)
    mse = 1/(2*len(e)) * e.T @ e
    return mse, w