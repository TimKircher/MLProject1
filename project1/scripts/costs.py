# -*- coding: utf-8 -*-
"""A function to compute the cost."""
from numpy import dot

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_rmse(mse):
    return np.sqrt(2*mse)

def calculate_logistic_loss(y,tx,w):
    """compute the loss: negative log likelihood."""
    loss = np.sum(np.log(1 + np.exp(tx @ w))) -  y.T @ tx @ w 
    return loss