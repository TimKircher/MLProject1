import numpy as np

def compute_error(y, tx, w):
    """
    Computes the error
    """
    error = y- tx @ w
    
    return error

def compute_mse_loss(error):
    mean_squared_error = np.mean(error ** 2) / 2
    return mean_squared_error

def compute_gradient(tx, error):
    
    gradient = -(tx @ error) / len(error)
    
    return gradient

def sigmoid(t):
    sigmoid_t = 1 / (1 + np.exp(-t))
    return sigmoid_t

#from course
def compute_logistic_loss(y, tx, w):
    predictions = sigmoid(tx @ w)
    neg_losses_per_datapoint = y * np.log(predictions) + (1-y) * np.log(1-predictions)
    loss = - neg_losses_per_datapoint.sum()
    return losses

#from course
def compute_logistic_gradient(y, tx, w):
    predictions = sigmoid(tx @ w)
    grad = tx @ (predictions - y)
    return grad