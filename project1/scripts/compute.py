import numpy as np

def compute_error(y, tx, w):
    """
    Computes the error
    """
    error =y - (tx @ w)
    return error

def compute_mse_loss(error):
    """
    Computes the mean squared error for a given error
    """
    mean_squared_error = np.mean(error ** 2) / 2
    return mean_squared_error

def compute_gradient(tx, error):
    """
    Computes the gradient for a given error 
    """
    gradient = -tx.T.dot(error) / len(error)
    
    return gradient

def compute_sigmoid(t):
    """
    Computes the output of the sigmoid function for a given input
    """
    t = np.clip(t, -8,8)
    sigmoid_t = 1 / (1 + np.exp(-t))
    return sigmoid_t

def compute_logistic_loss(y, tx, w):
    """
    Computes the logistic loss of a model
    """
    predictions = compute_sigmoid(tx @ w)
    neg_losses_per_datapoint = y * np.log(predictions) + (1-y) * np.log(1-predictions)
    loss = - neg_losses_per_datapoint.sum()
    return loss

def compute_logistic_gradient(y, tx, w):
    """
    Computes the logistic gradient of a model
    """
    predictions = compute_sigmoid(tx @ w)
    grad = tx.T @ (predictions - y)
    return grad

def compute_leaderboard_score(y_true,y_pred):
    """
    Helper function estimating the categorical accuracy on the leaderscore
    """
    N_tot = y_pred.shape[0]
    N_true = len(np.where(y_pred == y_true)[0])
    categorical_acuracy = N_true/N_tot
    return categorical_acuracy