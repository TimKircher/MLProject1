from tools import *
from proj1_helpers import batch_iter

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    """
    w = initial_w
    
    for iter in range(max_iters):
        
        error = compute_error(y, tx, w)        
        gradient = compute_gradient(tx, error)
        w -= gamma * gradient
        
    loss = compute_mse_loss(error)
    return w, loss
       
        
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    """
    w = intial_w
    
    for iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            
            error = compute_error(y, tx, w)           
            gradient = compute_gradient(tx, error)
            w -= gamma * gradient
            
    loss = compute_mse_loss(error)
    return w, loss
          
    
    
def least_squares(y, tx):
    """
    Least squares regression using normal equations
    """
    #From least_squares_mathias
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)   #system of type A*x = b solve(A,b)    
    error = compute_error(y, tx, w)
    loss = compute_mse_loss(error)
    
    return w, loss
    
def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    """
    #from ridge_regression_matthias
    w = np.linalg.solve(tx.T @ tx + (lambda_*2*len(y)) * np.identity(tx.shape[1]), tx.T @ y)   #system of type A*x = b solve(A,b)    
    error = compute_error(y, tx, w)
    loss = compute_mse_loss(error)
    
    return w, loss
    
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent
    """
    w = initial_w
    
    for iter in range(max_iters):
        
        loss = compute_logistic_loss(y, tx, w)
        gradient = compute_logistic_gradient(y, tx, w)
        w -= gamma * gradient
    
    return w, loss
        
    
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent
    """ 
    w = initial_w
    
    for iter in range(max_iters):
        
        loss = compute_logistic_loss(y, tx, w) + lambda_ / 2 * np.squeeze(w.T @ w) #Last part might need x2
        gradient = compute_logistic_gradient(y, tx, w) + lambda_ * w #Last part might need x2
        w -= gamma * gradient    
    
    return w, loss