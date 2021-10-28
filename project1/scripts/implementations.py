from compute import *
from proj1_helpers import batch_iter

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    """
    w = initial_w
    for iter_ in range(max_iters):
        
        error = compute_error(y, tx, w) 
        loss = compute_mse_loss(error)
        gradient = compute_gradient(tx, error)
        w = w - gamma * gradient
        
        if iter_ % 250 == 0:
                print("Current iteration :%d, loss= %.4f" %(iter_, loss)) 
                
        if loss < 0.1 :
            break
    return w, loss
       
        
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    """
    w = initial_w
    
    for iter_ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            
            error = compute_error(y_batch, tx_batch, w)  
            loss = compute_mse_loss(error)
            gradient = compute_gradient(tx_batch, error)
            w -= gamma * gradient
            
            if iter_ % 250 == 0:
                print("Current iteration :%d, loss= %.4f" %(iter_, loss)) 
            
            if loss < 0.1:
                break
    return w, loss
          
    
    
def least_squares(y, tx):
    """
    Least squares regression using normal equations
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)   #system of type A*x = b solve(A,b)    
    error = compute_error(y, tx, w)
    loss = compute_mse_loss(error)
    
    return w, loss
    
def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    """
    w = np.linalg.solve(tx.T @ tx + (lambda_*2*len(y)) * np.identity(tx.shape[1]), tx.T @ y)   #system of type A*x = b solve(A,b)    
    error = compute_error(y, tx, w)
    loss = compute_mse_loss(error)
    
    return w, loss
    
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent
    """
    w = initial_w
    
    for iter_ in range(max_iters):
        
        loss = compute_logistic_loss(y, tx, w)
        gradient = compute_logistic_gradient(y, tx, w)
        w -= gamma * gradient
        
        if iter_ % 250 == 0:
                print("Current iteration :%d, loss= %.4f" %(iter_, loss)) 
    
    return w, loss
        
    
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent
    """ 
    w = initial_w
    
    for iter_ in range(max_iters):
        
        loss = compute_logistic_loss(y, tx, w) + lambda_  * np.squeeze(w.T @ w) #Last part might need x2
        gradient = compute_logistic_gradient(y, tx, w) + 2*lambda_ * w #Last part might need x2
        w -= gamma * gradient   
        
        if iter_ % 250 == 0:
                print("Current iteration :%d, loss= %.4f" %(iter_, loss)) 
    
    return w, loss