import numpy as np

def sigmoid(t):
    """apply the sigmoid function on t."""
    t = np.clip(t, -25,25) #This line is useful for large values
    return 1./(1.+np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    loss = - np.sum([ prediciton * np.log(sigmoid(sample.T @ w)) + (1-prediciton)*np.log(1-sigmoid(sample.T @ w )) for prediciton, sample in zip(y,tx)])
    return loss    

def calculate_loss_fast(y_ext, x_ext,w):
    """compute the loss: negative log likelihood."""
    y = np.copy(y_ext)
    y[y > 0] = 1
    y[y < 0] = 0
    
    x = np.copy(x_ext)
    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x_ext]
    
    loss = np.sum(np.log(1 + np.exp(np.clip(tx @ w,-25,25)))) -  y.T @ tx @ w 
    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    gradient = tx.T @ ((sigmoid(tx @ w))-y)
    
    return gradient


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """ 
    gradient = calculate_gradient(y,tx,w)
    w -= gamma * gradient

    return w



def logistic_regression_gradient_descent_demo(y, x):
    # init parameters
    max_iter = 10000
    threshold = 1e-8
    gamma = 0.01
    losses = []
    x_ext = np.copy(x)
    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x_ext]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

def logistic_regression(y, x_ext, initial_w=None, max_iters=1000, gamma=0.01):

    x = np.copy(x_ext)
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    
    if initial_w is not None:
        w = initial_w
    else:
        w = np.zeros((tx.shape[1], 1))
        
    for iter in range(max_iters):
        w = learning_by_gradient_descent(y, tx, w, gamma)
        if iter % 100 == 0:
            loss = calculate_loss_fast(y, x, w)
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        
    loss = calculate_loss_fast(y, x, w)
    print("Last iteration={i}, loss={l}".format(i=max_iters, l=loss))
    
    return w


def evaluate(tx,w):
    y = sigmoid(tx @  w)
    y[y > 0.5] = 1
    y[y <= 0.5 ] = -1
    return y