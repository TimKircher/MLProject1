#k_fold cross validation as implemented by Matthias and given in excercises
import numpy as np
from least_squares_matthias import least_squares

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    test_indices = k_indices[k].flatten()
    train_indices = np.delete(k_indices,k,0).flatten()
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # form data with polynomial degree: TODO
    # ***************************************************
    test_data = x[test_indices]
    train_data = x[train_indices]
    test_poly = build_poly(test_data,degree)
    train_poly = build_poly(train_data,degree)
    y_test = y[test_indices]
    y_train = y[train_indices]
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    
    ridge_mse, ridge_w = ridge_regression(y_train, train_poly, lambda_)
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate the loss for train and test data: TODO
    # ***************************************************
    loss_tr = compute_mse(y_train, train_poly,ridge_w)
    loss_te = compute_mse(y_test, test_poly,ridge_w)
    return loss_tr, loss_te


def cross_validation_demo(y,x):
    seed = 1
    degree = 7
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation: TODO
    # *************************************************** 
    for lambda_ in lambdas:
        tmp_tr = []
        tmp_te = []
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree)
            tmp_tr.append(loss_tr)
            tmp_te.append(loss_te)
        rmse_tr.append(np.sqrt(2*np.mean(tmp_tr)))
        rmse_te.append(np.sqrt(2*np.mean(tmp_te)))