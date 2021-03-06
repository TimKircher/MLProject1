from proj1_helpers import *
from build_polynomial import build_poly
from abc import ABCMeta, abstractmethod
from k_fold_validate_matthias import build_k_indices
from ridge_regression_matthias import ridge_regression
from logistic_regression import logistic_regression, calculate_loss_fast, sigmoid
from costs import compute_mse
import numpy as np

class LinearModel:
    """Abstract class that serves as the building block for any linear model
       
       LinearModel(Data_cleaner_object)
       
       retrieves data from Data_cleaner_object
       
       Attributes
       ----------
       data_container : Data_cleaner object
                        containes the datasets data, predictions, and ids
       w              : Numpy array shape (D_features,)
                        weight vector of linear model, NOT USED
       mse            : float
                        mean squared error, NOT USED
    
        Methods
        -------
        add_data(data_container): 
                Adds Data_cleaner object if it was not given during initialization
        
        _cross_validate(k_indices, k):
                Runs the kth crossvalidation of a k_fold cross validation
                e.g. dataset separated in [1,2,3,4] sets.
                if k is 2:
                    training is run on [1,3,4]
                    and test on  [2]
                training loss MSE and test loss MSE is returned
                
        cross_validation(k_fold, seed=1):
                runs k_fold cross validation on dataset that is stored in data_container
                calls _cross_validate for kth evaluation
                returns RMSE of the average k_fold training MSE and test MSE
                TODO: is this correct? 
            
        _run():
                abstract method that runs the actual model
                to be replaced in the child classes
       
       
    """
    def __init__(self, data_container=None,loss=None):
        self.data_container = data_container
        self.w = None
        self.mse = None
        self.loss = loss
    
    def add_data(self,data_container):
        """function that returns
        """
        self.data_container = data_container
    
    def _cross_validate(self, k_indices, k, **kwargs):
        test_indices = k_indices[k].flatten()
        train_indices = np.delete(k_indices,k,0).flatten()
        test_data = self.data_container.tX[test_indices]
        train_data = self.data_container.tX[train_indices]
        y_test = self.data_container.y[test_indices]
        y_train = self.data_container.y[train_indices]
        
        w = self._run(y=y_train, tx=train_data, **kwargs)
        
        loss_tr = self.loss(y_train, train_data,w)
        loss_te = self.loss(y_test, test_data,w)
        return loss_tr, loss_te
        
    def cross_validation(self, k_fold, seed=1, **kwargs):
        k_indices = build_k_indices(self.data_container.y, k_fold, seed)
        tmp_tr = []
        tmp_te = []
        for k in range(k_fold):
            loss_tr, loss_te = self._cross_validate(k_indices, k, **kwargs)
            tmp_tr.append(loss_tr)
            tmp_te.append(loss_te)
        
        return np.mean(tmp_tr), np.mean(tmp_te)
        
        
    @abstractmethod
    def _run(self, *args, **kwargs):
        raise NotImplementedError("Must override with actual model")
                


class RidgeRegression(LinearModel):
    """ Child class of the LinearModel class
        _run runs the ridge_regression model
        
        pass lambda_ in **kwargs for cross_validation, or _run
        
        Methods
        -------
        _run(y,tx,)
            performs ridge regression
            pass lambda_ here 
            
        
    """
    def __init__(self, data_container=None,loss=compute_mse):
        super().__init__(data_container=data_container,loss=loss)
    
    def _run(self, y=None, tx=None, *args, **kwargs):
        
        if (y is None) and (tx is None) :
            y = self.data_container.y
            tx = self.data_container.tX
        
        lambda_ = kwargs.get("lambda_")
        
        _, w = ridge_regression( y, tx,lambda_=lambda_)
        return w

class LogisiticRegression(LinearModel):
    def __init__(self, data_container=None,loss=calculate_loss_fast):
        super().__init__(data_container=data_container,loss=loss)
        
        #we should transform tx directly here
    
    def _run(self, y=None, tx=None, *args, **kwargs):
        
        if (y is None) and (tx is None) :
            y = np.copy(self.data_container.y)
            tx = np.copy(self.data_container.tX)
        
        #check this
        y_int = np.copy(y)
        y_int[y_int > 0 ] = 1.
        y_int[y_int < 0 ] = 0.
        y_int = y_int[:,np.newaxis]
        
        gamma = kwargs.get("gamma")
        max_iters = kwargs.get("max_iters")
        initial_w = kwargs.get("initial_w") #returns None if key not in dict
        
        w = logistic_regression(y_int, tx, initial_w=initial_w, max_iters=max_iters, gamma=gamma)
        
        return w
    
    
    def evaluate(self,w,x_test):
        
        tx = np.c_[np.ones((x_test.shape[0], 1)), x_test]
        y = sigmoid(tx @  w)
        
        y[y > 0.5] = 1
        y[y <= 0.5 ] = -1
        return y