from proj1_helpers import *
from build_polynomial import build_poly
from abc import ABCMeta, abstractmethod
from k_fold_validate_matthias import build_k_indices
from ridge_regression_matthias import ridge_regression
from costs import compute_mse
import numpy as np

class LinearModel:
    def __init__(self, data_container=None):
        self.data_container = data_container
        self.w = None
        self.mse = None
    
    def add_data(self,data_container):
        self.data_container = data_container
    
    def _cross_validate(self, k_indices, k, **kwargs):
        test_indices = k_indices[k].flatten()
        train_indices = np.delete(k_indices,k,0).flatten()
        test_data = self.data_container.tX[test_indices]
        train_data = self.data_container.tX[train_indices]
        y_test = self.data_container.y[test_indices]
        y_train = self.data_container.y[train_indices]
        
        w = self._run(y=y_train, tx=train_data, **kwargs)
        
        loss_tr = compute_mse(y_train, train_data,w)
        loss_te = compute_mse(y_test, test_data,w)
        return loss_tr, loss_te
        
    def cross_validation(self, k_fold, seed=1, **kwargs):
        k_indices = build_k_indices(self.data_container.y, k_fold, seed)
        tmp_tr = []
        tmp_te = []
        for k in range(k_fold):
            loss_tr, loss_te = self._cross_validate(k_indices, k, **kwargs)
            tmp_tr.append(loss_tr)
            tmp_te.append(loss_te)
        
        return np.sqrt(2*np.mean(tmp_tr)), np.sqrt(2*np.mean(tmp_te))
        
        
    @abstractmethod
    def _run(self, *args, **kwargs):
        raise NotImplementedError("Must override with actual model")
                


class RidgeRegression(LinearModel):
    def _run(self, y=None, tx=None, *args, **kwargs):
        
        if (y is None) and (tx is None) :
            y = self.data_container.y
            tx = self.data_container.tX
        
        lambda_ = kwargs.get("lambda_")
        
        _, w = ridge_regression( y, tx,lambda_=lambda_)
        return w

