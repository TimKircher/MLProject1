from linear_model_base import RidgeRegression
import numpy as np
from data_cleaner import Data_Cleaner
from proj1_helpers import *

def build_interactions(x):
    x_out = np.array(x)
    for i in range(int(x.shape[1])):
        x_i = x[:,0]
        x = np.delete(x, 0, 1)
        x_interact = (x_i*x.T).T
        
        x_out = np.hstack([x_out,x_interact])
        
    return x_out


def build_poly(x, degree, add_degree_zero=False):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    if add_degree_zero:
        xN = np.hstack([np.ones([x.shape[0],1]),x])
    else:
        xN = x
    if degree>0:
        for i in range(degree-1):
            xN = np.hstack([xN, x**(i+2)])
    return np.array(xN)

#best lambda from crossvalidation of lambda from -15 to -5
best_lambda = np.array([5.17947468e-10])

#generate min/max for scaling test set according to original training 
data = Data_Cleaner("../data/train.csv")
#fill -999 with nan and replace nan with 1, due to log scaling 
data._fill_with_NaN()
data.fix_mass_MMC()
data.replace_with_one()
#find columns with multiscale input (max greater than 100)
log_columns = np.max(data.tX, axis=0)>100
#log(x+1) of columns with multiscale data, to ensure no x <= 0
data.tX[:,log_columns] = np.log(data.tX[:,log_columns]+1)
#generate poly features and interaction features
data.tX = build_poly(data.tX,2)
data.tX = build_interactions(data.tX)
#split data
tX_train, tX_test, y_train, y_test = data.split_data(80)
#generate data cleaner
data_train = Data_Cleaner()
data_train.tX = tX_train
data_train.y = y_train
#outlier detection
data_train.treat_outliers(1.5,92.5)
#extract min/max
minimum, maximum = data_train.getMinMax()
data_train.standardize()

#generate predictions for testset
data_upload = Data_Cleaner("../data/test.csv")
#fill -999 with nan and replace nan with 1, due to log scaling 
data_upload._fill_with_NaN()
data_upload.fix_mass_MMC()
data_upload.replace_with_one()
#find columns with multiscale input (max greater than 100)
log_columns = np.max(data_upload.tX, axis=0)>100
#log(x+1) of columns with multiscale data, to ensure no x <= 0
data_upload.tX[:,log_columns] = np.log(data_upload.tX[:,log_columns]+1)

#generate poly features and interaction features
data_upload.tX = build_poly(data_upload.tX,2)
data_upload.tX = build_interactions(data_upload.tX)
#outlier detection
data_upload.treat_outliers(1.5,92.5)
#minmax scaling according to train set
#split due to memory constraints
data_upload.tX = (data_upload.tX - minimum)
data_upload.tX = data_upload.tX / (maximum - minimum)

#generate weights
Model = RidgeRegression(data_train)
weights = Model._run(lambda_ = best_lambda)
y_pred = predict_labels(weights, data_upload.tX)
create_csv_submission(data_upload.ids, y_pred, "../results/FinalSubmission.csv")