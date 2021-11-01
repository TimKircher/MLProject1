# MLProject1

Repository for the machine learning project 1 of the CS-433 course at EPFL
Authors : Matthias Kellner, Tim Kircher, Nearchos Potamitis

Our code is divided in 5 folders : 

1. data : Contains the data provided by the Higgs challenge (train and test set) to train and test our models.
2. Documents : Contains the project description given by the course and the Higgs challenge documentation
3. Notebooks : Contains
- data_exploration.ipynb : Notebook used for the data exploration associated with the training provided by the Higgs challenge
- implementations_pipeline.ipynb : Notebook used to compare the accuracy of our models using the 6 basic method implementations with different 
                                                 feature engineering techniques
4. plots : Contains different kinds of plots describing the provided training dataset 
5. scripts : Contains 
- implementations.py : The 6 basic method implementations required
- compute.py : Contains functions associated with computing a value or a vector (Error vector, MSE, Gradient etc.)
- proj1_helpers.py : Contains functions provided by the course labs aimed to help our project
- data_cleaner.py : Contains a class that is responsible for handling the data ( Cleaning, Feature Engineering, Splitting the data, etc.)
- run.py : The run.py file required in the submission guidlines, reproducing our best submission
- k_fold_validate_matthias.py : Builds the random indices for cross validation
- linear_model_base.py : Abstract base class and child class for least squares and ridge regression. The class is used for the final submission and is intended to be easily extendable to new linear and nonlinear models
- least_squares_matthias.py : An implementation of the closed-form least squares method
