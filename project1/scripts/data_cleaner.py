from proj1_helpers import *
from build_polynomial import build_poly

class Data_Cleaner:
    """ Class for data cleaning and feature processing of the higgs dataset
    
    Data_cleaner(FILEPATH)
    
    Returns the datacleaner object with data, labels and ids.
    
    Attributes
    ----------
    data          : None
                    Not used
    feature_names : dict
                    dictionary {feature_name: feature collumn index} 
                    containing the feature names 
    y             : Numpy array shape (N_samples,)
                    containing the datasets predictions, if given
    tX            : Numpy array shape (N_samples, D_features)
                    containing the datasets data
    ids           : Numpy array shape (N_samples,)
                    containing the datasets ids       
    DATAPATH      : string
                    The path where the dataset.csv file can be found
    
    Methods
    -------
    _load_data(DATAPATH)
        loads the dataset if the DATAPATH was not given during initialization of object.
        overwrites current ids, tX and y
    
    _fill_with_NAN()
        fills data values of -999. with np.NaN
    
    replace_with_mean()
        replaces np.NaN values with features nanmean
    
    normalize()
        standardizes! data by substracting with features mean and dividing by its standard deviation
        can not handle NaNs
        
    standadize()
        standardizes! data to range 0 to 1
        
    transform_to_pca(max_var,max_eigenvalue)
        performs principal component analysis using the covariance matrix.
        builds projection matrix of the principal components that have a combined explained variance 
        ratio larger than max_var.
        the projection matrix can also restricted to max_eigenvalues principal components.
    
        
    build_polynomials_from_degree_array(degrees)
        basis expansion of the features from an array that contains possible optimal degrees of expansion
        degrees must be a numpy array of shape (D_features,)
        e.g. degrees is a numpy array ([0,10,0,0 .... 3,0]
        meaning that the second feature will be expanded in a polynomial basis of 1st to 10th order
        a degree of 0 of a feature means, that the feature collum will be deleted

    """
    
    def _load_data(self, DATAPATH):
        """
        Function that loads the dataset in .csv format from DATAPATH
        """
        self.y, self.tX, self.ids = load_csv_data(DATAPATH)
        with open(DATAPATH) as fileobj:
            feature_names = fileobj.readline().rstrip("\n")
            feature_names = feature_names.split(sep=",")[2:] # remove ID and prediction
            #for easier access to variables make index dict
            self.feature_names = {x:v for v,x in enumerate(feature_names)}
            
    def __init__(self, DATAPATH=None):

        self.data = None
        self.feature_names = None
        self.y = None
        self.tX = None
        self.ids = None
        self.DATAPATH = DATAPATH
        
        if self.DATAPATH:
            self._load_data(self.DATAPATH)
            
            
            
    def _fill_with_NaN(self):
        """
        Fills the values that are -999. with np.NaN
        """
        for feature_name, index in self.feature_names.items():
            self.tX[:,index][self.tX[:,index] == -999.] = np.NaN
    
    def replace_with_mean(self):
        """
        Replaces np.NaN values with collum (feature) mean 
        """
        #TODO: Replace with mode, median, binary, hash ?
        
        #self._fill_with_NaN() #make auto_later
        #also handles all NaN collums -> replaces with 0
        self.tX = np.where(np.isnan(self.tX), np.ma.array(self.tX, mask=np.isnan(self.tX)).mean(axis=0), self.tX)
        
    def fix_mass_MMC(self, add_impute_array = True):
        """
        Replaces np.NaN values of first column (feature) with median value of the rest of them.
        Additionaly, it creates another column whose value is 0 if the variable of the first
        column is impute and 1 if it's not.
        """
        mass_MMC = self.tX[:,0]
        tx = self.tX
        
        median = np.nanmedian(mass_MMC)        
        imputed_array = np.ones(self.tX.shape[0])
        imputed_array[np.isnan(mass_MMC)] = 0
        
        mass_MMC[np.isnan(mass_MMC)] = median
        if add_impute_array:
            self.tX = np.c_[self.tX,imputed_array]
                  
    def replace_with_zero(self):
        """
        Replaces np.NaN values with 0 
        """
        self.tX[np.isnan(self.tX)] = 0
        
    def replace_with_one(self):
        """
        Replaces np.NaN values with 1 
        """
        self.tX[np.isnan(self.tX)] = 1
    
    def normalize(self):
        """
        Standardizes data
        """
        mean = np.nanmean(self.tX,axis=0)
        std = np.nanstd(self.tX,axis=0)
        
        self.tX = (self.tX-mean)/std
    
    def standardize(self):
        """
        Standardizes data
        """
        minimum = np.min(self.tX,axis=0)
        maximum = np.max(self.tX,axis=0)
        
        self.tX = (self.tX-minimum)/(maximum-minimum)
        
    def split_data(self,percent):
        """
        Given a percentage it splits the dataset at that percentage
        """
        rows = len(self.y)
        split_index = int(np.floor(percent / 100 * rows))
        training_x = self.tX[:split_index,:]
        testing_x = self.tX[split_index:,:]
        training_y = self.y[:split_index]
        testing_y = self.y[split_index:]   
        return training_x, testing_x, training_y, testing_y
    
    def treat_outliers (self, bot_percentage_bound, top_percentage_bound):
        """
        Given a top and bottom percentage bound it sets all outliers over or lower
        of the corresponding bound to the value of the bound
        """
        for iter_ in range(self.tX.shape[1]):
            feature = self.tX[:,iter_]
            bot_bound = np.nanpercentile(feature, bot_percentage_bound)
            top_bound = np.nanpercentile(feature, top_percentage_bound)
            bot_criterion = np.where(feature < bot_bound)
            top_criterion = np.where(feature > top_bound)
            feature[bot_criterion] = bot_bound
            feature[top_criterion] = top_bound
     
    def getMinMax(self):
        """
        Get min and max to scale testset
        """
        return np.min(self.tX, axis=0), np.max(self.tX, axis=0)
    
    def getMeanStd(self):
        """
        Standardizes data
        """
        return np.mean(self.tX, axis=0), np.std(self.tX, axis=0)
    
    def build_interactions(self):
        x = self.tX
        x_out = np.array(x)
        for i in range(int(x.shape[1])):
            x_i = x[:,0]
            x = np.delete(x, 0, 1)
            x_interact = (x_i*x.T).T
            x_out = np.hstack([x_out,x_interact])
        self.tX = x_out

    def build_polynomial(self,degree, add_degree_zero=False):
        """
        polynomial basis functions for input data x, for j=0 up to j=degree.
        """
        x = self.tX
        if add_degree_zero:
            xN = np.hstack([np.ones([x.shape[0],1]),x])
        else:
            xN = x
        if degree>0:
            for i in range(degree-1):
                xN = np.hstack([xN, x**(i+2)])
        self.tX = np.array(xN)

    
    
    
    
    
    
    
    def transform_to_pca(self,max_var=0.95,max_eigenvalue=None):
        """
        PCA transformation
        """
        #self.normalize() make auto_later
        cov_mat = np.cov(self.tX.T) #calculate covariance matrix
        eigval_pca, eigvec_pca = np.linalg.eig(cov_mat) #can not be orderd, but they are here
        total_eigval = np.sum(eigval_pca)
        percentages = [eigval/total_eigval for eigval in eigval_pca]
        percentages_cumulative = np.cumsum(percentages)
        greater_var = np.argmax(percentages_cumulative > max_var)
        
        if max_var and not max_eigenvalue:
            projection_mat = eigvec_pca[:,:greater_var]
            print(greater_var)
        else:
            projection_mat = eigvec_pca[:,:max_eigenvalue]
        
        self.tX = self.tX @  projection_mat
    
        #standardize before or after building polynomial features?
        #-> https://datascience.stackexchange.com/questions/9020/do-i-have-to-standardize-my-new-polynomial-features
        #- do after!! otherwise features are an order of magnitude smaller
        
        #standardize binary values?
        #rather not, does not make sense...
        #https://stats.stackexchange.com/questions/59392/should-you-ever-standardise-binary-variables
    
    def build_polynomials_from_degree_array(self,degrees):
        """
        Building polynomial features from array
        """
        #self.replace_with_mean() make auto_later
        data_polys = []
        
        for feature_index, degree in enumerate(degrees):
            if degree == 0:
                pass
            else:
                data_polys.append(build_poly(self.tX[:,feature_index],degree))
        self.tX = np.concatenate(data_polys,axis=1)