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
        """function that loads the dataset in .csv format from DATAPATH
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
        """fills the values that are -999. with np.NaN
        """
        for feature_name, index in self.feature_names.items():
            self.tX[:,index][self.tX[:,index] == -999.] = np.NaN
        
        #we have to check if we only need to replace -999.
        """if feature_name not in ["PRI_jet_all_pt","DER_lep_eta_centrality"]:
                self.tX[:,index][self.tX[:,index] == -999.] = np.NaN
            else:
                self.tX[:,index][self.tX[:,index] == -999.] = np.NaN
                self.tX[:,index][self.tX[:,index] == 0] = np.NaN"""
    
    def replace_with_mean(self):
        """replaces np.NaN values with collum (feature) mean 
        """
        #TODO: Replace with mode, median, binary, hash ?
        
        #self._fill_with_NaN() #make auto_later
        #also handles all NaN collums -> replaces with 0
        self.tX = np.where(np.isnan(self.tX), np.ma.array(self.tX, mask=np.isnan(self.tX)).mean(axis=0), self.tX)
            
    def normalize(self):
        """standardizes data
        """
        self.tX -= np.nanmean(self.tX,axis=0)
        self.tX /= np.nanstd(self.tX,axis=0)
        
    
    def transform_to_pca(self,max_var=0.95,max_eigenvalue=None):
        """pca transformation
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
        """building polynomial features from array
        """
        #self.replace_with_mean() make auto_later
        data_polys = []
        
        for feature_index, degree in enumerate(degrees):
            if degree == 0:
                pass
            else:
                data_polys.append(build_poly(self.tX[:,feature_index],degree))
        self.tX = np.concatenate(data_polys,axis=1)