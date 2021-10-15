from proj1_helpers import *
from build_polynomial import build_poly

class Data_Cleaner:
    
    def _load_data(self, DATAPATH):
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
        for feature_name, index in self.feature_names.items():
            self.tX[:,index][self.tX[:,index] == -999.] = np.NaN
        
        #we have to check if we only need to replace -999.
        """if feature_name not in ["PRI_jet_all_pt","DER_lep_eta_centrality"]:
                self.tX[:,index][self.tX[:,index] == -999.] = np.NaN
            else:
                self.tX[:,index][self.tX[:,index] == -999.] = np.NaN
                self.tX[:,index][self.tX[:,index] == 0] = np.NaN"""
    
    def replace_with_mean(self):
        #self._fill_with_NaN() #make auto_later
        #also handles all NaN collums -> replaces with 0
        self.tX = np.where(np.isnan(self.tX), np.ma.array(self.tX, mask=np.isnan(self.tX)).mean(axis=0), self.tX)
            
    def normalize(self):
        self.tX -= np.nanmean(self.tX,axis=0)
        self.tX /= np.nanstd(self.tX,axis=0)
        
    
    def easy_clean(self):
        #loaded -> replace_with_mean -> normalize
        self.replace_with_mean()
        self.normalize()
        
    
    def transform_to_pca(self,max_var=0.95,max_eigenvalue=None):
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
    
    def make_features(self,max_percentage=0.99,degree=1):
        #self._fill_with_NaN() make auto later
        poly_features = []
        for feature_name, index in self.feature_names.items():
            percentage = np.count_nonzero(~np.isnan(self.tX[:,index]))/len(self.tX[:,index])
            print(percentage)
            if percentage < 0.99:
                indices = np.isnan(self.tX[:,index])
                self.tX[:,index][indices] = 0.
                self.tX[:,index][~indices] = 1.
            else:
                if index != self.feature_names["PRI_jet_num"]:
                    #check this!!
                    
                    poly_features.append(build_poly(self.tX[:,index],degree)[:,1:])
        #standardize before or after building polynomial features?
        #-> https://datascience.stackexchange.com/questions/9020/do-i-have-to-standardize-my-new-polynomial-features
        #- do after!! otherwise features are an order of magnitude smaller
        
        #standardize binary values?
        #rather not, does not make sense...
        #https://stats.stackexchange.com/questions/59392/should-you-ever-standardise-binary-variables
        
        polys = np.concatenate(poly_features,axis=1)
        self.tX  = np.concatenate([self.tX, polys], axis=1)
    
    def build_polynomials_from_degree_array(self,degrees):
        #self.replace_with_mean() make auto_later
        data_polys = []
        
        for feature_index, degree in enumerate(degrees):
            if degree == 0:
                pass
            else:
                data_polys.append(build_poly(self.tX[:,feature_index],degree))
        self.tX = np.concatenate(data_polys,axis=1)