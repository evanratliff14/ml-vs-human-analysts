from abc import abstractmethod
from model import Model
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

class Xgb(Model):
    def __init__(self, points_type, hist = True):
        super().__init__(points_type = points_type)
        self.hist = hist

    def set_model(self):
        # n estimators is number of trees in the ensemble
        # use max leaf nodes instead of max depth??
        # validation_fraction is 0.0 because we are using our own cross validation methods
        # can use impurity decreate, max depth, or max leaf nodes. impurity decrease measures the MSE loss of a node
        # can use a combo of size and impurity based limits
        # n_iter_no_change, validation fraction focus on early stopping (validation fraction only used is n is integer)
        if self.hist:
            # can insert 'college' into categorical features
            self.model = HistGradientBoostingRegressor(loss='squared_error', quantile=None,
                learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, 
                l2_regularization=0.0, max_features=1.0, max_bins=255, categorical_features='from_dtype', 
                monotonic_cst=None, interaction_cst=None, warm_start=False, early_stopping='auto', scoring='loss', 
                validation_fraction=0.1, n_iter_no_change=10, tol=0.1, verbose=0, random_state=42)
        elif not self.hist:
            self.model = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100,
                criterion='friedman_mse', min_samples_split=250, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_depth=None, min_impurity_decrease=64.0, init=None, random_state=42, max_features=None, alpha=0.9, 
                verbose=0, max_leaf_nodes=15, warm_start=False, validation_fraction=0.1, n_iter_no_change=None,
                tol=0.1, ccp_alpha=0.0)

        @abstractmethod
        def set_features():
            pass
        
        @abstractmethod
        def train():
            pass
        
