from abc import abstractmethod
from model import Model
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector
import copy
import logging
import os

class Seasonal(Model):
    def __init__(self, points_type, position, type = 'xgb'):
        logging.info("Initializing...")
        super().__init__(points_type)
        self.type = type

        train = self.X_train
        test = self.X_test
        train[self.label] = self.y_train
        test[self.label] = self.y_test
        train, test = train.loc[train['position'] == position], test.loc[test['position'] == position]
        train.drop('position', axis=1, inplace=True)
        test.drop('position', axis=1)


        # Imputation is required for gradient boosting class
        # iterative imputer is a bayesian ridge regression model
        # random state ON to control seeding/ control variables for feature evaluation
        if type is not 'xgb_hist':
            logging.info("Imputing missing values...")
            imputer =IterativeImputer(max_iter=500, n_nearest_featurses=5, 
                initial_strategy='median', random_state=42, 
                add_indicator=False)
            imputer.set_output(transform = 'pandas')
            self.X_test = imputer.fit_transform(self.X_test)
            self.X_train = imputer.fit_transform(self.X_train)
    
        self.set_model()
        logging.info("Model is ready to train")

    def set_model(self):
        # n estimators is number of trees in the ensemble
        # use max leaf nodes instead of max depth??
        # validation_fraction is 0.0 because we are using our own cross validation methods
        # can use impurity decreate, max depth, or max leaf nodes. impurity decrease measures the MSE loss of a node
        # can use a combo of size and impurity based limits
        # n_iter_no_change, validation fraction focus on early stopping (validation fraction only used is n is integer)
        type = self.type
        if type is 'xgb_hist':
            # can insert 'college' into categorical features
            self.model = HistGradientBoostingRegressor(loss='squared_error', quantile=None,
                learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, 
                l2_regularization=0.0, max_features=1.0, max_bins=255, categorical_features='from_dtype', 
                monotonic_cst=None, interaction_cst=None, warm_start=False, early_stopping='auto', scoring='loss', 
                validation_fraction=0.1, n_iter_no_change=10, tol=0.1, verbose=0, random_state=42)
        elif type is 'xgb':
            self.model = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100,
                criterion='friedman_mse', min_samples_split=250, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_depth=None, min_impurity_decrease=64.0, init=None, random_state=42, max_features=None, alpha=0.9, 
                verbose=0, max_leaf_nodes=15, warm_start=False, validation_fraction=0.1, n_iter_no_change=None,
                tol=0.1, ccp_alpha=0.0)

    # may pass anything that uses model interface, including sequential feature selector
    def train_model(self, model):
        features = self.features
        if 'position' in features:
            features.remove('position')
        model.fit(self.X_train[features], self.X_train[self.label])
        self.model = model


    def test(self):
        # staged predict: returns each stage of the prediction of the test set, vs just the final
        self.X_test['predictions'] = self.model.predict(self.X_test[self.features])
        self.X_train['predictions'] = self.model.predict(self.X_train[self.features])
                
        stage_errors = []
        for i, y_pred_stage in enumerate(self.model.staged_predict(self.X_test)):
            mse = mean_squared_error(self.X_test[self.label], y_pred_stage)
            stage_errors.append(mse)
            logging.info(f"Iteration {i+1}: MSE = {mse}")

    def set_features(self):
        logging.info("Setting features...")
        if not os.path.isfile('features.txt'):
            # preserve initial model type
            model = copy.deepcopy(self.model)
            selector = SequentialFeatureSelector(model, n_features_to_select='auto', tol=0.4, direction='forward',
                                                scoring="neg_median_absolute_error", cv=5, n_jobs=-1)
            self.train_model(selector)
            indices_selected_arr = selector.get_support(indices = True).tolist()
            features = []
            with open('features.txt', 'w') as file:
                for i in indices_selected_arr:
                    features.append(self.features[i])
                    file.write(f"{self.features[i]}\n")
            self.features = features
        else:
            features = [line.strip() for line in open("features.txt")]
            self.features = features
        logging.info("Features set.")

    def __str__(self):
        return super().__str__()
        
    def cross_validate(self):
        return super().cross_validate()
        

        
