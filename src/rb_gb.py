
import pandas as pd
from xgb import Xgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector
import copy
import logging
import os

class Rb_Xgb(Xgb):
    def __init__(self, points_type, hist = True):
        logging.info("Initializing...")
        super().__init__(points_type, hist=hist)

        rb_train = self.X_train
        rb_test = self.X_test
        rb_train[self.label] = self.y_train
        rb_test[self.label] = self.y_test
        rb_train= rb_train.loc[rb_train['position'] == 'RB']
        rb_test= rb_test.loc[rb_test['position'] == 'RB']
        rb_train, rb_test = rb_train.drop('position', axis=1, inplace=True), rb_test.drop('position', axis=1, inplace=True)

        # Imputation is required for gradient boosting class
        # iterative imputer is a bayesian ridge regression model
        # random state ON to control seeding/ control variables for feature evaluation
        if not hist:
            logging.info("Imputing missing values...")
            imputer =IterativeImputer(max_iter=500, n_nearest_features=5, 
                initial_strategy='median', random_state=42, 
                add_indicator=False)
            imputer.set_output(transform = 'pandas')
            self.X_test = imputer.fit_transform(self.X_test)
            self.X_train = imputer.fit_transform(self.X_train)
       
        self.set_model()
        logging.info("Model is ready to train")
        
    # may pass anything that uses model interface, including sequential feature selector
    def train_model(self, model):
        features = self.features
        features.remove('position')
        self.model.fit(self.X_train[features], self.X_train[self.label])


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
            # make feature evaluation robust to outliers
            # fantasy point evaluation is based on absolute points - risk is not part of strategy
            model = copy.deepcopy(self.model)
            selector = SequentialFeatureSelector(model, n_features_to_select='auto', tol=0.4, direction='forward',
                                                scoring="neg_median_squared_error", cv=5, n_jobs=-1)
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
    def set_model(self):
        super().set_model()
        
    