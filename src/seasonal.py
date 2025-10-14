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
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Seasonal(Model):
    def __init__(self, points_type, position, type = 'xgb'):
        logging.info("Initializing...")
        super().__init__(points_type)
        self.type = type
        self.position = position

        train_test_data = self.train_test_data
        eval = self.eval
        test_size = 0.2
        features = self.features
        

        train_test_data, eval= train_test_data.loc[train_test_data['position'] == position], eval.loc[eval['position'] == position]
        # train.drop('position', axis=1, inplace=True)
        # test.drop('position', axis=1, inplace=True)


        X_train, X_test, y_train, y_test = train_test_split(train_test_data[features], train_test_data[self.label], 
            test_size=test_size, random_state = 42)
        
        logging.info(f"Train and test data has {(1-test_size)*train_test_data.shape[0]} train rows and {(test_size)*train_test_data.shape[0]} test rows")

        X_eval = eval[features]
        y_eval = eval[self.label]

        # create new dfs for rejoining
        train = pd.DataFrame()
        test = pd.DataFrame()
        eval = pd.DataFrame()
        
        # Imputation is required for gradient boosting class
        # iterative imputer is a bayesian ridge regression model
        # random state ON to control seeding/ control variables for feature evaluation
        if type =='xgb_hist':
            
            train[list(X_train.columns)] = X_train[list(X_train.columns)]
            test[list(X_test.columns)] = X_test[list(X_test.columns)]
            eval[list(X_eval.columns)] = X_eval[list(X_eval.columns)]
            train[self.label] = y_train
            test[self.label] = y_test
            eval[self.label] = y_eval
        # elif os.path.isfile('test.parquet') and os.path.isfile('train.parquet') and os.path.isfile('eval.parquet'):
        #     train = pd.read_parquet('train.parquet').fillna(0, inplace=False)
        #     test = pd.read_parquet('test.parquet').fillna(0, inplace=False)
        #     eval = pd.read_parquet('eval.parquet').fillna(0, inplace=False)

        #     train[self.label] = self.y_train
        #     test[self.label]  = self.y_test
        #     eval[self.label] = self.eval[self.label]
        #     self.eval = eval
        #     self.train = train
        #     self.test = test
        # else:
        else:

            logging.info("Imputing missing values...")
            imputer =IterativeImputer(max_iter=500, n_nearest_features=5, 
                initial_strategy='median', random_state=42, 
                add_indicator=False)
            imputer.set_output(transform = 'pandas')

            numeric = [feat for feat in self.features if feat not in self.categorical_identifiers and feat !=self.label]
            
            # Fit on train numeric only
            imputer.fit(X_train[numeric])

            # Transform both (use .loc to avoid accidental reindexing)
            X_train.loc[:, numeric] = imputer.transform(X_train[numeric])
            X_test.loc[:, numeric]  = imputer.transform(X_test[numeric])
            X_eval.loc[:, numeric] = imputer.transform(X_eval[numeric])

            # rejoin
            train[list(X_train.columns)] = X_train[list(X_train.columns)]
            test[list(X_test.columns)] = X_test[list(X_test.columns)]
            eval[list(X_eval.columns)] = X_eval[list(X_eval.columns)]
            train[self.label] = y_train
            test[self.label] = y_test
            eval[self.label] = y_eval
            


            # Save parquets without the pandas index column
            train.to_parquet('train.parquet', index=False )
            test.to_parquet('test.parquet', index=False)
            eval.to_parquet('eval.parquet', index=False)
            
    
        self.train = train
        self.test = test
        self.eval = eval
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
        if type =='xgb_hist':
            # can insert 'college' into categorical features
            self.model = HistGradientBoostingRegressor(loss='squared_error', quantile=None,
                learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, 
                l2_regularization=0.0, max_features=10, max_bins=255, categorical_features='from_dtype', 
                monotonic_cst=None, interaction_cst=None, warm_start=False, early_stopping='auto', scoring='loss', 
                validation_fraction=0.1, n_iter_no_change=10, tol=0.1, verbose=0, random_state=42)
        elif type == 'xgb':
            self.model = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=350,
                criterion='friedman_mse', min_samples_split=250, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_depth=5, min_impurity_decrease=64.0, init=None, random_state=42, max_features=20, alpha=0.9, 
                verbose=0, max_leaf_nodes=15, warm_start=False, validation_fraction=0.1, n_iter_no_change=None,
                tol=0.1, ccp_alpha=0.0)

    # may pass anything that uses model interface, including sequential feature selector
    def train_model(self, model, features = None):
        logging.info("Training model")
        if features is None:
            features = self.features
        
        # remove categorical identifiers
        features = [f for f in features if f not in self.categorical_identifiers]

        model.fit(self.train[features], self.train[self.label])
        joblib.dump(model, f"{self.position}_{self.type}.joblib")
        self.model = model
        logging.info("Training complete")


    def test_model(self, features=None):
        logging.info("Testing model...")
        if features is None:
            features = self.features

        features = [f for f in features if f not in self.categorical_identifiers]

        # staged predict: returns each stage of the prediction of the test set, vs just the final
        self.train['predictions'] = self.model.predict(self.train[features])
        self.test['predictions'] = self.model.predict(self.test[features])
        self.eval['predictions'] = self.model.predict(self.eval[features])
        # self.eval['predictions'] = self.model.predict(self.eval[features])
                
        stage_errors = []
        for i, y_pred_stage in enumerate(self.model.staged_predict(self.test[features])):
            mse = mean_squared_error(self.test[self.label], y_pred_stage)
            stage_errors.append(mse)
            logging.info(f"Iteration {i+1}: MSE = {mse}")
        

    def set_features(self):
        logging.info("Setting features...")
        if not os.path.isfile(f'{self.position}_features.txt'):
            return Exception(f"{self.position}_features.txt does not exist in filepath!")
        else:
            features = [line.strip() for line in open(f"{self.position}_features.txt")]
            self.features = features
        logging.info("Features set.")

    def __str__(self):
        return super().__str__()
        
    def cross_validate(self):
        try:
            self.test_mse = mean_squared_error(self.test['predictions'], self.test[self.label])
            self.test_mae = mean_absolute_error(self.test['predictions'],self.test[self.label])
            self.train_mse = mean_squared_error(self.train['predictions'],self.train[self.label])
            self.train_mae = mean_absolute_error(self.train['predictions'],self.train[self.label])
            self.eval_mse = mean_squared_error(self.eval['predictions'],self.eval[self.label])
            self.eval_mae = mean_absolute_error(self.eval['predictions'],self.eval[self.label])
        except Exception as e:
            print("Error in cross_validate" + str(e))

    def corr(self):
        fantasy_data = self.fantasy_data
        features = [feat for feat in self.features if feat not in self.categorical_identifiers]

        plt.figure(figsize=(20, 10))  # width, height in inches

        sns.heatmap(fantasy_data[features].corr(numeric_only=True).abs(), cmap='coolwarm')
        plt.savefig('corr_plot.png')
        plt.show()

    def __str__(self):
        model_string = "Features: \n"
        model_string = model_string + str(self.features) + "\n"
        # intentional side effect
        logging.info(self.test[['player_name', 'predictions', 'season']])
        display = pd.concat(objs = [self.test, self.eval])
        display = display[['player_name', 'predictions', 'season']].sort_values(
            by='predictions',
            ascending = True,
            inplace=False  # descending predictions, ascending season
        )       
        display = display.sort_values(
            by='season',
            ascending = False,
            inplace=False  # descending predictions, ascending season
        )  
        
        display.to_csv('predictions.csv')
        self.cross_validate()
        model_string += "Test MSE: " + str(self.test_mse) + "\n"
        model_string += "Test MAE: " + str(self.test_mae) + "\n"
        model_string += "Train MSE: " + str(self.train_mse) + "\n"
        model_string += "Test MSE: " + str(self.train_mae) + "\n"
        model_string += "Eval MSE: " + str(self.eval_mse) + "\n"
        model_string += "Eval MSE: " + str(self.eval_mae) + "\n"


        return model_string

        
