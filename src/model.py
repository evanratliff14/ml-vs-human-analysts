from abc import abstractmethod
import pandas as pd
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import copy
import nflreadpy as nfl
import pyarrow
import logging

class Model:
    def __init__(self, points_type, **kwargs):
        logging.info("Reading data from parquet")
        fantasy_data = pd.read_parquet('data.parquet', engine = "pyarrow")
        #refactor for categorical features
        features = [feat for feat in list(fantasy_data.columns) if pd.api.types.is_numeric_dtype(fantasy_data[feat])]
        self.features = features
        logging.info(features)
        self.label = f"future_{points_type}/game"
        self.eval_data = fantasy_data.loc[fantasy_data['season'] == nfl.get_current_season()-1 ]
        train_test_data = fantasy_data.loc[fantasy_data['season'] < nfl.get_current_season()-1 ]

        logging.info("Preparing data for cross validation")
        # shuffle and stratify to get results that will extrapolate to any year
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(train_test_data[features].drop(self.label, inplace=False, axis=1), train_test_data[self.label], 
            test_size=0.2)
        self.points_type = points_type
        self.model = None

    @abstractmethod
    def train(self):
        pass

    def cross_validate(self):
        try:
            self.test_mse = mean_squared_error(self.X_test['predictions'], self.y_test[self.label])
            self.test_mae = mean_absolute_error(self.X_test['predictions'],self.y_test[self.label])
            self.train_mse = mean_squared_error(self.X_train['predictions'],self.y_train[self.label])
            self.train_mae = mean_absolute_error(self.X_train['predictions'],self.y_test[self.label])
        except Exception as e:
            print("Error in cross_validate" + str(e))
    
    def __str__(self):
        model_string = "Features: \n"
        model_string = model_string + str(self.features) + "\n"
        # intentional side effect
        print(self.test_data[['player_name', 'predictions',self.points_type, 'season']])
        self.test_data[['player_name', 'predictions', self.points_type, 'season']].to_csv('predictions.csv')
        self.cross_validate()
        model_string += "Test MSE: " + str(self.test_mse) + "\n"
        model_string += "Test MAE: " + str(self.test_mae) + "\n"
        model_string += "Train MSE: " + str(self.train_mse) + "\n"
        model_string += "Test MSE: " + str(self.train_mae) + "\n"
        return model_string

    @abstractmethod
    def set_features(self):
        pass