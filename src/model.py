from abc import abstractmethod
import pandas as pd
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
import copy
import nflreadpy as nfl
import pyarrow

class Model:
    def __init__(self, points_type):
        fantasy_data = pd.read_parquet('data.parquet', engine = "pyarrow")
        self.features = (list(fantasy_data.columns))
        self.label = f"future_{points_type}/game"
        self.eval_data = fantasy_data.loc[fantasy_data['season' == nfl.get_current_season ]]
        train_test_data = fantasy_data.loc[fantasy_data['season'] <nfl.get_current_season ]

        # shuffle and stratify to get results that will extrapolate to any year
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(train_test_data[self.features], train_test_data[self.label], 
            test_size=0.2, shuffle = True, stratify=True)
        self.points_type = points_type
        self.model = None

    @abstractmethod
    def train(self):
        pass

    def cross_validate(self):
        try:
            self.test_mse = mean_squared_error(self.X_test[self.test_data], self.y_test[self.label])
            self.test_mae = mean_absolute_error(self.X_test[self.points_type],self.y_test[self.label])
            self.train_mse = mean_squared_error(self.X_train[self.points_type],self.y_train[self.label])
            self.train_mae = mean_absolute_error(self.X_train[self.points_type],self.y_test[self.label])
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