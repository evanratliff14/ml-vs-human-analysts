import pickle
from abc import sABC, abstractmethod
import pandas as pd
import pickle
from model import Model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector
import copy

class Model:
    def __init__(self, points_type):
        self.train_years = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 1010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
        self.test_years = [2021, 2022]
        self.fantasy_data = pd.read_pickle("fantasy_data.pkl")
        self.features = None
        self.train_data = None
        self.test_data = None
        self.full_data = None
        self.points_type = points_type

    @abstractmethod
    def train(self):
        pass

    def cross_validate(self):
        try:
            self.test_mse = mean_squared_error(self.test_data[self.points_type],self.test_data['predictions'])
            self.test_mae = mean_absolute_error(self.test_data[self.points_type],self.test_data['predictions'])
            self.train_mse = mean_squared_error(self.train_data[self.points_type],self.train_data['predictions'])
            self.train_mae = mean_absolute_error(self.train_data[self.points_type],self.train_data['predictions'])
        except:
            print("Error in cross_validate")
    
    @override
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