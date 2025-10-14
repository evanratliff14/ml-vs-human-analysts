from abc import abstractmethod
from re import S
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
        self.label = f"future_{points_type}/game"


        fantasy_data = pd.read_parquet('data.parquet')
        self.fantasy_data = fantasy_data

        #clean data at this step - we don't want to call fantasy_df too many times to clean data
        fantasy_data.dropna(subset=[self.label], axis=0, inplace=True)
        threshold = 0.5
        fantasy_data = fantasy_data.dropna(axis=1, thresh=len(fantasy_data) * (1 - threshold))

        #refactor for categorical features
        features = [feat for feat in list(fantasy_data.columns) if pd.api.types.is_numeric_dtype(fantasy_data[feat])]
        features.append('position')
        features = features
        logging.info(f"Total numeric columns and position {features}")

        self.eval_data = fantasy_data.loc[fantasy_data['season'] == nfl.get_current_season()-1 ]
        train_test_data = fantasy_data.loc[fantasy_data['season'] < nfl.get_current_season()-1 ]

        test_size = 0.2
        logging.info(f"Train and test data has {(1-test_size)*train_test_data.shape[0]} train rows and {(test_size)*train_test_data.shape[0]} test rows")

        logging.info("Preparing data for cross validation")
        # shuffle and stratify to get results that will extrapolate to any year
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(train_test_data[features].drop(self.label, inplace=False, axis=1), train_test_data[self.label], 
            test_size=test_size)

        # prevent set_features from cheating during feature selection
        self.X_train.drop(labels=[col for col in self.X_train.columns if 'future' in col.lower()], axis=1, inplace=True)
        self.X_test.drop(labels=[col for col in self.X_test.columns if 'future' in col.lower()],axis=1, inplace=True)

        features = [col for col in features if 'future' not in col.lower()]
        self.features = features
        
        #init - these are the names seasonal uses, the vars that __str__ calls
        self.test = pd.DataFrame()
        self.train = pd.DataFrame()

        self.points_type = points_type
        self.model = None

    @abstractmethod
    def train_model(self, model, features=None):
        pass

    @abstractmethod
    def test_model(self, features=None):
        pass

    def cross_validate(self):
        try:
            self.test_mse = mean_squared_error(self.X_test['predictions'], self.y_test[self.label])
            self.test_mae = mean_absolute_error(self.X_test['predictions'],self.y_test[self.label])
            self.train_mse = mean_squared_error(self.X_train['predictions'],self.y_train[self.label])
            self.train_mae = mean_absolute_error(self.X_train['predictions'],self.y_test[self.label])
        except Exception as e:
            print("Error in cross_validate" + str(e))
    
    @abstractmethod
    def __str__(self):
       pass

    @abstractmethod
    def set_features(self):
        pass