# import sklearn
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.impute import IterativeImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error

##BEFORE I RUN FOR THE FIRST TIME: INSTALL JUPYTER LAB AND PYTHON EXTENSIONS + OPEN IN JUPYTER LAB
#MIGHT HAVE TO RENAME USER

class rb_gb:
    def __init__():
        self.__train_years = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 1010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
        self.__test_years = [2021, 2022]
        with open('fantasy_data.pkl', 'rb') as file:
            fantasy_data = pickle.load(file)

        rb_data = fantasy_data.loc[fantasy_data['position'] == 'RB']

        # Imputation is required for gradient boosting class
        # iterative imputer is a bayesian ridge regression model
        # random state ON to control seeding/ control variables for feature evaluation
        rb_data =IterativeImputer(max_iter=100, n_nearest_features=5, 
            initial_strategy='median', random_state=42, 
            add_indicator=False)

        self.__rb_test = rb_data.loc[rb_data['season'].isin[self.__train_years]]
        self.__rb_train = rb_data.loc[rb_data['season'].isin[self.__test_years]]

        self.__features = [
            'completions', 'attempts', 'passing_yards', 
            'passing_tds', 'interceptions', 'sacks', 'sack_yards', 'sack_fumbles', 'sack_fumbles_lost', 
            'passing_air_yards', 'passing_yards_after_catch', 'passing_epa', 'passing_2pt_conversions', 
            'pacr', 'dakota', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles', 
            'rushing_fumbles_lost', 'rushing_epa', 'rushing_2pt_conversions', 'receptions', 'targets', 
            'receiving_yards', 'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost', 
            'receiving_air_yards', 'receiving_yards_after_catch', 'receiving_first_downs', 'receiving_epa', 
            'receiving_2pt_conversions', 'racr', 'target_share', 'air_yards_share', 'wopr_x', 
            'special_teams_tds', 'fantasy_points', 'fantasy_points_ppr', 'games', 'tgt_sh', 'ay_sh', 
            'wopr_y', 'ry_sh', 'rtd_sh', 'rfd_sh', 'rtdfd_sh', 'w8dom', 'player_name', 'position', 
            'fantasy_points_half_ppr', 'ir_games', 'out_games', 'avg_separation', 'avg_yac_above_expectation', 
            'catch_percentage', 'rush_yards_over_expected_per_att', 'rush_pct_over_expected', 
            'completion_percentage_above_expectation', 'pass_tds/game', 'pass_air_yards/game', 'carries/game
            ]

        
    def __train_model():
        rb_gb_model = GradientBoostingRegressor()


        


# use sklearn gradient booster
# set up corr matrix
# todo: set on only top k
# set up permutation importance
# todo: set up shap values

## output:
# corr matrix
# outputs predictions
# outputs feature imp
# outputs MSE, MAE