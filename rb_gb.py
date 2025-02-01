# import sklearn
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.impute import IterativeImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error

# feature selection
from sklearn.feature_selection import SequentialFeatureSelector


##BEFORE I RUN FOR THE FIRST TIME: INSTALL JUPYTER LAB AND PYTHON EXTENSIONS + OPEN IN JUPYTER LAB
#MIGHT HAVE TO RENAME USER
# COMMIT EVERY DAY

class rb_gb:
    def __init__(self):
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
        self.__rb_data = rb_data

        self.__features = [
            'pacr', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles', 
            'rushing_fumbles_lost', 'rushing_epa', 'rushing_2pt_conversions', 'receptions', 'targets', 
            'receiving_yards', 'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost', 
            'receiving_air_yards', 'receiving_yards_after_catch', 'receiving_first_downs', 'receiving_epa', 
            'receiving_2pt_conversions', 'racr', 'target_share', 'air_yards_share', 'wopr_x', 
            'games', 'tgt_sh', 'ay_sh', 
            'wopr_y', 'ry_sh', 'rtd_sh', 'rfd_sh', 'rtdfd_sh', 'w8dom', 
            'fantasy_points_half_ppr', 'ir_games', 'out_games', 'avg_separation', 'avg_yac_above_expectation', 
            'catch_percentage', 'rush_yards_over_expected_per_att', 'rush_pct_over_expected', 
            'carries/game', 
            'yards/carry', 'rushing_tds/game', 'receiving_tds/game', 'turnovers/game', 'adot', 'targets_game', 
            'team', 'depth_chart_position', 'height', 'weight', 'years_exp', 'age', 'depth_team', 'win_total_adj_line', 
            'gsis_id', 'college', 'pick', 'draft_value', 'forty', 'bench', 'vertical', 'broad_jump', 'cone', 
            'shuttle', 'fantasy_points_future', 'fantasy_points_ppr_future', 
            'fantasy_points_half_ppr_future', 'games_future', 'team_future', 'future_ppr/game', 'future_half_ppr/game', 
            'ppr/game', 'half_ppr/game', 'receiving_epa_team_tes', 'rushing_epa_team_qbs',  'receiving_epa_team_wrs', 'receiving_epa_team_rbs', 'rushing_epa_team_rbs'
            ]
        self.__set_features()
        self.__train_model()
        
    def __train_model(self):
        # what is friedman mse and why is it better?
        # min samples split --> less overfitting?
        # random state on?
        # n estimators is number of trees in the ensemble
        # use max leaf nodes instead of max depth??
        # validation_fraction is 0.0 because we are using our own cross validation methods
        # can use impurity decreate, max depth, or max leaf nodes. impurity decrease measures the MSE loss of a node
        # can use a combo of size and impurity based limits
        self.model = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100,
            criterion='friedman_mse', min_samples_split=None, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
            max_depth=None，min_impurity_decrease=64.0, init=None, random_state=42, max_features=None, alpha=0.9, 
            verbose=0, max_leaf_nodes=15, warm_start=False, validation_fraction=0.0, n_iter_no_change=None,
            tol=0.0001, ccp_alpha=0.0)[self._rb_train[self._features], self._rb_train['fantasy_points_half_ppr_future']]


    def test(self):
        self._rb_test['results'] = self.model(self._rb_test[features], self._rb_test['fantasy_points_half_ppr_future'])

    def __set_features(self):
        selector = SequentialFeatureSelector(self.model, *, n_features_to_select='auto', tol=None, direction='forward', scoring="mean_absolute_error", cv=5, n_jobs=-1)[rb_train]
        indices_selected_arr = selector.getSupport(indices = True).toList()
        features = []
        for i in indices_selected_arr:
            features.append(self.__features[i])
        self.__features = features





    # def shap(self):

    # def permutation_importance

        


# use sklearn gradient booster
# set up corr matrix
# todo: set on only top k
# set up permutation importance
#calculate other rbs_epa
# todo: set up shap values

## output:
# corr matrix
# outputs predictions
# outputs feature imp
# outputs MSE, MAE