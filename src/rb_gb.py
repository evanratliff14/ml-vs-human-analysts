# COMMIT EVERY DAY
#add log statements

import pandas as pd
import pickle
from model import Model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector
import copy

class Rb_XGBoost(Model):
    def __init__(self, hist = True):
        print("Initializing...")
        super().__init__()

        self.__features = [
            'pacr', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles', 
            'rushing_fumbles_lost', 'rushing_epa', 'receptions', 'targets', 
            'receiving_yards', 'receiving_tds', 'receiving_fumbles', 
            'receiving_air_yards', 'receiving_yards_after_catch', 'receiving_first_downs', 'receiving_epa', 
            'racr', 'target_share', 'air_yards_share', 'wopr_x', 
            'tgt_sh', 'ay_sh', 
            'wopr_y', 'ry_sh', 'rtd_sh', 'rfd_sh', 'rtdfd_sh', 'w8dom', 
            'fantasy_points_half_ppr', 'ir_games', 'out_games',
            'carries/game', 
            'yards/carry', 'rushing_tds/game', 'receiving_tds/game', 'turnovers/game', 'adot', 'targets_game', 
            'height', 'weight', 'years_exp', 'age', 'depth_team', 'win_total_adj_line', 
            'draft_value', 'forty', 'bench', 'vertical', 'broad_jump', 'cone', 
            'shuttle',
            'ppr/game', 'half_ppr/game', 'receiving_epa_team_tes', 'rushing_epa_team_qbs',  
            'receiving_epa_team_wrs', 'receiving_epa_team_rbs', 'rushing_epa_team_rbs', 'dakota_team_qbs'
            ]

        rb_data = self.fantasy_data.loc[self.fantasy_data['position'] == 'RB']
        rb_data = rb_data.dropna(subset = ['fantasy_points_half_ppr_future'])

        # Imputation is required for gradient boosting class
        # iterative imputer is a bayesian ridge regression model
        # random state ON to control seeding/ control variables for feature evaluation
        if hist == False:
            print("Imputing missing values...")
            imputer =IterativeImputer(max_iter=500, n_nearest_features=5, 
                initial_strategy='median', random_state=42, 
                add_indicator=False)
            imputer.set_output(transform = 'pandas')
            rb_data[self.features] = imputer.fit_transform(rb_data[self.features])
       

        print("Preparing cross-validation sets")
        self.test_data = rb_data.loc[rb_data['season'].isin(self.train_years)]
        self.train_data = rb_data.loc[rb_data['season'].isin(self.test_years)]
        self.full_data = rb_data
        print("Model is ready to train")
        
    def set_model(self):
        # what is friedman mse and why is it better?
        # min samples split --> less overfitting?
        # random state on?
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
            
    def train_model(self):
        self.model.fit(self.__rb_train[self.__features], self.__rb_train['fantasy_points_half_ppr_future'])


    def test(self):
        # staged predict: returns each stage of the prediction of the test set, vs just the final
        self.test_data['predictions'] = self.model.predict(self.test_data[self.features])
        self.train_data['predictions'] = self.model.predict(self.train_data[self.features])
                
        # stage_errors = []
        # for i, y_pred_stage in enumerate(self.model.staged_predict(self.__rb_test)):
        #     mse = mean_squared_error(self.__rb_test['fantasy_points_half_ppr'], y_pred_stage)
        #     stage_errors.append(mse)
        #     print(f"Iteration {i+1}: MSE = {mse}")

    def set_features(self):
        print("Setting features...")
        # make feature evaluation robust to outliers
        # fantasy point evaluation is based on absolute points - risk is not part of strategy
        model = copy.deepcopy(self.model)
        selector = SequentialFeatureSelector(model, n_features_to_select='auto', tol=0.4, direction='forward',
                                             scoring="neg_median_absolute_error", cv=5, n_jobs=-1)
        selector.fit(self.__rb_train[self.__features], self.__rb_train['fantasy_points_half_ppr_future'] )
        indices_selected_arr = selector.get_support(indices = True).tolist()
        features = []
        for i in indices_selected_arr:
            features.append(self.__features[i])
        self.__features = features
        print("Features set.")

    def __str__(self):
        return super().__str__()
        
    def cross_validate(self):
        return super().cross_validate()
        
    