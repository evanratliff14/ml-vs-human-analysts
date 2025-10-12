import pandas as pd
import nflreadpy as nfl
import numpy as np
import pickle
from datetime import datetime
import pyarrow

class FantasyDataFrame:
    def __init__(self):
        print('Initializing...')
        # normalize to football year season
        self.years = [year for year in range(2016, nfl.get_current_season())]
        self.pos = ['TE', 'RB', 'WR', 'QB']
        self.load_data()

    def load_data(self):
        # TODO: merge teams for next season
        # TODO: Merge other_epas
        

        # use players_stats as base file
        players_stats = nfl.load_player_stats(self.years).to_pandas()
        players_stats = players_stats[players_stats['season_type']=='REG']
        print(players_stats.head)
        
        self.players_stats = players_stats
        self.map_ids()
        # create custom stats
        players_stats['fantasy_points_standard'] = players_stats['fantasy_points_ppr']-(players_stats['receptions']*1)
        players_stats['fantasy_points_half_ppr'] = players_stats['fantasy_points_ppr']-(players_stats['receptions']*0.5)
        
        # track games played and games missed on IR
        injuries = nfl.load_injuries([year for year in self.years if year >=2009])[['gsis_id', 'season', 'week','report_status', 'position']].to_pandas()

        # count games on ir and games played
        injuries['ir_games'] = injuries.groupby(['gsis_id', 'position', 'season'])['report_status'].transform(lambda x: (x == 'RES').sum())
        injuries['out_games'] = injuries.groupby(['gsis_id', 'position', 'season'])['report_status'].transform(lambda x: (x == 'INA').sum())
        injuries['total_missed_games'] = injuries['out_games'] + injuries['ir_games']

        print('Importing games spent on IR...')

        players_stats = players_stats.merge(
            injuries[['gsis_id', 'position', 'season', 'ir_games', 'out_games', 'total_missed_games']].drop_duplicates(),
            left_on = ['player_id', 'position', 'season'],
            right_on = ['gsis_id', 'position', 'season'],
            how = 'left'   
        )

        # Fill NaN with 0 and convert to int
        players_stats['ir_games'] = players_stats['ir_games'].fillna(0).astype(int)
        players_stats['out_games'] = players_stats['out_games'].fillna(0).astype(int)
        players_stats['total_missed_games'] = players_stats['total_missed_games'].fillna(0).astype(int)
        players_stats['games'] = np.where(players_stats['season'] >= 2021, 17, 16)
        players_stats['games'] = players_stats['games'] - players_stats['total_missed_games']



        print("Importing Next Gen Stats...")

        # track next gen stats
        # receiving
        ngs_receiving = nfl.load_nextgen_stats([year for year in self.years if year >=2016], 'receiving').to_pandas()
        players_stats = players_stats.merge(
            ngs_receiving[['player_gsis_id', 'season', 'avg_cushion', 'avg_separation', 'percent_share_of_intended_air_yards', 'catch_percentage', 'avg_yac', 'avg_expected_yac', 'avg_yac_above_expectation']],
            left_on=['player_id', 'season'],
            right_on=['player_gsis_id', 'season'],
            how='left'
        ).drop(columns=['player_gsis_id'])


        # rushing
        ngs_rushing = nfl.load_nextgen_stats([year for year in self.years if year >=2016], 'rushing').to_pandas()
        players_stats = players_stats.merge(
            ngs_rushing[['player_gsis_id', 'season', 'efficiency', 'percent_attempts_gte_eight_defenders', 'avg_time_to_los', 'expected_rush_yards', 'rush_yards_over_expected', 'rush_pct_over_expected', 'rush_yards_over_expected_per_att']],
            left_on=['player_id', 'season'],
            right_on=['player_gsis_id', 'season'],
            how='left'
        ).drop(columns=['player_gsis_id'])
                                                                    

        # passing
        ngs_passing = nfl.load_nextgen_stats([year for year in self.years if year >=2016], 'passing').to_pandas()
        players_stats = players_stats.merge(
            ngs_passing[['player_gsis_id', 'season', 'avg_air_distance', 'max_air_distance', 'avg_time_to_throw', 'avg_completed_air_yards', 'avg_intended_air_yards', 'avg_air_yards_differential', 'aggressiveness', 'max_completed_air_distance', 'avg_air_yards_to_sticks', 'passer_rating', 'completion_percentage', 'expected_completion_percentage', 'completion_percentage_above_expectation', 'avg_air_distance', ]],
            left_on=['player_id', 'season'],
            right_on=['player_gsis_id', 'season'],
            how='left'
        ).drop(columns=['player_gsis_id'])


        # create per-___ stats
        players_stats['pass_tds/game'] = players_stats['passing_tds']/players_stats['games']
        players_stats['pass_air_yards/game'] = players_stats['passing_air_yards']/players_stats['games']
        players_stats['carries/game'] = players_stats['carries']/players_stats['games']
        players_stats['yards/carry'] = players_stats['rushing_yards']/players_stats['carries']
        players_stats['rushing_tds/game'] = players_stats['rushing_tds']/players_stats['games']
        players_stats['receiving_tds/game'] = players_stats['receiving_tds']/players_stats['games']
        players_stats['turnovers/game'] = (players_stats['rushing_fumbles'] + players_stats['sack_fumbles']+players_stats['passing_interceptions'] )/players_stats['games']
        players_stats['adot'] = players_stats['passing_air_yards']/players_stats['attempts']
        players_stats['targets_game'] = players_stats['targets']/players_stats['games']

        print("Importing seasonal roster data...")
        # import roster data
        # players_stats['team'] = ""
        #TODO: keep cleaning data from here. also group stats for this model by season and gsis_id
        rosters = nfl.load_rosters(self.years)
        players_stats = players_stats.merge(
            rosters[['gsis_id', 'season', 'position', 'team', 'depth_chart_position', 'height', 'weight', 'years_exp', 'college', 'status', 'birthdate']],
            on=['player_gsis_id', 'season', 'position'],
            how='left'
        )

        # import position on depth chart
        depth_charts = nfl.load_depth_charts(self.years)
        depth_charts = depth_charts.loc[depth_charts['week']==1.0]
        players_stats = players_stats.merge(
            depth_charts[['gsis_id', 'season', 'depth_team']],
            left_on = ['player_gsis_id', 'season'],
            right_on= ['gsis_id', 'season'],
            how='left',
            suffixes = ["", '']
        )
        # drop because NaN causes error for astype
        players_stats.dropna(subset = ['depth_team'], inplace = True)
        players_stats['depth_team'] = players_stats['depth_team'].astype('int')

        # will assign win totals based on team column (might need heavy mapping w dicts)
        #https://www.nfeloapp.com/nfl-power-ratings/nfl-win-totals/
        print("Importing Vegas Win Total Lines...")
        win_totals = pd.read_csv('nfl-win-totals.csv-2025')[['season', 'team', 'Adj. Total']]
        win_totals['team'] = win_totals['team'].str.strip()
        players_stats['team'] = players_stats['team'].str.strip()
 
        players_stats = players_stats.merge(
            win_totals[['team', 'season', 'Adj. Total']],
            on=['team', 'season'],
            how='left'
        ).rename(columns={'Adj. Total': 'win_total_adj_line'})

        ## rookies
        print("Importing draft data...")
        # import rec_yards, rec_tds, etc. college production
        draft_picks = nfl.clean_nfl_data(nfl.import_draft_picks(self.years))
        # Merge draft_picks into players_stats based on the player_id and gsis_id columns
        players_stats = players_stats.merge(
            draft_picks[['gsis_id', 'college', 'pick']], 
            left_on='player_id',  
            right_on='gsis_id',   
            how='left'            
        )

        # import PFF non-linear draft value model, map this to self.draft_picks
        players_stats['draft_value'] = np.NaN
        draft_values = nfl.clean_nfl_data(nfl.import_draft_values())
        draft_values = draft_values[['pick', 'pff']]
        draft_values.set_index('pick', inplace=True)
        draft_value_dict = draft_values['pff'].to_dict()
        players_stats['draft_value'] = players_stats['pick'].map(draft_value_dict)

        print("Importing combine data...")
        # get combine stats: for RBs, WRs, TEs, and map by player_name and pos
        combine = nfl.clean_nfl_data(nfl.import_combine_data(years = self.years, positions = self.pos))

        combine = combine.rename(columns={'pos': 'position'})
        combine['player_name'] = combine['player_name'].str.strip()
        combine['position'] = combine['position'].str.strip()
        # use vectorized operations for faster mapping
        players_stats = players_stats.merge(combine[['player_name', 'position', 'forty', 'bench', 'vertical', 'broad_jump', 'cone', 'shuttle']], 
                                    on=['player_name', 'position'], 
                                    how='left')
        # drop duplicates
        players_stats = players_stats.drop_duplicates(subset=['player_name', 'season', 'position'])

        ## TODO: support wopr_x, wopr_y
        # drop duplicate _x columns created by .merge
        for column in players_stats.columns:
            if (column[-2:] == '_x') & (column != 'wopr_x'):
                players_stats.drop(column, axis=1, inplace=True)
        # delete the _y created from the .merge function
            if (column[-2:] == '_y') & (column != 'wopr_y'):
                players_stats.rename(columns={column:column[:-2]}, inplace=True)

        players_stats['next_season'] = players_stats['season']-1

        ## assign future stats for y values
        #important: get future team to load other_epas
        players_stats = players_stats.merge(
            players_stats[['player_id', 'next_season', 'fantasy_points','fantasy_points_ppr', 'fantasy_points_half_ppr','games', 'team']],
            left_on = ['player_id', 'season'],
            right_on = ['player_id', 'next_season'],
            how = 'left',
            suffixes = ('', '_future')                                                                                                      
        )

        players_stats['future_ppr/game'] = players_stats['fantasy_points_ppr_future']/players_stats['games_future']
        players_stats['future_half_ppr/game'] = players_stats['fantasy_points_half_ppr_future']/players_stats['games_future']
        players_stats['ppr/game'] = players_stats['fantasy_points_ppr']/players_stats['games']
        players_stats['half_ppr/game'] = players_stats['fantasy_points_half_ppr']/players_stats['games']

        # calculate each combined position epa (for depth chart 1 and 2)
        # each position model class will need to subtract their own epa to get the result, other_rbs_epa
        print("Importing player contextual stats...")

        tes = players_stats.loc[players_stats['position']=='TE']
        tes = players_stats.loc[players_stats['depth_team'] <=2]
        players_stats = players_stats.merge(
            tes[['player_id', 'season', 'team_future', 'receiving_epa']],
            left_on = ['player_id', 'season', 'team_future'],
            right_on = ['player_id', 'season', 'team_future'],
            how = 'left',
            suffixes = ['', '_team_tes']
        )


        qbs = players_stats.loc[players_stats['position']=='QB']
        qbs = players_stats.loc[players_stats['depth_team'] ==1]
        players_stats = players_stats.merge(
            qbs[['player_id', 'season', 'team_future', 'rushing_epa', 'dakota', 'passing_epa']],
            left_on = ['player_id', 'season', 'team_future'],
            right_on = ['player_id', 'season', 'team_future'],
            how = 'left',
            suffixes = ['', '_team_qbs']
        )


        wrs = players_stats.loc[players_stats['position']=='WR']
        wrs = players_stats.loc[players_stats['depth_team'] <=2]
        players_stats = players_stats.merge(
            wrs[['player_id', 'season', 'team_future', 'receiving_epa']],
            left_on = ['player_id', 'season', 'team_future'],
            right_on = ['player_id', 'season', 'team_future'],
            how = 'left',
            suffixes = ['', '_team_wrs']
        )


        rbs = players_stats.loc[players_stats['position']=='RB']
        rbs = players_stats.loc[players_stats['depth_team'] <=2]
        players_stats = players_stats.merge(
            rbs[['player_id', 'season', 'team_future', 'receiving_epa', 'rushing_epa']],
            left_on = ['player_id', 'season', 'team_future'],
            right_on = ['player_id', 'season', 'team_future'],
            how = 'left',
            suffixes = ['', '_team_rbs']
        )
        

        self.players_stats = players_stats

    def map_ids(self):
        df = self.players_stats
        ## map all values in column id to names
        # import df of columns 'id', 'name', 'position'
        mappings = nfl.load_ff_playerids().to_pandas()[['name', 'position', 'gsis_id']]
        mappings.set_index('gsis_id', inplace=True)

        # for dict transformation eliminate duplicate key-value pairs
        mappings.drop_duplicates(inplace=True)

        # df to dict for mapping
        mappings_name_dict = mappings['name'].to_dict()
        mappings_pos_dict = mappings['position'].to_dict()

        # use .map to map flat dictionary, "name+position" -->id
        print("Mapping IDs...")
        df['player_name'] = df['player_id'].map(mappings_name_dict)
        df['position'] = df['player_id'].map(mappings_pos_dict)

        # drop unmapped-id players from df (fantasy defenses, other edgecases)
        df.dropna(subset=['position'], inplace=True)
        self.players_stats = df
    def convert_to_seasonal(self, inplace=False):
        if not self.players_stats:
            return Exception("Player seasonal stats have not been loaded yet!")

        temp = players_stats.groupby(['gsis_id', 'season']).transform(lambda x: x.sum())
        if inplace:
            players_stats = temp
        else:
            return temp


