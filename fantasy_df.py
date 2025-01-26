import pandas as pd
import nfl_data_py as nfl
import numpy as np

class FantasyDataFrame:
    def __init__(self):
        print('Initializing...')
        self.__years = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
        self.__pos = ['TE', 'RB', 'WR', 'QB']
        self.__load_data()

    def __load_data(self):
        # TODO: merge teams for next season
        # TODO: Merge other_epas
        

        # use players_stats as base file
        players_stats = nfl.import_seasonal_data(years=self.__years)
        players_stats = players_stats[players_stats['season_type']=='REG']
        nfl.clean_nfl_data(players_stats)
        players_stats.drop(['passing_first_downs',
       'rushing_first_downs',  'yac_sh', 'dom', 'yptmpa', 'ppr_sh'], axis=1, inplace=True)
        
        self.players_stats = players_stats
        self.__map_ids()
        # create custom stats
        players_stats['fantasy_points_half_ppr'] = players_stats['fantasy_points_ppr']-(players_stats['receptions']*0.5)
        
        # track games played and games missed on IR
        weekly_rosters = nfl.clean_nfl_data(nfl.import_weekly_rosters(self.__years))[['player_name', 'season', 'week','status', 'position']]

        # count games on ir and games played
        weekly_rosters['ir_games'] = weekly_rosters.groupby(['player_name', 'position', 'season'])['status'].transform(lambda x: (x == 'RES').sum())
        weekly_rosters['out_games'] = weekly_rosters.groupby(['player_name', 'position', 'season'])['status'].transform(lambda x: (x == 'INA').sum())

        print('Importing games spent on IR...')

        players_stats = players_stats.merge(
            weekly_rosters[['player_name', 'position', 'season', 'ir_games', 'out_games']].drop_duplicates(),
            on = ['player_name', 'position', 'season'],
            how = 'left'   
        )

        # Fill NaN with 0 and convert to int
        players_stats['ir_games'] = players_stats['ir_games'].fillna(0).astype(int)

        print("Importing Next Gen Stats...")

        # track next gen stats
        # receiving
        ngs_receiving = nfl.import_ngs_data('receiving', self.__years)
        players_stats = players_stats.merge(
            ngs_receiving[['player_gsis_id', 'season', 'avg_separation', 'avg_yac_above_expectation', 'catch_percentage']],
            left_on=['player_id', 'season'],
            right_on=['player_gsis_id', 'season'],
            how='left'
        ).drop(columns=['player_gsis_id'])


        # rushing
        ngs_rushing = nfl.import_ngs_data('rushing', self.__years)
        players_stats = players_stats.merge(
            ngs_rushing[['player_gsis_id', 'season', 'rush_yards_over_expected_per_att', 'rush_pct_over_expected']],
            left_on=['player_id', 'season'],
            right_on=['player_gsis_id', 'season'],
            how='left'
        ).drop(columns=['player_gsis_id'])
                                                                    

        # passing
        ngs_passing = nfl.import_ngs_data('passing', self.__years)
        players_stats = players_stats.merge(
            ngs_passing[['player_gsis_id', 'season', 'completion_percentage_above_expectation']],
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
        players_stats['turnovers/game'] = (players_stats['rushing_fumbles'] + players_stats['sack_fumbles']+players_stats['interceptions'] )/players_stats['games']
        players_stats['adot'] = players_stats['passing_air_yards']/players_stats['attempts']
        players_stats['targets_game'] = players_stats['targets']/players_stats['games']

        print("Importing seasonal roster data...")
        # import roster data
        # players_stats['team'] = ""
        rosters = nfl.clean_nfl_data(nfl.import_seasonal_rosters(self.__years))
        players_stats = players_stats.merge(
            rosters[['player_name', 'season', 'position', 'team', 'depth_chart_position', 'height', 'weight', 'years_exp', 'age']],
            on=['player_name', 'season', 'position'],
            how='left'
        )

        # import position on depth chart
        depth_charts = nfl.clean_nfl_data(nfl.import_depth_charts(self.__years))
        depth_charts = depth_charts.loc[depth_charts['week']==1.0]
        players_stats = players_stats.merge(
            depth_charts[['gsis_id', 'season', 'depth_team']],
            left_on = ['player_id', 'season'],
            right_on= ['gsis_id', 'season'],
            how='left',
            suffixes = ["", '_y']
        )

        # will assign win totals based on team column (might need heavy mapping w dicts)
        #https://www.nfeloapp.com/nfl-power-ratings/nfl-win-totals/
        print("Importing Vegas Win Total Lines...")
        win_totals = pd.read_csv('nfl-win-totals.csv')[['season', 'team', 'line_adj']]
        win_totals['team'] = win_totals['team'].str.strip()
        players_stats['team'] = players_stats['team'].str.strip()
 
        players_stats = players_stats.merge(
            win_totals[['team', 'season', 'line_adj']],
            on=['team', 'season'],
            how='left'
        ).rename(columns={'line_adj': 'win_total_adj_line'})

        ## rookies
        print("Importing draft data...")
        # import rec_yards, rec_tds, etc. college production
        draft_picks = nfl.clean_nfl_data(nfl.import_draft_picks(self.__years))
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
        combine = nfl.clean_nfl_data(nfl.import_combine_data(years = self.__years, positions = self.__pos))

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
        players_stats = players_stats.merge(
        players_stats[['player_id', 'next_season', 'fantasy_points','fantasy_points_ppr', 'fantasy_points_half_ppr','ir_games','games']],
        left_on = ['player_id', 'season'],
        right_on = ['player_id', 'next_season'],
        how = 'left',
        suffixes = ('', '_future')

        )

        

        # create fpts/game metrics (future and current)
        players_stats['future_ppr/game'] = players_stats['fantasy_points_ppr_future']/players_stats['games_future']
        players_stats['future_half_ppr/game'] = players_stats['fantasy_points_half_ppr_future']/players_stats['games_future']
        players_stats['ppr/game'] = players_stats['fantasy_points_ppr']/players_stats['games']
        players_stats['half_ppr/game'] = players_stats['fantasy_points_half_ppr']/players_stats['games']

        self.players_stats = players_stats

    # def __count_status(self, row):
    # to add in, need .apply
        # players_stats = self.players_stats
        # games_played = players_stats.loc[(players_stats['player_name'] == row['player_name']) & ((players_stats['season']==row['season'])) & (players_stats['position']==row['position']), 'games_played'].squeeze()
        # ir_games = players_stats.loc[(players_stats['player_name'] == row['player_name']) & ((players_stats['season']==row['season'])) & (players_stats['position']==row['position']), 'ir_games'].squeeze()
        # if row['status'] == 'ACT':
        #     players_stats.loc[(players_stats['player_name']==row['player_name']) & (players_stats['season']==row['season']) & (players_stats['position']==row['position']), 'games_played'] = games_played + 1
        #     self.players_stats = players_stats
        # elif row['status'] == 'RES':
        #     players_stats.loc[(players_stats['player_name']==row['player_name']) & (players_stats['season']==row['season']) & (players_stats['position']==row['position']), 'ir_games'] = ir_games + 1
        #     self.players_stats = players_stats

    def __map_ids(self):
        df = self.players_stats
        ## map all values in column id to names
        # import df of columns 'id', 'name', 'position'
        mappings = nfl.import_ids()[['name', 'position', 'gsis_id']]
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

fdf = FantasyDataFrame()
fdf.players_stats.to_csv('fantasy_data.csv') 