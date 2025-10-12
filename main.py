from fantasy_df import FantasyDataFrame
# from rb_gb import Rb_XGBoost

class __main__:
    def __init__(self):
        fdf = FantasyDataFrame()
        fdf.convert_to_seasonal(inplace = True)

        fdf.players_stats.to_pickle('fantasy_data_2025.pkl')

        fdf.players_stats.to_csv('fantasy_data_2025.csv')
        self.fdf = fdf
        # self.rb_gb = Rb_XGBoost()

    def run(self):
        pass

        

if __name__ == '__main__':
    __main__().run()