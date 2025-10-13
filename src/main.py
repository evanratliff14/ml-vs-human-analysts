from fantasy_df import FantasyDataFrame
from rb_gb import Rb_Xgb
import logging
import pyarrow
import os

class __main__:
    def __init__(self):
        if not os.path.isfile('players_stats.csv'):
            fdf = FantasyDataFrame()
            self.fdf = fdf
            logging.info("Creating csv...")
            fdf.players_stats.to_csv('players_stats.csv')
        
        self.rb_gb = Rb_Xgb(points_type='half_ppr', hist=True)

    def run(self):
        rb_gb = self.rb_gb
        
        
        rb_gb.set_features()
        rb_gb.train(rb_gb.model)
        rb_gb.test()
        print(rb_gb)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    __main__().run()