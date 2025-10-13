from fantasy_df import FantasyDataFrame
from xgb import Xgb
import logging
import pyarrow
import os

class __main__:
    def __init__(self):
        if not os.path.isfile('data.parquet'):
            fdf = FantasyDataFrame()
            self.fdf = fdf
            logging.info("Creating parquet...")
            fdf.players_stats.to_parquet('data.parquet')
        
        self.rb_xgb = Xgb(points_type='half_ppr', position = 'RB', type = 'xgb')

    def run(self):
        rb_xgb = self.rb_xgb
        
        
        rb_xgb.set_features()
        rb_xgb.train(rb_xgb.model)
        rb_xgb.test()
        print(rb_xgb)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    __main__().run()