from fantasy_df import FantasyDataFrame
from rb_gb import Rb_XGBoost
import logging
import pyarrow
import os

class __main__:
    def __init__(self):
        if not os.path.isfile('data.parquet'):
            fdf = FantasyDataFrame()
            self.fdf = fdf
            logging.info("Creating parquet...")
            fdf.players_stats.to_parquet('data.parquet', engine='pyarrow', index=False)
        
        self.rb_gb = Rb_XGBoost(points_type='half_ppr')

    def run(self):
        rb_gb = self.rb_gb
        
        
        rb_gb.set_features()
        rb_gb.train(rb_gb.model)
        rb_gb.test()
        print(rb_gb)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    __main__().run()