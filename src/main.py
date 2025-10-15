from fantasy_df import FantasyDataFrame
from seasonal import Seasonal
import logging
import pyarrow
import os

class __main__:
    def __init__(self):
        if not os.path.isfile('data.parquet'):
            fdf = FantasyDataFrame()
            self.fdf = fdf
            logging.info("Creating parquet...")
            fdf.players_stats.to_parquet('data.parquet', index=False)
        
        self.rb_seasonal = Seasonal(points_type='ppr', position = 'RB', type = 'xgb')

    def run(self):
        rb_seasonal = self.rb_seasonal
        rb_seasonal.corr()

        # only outputting standard/game right now
        # rb_seasonal.set_features()
        rb_seasonal.train_model(rb_seasonal.model)
        rb_seasonal.test_model()
        rb_seasonal.cross_validate()
        print(rb_seasonal)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    __main__().run()