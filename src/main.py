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
            fdf.players_stats.to_parquet('data.parquet')
        
        self.rb_seasonal = Seasonal(points_type='half_ppr', position = 'RB', type = 'xgb')

    def run(self):
        rb_seasonal = self.rb_seasonal
        rb_seasonal.corr()

        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 10))  # width, height in inches

        sns.heatmap(rb_seasonal.fantasy_data.corr(numeric_only=True).abs(), cmap='coolwarm')
        plt.savefig('corr_plot.png')
        plt.show()

        # only outputting standard/game right now
        # rb_seasonal.set_features()
        rb_seasonal.train_model(rb_seasonal.model)
        rb_seasonal.test_model()
        # rb_seasonal.cross_validate()
        # print(rb_seasonal)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    __main__().run()