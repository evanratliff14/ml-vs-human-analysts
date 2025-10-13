from fantasy_df import FantasyDataFrame
# from rb_gb import Rb_XGBoost
import logging
import pyarrow

class __main__:
    def __init__(self):
        try:
            fdf = FantasyDataFrame()
            self.fdf = fdf

        except Exception as e:
            logging.exception(e)
        
        # self.rb_gb = Rb_XGBoost()

    def run(self):
        fdf = self.fdf
        if self.fdf:
            logging.info("Creating parquet...")
            fdf.players_stats.to_parquet('data.parquet', engine='pyarrow', index=False)
        else:
            logging.exception("fdf does not exist")


        

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    __main__().run()