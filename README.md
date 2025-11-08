**# ml_vs_human_analysts

A repository leveraging 10 years worth of classical NFL stats and Next Gen Stats into fantasy football machine learning predictions against human analysts’ 

Uses half-ppr, standard, and full-ppr PER game as RMSE and MAE metrics. Eval data:= current nfl season, test:= {current_season-1, current_season-2}, train := [year for year in range(2016,current_season-2)]

## Overview
web/ for frontend
api/ for exposed app.py endpoint
src/ for backend (data, ML)

## Run models locally
git clone https://github.com/evanratliff14/ml_vs_human_analysts.git
cd src
python3 -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

## PYTHON
make sure python3 version is supported (recommended ~3.11, but multiple versions will work)

``bash
python3 model_executor.py
flask run (to run the backend in development)

## to tweak model features
edit/add/remove lines in {position}_features.txt

## other navigation
/data
    - all data accessible via API (lightweight parquets)

/viewable
    - historical model predictions for easy open (csv)

/cache
    - files that are saved just as to save loading time for model_executor.py

.parquet --> data for non-reloading when calling model_executor.py (calls FantasyDataFrame)
.joblib --> lightweight models
.csv --> heavier weight than parquet, used for local display

# data
credit nflreadpy and nflverse
win totals credit https://www.nfeloapp.com/nfl-power-ratings/nfl-win-totals/


