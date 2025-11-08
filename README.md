**# ml_vs_human_analysts

A project comparing machine learning predictions against human analysts’ performance in fantasy football.

## Overview
This repo contains code, data, and notebooks to evaluate how well machine learning models forecast NFL fantasy football outcomes compared to human analysts.

## Structure
- `PLAN.md` — Project goals & roadmap  
- `fantasy_data.csv`, `fantasy_data.pkl` — Fantasy football stats  
- `nfl-win-totals.csv` — NFL win totals data  
- `RB_Prediction.csv` — Running back prediction data  
- `fantasy_df.py` — Data cleaning/processing  
- `rb_gb.py` — Model training & evaluation  
- `gb_nb.ipynb` — Jupyter notebook for analysis  
- `planning.txt` — Notes & brainstorming  

## Setup
```bash
git clone https://github.com/evanratliff14/ml_vs_human_analysts.git
cd ml_vs_human_analysts
python3 -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt  # or: pip install pandas numpy scikit-learn jupyter
**

## Usage
python fantasy_df.py
python rb_gb.py
jupyter notebook gb_nb.ipynb
