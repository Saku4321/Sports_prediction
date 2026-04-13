# ⚽ Football Match Predictor

A machine learning application for predicting Premier League match outcomes, combining statistical models with AI-powered team morale analysis.

## Overview

The app predicts match results (home win / draw / away win) using:
- **XGBoost classifier** trained on 25 seasons of Premier League data (2000–2025)
- **ELO rating system** to measure current team strength
- **Morale scoring** via Claude API or a fine-tuned DeBERTa model, based on recent news headlines

## Features

- Match outcome probability prediction
- Two morale analysis models to choose from: Claude API or local DeBERTa
- ELO history charts for any two teams
- Automatic data update on app startup (downloads latest match data, retrains XGBoost)
- Manual refresh button in the sidebar

## Project Structure

```
├── data/
│   └── Premier_League/
│       ├── raw/          # Raw CSV files per season (football-data.co.uk)
│       ├── PremierLeague_WszystkieSezony.csv
│       └── PremierLeague_Match_Data_Ready_For_ML.csv
├── models/
│   ├── xgb_Premier_League.pkl
│   └── deberta-morale-final/    # Fine-tuned DeBERTa (not in repo, download separately)
├── notebooks/
│   ├── 01_eda.ipynb             # Data pipeline & feature engineering
│   ├── machine_learning.ipynb   # XGBoost training
│   ├── visualization_eda.ipynb  # EDA charts
│   └── deberta_fine_tuning.ipynb # DeBERTa fine-tuning (Google Colab)
└── source/
    ├── app.py                   # Streamlit app
    ├── predict.py               # Prediction logic
    ├── llm_claude_morale.py     # Morale scoring (Claude + DeBERTa)
    ├── scraper.py               # News scraper (Google News RSS)
    └── prepeare_data.py         # Fine-tuning dataset preparation
```

## Setup

```bash
git clone https://github.com/Saku4321/Sports_prediction.git
cd Sports_prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the root directory:
```
ANTHROPIC_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

## Running the App

```bash
cd source
streamlit run app.py
```

On first launch the app will automatically download all match data and retrain the model — this takes a few minutes.

## Models

### XGBoost
Trained on 25 seasons of Premier League data with features including last 5 match form, ELO ratings, H2H history, days of rest, and shot statistics.

### DeBERTa Morale Model
Fine-tuned `microsoft/deberta-v3-base` for regression on a custom dataset of 1000 football news scenarios. Predicts team morale score (1–10) from news headlines. Fine-tuning was done on Google Colab (GPU required). Download the model weights separately and place in `models/deberta-morale-final/`.

## Data Source

Match data from [football-data.co.uk](https://www.football-data.co.uk). News headlines scraped from Google News RSS.