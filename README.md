# ⚽ Football Match Predictor

A machine learning application for predicting football match outcomes across **5 major European leagues**, combining statistical models with AI-powered team morale analysis.

## Overview

The app predicts match results (home win / draw / away win) using:
- **XGBoost classifier** — one model per league, trained on ~25 seasons of historical data
- **ELO rating system** to measure current team strength
- **Morale scoring** via Claude API or a fine-tuned DeBERTa model, based on recent news headlines

**Supported leagues:** Premier League 🏴, La Liga 🇪🇸, Bundesliga 🇩🇪, Serie A 🇮🇹, Ligue 1 🇫🇷

## Features

- Match outcome probability prediction for any two teams in a league
- Two morale analysis models to choose from: **Claude API** (reasoning, world knowledge) or **local DeBERTa** (fast, free, offline)
- ELO history charts for any two teams over a selectable date range
- One-click data refresh in the sidebar (downloads latest matches, reprocesses, retrains XGBoost)

## Project Structure

```
├── data/
│   └── <League>/                         # one folder per league
│       ├── raw/<code>_LIVE.csv           # latest season (football-data.co.uk)
│       ├── <League>_Match_Data_Ready_For_ML.csv
│       └── elo_ratings.json              # current ELO per team
├── models/
│   ├── xgb_<League>.pkl                  # one XGBoost model per league
│   └── deberta-morale-final/             # fine-tuned DeBERTa (not in repo — retrain via notebook)
├── notebooks/
│   ├── 01_eda.ipynb                      # data pipeline & feature engineering
│   ├── machine_learning.ipynb            # XGBoost training
│   ├── visualization_eda.ipynb           # EDA charts
│   └── deberta_fine_tuning.ipynb         # DeBERTa fine-tuning (local, documented)
├── source/
│   ├── app.py                            # Streamlit app
│   ├── config.py                         # leagues, paths, constants
│   ├── prediction/
│   │   ├── predict.py                    # XGBoost + ELO + morale prediction logic
│   │   └── llm_claude_morale.py          # morale scoring (Claude + DeBERTa)
│   ├── data_tools/
│   │   ├── scraper.py                    # live news scraper (Google News RSS)
│   │   ├── collect_real_headlines.py     # GDELT headline collection
│   │   ├── label_real_headlines.py       # Claude Haiku morale labeling
│   │   └── generate_dataset.py           # synthetic dataset generation
│   └── pipeline/
│       ├── download_data.py              # download latest match data
│       ├── process_league.py             # feature engineering + ELO
│       └── train_model.py                # XGBoost training per league
├── requirements.txt                      # app environment
└── requirements-train.txt                # DeBERTa training environment (pinned)
```

## Setup

### App environment

```bash
git clone https://github.com/Saku4321/Sports_prediction.git
cd Sports_prediction
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the root directory:
```
ANTHROPIC_API_KEY=your_key_here
HF_TOKEN=your_token_here          # optional — only for downloading the DeBERTa base model
```

### Running the App

```bash
streamlit run source/app.py
```

Use the sidebar **Update Data** button to download the latest matches and retrain the per-league XGBoost models.

## Models

### XGBoost (match outcome)
One model per league, trained on ~25 seasons of data. Features include last-5 match form, ELO ratings, head-to-head history, days of rest, and shot statistics. Morale is applied as a post-hoc adjustment to the predicted probabilities.

### DeBERTa (team morale)
Fine-tuned `microsoft/deberta-v3-base` (regression, `num_labels=1`) that predicts a team's morale (1–10) from recent headlines. Input format: `"{team}: headline . headline . ..."` — the team name is part of the input so the model judges sentiment *from that team's perspective*.

**Training data:** real headlines collected from the [GDELT](https://www.gdeltproject.org/) DOC API and labeled 1–10 by Claude Haiku (633 clean examples after noise filtering).

**Training pipeline** (see `notebooks/deberta_fine_tuning.ipynb`):
1. Pre-training on ~11k synthetic examples (sequential transfer learning)
2. 5-fold cross-validation on real data (honest generalization estimate)
3. Final model trained on 90% with a 10% holdout for early stopping

**Performance:** MAE ≈ **0.97**, Spearman ≈ **0.80** (vs. a predict-the-mean baseline MAE of 1.62).

> The trained weights (~700 MB) are not committed to the repo. Re-create them by running `deberta_fine_tuning.ipynb`, which saves to `models/deberta-morale-final/`.

#### Training environment ⚠️
DeBERTa-v3 fine-tuning on this small dataset is numerically unstable on `transformers 5.x` (produces NaN). Use the pinned environment in `requirements-train.txt` (**Python 3.12**, `transformers==4.44.0`):

```bash
py -3.12 -m venv .venv-train
.venv-train\Scripts\activate
pip install -r requirements-train.txt
# torch is pinned to a CUDA build — if it fails, install with:
# pip install torch --index-url https://download.pytorch.org/whl/cu128
```

The app itself runs fine on the standard `.venv` — DeBERTa inference is version-independent; only training requires the pinned stack.

### Claude vs DeBERTa — when to use which
DeBERTa does fast, shallow sentiment matching; Claude reasons with world knowledge. For example, the headline *"New to the Sky Bet Championship: ... Wolves"* implies relegation (a morale disaster) — Claude infers this, while DeBERTa, lacking that world knowledge, rates it neutrally. The in-app toggle lets you pick: DeBERTa for speed/cost, Claude for nuance.

## Data Source

Match data from [football-data.co.uk](https://www.football-data.co.uk). Live news headlines scraped from Google News RSS; training headlines from the [GDELT Project](https://www.gdeltproject.org/).
