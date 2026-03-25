import joblib
import pandas as pd
import numpy as np
import json

def load_model(model_path: str= "../models/xgb_model.pkl"):
    return joblib.load(model_path)

def get_current_elo(home_team: str, away_team: str, elo_path: str = "../data/Premier_League/elo_ratings.json") -> tuple:
    with open(elo_path) as f:
        elo_dict = json.load(f)
    home_elo = elo_dict.get(home_team, 1350)
    away_elo = elo_dict.get(away_team, 1350)
    return home_elo, away_elo

def get_match_features(home_team: str, away_team: str, data_path: str = "../data/Premier_League/PremierLeague_Match_Data_Ready_For_ML.csv") -> pd.DataFrame:
    df = pd.read_csv(data_path)
    home_matches = df[df["HomeTeam"] == home_team].tail(1)
    away_matches = df[df["AwayTeam"] == away_team].tail(1)
    if home_matches.empty or away_matches.empty:
        raise Exception(f"No data for team {home_team} or team {away_team}")

    features = home_matches.copy()
    away_columns = [c for c in df.columns if c.startswith("Away")]
    for col in away_columns:
        if col in away_matches.columns:
            features[col] = away_matches[col].values

    h2h_matches = df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)].tail(1)
    if not h2h_matches.empty:
        h2h_cols = [c for c in df.columns if 'H2H' in c]
        for col in h2h_cols:
            if col in h2h_matches.columns:
                features[col] = h2h_matches[col].values

    home_elo, away_elo = get_current_elo(home_team, away_team)
    features['Home_ELO'] = home_elo
    features['Away_ELO'] = away_elo
    features['ELO_difference'] = (home_elo + 70) - away_elo

    drop_cols = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'Season', 'Target',
                 'FTR', 'FTHG', 'FTAG', 'HTR', 'HTHG', 'HTAG',
                 'Referee', 'Away_H2H_Overall_Pts', 'Away_H2H_Overall_GF', 'Away_H2H_Overall_GA', 'HomePoints', 'AwayPoints']

    features = features.drop(columns=[c for c in drop_cols if c in features.columns])
    return features

def apply_morale(proba: np.ndarray, home_morale: int, away_morale: int) -> np.ndarray:
    home_factor = 0.75 + (home_morale / 10) * 0.50
    away_factor = 0.75 + (away_morale / 10) * 0.50

    proba = proba.copy()
    proba[2] *= home_factor
    proba[0] *= away_factor
    proba = np.clip(proba, 0, 1)
    proba = proba / np.sum(proba)

    return proba

def predict_match(home_team: str, away_team:str, home_morale: int, away_morale: int) -> dict:
    model = load_model()
    features = get_match_features(home_team, away_team)
    proba_raw = model.predict_proba(features)[0]
    proba_adjusted = apply_morale(proba_raw, home_morale, away_morale)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_morale": home_morale,
        "away_morale": away_morale,
        "away_win": round(proba_adjusted[0] * 100, 1),
        "draw": round(proba_adjusted[1] * 100, 1),
        "home_win": round(proba_adjusted[2] * 100, 1),
    }

if __name__ == "__main__":
    result = predict_match("West Ham", "Wolves", home_morale=5, away_morale=5)
    print(f"{result['home_team']}, vs {result['away_team']}")
    print(f"Win chance {result['home_team']}: {result['home_win']:.1f}%")
    print(f"Draw chance: {result['draw']:.1f}%")
    print(f"Win chance: {result['away_team']}: {result['away_win']:.1f}%")