import pandas as pd
import os
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from config import BASE_DIR, LEAGUES

def load_data(league: str) -> tuple[pd.DataFrame, pd.Series]:
    path = os.path.join(BASE_DIR, "data", league, f"{league.replace('_','')}_Match_Data_Ready_For_ML.csv")
    df = pd.read_csv(path, parse_dates=['Date'])

    df = df.drop(columns=['Date', 'HomeTeam', 'AwayTeam', 'Season'])

    X = df.drop(columns=['Target'])
    y = df['Target']
    return X, y
def train(X_train, y_train) -> XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    return model

def save_model(model: xgb.XGBClassifier, league: str):
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    joblib.dump(model, os.path.join(BASE_DIR, "models", f"xgb_{league}.pkl"))

def train_model(league: str):
    X, y = load_data(league)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train(X_train, y_train)
    save_model(model, league)

if __name__ == "__main__":
    for league in LEAGUES:
        train_model(league)