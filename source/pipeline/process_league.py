import pandas as pd
import glob
import os
import json
from config import BASE_DIR, CORE_COLUMNS, LEAGUES


def load_raw_data(league: str) -> pd.DataFrame:
    folder = os.path.join(BASE_DIR, "data", league, "raw")
    code = LEAGUES[league]["code"]
    files = glob.glob(os.path.join(folder, f"{code}*.csv"))

    all_matches = []

    for file in files:
        df = pd.read_csv(file, encoding='latin-1', on_bad_lines='skip')

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce', format='mixed')

        cols_to_keep = [col for col in CORE_COLUMNS if col in df.columns]
        all_matches.append(df[cols_to_keep])

    master_db = pd.concat(all_matches, ignore_index=True)
    master_db = master_db.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTR'])
    master_db = master_db.drop_duplicates(subset =['Date', 'HomeTeam', 'AwayTeam'])
    master_db = master_db.sort_values(by='Date').reset_index(drop=True)

    return master_db


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Season columns
    def get_season(date):
        if pd.isna(date): return "Unknown"
        return f"{date.year}/{date.year+1}" if date.month >= 7 else f"{date.year -1}/{date.year}"
    df['Season'] = df['Date'].apply(get_season)

    # Target + points
    df['Target'] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
    df['HomePoints'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    df['AwayPoints'] = df['FTR'].map({'A': 3, 'D': 1, 'H': 0})

    # Days of rest
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    df['HomeTeam'] = df['HomeTeam'].astype(str).str.split('_').str[0]
    df['AwayTeam'] = df['AwayTeam'].astype(str).str.split('_').str[0]

    home_dates = df[['Date', 'HomeTeam']].rename(columns={'HomeTeam': 'Team'})
    away_dates = df[['Date', 'AwayTeam']].rename(columns={'AwayTeam': 'Team'})
    all_matches = pd.concat([home_dates, away_dates]).sort_values(by=['Team', 'Date']).reset_index(drop=True)

    all_matches['Raw_DaysOfRest'] = all_matches.groupby('Team')['Date'].diff().dt.days
    all_matches['Is_New_Spell'] = (all_matches['Raw_DaysOfRest'] > 300).astype(int)
    all_matches['Spell_ID'] = all_matches.groupby('Team')['Is_New_Spell'].cumsum()

    all_matches['Team_Spell'] = all_matches['Team'] + '_' + all_matches['Spell_ID'].astype(str)

    all_matches['DaysOfRest'] = all_matches['Raw_DaysOfRest'].apply(
        lambda x: 14.0 if pd.isna(x) or x > 70 else x
    )

    df = df.merge(all_matches[['Date', 'Team', 'DaysOfRest', 'Team_Spell']],
                  left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left')
    df.rename(columns={'DaysOfRest': 'Home_DaysOfRest'}, inplace=True)

    df['HomeTeam'] = df['Team_Spell']
    df.drop(['Team', 'Team_Spell'], axis=1, inplace=True)

    df = df.merge(all_matches[['Date', 'Team', 'DaysOfRest', 'Team_Spell']],
                  left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left')
    df.rename(columns={'DaysOfRest': 'Away_DaysOfRest'}, inplace=True)
    df['AwayTeam'] = df['Team_Spell']
    df.drop(['Team', 'Team_Spell'], axis=1, inplace=True)

    # Home Last 5
    df = df.sort_values(by='Date').reset_index(drop=True)

    home_rolling = {
        'Home_Last5_Home_Pts':  ('HomeTeam', 'HomePoints'),
        'Home_Last5_GF':    ('HomeTeam', 'FTHG'),
        'Home_Last5_GA':    ('HomeTeam', 'FTAG'),
        'Home_Last5_Home_Shots':    ('HomeTeam', 'HS'),
        'Home_Last5_Home_ShotsOT':  ('HomeTeam', 'HST'),
        'Home_Last5_Home_ShotsAgainst': ('HomeTeam', 'AS'),
        'Home_Last5_HomeShotsOTAgainst':    ('HomeTeam', 'AST'),
        'Home_Last5_Corners':   ('HomeTeam', 'HC'),
        'Home_Last5_CornersAgainst':    ('HomeTeam', 'AC'),
        'Home_Last5_HomeFouls': ('HomeTeam', 'HF'),
        'Home_Last5_HomeFoulsAgainst':  ('HomeTeam', 'AF'),
        'Home_Last5_HomeYellows':   ('HomeTeam', 'HY'),
        'Home_Last5_HomeYellowsAgainst':    ('HomeTeam', 'AY'),
        'Home_Last5_HomeReds':  ('HomeTeam', 'HR'),
        'Home_Last5_HomeRedsAgainst':   ('HomeTeam', 'AR'),
    }
    for new_col, (group_col, stat_col) in home_rolling.items():
        df[new_col]= df.groupby(group_col)[stat_col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

    # Away Last 5
    away_rolling = {
        'Away_Last5_AwayPts':   ('AwayTeam', 'AwayPoints'),
        'Away_Last5_GF':    ('AwayTeam', 'FTAG'),
        'Away_Last5_GA':    ('AwayTeam', 'FTHG'),
        'Away_Last5_AwayShots': ('AwayTeam', 'AS'),
        'Away_Last5_AwayShotsOT':   ('AwayTeam', 'AST'),
        'Away_Last5_AwayShotsAgainst':  ('AwayTeam', 'HS'),
        'Away_Last5_AwayShotsOTAgainst':    ('AwayTeam', 'HST'),
        'Away_Last5_Corners':   ('AwayTeam', 'AC'),
        'Away_Last5_CornersAgainst':    ('AwayTeam', 'HC'),
        'Away_Last5_AwayFouls': ('AwayTeam', 'AF'),
        'Away_Last5_AwayFoulsAgainst':  ('AwayTeam', 'HF'),
        'Away_Last5_AwayYellows':   ('AwayTeam', 'AY'),
        'Away_Last5_AwayYellowsAgainst':    ('AwayTeam', 'HY'),
        'Away_Last5_AwayReds':  ('AwayTeam', 'AR'),
        'Away_Last5_AwayRedsAgainst':   ('AwayTeam', 'HR'),
    }
    for new_col, (group_col, stat_col) in away_rolling.items():
        df[new_col] = df.groupby(group_col)[stat_col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

    # Overall Last 5
    home_form = df[
        ['Date', 'HomeTeam', 'HomePoints', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY',
         'HR', 'AR']].copy()
    home_form.columns = ['Date', 'Team', 'Points', 'GoalsFor', 'GoalsAgainst', 'ShotsFor', 'ShotsAgainst', 'ShotsOTFor',
                         'ShotsOTAgainst', 'CornersFor', 'CornersAgainst', 'FoulsFor', 'FoulsAgainst', 'YellowsFor',
                         'YellowsAgainst', 'RedsFor', 'RedsAgainst']

    away_form = df[
        ['Date', 'AwayTeam', 'AwayPoints', 'FTAG', 'FTHG', 'AS', 'HS', 'AST', 'HST', 'AC', 'HC', 'AF', 'HF', 'AY', 'HY',
         'AR', 'HR']].copy()
    away_form.columns = ['Date', 'Team', 'Points', 'GoalsFor', 'GoalsAgainst', 'ShotsFor', 'ShotsAgainst', 'ShotsOTFor',
                         'ShotsOTAgainst', 'CornersFor', 'CornersAgainst', 'FoulsFor', 'FoulsAgainst', 'YellowsFor',
                         'YellowsAgainst', 'RedsFor', 'RedsAgainst']

    team_form = pd.concat([home_form, away_form]).sort_values(by=['Team', 'Date']).reset_index(drop=True)
    stat_columns = ['Points', 'GoalsFor', 'GoalsAgainst', 'ShotsFor', 'ShotsAgainst', 'ShotsOTFor', 'ShotsOTAgainst',
                    'CornersFor', 'CornersAgainst', 'FoulsFor', 'FoulsAgainst', 'YellowsFor', 'YellowsAgainst',
                    'RedsFor', 'RedsAgainst']

    for col in stat_columns:
        team_form[f'Overall_Last5_{col}'] = team_form.groupby('Team')[col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    team_form = team_form[['Date', 'Team'] + [f'Overall_Last5_{col}' for col in stat_columns]]

    df = df.merge(team_form, left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left')
    df.rename(columns=lambda x: f"Home_{x}" if x in team_form.columns and x not in ['Date', 'Team'] else x,
              inplace=True)
    df.drop('Team', axis=1, inplace=True)

    df = df.merge(team_form, left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left')
    df.rename(columns=lambda x: f"Away_{x}" if x in team_form.columns and x not in ['Date', 'Team'] else x,
              inplace=True)
    df.drop('Team', axis=1, inplace=True)

    # H2H
    df = df.sort_values(by='Date').reset_index(drop=True)
    df['H2H_Home_Pts'] = df.groupby(['HomeTeam', 'AwayTeam'])['HomePoints'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['H2H_Home_GF'] = df.groupby(['HomeTeam', 'AwayTeam'])['FTHG'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['H2H_Home_GA'] = df.groupby(['HomeTeam', 'AwayTeam'])['FTAG'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())

    h2h_home = df[['Date', 'HomeTeam', 'AwayTeam', 'HomePoints', 'FTHG', 'FTAG']].copy()
    h2h_home.columns = ['Date', 'Team', 'Opponent', 'Points', 'GF', 'GA']

    h2h_away = df[['Date', 'AwayTeam', 'HomeTeam', 'AwayPoints', 'FTAG', 'FTHG']].copy()
    h2h_away.columns = ['Date', 'Team', 'Opponent', 'Points', 'GF', 'GA']

    h2h_all = pd.concat([h2h_home, h2h_away]).sort_values(by=['Team', 'Opponent', 'Date']).reset_index(drop=True)
    h2h_all['H2H_Overall_Pts'] = h2h_all.groupby(['Team', 'Opponent'])['Points'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    h2h_all['H2H_Overall_GF'] = h2h_all.groupby(['Team', 'Opponent'])['GF'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    h2h_all['H2H_Overall_GA'] = h2h_all.groupby(['Team', 'Opponent'])['GA'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    h2h_all = h2h_all[['Date', 'Team', 'Opponent', 'H2H_Overall_Pts', 'H2H_Overall_GF', 'H2H_Overall_GA']]

    df = df.merge(h2h_all, left_on=['Date', 'HomeTeam', 'AwayTeam'], right_on=['Date', 'Team', 'Opponent'], how='left')
    df.rename(columns={
        'H2H_Overall_Pts': 'Home_H2H_Overall_Pts', 'H2H_Overall_GF': 'Home_H2H_Overall_GF', 'H2H_Overall_GA': 'Home_H2H_Overall_GA',
    }, inplace=True)
    df.drop(['Team', 'Opponent'], axis=1, inplace=True)

    df = df.merge(h2h_all, left_on=['Date', 'AwayTeam', 'HomeTeam'], right_on=['Date', 'Team', 'Opponent'], how='left')
    df.rename(columns={
        'H2H_Overall_Pts': 'Away_H2H_Overall_Pts', 'H2H_Overall_GF': 'Away_H2H_Overall_GF', 'H2H_Overall_GA': 'Away_H2H_Overall_GA',
    }, inplace=True)
    df.drop(['Team', 'Opponent'], axis=1, inplace=True)

    return df

def calculate_elo(df: pd.DataFrame, base_elo: int = 1500, hfa: int = 65) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    K_FACTOR = 20
    SCALE_FACTOR = 400

    def expected_result(elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / SCALE_FACTOR))

    def get_g_modifier(goal_diff):
        if goal_diff <= 1:
            return 1.0
        elif goal_diff == 2:
            return 1.5
        else:
            return (11 + goal_diff) / 8.0

    elo_dict = {}

    first_season = df['Season'].iloc[0]

    home_elos_before = []
    away_elos_before = []

    for _, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']

        if home not in elo_dict:
            elo_dict[home] = base_elo if row['Season'] == first_season else base_elo - 150
        if away not in elo_dict:
            elo_dict[away] = base_elo if row['Season'] == first_season else base_elo - 150

        elo_h = elo_dict[home]
        elo_a = elo_dict[away]

        home_elos_before.append(elo_h)
        away_elos_before.append(elo_a)

        elo_h_adj = elo_h + hfa
        exp_h = expected_result(elo_h_adj, elo_a)
        exp_a = expected_result(elo_a, elo_h_adj)

        goals_h, goals_a = row['FTHG'], row['FTAG']
        if goals_h > goals_a:   res_h, res_a = 1.0, 0.0
        elif goals_h == goals_a:     res_h, res_a = 0.5, 0.5
        else:    res_h, res_a = 0.0, 1.0

        G = get_g_modifier(abs(int(goals_h) - int(goals_a)))

        elo_dict[home] = elo_h + K_FACTOR * G * (res_h - exp_h)
        elo_dict[away] = elo_a + K_FACTOR * G * (res_a - exp_a)

    df['Home_ELO'] = home_elos_before
    df['Away_ELO'] = away_elos_before
    df['ELO_difference'] = (df['Home_ELO'] + hfa) - df['Away_ELO']

    df['HomeTeam'] = df['HomeTeam'].str.split('_').str[0]
    df['AwayTeam'] = df['AwayTeam'].str.split('_').str[0]
    elo_dict_clean = {k.split('_')[0]: v for k, v in elo_dict.items()}

    return df, elo_dict_clean

def impute_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    min_year = df['Date'].dt.year.min()
    safe_historical_data = df[df['Date'].dt.year <= min_year + 2]

    cols_form = [col for col in df.columns if 'Last5' in col]

    for col in cols_form:
        low_baseline = safe_historical_data[col].quantile(0.25)

        if pd.isna(low_baseline):
            low_baseline = 0.0
        df[col] = df[col].fillna(low_baseline)

    cols_h2h_pts = [col for col in df.columns if 'H2H' in col and 'Pts' in col]
    cols_h2h_goals = [col for col in df.columns if 'H2H' in col and ('GF' in col or 'GA' in col)]

    for col in cols_h2h_pts:
        df[col] = df[col].fillna(1.0)

    for col in cols_h2h_goals:
        df[col] = df[col].fillna(0.0)

    cols_to_drop = [col for col in df.columns if 'Away_H2H' in col]
    cols_to_drop += [col for col in [
        'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR', 'FTR',
        'HS', 'AS', 'HST', 'AST',
        'HF', 'AF', 'HC', 'AC',
        'HY', 'AY', 'HR', 'AR',
        'HomePoints', 'AwayPoints',
        'Div', 'Referee'
    ] if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df

def save(df: pd.DataFrame, elo_dict: dict, league: str):
    folder = os.path.join(BASE_DIR, "data", league)

    save_path = os.path.join(folder,f"{league.replace('_', '')}_Match_Data_Ready_For_ML.csv")
    df.to_csv(save_path, index=False)

    with open (os.path.join(BASE_DIR, "data", league,"elo_ratings.json"), 'w') as f:
        json.dump(elo_dict, f)


def process_league(league: str):
    config = LEAGUES[league]
    df = load_raw_data(league)
    df = engineer_features(df)
    df, elo_dict = calculate_elo(df, base_elo=config["base_elo"], hfa=config["hfa"])
    df = impute_missing_features(df)
    save(df, elo_dict, league)

if __name__ == "__main__":
    for league in LEAGUES:
        process_league(league)