import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Current live season — auto-calculated from date
_now = datetime.now()
_season_start = _now.year if _now.month >= 7 else _now.year - 1
LIVE_SEASON = str(_season_start)[-2:] + str(_season_start + 1)[-2:]

# All completed historical seasons from 2000/01 up to (but not including) live season
SEASONS = [str(y).zfill(2) + str(y + 1).zfill(2) for y in range(0, _season_start - 2000)]

LEAGUES = {
    "Premier_League": {"code": "E0",  "base_elo": 1600, "hfa": 60},
    "La_Liga":        {"code": "SP1", "base_elo": 1580, "hfa": 75},
    "Bundesliga":     {"code": "D1",  "base_elo": 1560, "hfa": 55},
    "Serie_A":        {"code": "I1",  "base_elo": 1550, "hfa": 65},
    "Ligue_1":        {"code": "F1",  "base_elo": 1520, "hfa": 70},
}

CORE_COLUMNS = [
    'Div', 'Date', 'HomeTeam', 'AwayTeam',
    'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
    'Referee', 'HS', 'AS', 'HST', 'AST',
    'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR'
]

# Team name (from CSV) → full name for news search
TEAM_SEARCH_NAMES = {
    # Premier League
    "Man City":       "Manchester City",
    "Man United":     "Manchester United",
    "Newcastle":      "Newcastle United",
    "Nott'm Forest":  "Nottingham Forest",
    "West Ham":       "West Ham United",
    "Leeds":          "Leeds United",
    "Wolves":         "Wolverhampton Wanderers",
    "Tottenham":      "Tottenham Hotspur",
    # La Liga
    "Ath Madrid":     "Atletico Madrid",
    "Ath Bilbao":     "Athletic Bilbao",
    "Sociedad":       "Real Sociedad",
    "Espanol":        "Espanyol",
    "Betis":          "Real Betis",
    "Vallecano":      "Rayo Vallecano",
    "Celta":          "Celta Vigo",
    "Alaves":         "Deportivo Alaves",
    "Oviedo":         "Real Oviedo",
    # Bundesliga
    "Ein Frankfurt":  "Eintracht Frankfurt",
    "M'gladbach":     "Borussia Monchengladbach",
    "Leverkusen":     "Bayer Leverkusen",
    "Dortmund":       "Borussia Dortmund",
    "FC Koln":        "Cologne",
    "Hamburg":        "Hamburger SV",
    "Mainz":          "Mainz 05",
    "Stuttgart":      "VfB Stuttgart",
    "St Pauli":       "St Pauli",
    # Serie A
    "Inter":          "Inter Milan",
    "Milan":          "AC Milan",
    "Roma":           "AS Roma",
    "Verona":         "Hellas Verona",
    "Como":           "Como 1907",
    "Pisa":           "Pisa",
    # Ligue 1
    "Paris SG":       "Paris Saint-Germain",
    "Marseille":      "Olympique Marseille",
    "Lyon":           "Olympique Lyon",
    "Monaco":         "AS Monaco",
    "Nice":           "OGC Nice",
}
