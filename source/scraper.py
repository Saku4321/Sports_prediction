import pandas as pd
import requests
from bs4 import BeautifulSoup

EXCLUDE_KEYWORDS = ["women", "woman", "ladies", "u21", "u23", "academy", "reserves"]


def get_teams_from_csv(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    return sorted(teams.tolist())

def get_news_for_team(team_name: str, n: int =5):
    url = f"https://news.google.com/rss/search?q={team_name}+men+football&hl=en&gl=GB&ceid=GB:en"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, features="xml")
    items = soup.find_all('item')[:n]
    results = []

    for item in items:
        title = item.find("title").text

        if any(kw in title.lower() for kw in EXCLUDE_KEYWORDS):
            continue

        results.append({
            "team": team_name,
            "title": item.title.text,
            "published": item.find("pubDate").text if item.find("pubDate") else None,
            "source": item.find("source").text if item.find("source") else None,
        })
        if len(results) == n:  # zatrzymaj gdy masz n wyników
            break
    return results
def get_news_for_match(home_team: str, away_team: str, n: int = 5):
    home_news = get_news_for_team(home_team, n)
    away_news = get_news_for_team(away_team, n)
    return pd.DataFrame(home_news + away_news)

if __name__ == "__main__":
    teams = get_teams_from_csv("../data/Premier_League/PremierLeague_Match_Data_Ready_For_ML.csv")
    print(f"Dostepne druzyny {teams}")
    df = get_news_for_match(teams[0],teams[1])
    print(df.to_string())