import os
import json
import requests
import time
from config import TEAM_SEARCH_NAMES, BASE_DIR, LEAGUES
from data_tools.scraper import EXCLUDE_KEYWORDS, get_teams_from_csv

TEAM_GDELT_ALIASES = {
    "Lyon": ["Lyon"],
    "Marseille":    ["Marseille"],
    "Mainz":    ["Mainz"],
    "M'gladbach":   ["Monchengladbach", "Borussia Monchengladbach", "Gladbach"],
    "Hamburg":  ["Hamburger SV", "Hamburg", "HSV"],
    "Verona":   ["Hellas Verona", "Verona"],
    "Werder Bremen":    ["Werder Bremen", "Werder"],
    "Como": ["Como 1907", "Como"],
    "Oviedo":   ["Real Oviedo", "Oviedo"],
    "Alaves":   ["Deportivo Alaves", "Alaves"],
}
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def get_gdelt_headlines(team_name: str, cap: int = 80) -> list[str] | None:

    if team_name in TEAM_GDELT_ALIASES:
        names = TEAM_GDELT_ALIASES[team_name]
    else:
        names = [TEAM_SEARCH_NAMES.get(team_name, team_name)]

    seen = set()
    headlines = []
    any_success = False

    for search_name in names:
        if len(headlines) >= cap:
            break
        if " " in search_name or "-" in search_name:
            query = f'"{search_name}" football sourcelang:english'
        else:
            query = f"{search_name} football sourcelang:english"
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": 250,
        }

        articles = None
        for attempt in range(20):
            try:
                r = requests.get(GDELT_URL, params=params, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 429 or "Please limit requests" in r.text:
                    print(f"    {team_name}: rate-limit - waiting 10s (attempt {attempt + 1})")
                    time.sleep(10)
                    continue
                articles = r.json().get("articles", [])
                break
            except Exception as e:
                print(f"    {team_name}: trying {attempt +1} time ({type(e).__name__}) - trying again in 10s")
                time.sleep(10)

        if articles is None:
            continue
        any_success = True
        for a in articles:
            title = (a.get("title") or "").strip()
            if not title:
                continue
            low = title.lower()
            if any(kw in low for kw in EXCLUDE_KEYWORDS):
                continue
            if low in seen:
                continue
            seen.add(low)
            headlines.append(title)
    if not any_success:
        return None
    return headlines[:cap]

def load_all_teams() -> list[str]:
    teams =[]
    for league, cfg in LEAGUES.items():
        path = os.path.join(BASE_DIR, "data", league, "raw", f"{cfg['code']}_LIVE.csv")
        teams.extend(get_teams_from_csv(path))
    return sorted(set(teams))

def collect_all_headlines(per_team_cap: int = 80):
    output_path = os.path.join(BASE_DIR, "data", "gdelt_headlines.json")
    teams = load_all_teams()

    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} existing headlines.")
    else:
        data = {}

    for i, team in enumerate(teams):
        if team in data:
            print(f"{i + 1} / {len(teams)}: {team} - already collected ({len(data[team])}), skipping")
            continue

        headlines = get_gdelt_headlines(team, cap=per_team_cap)
        if headlines is None:
            print(f"{i + 1} / {len(teams)}: {team} - CONNECTION ERROR, skipping")
            continue

        data[team] = headlines
        print(f"{i + 1} / {len(teams)}: {team} - collected ({len(headlines)})")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        time.sleep(10)

    total = sum(len(v) for v in data.values())
    print(f"\n Finished! {len(data)} teams, {total} headlines total.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    collect_all_headlines()