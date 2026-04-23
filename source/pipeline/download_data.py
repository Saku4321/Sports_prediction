import os
import urllib.request
from config import BASE_DIR, LEAGUES, SEASONS, LIVE_SEASON


def download_league(league_name: str, config: dict):
    folder = os.path.join(BASE_DIR, "data", league_name, "raw")
    os.makedirs(folder, exist_ok=True)
    code = config["code"]

    # Historical seasons — skip if already exists
    for i, season in enumerate(SEASONS, start=1):
        file_path = os.path.join(folder, f"{code}({i}).csv")
        if not os.path.exists(file_path):
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            try:
                urllib.request.urlretrieve(url, file_path)
                print(f"  [{league_name}] Downloaded {season}")
            except Exception as e:
                print(f"  [{league_name}] Skipping {season} — {e}")
        else:
            print(f"  [{league_name}] Skipping {season} — already exists")

    # Live file — always fresh
    live_path = os.path.join(folder, f"{code}_LIVE.csv")
    url = f"https://www.football-data.co.uk/mmz4281/{LIVE_SEASON}/{code}.csv"
    urllib.request.urlretrieve(url, live_path)
    print(f"  [{league_name}] Live file updated")


def download_all():
    for league_name, config in LEAGUES.items():
        print(f"\nDownloading {league_name}...")
        download_league(league_name, config)
    print("\nAll leagues downloaded.")


if __name__ == "__main__":
    download_all()