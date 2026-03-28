import re
import anthropic
import os
from scraper import get_teams_from_csv
from predict import BASE_DIR

DATA_PATH_CS = os.path.join(BASE_DIR, "data", "Premier_League", "Not_Merged", "E0_25_26_LIVE.csv")
TEAMS = get_teams_from_csv(DATA_PATH_CS)

def parse_scenarios(md_path: str) -> list[dict]:
    with open(md_path, encoding = "utf-8") as f:
        content = f.read()

    scenarios = []
    blocks = content.split("---")

    for block in blocks:
        name_match = re.search(r'## \d+\. (.+)', block)

        morale_match =  re.search(r'\*\*Morale:\*\* (\d+)[–-](\d+)/10', block)

        headlines = re.findall(r'- "(.+)"', block)

        if name_match and morale_match and headlines:
            scenarios.append({
                "name": name_match.group(1).strip(),
                "morale_range":(
                    int(morale_match.group(1)),
                    int(morale_match.group(2))
                ),
                "example_headlines": headlines

            })

    return scenarios



if __name__ == "__main__":
    scenarios = parse_scenarios("../data/50_premier_league_scenarios.md")
    for scenario in scenarios:
        print(scenario)
