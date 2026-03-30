import re
import json
import textwrap
import anthropic
import random
from dotenv import load_dotenv
import os
from scraper import get_teams_from_csv
from predict import BASE_DIR
from collections import Counter
import time

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

DATA_PATH_CS = os.path.join(BASE_DIR, "data", "Premier_League", "Not_Merged", "E0_25_26_LIVE.csv")
TEAMS = get_teams_from_csv(DATA_PATH_CS)

def parse_scenarios(md_path: str) -> list[dict]:
    with open(md_path, encoding="utf-8") as f:
        content = f.read()

    scenarios = []
    blocks = content.split("---")

    for block in blocks:
        name_match = re.search(r'## \d+\. (.+)', block)
        morale_match = re.search(r'\*\*Morale:\*\* (\d+)[–-](\d+)/10', block)
        headlines = re.findall(r'- "(.+)"', block)

        if name_match and morale_match and headlines:
            scenarios.append({
                "name": name_match.group(1).strip(),
                "morale_range": (
                    int(morale_match.group(1)),
                    int(morale_match.group(2))
                ),
                "example_headlines": headlines
            })
    return scenarios

def generate_example(scenario: dict, team: str, max_retries: int = 2) -> dict | None:
    morale_min, morale_max = scenario["morale_range"]
    example_headlines = "\n".join(f'- "{h}"' for h in scenario["example_headlines"])

    prompt = textwrap.dedent(f"""\
You are a football data labeler generating training data for a morale prediction model.

Scenario: {scenario['name']}
Team: {team}

Here are example headlines showing the correct style for this scenario:
{example_headlines}

Generate 5 NEW and DIFFERENT headlines about {team} that reflect this scenario.
Headlines must look like real BBC Sport, Sky Sports or The Guardian headlines.
Do NOT copy the example headlines - generate original ones.
Use generic terms instead of specific player names to avoid factual errors.
For example: "key forward", "first-choice midfielder", "starting goalkeeper".
Only use player names if you are 100% sure they play that position for {team}.
Pick a score randomly within the allowed range - do not always choose the minimum or maximum.
    
Then assign a morale score between {morale_min} and {morale_max} out of 10.

Respond in this EXACT format with no other text:
HEADLINES:
- [headline 1]
- [headline 2]
- [headline 3]
- [headline 4]
- [headline 5]
SCORE: [single integer between {morale_min} and {morale_max}]\
""").strip()
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )

            response = message.content[0].text.strip()
            lines = response.split("\n")

            headlines = []
            score = None
            for line in lines:
                line = line.strip()
                if line.startswith("- "):
                    headlines.append(line[2:].strip('"').strip('\\"'))
                elif line.startswith("SCORE:"):
                    try:
                        score = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass

            if len(headlines) == 5 and score is not None and morale_min <= score <= morale_max:
                return {
                    "team": team,
                    "scenario": scenario["name"],
                    "headlines": headlines,
                    "morale_score": score,
                }
            print(f"Attempt {attempt + 1}, failed, retrying")

        except Exception as e:
            print(f"API error attempt {attempt + 1}: {e}")

    return None

def generate_dataset(target: int = 1000):
    output_path = os.path.join(BASE_DIR, "data", "morale_dataset.json")
    md_path = os.path.join(BASE_DIR, "data", "50_premier_league_scenarios.md")
    scenarios = parse_scenarios(md_path)

    print(f"Found {len(scenarios)} scenarios.")
    if not scenarios:
        print("No scenarios found. Exiting.")
        return

    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} existing examples, need {target - len(dataset)} more.")
    else:
        dataset = []
    errors = 0
    examples_per_scenario = max(1, target // len(scenarios))
    scenario_counts = Counter(d["scenario"] for d in dataset)

    print(f"Target: {examples_per_scenario} per scenario ({target} total)")

    for i, scenario in enumerate(scenarios):
        already = scenario_counts.get(scenario["name"],0)
        need = examples_per_scenario - already

        if need <= 0:
            print(f"{i + 1}/{len(scenarios)}: {scenario['name']} - complete ({already})")
            continue

        print(f"{i+1}/{len(scenarios)}: {scenario['name']} - generating {need} more (have {already})")
        for j in range(need):
            team = random.choice(TEAMS)
            result = generate_example(scenario, team)
            if result:
                dataset.append(result)
                if len(dataset) % 50 == 0:
                    print(f" Saved {len(dataset)} examples.")
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(dataset, f, ensure_ascii=False, indent=2)
            else:
                errors += 1
                print(" Response parsing failed")
            time.sleep(2.0)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Generated {len(dataset)} examples, with {errors} errors.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_dataset(target=1000)
