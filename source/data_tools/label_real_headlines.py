import os
import json
import anthropic
import time
from config import BASE_DIR
from dotenv import load_dotenv

def group_headlines(chunk_size: int = 5, min_size: int =3) -> list[dict]:
    input_path = os.path.join(BASE_DIR, "data", "gdelt_headlines.json")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for team, headlines in data.items():
        for i in range(0, len(headlines), chunk_size):
            chunk = headlines[i:i + chunk_size]
            if len(chunk) < min_size:
                continue
            examples.append({"team": team, "headlines": chunk})

    print(f"From {len(data)} teams, made {len(examples)} examples.")
    return examples

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def label_example(example: dict, max_retries: int = 3) -> dict | None:
    team = example["team"]
    headlines = example["headlines"]
    headlines_text = "\n".join(f" - {h}" for h in headlines)

    prompt = f"""You are a football analyst. Based on the following recent news headlines,
      rate the current morale and situation of {team} on a scale from 1 to 10.

      Judge morale STRICTLY from {team}'s perspective. The same match looks different
      from each side - a heavy win is high morale for the winner but low morale for the loser.

      Some headlines may not be about {team} at all (they may mention other clubs).
      Ignore those. Base your score only on headlines that are actually about {team}.

      If NONE of the headlines are about {team}, do not guess - respond with exactly:
      SCORE: NA

      Otherwise respond in this EXACT format with no other text:
      SCORE: [single integer 1-10]
      REASONING: [1-2 sentences in English]

      Headlines:
      {headlines_text}"""
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model = "claude-haiku-4-5",
                max_tokens = 218,
                messages = [{"role": "user", "content": prompt}],
            )
            response = "".join(b.text for b in message.content if hasattr(b, "text"))
            lines = response.strip().split("\n")

            score = None
            for line in lines:
                if line.strip().startswith("SCORE:"):
                    val = line.split(":", 1)[1].strip()
                    if val.upper() == "NA":
                        return None
                    try:
                        score = int(val)
                    except ValueError:
                        pass
            if score is not None and 1 <=score <= 10:
                return {
                    "team": team,
                    "headlines": headlines,
                    "morale_score": score,
                }
            print(f"Attempt {attempt + 1}, bad parse, retrying")

        except Exception as e:
            print(f"    {team} API error attempt {attempt + 1}: {e}")

    return None

def build_dataset():
    output_path = os.path.join(BASE_DIR, "data", "real_morale_dataset.json")
    examples = group_headlines()

    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} existing examples.")
    else:
        dataset = []

    done = {(d["team"], tuple(d["headlines"])) for d in dataset}
    errors = 0

    for i, ex in enumerate(examples):
        if (ex["team"], tuple(ex["headlines"])) in done:
            continue

        result = label_example(ex)
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
    build_dataset()