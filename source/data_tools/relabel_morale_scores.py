import json
import random
import re
import os
import shutil
import collections
from config import BASE_DIR

DATASET_PATH = os.path.join(BASE_DIR, "data", "morale_dataset.json")
SCENARIOS_PATH = os.path.join(BASE_DIR, "data", "match_scenarios.md")
SEED = 42

def load_ranges(md_path: str) -> dict[str, tuple[int,int]]:
    with open(md_path, encoding="utf-8") as f:
        content = f.read()

    ranges = {}

    for block in content.split("---"):
        name_match = re.search(r'## \d+\. (.+)', block)
        range_match = re.search(r'\*\*Morale:\*\* (\d+)\D+(\d+)/10', block)
        if name_match and range_match:
            lo, hi = int(range_match.group(1)), int (range_match.group(2))
            ranges[name_match.group(1).strip()] = (lo, hi)
    return ranges


def relabel():
    ranges = load_ranges(SCENARIOS_PATH)
    print(f"Loaded {len(ranges)} scenario ranges.")

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} examples.")

    missing = sorted({d["scenario"] for d in dataset if d["scenario"] not in ranges})
    if missing:
        raise ValueError(f"Scenarios in dataset not found in .md (name mismatch?): {missing}")

    #backup once before overwriting
    backup = DATASET_PATH + ".bak"
    if not os.path.exists(backup):
        shutil.copy(DATASET_PATH, backup)
        print(f"Backup saved -> {backup}")

    groups = collections.defaultdict(list)
    for i, d in enumerate(dataset):
        groups[d["scenario"]].append(i)

    rng = random.Random(SEED)
    for scenario, idxs in groups.items():
        lo, hi = ranges[scenario]
        values = list(range(lo, hi +1))
        rng.shuffle(idxs)
        for pos, i in enumerate(idxs):
            dataset[i]["morale_score"] = values[pos % len(values)]

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    dist = collections.Counter(d["morale_score"] for d in dataset)
    print("New morale distribution:")
    for score in range(1,11):
        print(f"    {score:>2}: {dist.get(score, 0)}")

if __name__ == "__main__":
    relabel()