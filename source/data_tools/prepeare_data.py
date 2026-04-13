import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from prediction.predict import BASE_DIR

def prepeare_data(
    input_path: str = None,
    output_dir: str = None
):
    if input_path is None:
        input_path = os.path.join(BASE_DIR, "data", "morale_dataset.json")

    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "data", "finetuning")

    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, encoding= "utf-8") as f:
        dataset = json.load(f)

    rows = []

    for item in dataset:
        text = " .".join(item["headlines"])
        label = float(item["morale_score"])
        rows.append({"text": text, "label": label})

    df = pd.DataFrame(rows)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)




if __name__ == "__main__":
    prepeare_data()