import os
import json
import pandas as pd

from tqdm import tqdm


def write_jsonl(data: list[dict[str, str]], file_path: str):
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item))
            file.write("\n")


tables_path = "../data_src/tables/pneuma_fetaqa"
summaries_name = "fetaqa_standard.jsonl"
content_summaries: list[dict[str, str]] = []
tables = [
    table[:-4] for table in sorted(os.listdir(tables_path)) if table.endswith(".csv")
]

for table in tqdm(tables, "processing tables"):
    df = pd.read_csv(f"{tables_path}/{table}.csv")
    summary = {
        "table": table,
        "summary": " | ".join(df.columns),
    }
    content_summaries.append(summary)

write_jsonl(content_summaries, f"summaries/standard/{summaries_name}")
