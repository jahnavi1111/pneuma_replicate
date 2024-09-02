import os
import sys
import pandas as pd

sys.path.append("..")

from tqdm import tqdm
from benchmark_generator.context.utils.jsonl import write_jsonl


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
