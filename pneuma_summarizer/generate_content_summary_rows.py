import sys

sys.path.append("..")


import math
import os
import pandas as pd

from tqdm import tqdm
from benchmark_generator.context.utils.jsonl import write_jsonl

name = "fetaqa"
table_path = "../data_src/tables/pneuma_fetaqa"
summary_path = f"summaries/rows/{name}.jsonl"
tables = sorted(os.listdir(table_path))


content_summaries = []
for table_idx, table in enumerate(tqdm(tables)):
    df = pd.read_csv(f"{table_path}/{table}", on_bad_lines="skip")
    sample_size = math.ceil(min(len(df), 5))

    selected_df = df.sample(n=sample_size, random_state=table_idx).reset_index(
        drop=True
    )
    for row_idx, row in selected_df.iterrows():
        formatted_row = " | ".join([f"{col}: {val}" for col, val in row.items()])
        content_summaries.append(
            {
                "table": table[:-4],
                "summary": formatted_row,
            }
        )
write_jsonl(content_summaries, summary_path)
