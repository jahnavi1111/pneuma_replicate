import os
import json
import duckdb

from tqdm import tqdm


def write_jsonl(data: list[dict[str, str]], file_path: str):
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item))
            file.write("\n")


con = duckdb.connect()
tables_path = "../data_src/tables/pneuma_public_bi"
content_summaries: list[dict[str, str]] = []
tables = [
    table[:-4] for table in sorted(os.listdir(tables_path)) if table.endswith(".csv")
]

for table in tqdm(tables, "processing tables"):
    df = (
        con.sql(f"select * from '{tables_path}/{table}.csv'")
        .to_df()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    summary = {
        "table": table,
        "summary": " | ".join(df.columns),
    }
    content_summaries.append(summary)

write_jsonl(content_summaries, "summaries/standard/public_standard.jsonl")
