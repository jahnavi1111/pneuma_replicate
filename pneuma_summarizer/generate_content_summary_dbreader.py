import os
import sys

sys.path.append("..")

from tqdm import tqdm
from sqlalchemy import create_engine
from llama_index.readers.database import DatabaseReader
from benchmark_generator.context.utils.jsonl import write_jsonl


tables_path = "../data_src/tables/pneuma_public_bi"
duckdb_filename = "public"
summaries_name = f"{duckdb_filename}_dbreader.jsonl"
content_summaries: list[dict[str, str]] = []

engine = create_engine(f"duckdb:///../other_systems/llama-index-RAG/{duckdb_filename}.duckdb")
db_reader = DatabaseReader(engine)


for table in tqdm(sorted([table[:-4] for table in os.listdir(tables_path)])):
    documents = db_reader.load_data(f"select * from {table}")
    for i in range(len(documents)):
        content_summaries.append({"table": table, "summary": documents[i].text})

write_jsonl(content_summaries, f"summaries/dbreader/{summaries_name}")
