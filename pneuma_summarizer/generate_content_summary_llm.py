import os
import sys
import json
import pandas as pd
import torch

sys.path.append("..")

from tqdm import tqdm
from benchmark_generator.context.utils.pipeline_initializer import initialize_pipeline
from benchmark_generator.context.utils.prompting_interface import prompt_pipeline

# Select GPU (if necessary)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import setproctitle
setproctitle.setproctitle("python")

pipe = initialize_pipeline("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16)


def write_jsonl(data: list[dict[str, str]], file_path: str):
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item))
            file.write("\n")


def get_col_description_prompt(columns: str, column: str):
    return f"""A table has the following columns:
/*
{columns}
*/
Describe briefly what the {column} column represents. If not possible, simply state "No description.\""""


def generate_descriptions(src_path: str, descriptions_path: str):
    tables = sorted([file[:-4] for file in os.listdir(src_path)])
    summaries: list[dict[str, str]] = []

    bar = tqdm(tables)
    for table in bar:
        bar.set_description(f"Processing table {table}")
        df = pd.read_csv(f"{src_path}/{table}.csv")
        cols = df.columns
        for col in cols:
            prompt = get_col_description_prompt(" | ".join(cols), col)
            description = prompt_pipeline(
                pipe,
                [[{"role": "user", "content": prompt}]],
                temperature=None,
                top_p=None,
                max_new_tokens=400,
            )[0][-1]["content"]

            new_row = {
                "table": table,
                "summary": f"{col}: {description}",
            }
            summaries.append(new_row)
            write_jsonl(summaries, f"{descriptions_path}.jsonl")


# Adjust paths
src_path = "../data_src/tables/pneuma_public_bi"
descriptions_path = "public_narrations"
generate_descriptions(src_path, descriptions_path)
