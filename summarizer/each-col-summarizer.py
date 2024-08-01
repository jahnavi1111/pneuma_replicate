# Select GPU (if necessary)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import setproctitle

setproctitle.setproctitle("python")

import pandas as pd
import torch
import sys

sys.path.append("..")

from tqdm import tqdm
from benchmark_generator.context.utils.pipeline_initializer import initialize_pipeline
from benchmark_generator.context.utils.prompting_interface import prompt_pipeline

pipe = initialize_pipeline("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16, "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")


def get_col_description_prompt(columns: str, column: str):
    prompt = f"""A dataset has the following columns:
/*
{columns}
*/
Describe briefly what the {column} column represents. If not possible, simply state "No description.\""""

    return prompt


def generate_descriptions(src_path: str, descriptions_path: str):
    tables = [file[:-4] for file in os.listdir(src_path)]
    tables.sort()
    cols_descriptions = pd.DataFrame(columns=["table", "description"])

    bar = tqdm(tables)
    for table in bar:
        bar.set_description(f"Processing table {table}")
        df = pd.read_csv(f"{src_path}/{table}.csv")
        cols = df.columns
        for col in cols:
            sample_size = min(len(df[col].dropna()), 5)
            values = df[col].dropna().sample(sample_size, random_state=42).to_list()
            prompt = get_col_description_prompt(" | ".join(cols), col)
            description = prompt_pipeline(
                pipe,
                [{"role": "user", "content": prompt}],
                temperature=None,
                top_p=None,
                max_new_tokens=400,
            )[-1]["content"]

            new_row = {
                "table": table,
                "description": f"{col}: {description}",
            }

            cols_descriptions = cols_descriptions._append(new_row, ignore_index=True)
            cols_descriptions.to_csv(f"{descriptions_path}.csv", index=False)


# Adjust paths
src_path = "../data_src/tables/public_bi_benchmark"
descriptions_path = "public_cols"
generate_descriptions(src_path, descriptions_path)
