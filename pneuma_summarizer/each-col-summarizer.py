# Select GPU (if necessary)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import setproctitle
# setproctitle.setproctitle("python")

import pandas as pd
import torch
import sys

sys.path.append("..")

from tqdm import tqdm
from benchmark_generator.context.utils.pipeline_initializer import initialize_pipeline
from benchmark_generator.context.utils.prompting_interface import prompt_pipeline

pipe = initialize_pipeline("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16)


def get_col_description_prompt(columns: str, column: str):
    return f"""A table has the following columns:
/*
{columns}
*/
Describe briefly what the {column} column represents. If not possible, simply state "No description.\""""


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
            prompt = get_col_description_prompt(" | ".join(cols), col)
            conversation = [{"role": "user", "content": prompt}]
            description = prompt_pipeline(
                pipe,
                conversation,
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
src_path = "public_bi_benchmark"
descriptions_path = "public_cols"
generate_descriptions(src_path, descriptions_path)
