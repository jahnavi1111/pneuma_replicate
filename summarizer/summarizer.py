import pandas as pd
import torch
import warnings
import os

from tqdm import tqdm

# These packages are available in benchmark_generator directory.
from pipeline.pipeline_initializer import initialize_pipeline
from pipeline.prompting_interface import prompt_pipeline


warnings.filterwarnings("ignore")
pipe = initialize_pipeline("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16)


def get_row_info_from_df(df: pd.DataFrame, row_idx=0):
    col = "col: " + " | ".join(df.columns)
    row = "row: " + " | ".join(df.iloc[row_idx].astype(str).str.strip())
    return col + "\n" + row


def get_prompt_row_summary(row_info: str):
    return f"""Given this row of a dataset:
/*
{row_info}
*/
Summarize it comprehensively into a single paragraph without adding any external information."""

name = "row_summaries_public_bi.csv"  # Adjust file name

def generate_row_summaries(src_path: str, row_summaries: pd.DataFrame):
    # Suppose row_summaries has 3 columns: 'table', 'row_idx', and 'summary'
    for table in os.listdir(src_path):
        print(f"Summarizing table: {table}")
        df = pd.read_csv(f"{src_path}/{table}")

        print("Summarizing the rows")
        for i in tqdm(range(len(df))):
            values = {"table": [table[:-4]], "row_idx": [i]}
            if row_summaries[["table", "row_idx"]].isin(values).all(axis=1).any():
                continue

            row_info = get_row_info_from_df(df, i)
            prompt = get_prompt_row_summary(row_info)
            row_summary = prompt_pipeline(
                pipe,
                [{"role": "user", "content": prompt}],
            )[
                -1
            ]["content"]
            new_row = {
                "table": table[:-4],
                "row_idx": i,
                "summary": row_summary,
            }
            row_summaries = row_summaries._append(new_row, ignore_index=True)
            row_summaries.to_csv(name, index=False)  # Checkpoint


row_summaries = pd.read_csv(name)
generate_row_summaries("public_bi_benchmark", row_summaries)
