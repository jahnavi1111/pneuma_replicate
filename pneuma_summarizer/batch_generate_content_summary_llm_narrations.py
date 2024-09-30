import os
import setproctitle
setproctitle.setproctitle("/opt/conda/bin/python3.8")

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import torch
import pandas as pd

sys.path.append("..")

from tqdm import tqdm
from collections import defaultdict
from benchmark_generator.context.utils.pipeline_initializer import initialize_pipeline
from benchmark_generator.context.utils.prompting_interface import prompt_pipeline
from benchmark_generator.context.utils.jsonl import write_jsonl, read_jsonl

pipe = initialize_pipeline("../models/llama", torch.bfloat16)
# Specific setting for Llama-3-8B-Instruct for batching
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
pipe.tokenizer.padding_side = "left"


def get_col_description_prompt(columns: str, column: str):
    return f"""A table has the following columns:
/*
{columns}
*/
Describe briefly what the {column} column represents. If not possible, simply state "No description.\""""

def is_fit_in_memory(conversations, batch_size: int):
    output = prompt_pipeline(
        pipe,
        conversations[:batch_size],
        batch_size=batch_size,
        context_length=8192,
        max_new_tokens=1,
        temperature=None,
        top_p=None,
    )
    if output[0][0]["content"] == "":
        return False
    else:
        return True

def get_optimal_batch_size(conversations):
    print(f"Looking for optimal batch size")
    max_batch_size = 50  # Change to a higher value if you have more capacity to explore batch size
    min_batch_size = 1
    while min_batch_size < max_batch_size:
        mid_batch_size = (min_batch_size + max_batch_size) // 2
        print(f"Current mid batch size: {mid_batch_size}")
        if is_fit_in_memory(conversations, mid_batch_size):
            min_batch_size = mid_batch_size + 1
        else:
            max_batch_size = mid_batch_size - 1
    optimal_batch_size = min_batch_size
    print(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size

def generate_llm_narration_summaries(src_path: str, descriptions_path: str):
    tables = sorted([file[:-4] for file in os.listdir(src_path)])
    try:
        summaries = read_jsonl(descriptions_path)
    except:
        summaries: list[dict[str, str]] = []

    conversations = []
    conv_tables = []
    conv_cols = []
    for table in tqdm(tables):
        # Skip already summarized tables
        if table in [summary["table"] for summary in summaries]:
            continue
        df = pd.read_csv(f"{src_path}/{table}.csv", on_bad_lines="skip")
        cols = df.columns
        for col in cols:
            prompt = get_col_description_prompt(" | ".join(cols), col)
            conversations.append([{"role": "user", "content": prompt}])
            conv_tables.append(table)
            conv_cols.append(col)

    if len(conversations) > 0:
        optimal_batch_size = get_optimal_batch_size(conversations)
        outputs = []
        for i in tqdm(range(0, len(conversations), optimal_batch_size)):
            outputs += prompt_pipeline(
                pipe,
                conversations[i:i+optimal_batch_size],
                batch_size=optimal_batch_size,
                context_length=8192,
                max_new_tokens=400,
                temperature=None,
                top_p=None,
            )

        col_narrations: dict[str,list[str]] = defaultdict(list)
        for output_idx, output in enumerate(outputs):
            col_narrations[conv_tables[output_idx]] += [f"{conv_cols[output_idx]}: {output[-1]["content"]}"]

        for table in tables:
            summaries.append({
                "id": f"{table}_SEP_contents_SEP_schema",
                "table": table,
                "summary": " | ".join(col_narrations[table]),
            })
            write_jsonl(summaries, descriptions_path)

if __name__ == "__main__":
    start = time.time()
    src_path = "../data_src/tables/pneuma_chembl_10K"
    descriptions_path = "chembl.jsonl"
    generate_llm_narration_summaries(src_path, descriptions_path)
    end = time.time()
    print(f"Total time: {end - start} seconds")

    start = time.time()
    src_path = "../data_src/tables/pneuma_adventure_works"
    descriptions_path = "adventure.jsonl"
    generate_llm_narration_summaries(src_path, descriptions_path)
    end = time.time()
    print(f"Total time: {end - start} seconds")

    start = time.time()
    src_path = "../data_src/tables/pneuma_public_bi"
    descriptions_path = "public.jsonl"
    generate_llm_narration_summaries(src_path, descriptions_path)
    end = time.time()
    print(f"Total time: {end - start} seconds")

    start = time.time()
    src_path = "../data_src/tables/pneuma_chicago_10K"
    descriptions_path = "chicago.jsonl"
    generate_llm_narration_summaries(src_path, descriptions_path)
    end = time.time()
    print(f"Total time: {end - start} seconds")

    start = time.time()
    src_path = "../data_src/tables/pneuma_fetaqa"
    descriptions_path = "fetaqa.jsonl"
    generate_llm_narration_summaries(src_path, descriptions_path)
    end = time.time()
    print(f"Total time: {end - start} seconds")
