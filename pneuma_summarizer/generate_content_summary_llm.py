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
# Specific setting for Llama-3-8B-Instruct for batching
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
pipe.tokenizer.padding_side = 'left'


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
        conversations = []
        for col in cols:
            prompt = get_col_description_prompt(" | ".join(cols), col)
            conversations.append([{"role": "user", "content": prompt}])
        
        for i in tqdm(range(0, len(conversations), 3)):
            outputs = prompt_pipeline(
                pipe, conversations[i:i+3], batch_size=3, context_length=8192, max_new_tokens=400, temperature=None, top_p=None
            )
            for output_idx, output in enumerate(outputs):
                summary = output[-1]["content"]
                new_row = {
                    "table": table,
                    "summary": f"{cols[i+output_idx]}: {summary}",
                }
                summaries.append(new_row)
                write_jsonl(summaries, f"{descriptions_path}")


# Adjust paths
src_path = "pneuma_fetaqa"
descriptions_path = "summaries/narrations/fetaqa_narrations.jsonl"
generate_descriptions(src_path, descriptions_path)
