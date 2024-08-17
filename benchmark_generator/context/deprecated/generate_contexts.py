import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import setproctitle
setproctitle.setproctitle("python")

import pandas as pd
import torch

from utils.pipeline_initializer import initialize_pipeline
from utils.prompting_interface import prompt_pipeline
from utils.csv_data_source import CsvDataSource
from tqdm import tqdm

pipe = initialize_pipeline("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16)
# Specific setting for Llama-3-8B-Instruct for batching
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
pipe.tokenizer.padding_side = 'left'

with open("questions.txt") as file:
    questions = [question.strip() for question in file.readlines()]


def get_generative_prompt(table: str, question: str, num_of_rows: int):
    return f"""Given a dataset, consisting of {num_of_rows} rows, with the following columns:
/*
{table}
*/
and this question:
/*
{question}
*/
Assume you are the creator of the table and have all the necessary information to respond to the question. Generate a concise answer to the question based on the table, satisfying the following criteria:
1. Completeness: The answer must definitively and comprehensively address all parts of the question.
2. Relevance: The answer must directly provide the information requested in the question without any extraneous details."""


def generate_contexts(benchmark_name: str, data_src: str, generation_params={}):
    csv_data_source = CsvDataSource(data_src)  # Adjust the csv source names
    for table in tqdm(iter(csv_data_source), desc="Processing tables"):
        print(f"Processing table: {table}")
        try:
            benchmark = pd.read_csv(benchmark_name)
        except FileNotFoundError:
            benchmark = pd.DataFrame(columns=["id", "table", "context_question", "context"])
        csv_file_name = table[0]

        conversations = []
        for i in tqdm(range(len(questions)), desc="Iterating questions"):
            values = {"table": [csv_file_name[:-4]], "context_question": [questions[i]]}

            # Skip if this combination of table and context question is answered already
            if benchmark[["table", "context_question"]].isin(values).all(axis=1).any():
                continue

            prompt = get_generative_prompt(table[1], questions[i], table[2])
            conversations.append([{"role": "user", "content": prompt}])
        
        if (len(conversations) > 0):
            outputs = prompt_pipeline(
                pipe, conversations, batch_size=20, context_length=8192, top_p=None, temperature=None, **generation_params
            )

            for output_idx, output in enumerate(outputs):
                answer = output[-1]["content"]
                row = pd.DataFrame(
                    {
                        "id": [f"{csv_file_name[:-4]}_{output_idx}"],
                        "table": [csv_file_name[:-4]],
                        "context_question": [questions[output_idx]],
                        "context": [answer],
                    }
                )
                benchmark = pd.concat([benchmark, row], ignore_index=True)
                benchmark.to_csv(benchmark_name, index=False)


name = "contexts_testing_adventure.csv"  # Adjust the contexts name
generate_contexts(
    name,
    "pneuma_adventure_works",  # Adjust data source path
)
