# Optional (only if we need to choose among multiple GPUs)
###########################################################
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import setproctitle
setproctitle.setproctitle("python")
###########################################################
import pandas as pd
import torch

from utils.pipeline_initializer import initialize_pipeline
from utils.prompting_interface import prompt_pipeline
from utils.csv_data_source import CsvDataSource
from tqdm import tqdm

with open("questions.txt") as file:
    questions = [question.strip() for question in file.readlines()]


def get_generative_prompt(dataset: str, question: str, num_of_rows: int):
    return f"""Given this dataset with its sample rows (out of {num_of_rows} rows):
*/
{dataset}
*/
and this question:
/*
{question}
*/
Assume you are the creator of the dataset and have all the necessary information to respond to the question. Generate a concise answer to the question based on the dataset, satisfying the following criteria:
1. Completeness: The answer must definitively and comprehensively address all parts of the question.
2. Relevance: The answer must directly provide the information requested in the question without any extraneous details."""


pipe = initialize_pipeline("mistralai/Mistral-Nemo-Instruct-2407", torch.bfloat16)


def generate_contexts(benchmark_name: str, data_src: str, generation_params={}):
    csv_data_source = CsvDataSource(data_src)  # Adjust the csv source names
    for table in tqdm(iter(csv_data_source), desc="Processing tables"):
        try:
            benchmark = pd.read_csv(benchmark_name)
        except FileNotFoundError:
            benchmark = pd.DataFrame(columns=["id", "table", "context_question", "context"])
        csv_file_name = table[0]
        dataset = "\n".join(table[1])
        num_of_rows = table[2]
        for i in tqdm(range(len(questions)), desc="Iterating questions"):
            question = questions[i]
            prompt = get_generative_prompt(dataset, question, num_of_rows)
            conversation = [{"role": "user", "content": prompt}]
            values = {"table": [csv_file_name[:-4]], "context_question": [question]}

            # Skip if this combination of table and context question is answered already
            if benchmark[["table", "context_question"]].isin(values).all(axis=1).any():
                continue

            answer = prompt_pipeline(
                pipe, conversation, context_length=128000, **generation_params
            )[-1]["content"]
            row = pd.DataFrame(
                {
                    "id": [f"{csv_file_name[:-4]}_{i}"],
                    "table": [csv_file_name[:-4]],
                    "context_question": [question],
                    "context": [answer],
                }
            )
            benchmark = pd.concat([benchmark, row], ignore_index=True)
            benchmark.to_csv(benchmark_name, index=False)


name = "contexts_chicago.csv"  # Adjust the contexts name
generate_contexts(
    name,
    "pneuma_chicago_10K",  # Adjust data source path
)
