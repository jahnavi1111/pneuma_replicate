# Import necessary packages
import pandas as pd
import torch
import random

# These packages are available in the benchmark_generator directory
from pipeline.pipeline_initializer import initialize_pipeline
from pipeline.prompting_interface import prompt_pipeline

pipe = initialize_pipeline("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16)


def get_row_representation(df: pd.DataFrame, row_idx=0):
    col = "col: " + " | ".join(df.columns)
    row = "row: " + " | ".join(df.iloc[row_idx].astype(str).str.strip())
    return col + "\n" + row


def get_random_row_indices(num_of_rows: int, num_of_indices: int):
    random.seed(42)
    return random.sample(range(num_of_rows), num_of_indices)


def get_patron_prompt(dataset_name: str, row_representation: str, question: str):
    """Return the prompt string for LLM-Patron"""
    return f"""Given this sample row of a dataset named {dataset_name}:
/*
{row_representation}
*/
and this question:
/*
{question}
*/
Argue how dataset {dataset_name}, through its sample row, can answer the question. Provide your response in the following format:
- Label: [yes/no]
- Argument: [Brief argument]

Argue briefly how dataset {dataset_name}, through its sample row, can answer the question. Begin your argument with "Yes" or "No" depending on whether you think this dataset can answer the question.
Do not forget to include the dataset name explicitly in your response."""


benchmark = pd.read_csv("BC1_chicago.csv")


def convert_argument_to_dict(argument: str):
    lines = argument.strip().split("\n")
    d = {}

    for line in lines:
        key, value = line.split(": ", 1)
        key = key[2:].strip()
        d[key] = value.strip()
    return d


def evaluate_benchmark(benchmark: pd.DataFrame, datasets_path: str, rows_considered=20):
    datasets = [
        f"{datasets_path}/{dataset_name}"
        for dataset_name in os.listdir(datasets_path)
        if dataset_name.endswith(".csv")
    ]
    accuracy_sum = 0
    for i in range(len(benchmark)):
        question = benchmark["question"][i]
        expected_dataset = f"{datasets_path}/{benchmark['table'][i]}.csv"
        relevant_datasets = []
        for dataset in datasets:
            print(f"Checking dataset {dataset}")
            df = pd.read_csv(dataset)
            row_indices = get_random_row_indices(len(df), rows_considered)
            can_answer = False
            for row_index in row_indices:
                row_representation = get_row_representation(df, row_index)
                prompt = get_patron_prompt(dataset, row_representation, question)
                conversation = [{"role": "user", "content": prompt}]
                response = convert_argument_to_dict(
                    prompt_pipeline(pipe, conversation, max_new_tokens=100)[-1][
                        "content"
                    ]
                )
                print(response)
                if response["Label"].lower() == "yes":
                    can_answer = not can_answer
                    break
            if can_answer:
                relevant_datasets.append(dataset)
        if expected_dataset in relevant_datasets:
            accuracy_sum += 1
        print("DEBUG")
        print(relevant_datasets)
        break
    return {"accuracy": accuracy_sum / len(benchmark)}


import warnings

warnings.filterwarnings("ignore")

evaluate_benchmark(benchmark, "chicago_open_data", rows_considered=1)
