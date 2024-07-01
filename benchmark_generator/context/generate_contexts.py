# Optional (only if we need to choose among multiple GPUs)
###########################################################
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import setproctitle
# setproctitle.setproctitle("python")
###########################################################
import pandas as pd
import torch

from utils.pipeline_initializer import initialize_pipeline
from utils.prompting_interface import prompt_pipeline
from utils.csv_data_source import CsvDataSource

with open("questions.txt") as file:
    questions = [question.strip() for question in file.readlines()]


def get_generative_prompt(dataset: str, question: str):
    return f"""Given this dataset:
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


pipe = initialize_pipeline("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16)


def generate_contexts(benchmark_name: str, data_src: str, generation_params={}):
    csv_data_source = CsvDataSource(data_src)  # Adjust the csv source names
    for table in iter(csv_data_source):
        try:
            benchmark = pd.read_csv(benchmark_name)
        except FileNotFoundError:
            benchmark = pd.DataFrame(columns=["table", "context_question", "context"])
        csv_file_name = table[0]
        print(f"Processing table {csv_file_name}")
        dataset = "\n".join(table[1])
        for i in range(len(questions)):
            print(f"Processing question {i}")
            question = questions[i]
            prompt = get_generative_prompt(dataset, question)
            conversation = [{"role": "user", "content": prompt}]
            values = {"table": [csv_file_name[:-4]], "context_question": [question]}

            # Skip if this combination of table and context question is answered already
            if benchmark[["table", "context_question"]].isin(values).all(axis=1).any():
                continue

            answer = prompt_pipeline(pipe, conversation, **generation_params)[-1][
                "content"
            ]
            row = pd.DataFrame(
                {
                    "table": [csv_file_name[:-4]],
                    "context_question": [question],
                    "context": [answer],
                }
            )
            benchmark = pd.concat([benchmark, row], ignore_index=True)
            benchmark.to_csv(benchmark_name, index=False)


name = "contexts_public_bi.csv"  # Adjust the benchmark name
generate_contexts(
    name,
    "public_bi_benchmark",  # Adjust data source path
    {"temperature": None, "top_p": None}
)
