# Import necessary packages
import pandas as pd
import torch

from pipeline.pipeline_initializer import initialize_pipeline
from pipeline.prompting_interface import prompt_pipeline

pipe = initialize_pipeline("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16)

def get_rephrase_prompt(metadata: str):
    return f"""Metadata:"{metadata}"
The metadata above desribes a dataset. Please create a question that asks for a dataset based on this description. Response in the following format:
Question: ..."""

name = "contexts_public_bi.csv"  # Adjust metadata sources
context_sources = pd.read_csv(name)

benchmark_name = "BX1_public_bi.csv"  # Adjust benchmark name
bx1 = pd.DataFrame(columns=["context","question","table"])

from tqdm import tqdm
for i in tqdm(range(len(context_sources))):
    table = context_sources["table"][i]
    context = context_sources["context"][i]

    prompt = get_rephrase_prompt(context)
    conversation = [{"role": "user", "content": prompt}]
    model_output = prompt_pipeline(pipe, conversation)[-1]["content"].split("Question: ")[-1]
    new_row = pd.DataFrame({
        "context": [context],
        "question": [model_output],
        "table": [table]
    })
    bx1 = pd.concat([bx1, new_row], ignore_index=True)
    bx1.to_csv(benchmark_name, index=False)
