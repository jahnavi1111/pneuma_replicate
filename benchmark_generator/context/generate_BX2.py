# Import necessary packages
import pandas as pd
import torch

from pipeline.pipeline_initializer import initialize_pipeline
from pipeline.prompting_interface import prompt_pipeline
from tqdm import tqdm

pipe = initialize_pipeline("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16)


def get_rephrase_prompt(question: str):
    return f"""Question:"{question}"
Rephrase the question above using completely different words. Response in the following format:
Question: ..."""


bx1 = pd.read_csv("BC1_chicago.csv")  # Adjust BX1 name
bx2 = pd.DataFrame(columns=["question", "table"])

bx2_name = "BC2_chicago.csv"  # Adjust expected BX2 name

for i in tqdm(range(len(bx1))):
    # context = bx1["context"][i]
    question = bx1["question"][i]
    table = bx1["table"][i]

    prompt = get_rephrase_prompt(question)
    conversation = [{"role": "user", "content": prompt}]
    model_output = prompt_pipeline(pipe, conversation)[-1]["content"].split(
        "Question: "
    )[-1]
    new_row = pd.DataFrame(
        {
            # "context": [context],
            "question": [model_output],
            "table": [table],
        }
    )
    bx2 = pd.concat([bx2, new_row], ignore_index=True)
    bx2.to_csv(bx2_name, index=False)
