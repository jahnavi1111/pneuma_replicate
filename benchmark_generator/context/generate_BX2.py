# Optional (only if we need to choose among multiple GPUs)
###########################################################
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import setproctitle
# setproctitle.setproctitle("python")
###########################################################
import pandas as pd
import torch

from utils.pipeline_initializer import initialize_pipeline
from utils.prompting_interface import prompt_pipeline
from tqdm import tqdm

pipe = initialize_pipeline("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16)


def get_rephrase_prompt(question: str):
    return f"""Original Question:"{question}"

Rephrase the above question with different wordings. Respond in the following format:
Rephrased Question: ..."""


bx1 = pd.read_csv("BX1_public_corrected.csv")  # Adjust BX1 name
bx2 = pd.DataFrame(columns=["context_id", "question", "answer_tables"])

bx2_name = "BX2_public.csv"  # Adjust BX2 name

for i in tqdm(range(len(bx1))):
    context_id = bx1["context_id"][i]
    question = bx1["question"][i]
    answer_tables = bx1["answer_tables"][i]

    prompt = get_rephrase_prompt(question)
    conversation = [{"role": "user", "content": prompt}]
    model_output = prompt_pipeline(
        pipe,
        conversation,
        temperature=None,
        top_p=None
    )[0][-1]["content"].split("Question: ")[-1]
    new_row = pd.DataFrame(
        {
            "context_id": [context_id],
            "question": [model_output],
            "answer_tables": [answer_tables],
        }
    )
    bx2 = pd.concat([bx2, new_row], ignore_index=True)
    bx2.to_csv(bx2_name, index=False)
