import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import setproctitle
setproctitle.setproctitle("python")

import pandas as pd
import torch

from utils.pipeline_initializer import initialize_pipeline
from utils.prompting_interface import prompt_pipeline
from tqdm import tqdm

pipe = initialize_pipeline("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16)
# Specific setting for Llama-3-8B-Instruct for batching
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
pipe.tokenizer.padding_side = "left"

def get_rephrase_prompt(question: str):
    return f"""Original Question:"{question}"

Rephrase the above question with different wordings. Respond in the following format:
Rephrased Question: ..."""


bx1 = pd.read_csv("BX1_adventure_corrected.csv")  # Adjust BX1 name
bx2 = pd.DataFrame(columns=["context_id", "question", "answer_tables"])

bx2_name = "BX2_adventure.csv"  # Adjust BX2 name

context_ids = []
answer_tables_aggr = []
conversations = []
for i in tqdm(range(len(bx1))):
    context_id = bx1["context_id"][i]
    question = bx1["question"][i]
    answer_tables = bx1["answer_tables"][i]

    prompt = get_rephrase_prompt(question)
    conversations.append([{"role": "user", "content": prompt}])
    context_ids.append(context_id)
    answer_tables_aggr.append(answer_tables)

for i in tqdm(range(0, len(conversations), 3)):
    outputs = prompt_pipeline(
        pipe, conversations[i:i+3], batch_size=3, context_length=8192, top_p=None, temperature=None
    )

    for output_idx, output in enumerate(outputs):
        new_question = output[-1]["content"].split("Question: ")[-1]
        new_row = pd.DataFrame(
            {
                "context_id": [context_ids[i+output_idx]],
                "question": [new_question],
                "answer_tables": [answer_tables_aggr[i+output_idx]],
            }
        )
        bx2 = pd.concat([bx2, new_row], ignore_index=True)
        bx2.to_csv(bx2_name, index=False)
