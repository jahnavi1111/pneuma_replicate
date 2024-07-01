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
from tqdm import tqdm

pipe = initialize_pipeline("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16)

def get_question_generating_prompt(metadata: str):
    return f"""Metadata: "{metadata}"

The above metadata describes a dataset. Please create a question that requests a dataset based on this description. For example, for the metadata "This dataset was created by X," the question would be "Provide a dataset that was created by X."

Respond in the following format:
Question: ... """


name = "contexts_chicago.csv"  # Adjust context sources name
context_sources = pd.read_csv(name)

benchmark_name = "BX1_chicago.csv"  # Adjust benchmark name
bx1 = pd.DataFrame(columns=["context", "question", "table"])

for i in tqdm(range(len(context_sources))):
    table = context_sources["table"][i]
    context = context_sources["context"][i]

    prompt = get_question_generating_prompt(context)
    conversation = [{"role": "user", "content": prompt}]
    model_output = prompt_pipeline(pipe, conversation, temperature=None, top_p=None)[
        -1
    ]["content"].split("Question: ")[-1]
    new_row = pd.DataFrame(
        {"context": [context], "question": [model_output], "table": [table]}
    )
    bx1 = pd.concat([bx1, new_row], ignore_index=True)
    bx1.to_csv(benchmark_name, index=False)
