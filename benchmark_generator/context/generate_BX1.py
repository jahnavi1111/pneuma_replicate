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

def get_question_generating_prompt(context: str):
    return f"""Context: "{context}"

The above context describes a table. Please create a question that requests a table based on this description. For example, given a context "This table was created by X", the question would be "Provide a table that was created by X."

Respond in the following format:
Question: ..."""


name = "contexts_chicago.csv"  # Adjust context sources name
context_sources = pd.read_csv(name)

# Sample 1020 contexts
unique_context_questions = context_sources["context_question"].nunique()
total_sample_size = 1020
num_of_samples_per_group = total_sample_size // unique_context_questions

context_sources = context_sources.groupby("context_question").apply(lambda x: x.sample(num_of_samples_per_group), include_groups=False).reset_index(drop=True)

benchmark_name = "BX1_chicago.csv"  # Adjust benchmark name
bx1 = pd.DataFrame(columns=["context_id", "question", "answer_tables"])


for i in tqdm(range(len(context_sources))):
    table = context_sources["table"][i]
    context = context_sources["context"][i]
    context_id = context_sources["id"][i]

    prompt = get_question_generating_prompt(context)
    conversation = [{"role": "user", "content": prompt}]
    model_output = prompt_pipeline(
        pipe,
        conversation,
        temperature=None,
        top_p=None
    )[0][-1]["content"].split("Question: ")[-1]
    new_row = pd.DataFrame(
        {"context_id": [context_id], "question": [model_output], "answer_tables": [[table]]}
    )
    bx1 = pd.concat([bx1, new_row], ignore_index=True)
    bx1.to_csv(benchmark_name, index=False)
