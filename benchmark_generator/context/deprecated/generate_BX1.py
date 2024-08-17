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


def get_question_generating_prompt(context: str):
    return f"""Context: "{context}"

The above context describes a table. Please create a question that requests a table based on this description. For example, given a context "This table was created by X", the question would be "Provide a table that was created by X."

Respond in the following format:
Question: ..."""


name = "contexts_adventure.csv"  # Adjust context sources name
context_sources = pd.read_csv(name)

# Sample 1020 contexts
unique_context_questions = context_sources["context_question"].nunique()
total_sample_size = 1020
num_of_samples_per_group = total_sample_size // unique_context_questions

context_sources = (
    context_sources.groupby("context_question")
    .apply(lambda x: x.sample(num_of_samples_per_group), include_groups=False)
    .reset_index(drop=True)
)

benchmark_name = "BX1_adventure.csv"  # Adjust benchmark name
bx1 = pd.DataFrame(columns=["context_id", "question", "answer_tables"])

conversations = []
context_ids = []
tables = []
for i in tqdm(range(len(context_sources))):
    table = context_sources["table"][i]
    context = context_sources["context"][i]
    context_id = context_sources["id"][i]

    prompt = get_question_generating_prompt(context)

    conversations.append([{"role": "user", "content": prompt}])
    context_ids.append(context_id)
    tables.append(table)

for i in tqdm(range(0, len(conversations), 3)):
    model_outputs = prompt_pipeline(
        pipe, conversations[i : i + 3], batch_size=3, temperature=None, top_p=None
    )

    for output_idx, output in enumerate(model_outputs):
        question = output[-1]["content"].split("Question: ")[-1]
        new_row = pd.DataFrame(
            {
                "context_id": [context_ids[i + output_idx]],
                "question": [question],
                "answer_tables": [[tables[i + output_idx]]],
            }
        )
        bx1 = pd.concat([bx1, new_row], ignore_index=True)
        bx1.to_csv(benchmark_name, index=False)
