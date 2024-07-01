# Optional (only if we need to choose among multiple GPUs)
###########################################################
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# import setproctitle
# setproctitle.setproctitle("python")
###########################################################
import pandas as pd
import torch

from utils.pipeline_initializer import initialize_pipeline
from utils.prompting_interface import prompt_pipeline

pipe = initialize_pipeline("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16)

def get_prompt(metadata: str, question: str):
    return f"""Metadata:"{metadata}"
Question:"{question}"
The metadata describes a specific dataset that we have access to. Does this dataset answer the question? Begin your response with yes/no."""

bx1 = pd.read_csv("BX1_chicago.csv")  # Adjust BX1 name
contexts = pd.read_csv("contexts_chicago.csv")  # Adjust contexts name
questions_to_get_contexts = []
for i in range(len(bx1)):
    table = bx1["table"][i]
    context = bx1["context"][i]
    filtered_df = contexts[(contexts['table'] == table) & (contexts["context"] == context)].reset_index(drop=True)
    question = filtered_df["context_question"][0]
    questions_to_get_contexts.append(question)
bx1["question_to_get_context"] = questions_to_get_contexts
bx1["relevant_tables"] = [""] * len(bx1)

for i in range(len(bx1)):
    question = bx1["question"][i]
    relevant_tables = [bx1["table"][i]]
    dfd_q = bx1["question_to_get_context"][i]

    print(f"Processing row: {i}")
    print(f"DFD Question of row {i}: {dfd_q}")
    print(f"Question of row {i}: {question}")

    filtered_bx1 = bx1[bx1["question_to_get_context"] == dfd_q]
    for j in filtered_bx1.index:
        print(f"Comparing {i} with context in row {j}")
        context = filtered_bx1["context"][j]
        context_table = filtered_bx1["table"][j]
        print(f"The context is from table {context_table}")

        prompt = get_prompt(context, question)
        conversation = [{"role": "user", "content": prompt}]
        model_output = prompt_pipeline(
            pipe, conversation, max_new_tokens=4, temperature=None, top_p=None
        )[-1]["content"]
        if model_output.strip().lower().startswith("yes") or model_output.strip().lower().startswith("**yes"):
            relevant_tables.append(context_table)
        print()

    relevant_tables = list(set(relevant_tables))
    relevant_tables.sort()
    bx1.loc[i, "relevant_tables"] = str(relevant_tables)
    bx1.to_csv("BX1_chicago_corrected.csv", index=False)  # Adjust name
