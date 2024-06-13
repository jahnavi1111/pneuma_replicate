from llama_index.core import (
    Settings,
    VectorStoreIndex,
    Document,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# These packages are available in the benchmark_generator directory
from pipeline.pipeline_initializer import initialize_pipeline
from pipeline.prompting_interface import prompt_pipeline

import torch
import warnings
import pandas as pd


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


pipe = initialize_pipeline("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16)


benchmark = pd.read_csv("BX1_chicago.csv")  # Adjust benchmark name
benchmark.head()


def create_context_documents(df):
    documents = []
    for idx in df.index:
        table = df["table"][idx]
        answer = df["context"][idx]
        document = Document(
            text=answer,
            metadata={"table": table},
            doc_id=f"doc_'{table}'_{idx}",
        )
        documents.append(document)
    return documents


def create_content_documents(df):
    documents = []
    for idx in df.index:
        table = df["table"][idx]
        table_summary = df["summary"][idx]
        document = Document(
            text=table_summary,
            metadata={"table": table},
            doc_id=f"doc_'{table}'_{idx}",
        )
        documents.append(document)
    return documents


import numpy as np


def get_sample_summaries(summaries: pd.DataFrame, sample_percentage=1):
    """
    This is to randomly sample certain percentage of the summaries.
    The return value is the df itself, but only the sampled rows remained.
    """
    # Prepare to sample summaries (category refers to the tables)
    category_counts = summaries["table"].value_counts()
    sample_sizes = np.ceil(category_counts * sample_percentage).astype(int)

    # Perform stratified sampling
    sampled_summaries = summaries.copy(deep=True)
    sampled_summaries["table_copy"] = sampled_summaries["table"]
    sampled_summaries = sampled_summaries.groupby("table_copy", group_keys=False).apply(
        lambda x: x.sample(n=sample_sizes[x.name], random_state=42),
        include_groups=False,
    )
    return sampled_summaries.reset_index(drop=True)


# For example, this is for context
documents = create_context_documents(benchmark)
len(documents)


vector_index = VectorStoreIndex(documents)
print("Index created")


def get_relevancy_prompt(metadata: str, query: str):
    return f"""Metadata M:"{metadata}"
Query Q: "{query}"
Does metadata M answer query Q? Begin your response with yes/no."""


def filter_retrieved_data(retrieved_data, query):
    filtered_data = []
    for data in retrieved_data:
        metadata = data.text
        prompt = get_relevancy_prompt(metadata, query)
        conversation = [{"role": "user", "content": prompt}]
        model_output: str = prompt_pipeline(pipe, conversation, max_new_tokens=128)[-1][
            "content"
        ]
        if model_output.lower().startswith("yes"):
            filtered_data.append(data)
    return filtered_data


def convert_retrieved_data_to_tables_ranks(retrieved_data, query):
    # Convert all retrieved data to the format (table, rank)
    rank = 1
    prev_score = retrieved_data[0].get_score()
    tables_ranks = []
    filterd_data = filter_retrieved_data(retrieved_data, query)
    for data in filterd_data:
        if data.get_score() < prev_score:
            rank += 1
        table = data.id_.split("'")[1]  # E.g., "chicago_open_data/22u3-xenr"
        tables_ranks.append((table, rank))
        prev_score = data.get_score()
    return tables_ranks


def evaluate(retriever, benchmark_df):
    accuracy_sum = 0
    precision_at_1_sum = 0
    reciprocal_rank_sum = 0
    for i in range(len(benchmark_df)):
        query = benchmark_df["question"][i]
        expected_table = benchmark_df["table"][i]
        retrieved_data = retriever.retrieve(query)
        tables_ranks = convert_retrieved_data_to_tables_ranks(retrieved_data, query)

        for j, (table, rank) in enumerate(tables_ranks):
            if table == expected_table:
                accuracy_sum += 1
                if rank == 1:
                    precision_at_1_sum += 1
                reciprocal_rank_sum += 1 / (j + 1)
                break

        # Checkpointing
        if i % 25 == 0:
            print(f"i: {i}")
            print(f"accuracy_sum: {accuracy_sum}")
            print(f"precision_at_1_sum: {precision_at_1_sum}")
            print(f"reciprocal_rank_sum: {reciprocal_rank_sum}")
    return {
        "accuracy": accuracy_sum / benchmark_df.shape[0],
        "Mean Precision@1": precision_at_1_sum / benchmark_df.shape[0],
        "MRR": reciprocal_rank_sum / benchmark_df.shape[0],
    }


warnings.filterwarnings("ignore")

retriever = vector_index.as_retriever(similarity_top_k=10)  # Adjust k
res = evaluate(retriever, benchmark)
print(res)
