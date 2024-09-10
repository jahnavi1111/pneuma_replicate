import setproctitle

setproctitle.setproctitle("python3.12")
import os
import time
import sys
import chromadb

sys.path.append("..")

from transformers import set_seed
from chromadb.api.client import Client
from sentence_transformers import SentenceTransformer
from sentence_transformers.SentenceTransformer import SentenceTransformer
from benchmark_generator.context.utils.jsonl import read_jsonl


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
set_seed(42, deterministic=True)


embedding_model = SentenceTransformer("../models/stella", local_files_only=True)


def indexing_vector(
    client: Client,
    embedding_model: SentenceTransformer,
    std_contents: list[dict[str, str]],
    contexts: list[dict[str, str]] = None,
    collection_name="benchmark",
    reindex=False,
):
    documents = []
    metadatas = []
    ids = []

    if not reindex:
        try:
            collection = client.get_collection(collection_name)
            return collection
        except:
            pass
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine", "hnsw:random_seed": 42}
    )

    tables = sorted({content["table"] for content in std_contents})
    for table in tables:
        cols = [
            content["summary"] for content in std_contents if content["table"] == table
        ]
        for content_idx, content in enumerate(cols):
            documents.append(content)
            metadatas.append({"table": f"{table}_SEP_contents_{content_idx}"})
            ids.append(f"{table}_SEP_contents_{content_idx}")

        if contexts is not None:
            filtered_contexts = [
                context["context"] for context in contexts if context["table"] == table
            ]
            for context_idx, context in enumerate(filtered_contexts):
                documents.append(context)
                metadatas.append({"table": f"{table}_SEP_{context_idx}"})
                ids.append(f"{table}_SEP_{context_idx}")

    for i in range(0, len(documents), 30000):
        embeddings = embedding_model.encode(
            documents[i : i + 30000],
            batch_size=100,
            show_progress_bar=True,
            device="cuda",
        )

        collection.add(
            embeddings=[embed.tolist() for embed in embeddings],
            metadatas=metadatas[i : i + 30000],
            documents=documents[i : i + 30000],
            ids=ids[i : i + 30000],
        )
    return collection


def start_indexing(dataset, contents, contexts):
    print(f"Indexing dataset: {dataset}")
    start = time.time()
    client = chromadb.PersistentClient(f"indices/index-{dataset}-pneuma-summarizer")
    indexing_vector(client, embedding_model, contents, contexts)
    end = time.time()
    print(f"Indexing time: {end-start} seconds")


if __name__ == "__main__":
    dataset = "public"
    contents = read_jsonl(
        "../pneuma_summarizer/summaries/standard/public_standard.jsonl"
    ) + read_jsonl("../pneuma_summarizer/summaries/rows/public.jsonl")
    contexts = read_jsonl("../data_src/benchmarks/context/public/contexts_public.jsonl")
    path = "../data_src/tables/pneuma_public_bi"
    start_indexing(dataset, contents, contexts)

    dataset = "chembl"
    contents = read_jsonl(
        "../pneuma_summarizer/summaries/standard/chembl_standard.jsonl"
    ) + read_jsonl("../pneuma_summarizer/summaries/rows/chembl.jsonl")
    contexts = read_jsonl("../data_src/benchmarks/context/chembl/contexts_chembl.jsonl")
    path = "../data_src/tables/pneuma_chembl_10K"
    start_indexing(dataset, contents, contexts)

    dataset = "adventure"
    contents = read_jsonl(
        "../pneuma_summarizer/summaries/standard/adventure_standard.jsonl"
    ) + read_jsonl("../pneuma_summarizer/summaries/rows/adventure.jsonl")
    contexts = read_jsonl(
        "../data_src/benchmarks/context/adventure/contexts_adventure.jsonl"
    )
    path = "../data_src/tables/pneuma_adventure_works"
    start_indexing(dataset, contents, contexts)

    dataset = "chicago"
    contents = read_jsonl(
        "../pneuma_summarizer/summaries/standard/chicago_standard.jsonl"
    ) + read_jsonl("../pneuma_summarizer/summaries/rows/chicago.jsonl")
    contexts = read_jsonl(
        "../data_src/benchmarks/context/chicago/contexts_chicago.jsonl"
    )
    path = "../data_src/tables/pneuma_chicago_10K"
    start_indexing(dataset, contents, contexts)

    dataset = "fetaqa"
    contents = read_jsonl(
        "../pneuma_summarizer/summaries/standard/fetaqa_standard.jsonl"
    ) + read_jsonl("../pneuma_summarizer/summaries/rows/fetaqa.jsonl")
    contexts = read_jsonl("../data_src/benchmarks/context/fetaqa/contexts_fetaqa.jsonl")
    path = "../data_src/tables/pneuma_fetaqa"
    start_indexing(dataset, contents, contexts)
