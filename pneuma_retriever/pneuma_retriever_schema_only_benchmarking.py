import setproctitle

setproctitle.setproctitle("python3.12")
import os
import time
import sys
import chromadb
import bm25s
import Stemmer
import torch
import numpy as np

sys.path.append("..")

from tqdm import tqdm
from collections import defaultdict
from transformers import set_seed
from chromadb.api.models.Collection import Collection
from benchmark_generator.context.utils.jsonl import read_jsonl
from benchmark_generator.context.utils.pipeline_initializer import initialize_pipeline
from benchmark_generator.context.utils.prompting_interface import prompt_pipeline


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_seed(42, deterministic=True)


pipe = initialize_pipeline("../models/llama", torch.bfloat16)
stemmer = Stemmer.Stemmer("english")

# Specific setting for Llama-3-8B-Instruct for batching
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
pipe.tokenizer.padding_side = "left"


def indexing_keyword(
    stemmer,
    narration_contents: list[dict[str, str]],
    contexts: list[dict[str, str]] = None,
):
    corpus_json = []
    tables = sorted({content["table"] for content in narration_contents})
    for table in tables:
        cols_descriptions = [
            content["summary"]
            for content in narration_contents
            if content["table"] == table
        ]
        for content_idx, content in enumerate(cols_descriptions):
            corpus_json.append(
                {
                    "text": content,
                    "metadata": {"table": f"{table}_SEP_contents_{content_idx}"},
                }
            )

        if contexts is not None:
            filtered_contexts = [
                context["context"] for context in contexts if context["table"] == table
            ]
            for context_idx, context in enumerate(filtered_contexts):
                corpus_json.append(
                    {
                        "text": context,
                        "metadata": {"table": f"{table}_SEP_{context_idx}"},
                    }
                )

    corpus_text = [doc["text"] for doc in corpus_json]
    corpus_tokens = bm25s.tokenize(
        corpus_text, stopwords="en", stemmer=stemmer, show_progress=False
    )

    retriever = bm25s.BM25(corpus=corpus_json)
    retriever.index(corpus_tokens, show_progress=False)
    return retriever


def process_nodes_bm25(items):
    # Normalize relevance scores and return the nodes in dict format
    results, scores = items
    scores: list[float] = scores[0]
    max_score = max(scores)
    min_score = min(scores)

    processed_nodes: dict[str, tuple[float, str]] = {}
    for i, node in enumerate(results[0]):
        if min_score == max_score:
            score = 1
        else:
            score = (scores[i] - min_score) / (max_score - min_score)
        processed_nodes[node["metadata"]["table"]] = (score, node["text"])
    return processed_nodes


def process_nodes_vec(items):
    # Normalize relevance scores and return the nodes in dict format
    scores: list[float] = [1 - dist for dist in items["distances"][0]]
    max_score = max(scores)
    min_score = min(scores)

    processed_nodes: dict[str, tuple[float, str]] = {}

    for idx in range(len(items["ids"][0])):
        if min_score == max_score:
            score = 1
        else:
            score = (scores[idx] - min_score) / (max_score - min_score)
        processed_nodes[items["ids"][0][idx]] = (score, items["documents"][0][idx])
    return processed_nodes


def hybrid_retriever(
    bm25_res,
    vec_res,
    k: int,
    question: str,
    use_reranker=False,
):
    processed_nodes_bm25 = process_nodes_bm25(bm25_res)
    processed_nodes_vec = process_nodes_vec(vec_res)

    node_ids = set(list(processed_nodes_bm25.keys()) + list(processed_nodes_vec.keys()))
    all_nodes: list[tuple[str, float, str]] = []
    for node_id in sorted(node_ids):
        bm25_score_doc = processed_nodes_bm25.get(node_id, (0.0, None))
        vec_score_doc = processed_nodes_vec.get(node_id, (0.0, None))

        combined_score = 0.5 * bm25_score_doc[0] + 0.5 * vec_score_doc[0]
        if bm25_score_doc[1] is None:
            doc = vec_score_doc[1]
        else:
            doc = bm25_score_doc[1]

        all_nodes.append((node_id, combined_score, doc))

    sorted_nodes = sorted(all_nodes, key=lambda node: (-node[1], node[0]))[:k]
    if use_reranker:
        sorted_nodes = rerank(sorted_nodes, question)
    return sorted_nodes


def get_relevance_prompt(desc: str, desc_type: str, question: str):
    if desc_type == "content":
        return f"""Given a table with the following columns:
*/
{desc}
*/
and this question:
/*
{question}
*/
Is the table relevant to answer the question? Begin your answer with yes/no."""
    elif desc_type == "context":
        return f"""Given this context describing a table:
*/
{desc}
*/
and this question:
/*
{question}
*/
Is the table relevant to answer the question? Begin your answer with yes/no."""


def rerank(nodes: list[tuple[str, float, str]], question: str):
    tables_relevance = defaultdict(bool)
    relevance_prompts = []
    node_tables = []

    for node in nodes:
        table_name = node[0]
        node_tables.append(table_name)
        if table_name.split("_SEP_")[1].startswith("contents"):
            relevance_prompts.append(
                [
                    {
                        "role": "user",
                        "content": get_relevance_prompt(node[2], "content", question),
                    }
                ]
            )
        else:
            relevance_prompts.append(
                [
                    {
                        "role": "user",
                        "content": get_relevance_prompt(node[2], "context", question),
                    }
                ]
            )

    arguments = prompt_pipeline(
        pipe,
        relevance_prompts,
        batch_size=1,
        context_length=8192,
        max_new_tokens=2,
        top_p=None,
        temperature=None,
    )
    for arg_idx, argument in enumerate(arguments):
        if argument[-1]["content"].lower().startswith("yes"):
            tables_relevance[node_tables[arg_idx]] = True

    new_nodes = [
        (table_name, score, doc)
        for table_name, score, doc in nodes
        if tables_relevance[table_name]
    ] + [
        (table_name, score, doc)
        for table_name, score, doc in nodes
        if not tables_relevance[table_name]
    ]
    return new_nodes


def get_question_key(benchmark_type: str, use_rephrased_questions: bool):
    if benchmark_type == "content":
        if not use_rephrased_questions:
            question_key = "question_from_sql_1"
        else:
            question_key = "question"
    else:
        if not use_rephrased_questions:
            question_key = "question_bx1"
        else:
            question_key = "question_bx2"
    return question_key


def evaluate_benchmark(
    benchmark: list[dict[str, str]],
    benchmark_type: str,
    k: int,
    collection: Collection,
    retriever,
    stemmer,
    use_reranker=False,
    use_rephrased_questions=False,
):
    start = time.time()
    hitrate_sum = 0
    wrong_questions = []

    if use_reranker:
        increased_k = k * 3
    else:
        increased_k = k

    question_key = get_question_key(benchmark_type, use_rephrased_questions)

    questions = []
    for data in benchmark:
        questions.append(data[question_key])
    embed_questions = np.loadtxt(
        f"embeddings/embed-{dataset}-questions-{benchmark_type}-{use_rephrased_questions}.txt"
    )
    embed_questions = [embed.tolist() for embed in embed_questions]

    for idx, datum in enumerate(tqdm(benchmark)):
        answer_tables = datum["answer_tables"]
        question_embedding = embed_questions[idx]

        query_tokens = bm25s.tokenize(
            questions[idx], stemmer=stemmer, show_progress=False
        )
        results, scores = retriever.retrieve(
            query_tokens, k=increased_k, show_progress=False
        )
        bm25_res = (results, scores)

        vec_res = collection.query(
            query_embeddings=[question_embedding], n_results=increased_k
        )

        all_nodes = hybrid_retriever(
            bm25_res, vec_res, increased_k, questions[idx], use_reranker
        )
        before = hitrate_sum
        for table, _, _ in all_nodes[:k]:
            table = table.split("_SEP_")[0]
            if table in answer_tables:
                hitrate_sum += 1
                break
        if before == hitrate_sum:
            wrong_questions.append(idx)
        # Checkpoint
        if idx % 20 == 0:
            print(f"Current Hit Rate Sum at index {idx}: {hitrate_sum}")
            print(
                f"Current wrongly answered questions at index {idx}: {wrong_questions}"
            )

    end = time.time()
    print(f"Hit Rate: {hitrate_sum/len(benchmark) * 100}")
    print(f"Benchmarking Time: {end - start} seconds")
    print(f"Wrongly answered questions: {wrong_questions}")


def start_benchmarking(
    dataset, contents_keyword, contexts, content_benchmark, context_benchmark
):
    print(f"Processing {dataset} dataset")
    start = time.time()
    client = chromadb.PersistentClient(
        f"indices/index-{dataset}-pneuma-summarizer-schema-only"
    )
    collection = client.get_collection("benchmark")
    retriever = indexing_keyword(stemmer, contents_keyword, contexts)
    end = time.time()
    print(f"Indexing time: {end-start} seconds")

    ks = [1]
    for k in ks:
        print(f"BC1 with k={k}")
        evaluate_benchmark(
            content_benchmark,
            "content",
            k,
            collection,
            retriever,
            stemmer,
            use_reranker=True,
        )
        print("=" * 50)

    for k in ks:
        print(f"BC2 with k={k}")
        evaluate_benchmark(
            content_benchmark,
            "content",
            k,
            collection,
            retriever,
            stemmer,
            use_rephrased_questions=True,
            use_reranker=True,
        )
        print("=" * 50)

    for k in ks:
        print(f"BX1 with k={k}")
        evaluate_benchmark(
            context_benchmark,
            "context",
            k,
            collection,
            retriever,
            stemmer,
            use_reranker=True,
        )
        print("=" * 50)

    for k in ks:
        print(f"BX2 with k={k}")
        evaluate_benchmark(
            context_benchmark,
            "context",
            k,
            collection,
            retriever,
            stemmer,
            use_rephrased_questions=True,
            use_reranker=True,
        )
        print("=" * 50)


if __name__ == "__main__":
    dataset = "public"
    contents_keyword = read_jsonl(
        "../pneuma_summarizer/summaries/narrations/public_narrations.jsonl"
    )
    contexts = read_jsonl("../data_src/benchmarks/context/public/contexts_public.jsonl")
    content_benchmark = read_jsonl(
        "../data_src/benchmarks/content/pneuma_public_bi_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        "../data_src/benchmarks/context/public/bx_public.jsonl"
    )
    path = "../data_src/tables/pneuma_public_bi"
    start_benchmarking(
        dataset, contents_keyword, contexts, content_benchmark, context_benchmark
    )

    dataset = "chembl"
    contents_keyword = read_jsonl(
        "../pneuma_summarizer/summaries/narrations/chembl_narrations.jsonl"
    )
    contexts = read_jsonl("../data_src/benchmarks/context/chembl/contexts_chembl.jsonl")
    content_benchmark = read_jsonl(
        "../data_src/benchmarks/content/pneuma_chembl_10K_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        "../data_src/benchmarks/context/chembl/bx_chembl.jsonl"
    )
    path = "../data_src/tables/pneuma_chembl_10K"
    start_benchmarking(
        dataset, contents_keyword, contexts, content_benchmark, context_benchmark
    )

    dataset = "adventure"
    contents_keyword = read_jsonl(
        "../pneuma_summarizer/summaries/narrations/adventure_narrations.jsonl"
    )
    contexts = read_jsonl(
        "../data_src/benchmarks/context/adventure/contexts_adventure.jsonl"
    )
    content_benchmark = read_jsonl(
        "../data_src/benchmarks/content/pneuma_adventure_works_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        "../data_src/benchmarks/context/adventure/bx_adventure.jsonl"
    )
    path = "../data_src/tables/pneuma_adventure_works"
    start_benchmarking(
        dataset, contents_keyword, contexts, content_benchmark, context_benchmark
    )

    dataset = "chicago"
    contents_keyword = read_jsonl(
        "../pneuma_summarizer/summaries/narrations/chicago_narrations.jsonl"
    )
    contexts = read_jsonl(
        "../data_src/benchmarks/context/chicago/contexts_chicago.jsonl"
    )
    content_benchmark = read_jsonl(
        "../data_src/benchmarks/content/pneuma_chicago_10K_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        "../data_src/benchmarks/context/chicago/bx_chicago.jsonl"
    )
    path = "../data_src/tables/pneuma_chicago_10K"
    start_benchmarking(
        dataset, contents_keyword, contexts, content_benchmark, context_benchmark
    )

    dataset = "fetaqa"
    contents_keyword = read_jsonl(
        "../pneuma_summarizer/summaries/narrations/fetaqa_narrations.jsonl"
    )
    contexts = read_jsonl("../data_src/benchmarks/context/fetaqa/contexts_fetaqa.jsonl")
    content_benchmark = read_jsonl(
        "../data_src/benchmarks/content/pneuma_fetaqa_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        "../data_src/benchmarks/context/fetaqa/bx_fetaqa.jsonl"
    )
    path = "../data_src/tables/pneuma_fetaqa"
    start_benchmarking(
        dataset, contents_keyword, contexts, content_benchmark, context_benchmark
    )
