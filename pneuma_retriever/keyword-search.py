import sys
import bm25s
import Stemmer

sys.path.append("..")

from tqdm import tqdm
from benchmark_generator.context.utils.jsonl import read_jsonl, write_jsonl

hitrates_data: list[dict[str, str]] = []
stemmer = Stemmer.Stemmer("english")


def indexing_keyword(
    stemmer,
    contents: list[dict[str, str]],
    contexts: list[dict[str, str]] = None,
):
    corpus_json = []
    tables = sorted({content["table"] for content in contents})
    for table in tables:
        cols_descriptions = [
            content["summary"] for content in contents if content["table"] == table
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


def get_question_key(benchmark_type: str, use_rephrased_questions: bool):
    if benchmark_type == "content":
        if not use_rephrased_questions:
            question_key = "question_from_sql_1"
            benchmark_name = "bc1"
        else:
            question_key = "question"
            benchmark_name = "bc2"
    else:
        if not use_rephrased_questions:
            question_key = "question_bx1"
            benchmark_name = "bx1"
        else:
            question_key = "question_bx2"
            benchmark_name = "bx2"
    return (question_key, benchmark_name)


def evaluate_benchmark(
    benchmark: list[dict[str, str]],
    benchmark_type: str,
    k: int,
    retriever,
    stemmer,
    dataset: str,
    use_rephrased_questions=False,
):
    hitrate_sum = 0
    question_key, benchmark_name = get_question_key(
        benchmark_type, use_rephrased_questions
    )

    questions = []
    for data in benchmark:
        questions.append(data[question_key])

    for idx, datum in enumerate(tqdm(benchmark)):
        answer_tables = datum["answer_tables"]

        query_tokens = bm25s.tokenize(
            questions[idx], stemmer=stemmer, show_progress=False
        )
        results, _ = retriever.retrieve(query_tokens, k=k, show_progress=False)
        for result in results[0]:
            table = result["metadata"]["table"].split("_SEP_")[0]
            if table in answer_tables:
                hitrate_sum += 1
                break
    print(f"Hit Rate: {round(hitrate_sum/len(benchmark) * 100, 2)}")
    hitrates_data.append(
        {
            "dataset": dataset,
            "benchmark": benchmark_name,
            "k": k,
            "hitrate": round(hitrate_sum / len(benchmark) * 100, 2),
        }
    )
    write_jsonl(hitrates_data, "keyword.jsonl")


def start(
    dataset: str,
    contents: list[dict[str, str]],
    contexts: list[dict[str, str]],
    content_benchmark: list[dict[str, str]],
    context_benchmark: list[dict[str, str]],
    ks: list[int],
):
    retriever = indexing_keyword(stemmer, contents, contexts)

    for k in ks:
        print(f"BC1 with k={k}")
        evaluate_benchmark(content_benchmark, "content", k, retriever, stemmer, dataset)
        print("=" * 50)

    for k in ks:
        print(f"BC2 with k={k}")
        evaluate_benchmark(
            content_benchmark, "content", k, retriever, stemmer, dataset, True
        )
        print("=" * 50)

    for k in ks:
        print(f"BX1 with k={k}")
        evaluate_benchmark(context_benchmark, "context", k, retriever, stemmer, dataset)
        print("=" * 50)

    for k in ks:
        print(f"BX2 with k={k}")
        evaluate_benchmark(
            context_benchmark, "context", k, retriever, stemmer, dataset, True
        )
        print("=" * 50)


def get_information(dataset: str):
    """
    Return the contents, contexts, and context benchmarks of a dataset
    """
    contents = read_jsonl(
        f"../../pneuma_summarizer/summaries/rows/{dataset}.jsonl"
    ) + read_jsonl(f"../../pneuma_summarizer/summaries/narrations/{dataset}.jsonl")
    contexts = read_jsonl(
        f"../../data_src/benchmarks/context/{dataset}/contexts_{dataset}.jsonl"
    )
    context_benchmark = read_jsonl(
        f"../../data_src/benchmarks/context/{dataset}/bx_{dataset}.jsonl"
    )
    return [contents, contexts, context_benchmark]


if __name__ == "__main__":
    ks = [1, 10]

    dataset = "chembl"
    content_benchmark = read_jsonl(
        "../../data_src/benchmarks/content/pneuma_chembl_10K_questions_annotated.jsonl"
    )
    contents, contexts, context_benchmark = get_information(dataset)
    start(dataset, contents, contexts, content_benchmark, context_benchmark, ks)

    dataset = "adventure"
    content_benchmark = read_jsonl(
        "../../data_src/benchmarks/content/pneuma_adventure_works_questions_annotated.jsonl"
    )
    contents, contexts, context_benchmark = get_information(dataset)
    start(dataset, contents, contexts, content_benchmark, context_benchmark, ks)

    dataset = "public"
    content_benchmark = read_jsonl(
        "../../data_src/benchmarks/content/pneuma_public_bi_questions_annotated.jsonl"
    )
    contents, contexts, context_benchmark = get_information(dataset)
    start(dataset, contents, contexts, content_benchmark, context_benchmark, ks)

    dataset = "chicago"
    content_benchmark = read_jsonl(
        "../../data_src/benchmarks/content/pneuma_chicago_10K_questions_annotated.jsonl"
    )
    contents, contexts, context_benchmark = get_information(dataset)
    start(dataset, contents, contexts, content_benchmark, context_benchmark, ks)

    dataset = "fetaqa"
    content_benchmark = read_jsonl(
        "../../data_src/benchmarks/content/pneuma_fetaqa_questions_annotated.jsonl"
    )
    contents, contexts, context_benchmark = get_information(dataset)
    start(dataset, contents, contexts, content_benchmark, context_benchmark, ks)
