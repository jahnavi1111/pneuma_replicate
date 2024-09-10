import json
import os
import shutil
from datetime import datetime
from time import time

from pneuma import Pneuma

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def read_jsonl(file_path: str):
    data: list[dict[str, str]] = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def write_jsonl(data: list[dict[str, str]], file_path: str):
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item))
            file.write("\n")


def get_question_key(benchmark_type: str, use_rephrased_questions: bool = False):
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


def main():
    dataset = "fetaqa"
    if dataset == "chicago":
        content_benchmark = read_jsonl(
            "../data_src/benchmarks/content/pneuma_chicago_10K_questions_annotated.jsonl"
        )
        data_path = "../data_src/tables/pneuma_chicago_10K"
    elif dataset == "public":
        content_benchmark = read_jsonl(
            "../data_src/benchmarks/content/pneuma_public_bi_questions_annotated.jsonl"
        )
        data_path = "../data_src/tables/pneuma_public_bi"
    elif dataset == "chembl":
        content_benchmark = read_jsonl(
            "../data_src/benchmarks/content/pneuma_chembl_10K_questions_annotated.jsonl"
        )
        data_path = "../data_src/tables/pneuma_chembl_10K"
    elif dataset == "adventure":
        content_benchmark = read_jsonl(
            "../data_src/benchmarks/content/pneuma_adventure_works_questions_annotated.jsonl"
        )
        data_path = "../data_src/tables/pneuma_adventure_works"
    elif dataset == "fetaqa":
        content_benchmark = read_jsonl(
            "../data_src/benchmarks/content/pneuma_fetaqa_questions_annotated.jsonl"
        )
        data_path = "../data_src/tables/pneuma_fetaqa"

    benchmark_type = "content"
    question_key = get_question_key(benchmark_type)
    out_path = "out_benchmark/storage"
    questions = []
    for data in content_benchmark:
        questions.append(data[question_key])
    metadata_path = ""
    pneuma = Pneuma(out_path=out_path)

    results = {}
    responses = {}

    # Generate Index
    print("Starting generating index...")
    start_time = time()
    response = pneuma.generate_index("benchmark_index")
    end_time = time()
    response = json.loads(response)
    print(
        f"Time to generate index with {len(response['data']['table_ids'])} tables: {end_time - start_time} seconds"
    )
    results["generate_index"] = {
        "table_count": len(response["data"]["table_ids"]),
        "vector_index_generation_time": response["data"][
            "vector_index_generation_time"
        ],
        "keyword_index_generation_time": response["data"][
            "keyword_index_generation_time"
        ],
        "time": end_time - start_time,
    }
    responses["generate_index"] = response

    # Query Index
    start_time = time()
    for question in questions:
        response = pneuma.query_index("benchmark_index", question, 3)
        response = json.loads(response)
    end_time = time()
    print(
        f"Time to query index with {len(questions)} questions: {end_time - start_time} seconds"
    )
    results["query_index"] = {
        "query_count": len(questions),
        "time": end_time - start_time,
        "query_throughput": len(questions) / (end_time - start_time),
    }

    # Write results to file
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    with open(
        f"{out_path}/../benchmark-{dataset}-{timestamp}.json", "w", encoding="utf-8"
    ) as f:
        json_results = {
            "dataset": dataset,
            "timestamp": timestamp,
            "results": results,
            "responses": responses,
        }
        json.dump(json_results, f, indent=4)


if __name__ == "__main__":
    main()
