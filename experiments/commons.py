import json


DATA_SRC = "../data_src"
CONTENTS_PATH = "pneuma_summarizer/summaries"
DATASETS = {
    "chembl": "pneuma_chembl_10K",
    "adventure": "pneuma_adventure_works",
    "public": "pneuma_public_bi",
    "chicago": "pneuma_chicago_10K",
    "fetaqa": "pneuma_fetaqa",
    "bird": "pneuma_bird",
}
CONTENT_BLOCK_PROCESSING_TYPE = {
    "schema_narrations": "splitted",
    "schema_concat": "splitted",
    "sample_rows": "merged",
    "dbreader": "merged",
}


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


def str_to_bool(value: str) -> bool:
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise ValueError("Invalid boolean value")


def get_documents(dataset: str, content_types: list[str], include_contexts: bool):
    """
    Return contents (row & schema) and optionally contexts of a dataset
    """
    contents = []
    for content_type in content_types:
        content_file_name = f"{dataset}_{CONTENT_BLOCK_PROCESSING_TYPE[content_type]}"
        contents.extend(read_jsonl(
            f"../{CONTENTS_PATH}/{content_type}/{content_file_name}.jsonl"
        ))
    contexts = None
    if include_contexts:
        contexts = read_jsonl(
            f"../{DATA_SRC}/benchmark/context/{dataset}/contexts_{dataset}_merged.jsonl"
        )
    return [contents, contexts]
