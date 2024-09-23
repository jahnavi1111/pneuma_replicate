import sys

sys.path.append("..")

from sentence_transformers import SentenceTransformer
from sentence_transformers.SentenceTransformer import SentenceTransformer
from benchmark_generator.context.utils.jsonl import read_jsonl, write_jsonl
from tqdm import tqdm

embedding_model = SentenceTransformer(
    "../models/stella", local_files_only=True, device="cpu"
)

EMBEDDING_MAX_TOKENS = 512


def split_schema_summaries(
    contents: list[dict[str, str]], summary_type: str, name: str
):
    processed_contents = []
    unique_tables = sorted(set([row["table"] for row in contents]))
    tokenizer = embedding_model.tokenizer
    for table in tqdm(unique_tables):
        schema_summary = [summary for summary in contents if summary["table"] == table][
            0
        ]["summary"]
        column_summaries = schema_summary.split(" | ")

        col_idx = 0
        while col_idx < len(column_summaries):
            processed_summary = column_summaries[col_idx]

            while (col_idx + 1) < len(column_summaries):
                temp = processed_summary + " | " + column_summaries[col_idx + 1]
                if len(tokenizer.encode(temp)) < EMBEDDING_MAX_TOKENS:
                    processed_summary = temp
                    col_idx += 1
                else:
                    break

            col_idx += 1
            processed_contents.append(
                {
                    "table": table,
                    "summary": processed_summary,
                }
            )
    print(f"Num of content summaries (BEFORE): {len(contents)}")
    print(f"Num of content summaries (AFTER): {len(processed_contents)}")
    write_jsonl(
        processed_contents, f"summaries/{summary_type}/{name}_splitted.jsonl"
    )


def merge_row_summaries(rows: list[dict[str, str]], name: str, summary_type: str):
    unique_tables = sorted(set([row["table"] for row in rows]))
    processed_rows = []

    tokenizer = embedding_model.tokenizer
    for table in tqdm(unique_tables):
        table_rows = [row for row in rows if row["table"] == table]

        rows_idx = 0
        while rows_idx < len(table_rows):
            processed_summary = table_rows[rows_idx]["summary"]

            while (rows_idx + 1) < len(table_rows):
                temp = processed_summary + " || " + table_rows[rows_idx + 1]["summary"]
                if len(tokenizer.encode(temp)) < EMBEDDING_MAX_TOKENS:
                    print(len(tokenizer.encode(temp)))
                    processed_summary = temp
                    rows_idx += 1
                else:
                    break

            rows_idx += 1
            processed_rows.append(
                {
                    "table": table,
                    "summary": processed_summary,
                }
            )

    print(f"Num of rows summaries (BEFORE): {len(rows)}")
    print(f"Num of rows summaries (AFTER): {len(processed_rows)}")

    write_jsonl(processed_rows, f"summaries/{summary_type}/{name}_merged.jsonl")


def merge_context_summaries(contexts: list[dict[str, str]], name: str):
    unique_tables = sorted(set([context["table"] for context in contexts]))
    processed_contexts = []

    tokenizer = embedding_model.tokenizer
    for table in tqdm(unique_tables):
        table_contexts = [context for context in contexts if context["table"] == table]

        context_idx = 0
        while context_idx < len(table_contexts):
            processed_context = table_contexts[context_idx]["context"]

            while (context_idx + 1) < len(table_contexts):
                temp = (
                    processed_context
                    + " || "
                    + table_contexts[context_idx + 1]["context"]
                )
                if len(tokenizer.encode(temp)) < EMBEDDING_MAX_TOKENS:
                    print(len(tokenizer.encode(temp)))
                    processed_context = temp
                    context_idx += 1
                else:
                    break

            context_idx += 1
            processed_contexts.append(
                {
                    "table": table,
                    "context": processed_context,
                }
            )
    print(f"Num of context summaries (BEFORE): {len(contexts)}")
    print(f"Num of context summaries (AFTER): {len(processed_contexts)}")

    write_jsonl(
        processed_contexts,
        f"../data_src/benchmarks/context/{name}/contexts_{name}_merged.jsonl",
    )


if __name__ == "__main__":
    name = f"chembl"
    std_contents = read_jsonl(f"summaries/standard/{name}.jsonl")
    narration_contents = read_jsonl(f"summaries/narrations/{name}.jsonl")
    row_contents = read_jsonl(f"summaries/rows/{name}.jsonl")
    dbreader_contents = read_jsonl(f"summaries/dbreader/{name}.jsonl")
    contexts = read_jsonl(f"../data_src/benchmarks/context/{name}/contexts_{name}.jsonl")

    split_schema_summaries(std_contents, "standard", name)
    split_schema_summaries(narration_contents, "narrations", name)
    merge_row_summaries(row_contents, name, "rows")
    merge_row_summaries(dbreader_contents, name, "dbreader")
    merge_context_summaries(contexts, name)

    name = f"adventure"
    std_contents = read_jsonl(f"summaries/standard/{name}.jsonl")
    narration_contents = read_jsonl(f"summaries/narrations/{name}.jsonl")
    row_contents = read_jsonl(f"summaries/rows/{name}.jsonl")
    dbreader_contents = read_jsonl(f"summaries/dbreader/{name}.jsonl")
    contexts = read_jsonl(f"../data_src/benchmarks/context/{name}/contexts_{name}.jsonl")

    split_schema_summaries(std_contents, "standard", name)
    split_schema_summaries(narration_contents, "narrations", name)
    merge_row_summaries(row_contents, name, "rows")
    merge_row_summaries(dbreader_contents, name, "dbreader")
    merge_context_summaries(contexts, name)

    name = f"public"
    std_contents = read_jsonl(f"summaries/standard/{name}.jsonl")
    narration_contents = read_jsonl(f"summaries/narrations/{name}.jsonl")
    row_contents = read_jsonl(f"summaries/rows/{name}.jsonl")
    dbreader_contents = read_jsonl(f"summaries/dbreader/{name}.jsonl")
    contexts = read_jsonl(f"../data_src/benchmarks/context/{name}/contexts_{name}.jsonl")

    split_schema_summaries(std_contents, "standard", name)
    split_schema_summaries(narration_contents, "narrations", name)
    merge_row_summaries(row_contents, name, "rows")
    merge_row_summaries(dbreader_contents, name, "dbreader")
    merge_context_summaries(contexts, name)

    name = f"chicago"
    std_contents = read_jsonl(f"summaries/standard/{name}.jsonl")
    narration_contents = read_jsonl(f"summaries/narrations/{name}.jsonl")
    row_contents = read_jsonl(f"summaries/rows/{name}.jsonl")
    dbreader_contents = read_jsonl(f"summaries/dbreader/{name}.jsonl")
    contexts = read_jsonl(f"../data_src/benchmarks/context/{name}/contexts_{name}.jsonl")

    split_schema_summaries(std_contents, "standard", name)
    split_schema_summaries(narration_contents, "narrations", name)
    merge_row_summaries(row_contents, name, "rows")
    merge_row_summaries(dbreader_contents, name, "dbreader")
    merge_context_summaries(contexts, name)

    name = f"fetaqa"
    std_contents = read_jsonl(f"summaries/standard/{name}.jsonl")
    narration_contents = read_jsonl(f"summaries/narrations/{name}.jsonl")
    row_contents = read_jsonl(f"summaries/rows/{name}.jsonl")
    dbreader_contents = read_jsonl(f"summaries/dbreader/{name}.jsonl")
    contexts = read_jsonl(f"../data_src/benchmarks/context/{name}/contexts_{name}.jsonl")

    split_schema_summaries(std_contents, "standard", name)
    split_schema_summaries(narration_contents, "narrations", name)
    merge_row_summaries(row_contents, name, "rows")
    merge_row_summaries(dbreader_contents, name, "dbreader")
    merge_context_summaries(contexts, name)
