import argparse
import sys

sys.path.append("..")

import bm25s
import time
import Stemmer
from commons import DATASETS, str_to_bool, get_documents


stemmer = Stemmer.Stemmer("english")


def indexing_keyword(
    stemmer,
    contents: list[dict[str, str]],
    contexts: list[dict[str, str]],
    content_types: list[str],
    include_contexts: bool,
):
    start = time.time()
    corpus_json = []
    tables = sorted({content["table"] for content in contents})
    for table in tables:
        table_contents = [content for content in contents if content["table"] == table]
        for content_idx, content in enumerate(table_contents):
            corpus_json.append(
                {
                    "text": content["summary"],
                    "metadata": {"table": f"{table}_SEP_contents_SEP_{content_idx}"},
                }
            )

        if contexts is not None:
            table_contexts = [
                context for context in contexts if context["table"] == table
            ]
            for context_idx, context in enumerate(table_contexts):
                corpus_json.append(
                    {
                        "text": context["context"],
                        "metadata": {"table": f"{table}_SEP_contexts-{context_idx}"},
                    }
                )

    corpus_text = [doc["text"] for doc in corpus_json]
    corpus_tokens = bm25s.tokenize(
        corpus_text, stopwords="en", stemmer=stemmer, show_progress=False
    )

    retriever = bm25s.BM25(corpus=corpus_json)
    retriever.index(corpus_tokens, show_progress=True)
    retriever.save(
        f"indices/fulltext-index-{dataset}-{'-'.join(content_types)}{'-context' if include_contexts else ''}"
    )
    end = time.time()
    print(f"Indexing time of dataset {dataset}: {end-start} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program indexes content and context documents.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="all",
        choices=["chembl", "adventure", "public", "chicago", "fetaqa", "bird"],
    )
    parser.add_argument(
        "-ctn",
        "--content-types",
        nargs="+",
        default=["sample_rows", "schema_narrations"],
        choices=["sample_rows", "schema_narrations", "schema_concat", "dbreader"],
    )
    parser.add_argument(
        "-ctx",
        "--include-contexts",
        type=str_to_bool,
        default=False,
        choices=[True, False],
    )
    dataset: str = parser.parse_args().dataset
    content_types: list[str] = parser.parse_args().content_types
    include_contexts: bool = parser.parse_args().include_contexts

    if dataset == "all":
        for dataset in DATASETS.keys():
            contents, contexts = get_documents(dataset, content_types, include_contexts)
            indexing_keyword(
                stemmer, contents, contexts, content_types, include_contexts
            )
    else:
        contents, contexts = get_documents(dataset, content_types, include_contexts)
        indexing_keyword(stemmer, contents, contexts, content_types, include_contexts)
