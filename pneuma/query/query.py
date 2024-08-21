import os
import sys
from collections import defaultdict
from pathlib import Path

import bm25s
import chromadb
import duckdb
import fire
import Stemmer
import torch
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.pipeline_initializer import initialize_pipeline
from utils.prompting_interface import prompt_pipeline
from utils.response import Response, ResponseStatus


class Query:
    def __init__(
        self,
        db_path: str = os.path.expanduser("~/Documents/Pneuma/out/storage.db"),
        index_path: str = os.path.expanduser("~/Documents/Pneuma/out/indexes"),
    ):
        self.db_path = db_path
        self.connection = duckdb.connect(db_path)
        # self.embedding_model = SentenceTransformer(
        #     "dunzhang/stella_en_1.5B_v5", trust_remote_code=True
        # )
        self.stemmer = Stemmer.Stemmer("english")

        # Small model for local testing purposes
        self.embedding_model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5", trust_remote_code=True
        )

        # self.pipe = initialize_pipeline(
        #     "meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16, hf_token
        # )
        # Specific setting for Llama-3-8B-Instruct for batching
        # self.pipe.tokenizer.pad_token_id = self.pipe.model.config.eos_token_id
        # self.pipe.tokenizer.padding_side = 'left'

        # Use small model for local testing
        self.pipe = initialize_pipeline("TinyLlama/TinyLlama_v1.1", torch.bfloat16)
        self.pipe.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"
        self.pipe.tokenizer.pad_token_id = self.pipe.model.config.eos_token_id
        self.pipe.tokenizer.padding_side = "left"

        self.index_path = index_path
        self.vector_index_path = os.path.join(index_path, "vector")
        self.keyword_index_path = os.path.join(index_path, "keyword")
        self.chroma_client = chromadb.PersistentClient(self.vector_index_path)

    def query(self, index_name: str, query: str, k: int = 10) -> str:
        try:
            chroma_collection = self.chroma_client.get_collection(index_name)
        except ValueError:
            return f"Index with name {index_name} does not exist."

        retriever = bm25s.BM25.load(
            os.path.join(self.keyword_index_path, index_name),
            load_corpus=True,
        )

        query_embedding = self.embedding_model.encode(query).tolist()
        vec_res = chroma_collection.query(
            query_embeddings=[query_embedding], n_results=k
        )

        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer, show_progress=False)
        results, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)
        bm25_res = (results, scores)

        all_nodes = self.__hybrid_retriever(bm25_res, vec_res, k, query)

        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"Query successful for index {index_name}.",
            data={"query": query, "response": all_nodes},
        ).to_json()

    def __hybrid_retriever(self, bm25_res, vec_res, k: int, query: str):
        processed_nodes_bm25 = self.__process_nodes_bm25(bm25_res)
        processed_nodes_vec = self.__process_nodes_vec(vec_res)

        node_ids = set(
            list(processed_nodes_bm25.keys()) + list(processed_nodes_vec.keys())
        )
        all_nodes: list[tuple[str, float, str]] = []
        for node_id in node_ids:
            bm25_score_doc = processed_nodes_bm25.get(node_id, (0.0, None))
            vec_score_doc = processed_nodes_vec.get(node_id, (0.0, None))

            combined_score = 0.5 * bm25_score_doc[0] + 0.5 * vec_score_doc[0]
            if bm25_score_doc[1] is None:
                doc = vec_score_doc[1]
            else:
                doc = bm25_score_doc[1]

            all_nodes.append((node_id, combined_score, doc))

        sorted_nodes = sorted(all_nodes, key=lambda node: (-node[1], node[0]))[:k]
        reranked_nodes = self.__rerank(sorted_nodes, query)
        return reranked_nodes

    def __process_nodes_bm25(self, items):
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

    def __process_nodes_vec(self, items):
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

    def __rerank(
        self,
        nodes: list[tuple[str, float, str]],
        query: str,
    ):
        tables_relevancy = defaultdict(bool)

        for node in nodes:
            node_id = node[0]
            # table_id = node_id.split("_SEP_")[0]
            node_type = node_id.split("_SEP_")[1]
            if node_type.startswith("contents"):
                if self.__is_table_content_relevant(node[2], query):
                    tables_relevancy[node_id] = True
            else:
                if self.__is_table_context_relevant(node[2], query):
                    tables_relevancy[node_id] = True
        new_nodes = [
            (node_id, score, doc)
            for node_id, score, doc in nodes
            if tables_relevancy[node_id]
        ] + [
            (node_id, score, doc)
            for tablnode_id, score, doc in nodes
            if not tables_relevancy[node_id]
        ]
        return new_nodes

    def __is_table_content_relevant(self, content: str, question: str):
        prompt = f"""Given a table with the following columns:
*/
{content}
*/
and this question:
/*
{question}
*/
Is the table relevant to answer the question? Begin your answer with yes/no."""

        answer: str = prompt_pipeline(
            self.pipe,
            [[{"role": "user", "content": prompt}]],
            context_length=8192,
            max_new_tokens=3,
            top_p=None,
            temperature=None,
        )[0][-1]["content"]

        if answer.lower().startswith("yes"):
            return True
        return False

    def __is_table_context_relevant(self, context: str, question: str):
        prompt = f"""Given this context describing a table:
*/
{context}
*/
and this question:
/*
{question}
*/
Is the table relevant to answer the question? Begin your answer with yes/no."""

        answer: str = prompt_pipeline(
            self.pipe,
            [[{"role": "user", "content": prompt}]],
            context_length=8192,
            max_new_tokens=3,
            top_p=None,
            temperature=None,
        )[0][-1]["content"]

        if answer.lower().startswith("yes"):
            return True
        return False


if __name__ == "__main__":
    fire.Fire(Query)
