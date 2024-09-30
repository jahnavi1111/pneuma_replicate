import sys
import ast

sys.path.append("..")

from scipy.spatial.distance import cosine
from collections import defaultdict
from enum import Enum
from benchmark_generator.context.utils.prompting_interface import prompt_pipeline


class RerankingMode(Enum):
    NONE = 0
    COSINE = 1
    LLM = 2
    DIRECT_SCORE = 3


class HybridRetriever:
    def __init__(self, reranker, reranking_mode: RerankingMode) -> None:
        self.reranker = reranker
        self.reranking_mode = reranking_mode

    def _process_nodes_bm25(self, items):
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

    def _process_nodes_vec(self, items):
        # Normalize relevance scores and return the nodes in dict format
        scores: list[float] = [1 - dist for dist in items["distances"][0]]
        documents: list[str] = items["documents"][0]
        source_ids: list[list[str]] = [ast.literal_eval(metadata["source_ids"]) for metadata in items["metadatas"][0]]
        max_score = max(scores)
        min_score = min(scores)

        processed_nodes: dict[str, tuple[float, str]] = {}

        for idx in range(len(scores)):
            if min_score == max_score:
                score = 1
            else:
                score = (scores[idx] - min_score) / (max_score - min_score)
            
            for source_id in source_ids[idx]:
                if source_id not in processed_nodes:
                    processed_nodes[source_id] = (score, documents[idx])
        return processed_nodes

    def _get_relevance_prompt(self, desc: str, desc_type: str, question: str):
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

    def _cosine_rerank(self, nodes: list[tuple[str, float, str]], question: str):
        # Each node is of the form (name, score, doc)
        names = []
        docs = []
        for node in nodes:
            names.append(node[0])
            docs.append(node[2])

        docs_embeddings = self.reranker.encode(
            docs,
            batch_size=100,
            device="cuda",
        )
        question_embedding = self.reranker.encode(question, device="cuda")
        similarities = [
            1 - cosine(question_embedding, docs_embedding)
            for docs_embedding in docs_embeddings
        ]

        reranked_nodes = sorted(
            zip(names, similarities, docs), key=lambda x: (-x[1], x[0])
        )
        return reranked_nodes

    def _llm_rerank(self, nodes: list[tuple[str, float, str]], question: str):
        # Each node is of the form (name, score, doc)
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
                            "content": self._get_relevance_prompt(
                                node[2], "content", question
                            ),
                        }
                    ]
                )
            else:
                relevance_prompts.append(
                    [
                        {
                            "role": "user",
                            "content": self._get_relevance_prompt(
                                node[2], "context", question
                            ),
                        }
                    ]
                )

        arguments = prompt_pipeline(
            self.reranker,
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

    def _direct_rerank(self, nodes: list[tuple[str, float, str]], question: str):
        # Each node is of the form (name, score, doc)
        names = []
        docs = []
        for node in nodes:
            names.append(node[0])
            docs.append(node[2])
        similarities = self.reranker.compute_score(
            [(question, doc) for doc in docs], normalize=True
        )
        reranked_nodes = sorted(
            zip(names, similarities, docs), key=lambda x: (-x[1], x[0])
        )
        return reranked_nodes

    def retrieve(
        self,
        bm25_res,
        vec_res,
        k: int,
        question: str,
        alpha=0.5,
    ):
        processed_nodes_bm25 = self._process_nodes_bm25(bm25_res)
        processed_nodes_vec = self._process_nodes_vec(vec_res)

        node_ids = set(
            list(processed_nodes_bm25.keys()) + list(processed_nodes_vec.keys())
        )
        all_nodes: list[tuple[str, float, str]] = []
        for node_id in sorted(node_ids):
            bm25_score_doc = processed_nodes_bm25.get(node_id, (0.0, None))
            vec_score_doc = processed_nodes_vec.get(node_id, (0.0, None))

            combined_score = alpha * bm25_score_doc[0] + (1 - alpha) * vec_score_doc[0]
            if bm25_score_doc[1] is None:
                doc = vec_score_doc[1]
            else:
                doc = bm25_score_doc[1]

            all_nodes.append((node_id, combined_score, doc))

        sorted_nodes = sorted(all_nodes, key=lambda node: (-node[1], node[0]))[:k]
        if self.reranking_mode == RerankingMode.COSINE:
            sorted_nodes = self._cosine_rerank(sorted_nodes, question)
        elif self.reranking_mode == RerankingMode.DIRECT_SCORE:
            sorted_nodes = self._direct_rerank(sorted_nodes, question)
        elif self.reranking_mode == RerankingMode.LLM:
            sorted_nodes = self._llm_rerank(sorted_nodes, question)
        return sorted_nodes
