import sys
from pathlib import Path

import chromadb
import duckdb
import fire
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.response import Response, ResponseStatus


class Query:
    def __init__(self, db_path: str, index_location: str = "../out/indexes"):
        self.db_path = db_path
        self.connection = duckdb.connect(db_path)
        # self.embedding_model = SentenceTransformer(
        #     "dunzhang/stella_en_1.5B_v5", trust_remote_code=True
        # )

        # Small model for local testing purposes
        self.embedding_model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5", trust_remote_code=True
        )
        self.index_location = index_location
        self.chroma_client = chromadb.PersistentClient(self.index_location)

    def query(self, index_name: str, query: str, k: int = 10):
        try:
            chroma_collection = self.chroma_client.get_collection(index_name)
        except ValueError:
            return f"Index with name {index_name} does not exist."

        query_embedding = self.embedding_model.encode(query).tolist()
        response = chroma_collection.query(
            query_embeddings=[query_embedding], n_results=k
        )

        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"Query successful for index {index_name}.",
            data=response,
        ).to_json()


if __name__ == "__main__":
    fire.Fire(Query)
