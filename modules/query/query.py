import chromadb
import duckdb
import fire
from sentence_transformers import SentenceTransformer


class Query:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = duckdb.connect(db_path)
        # self.embedding_model = SentenceTransformer(
        #     "dunzhang/stella_en_1.5B_v5", trust_remote_code=True
        # )

        # Small model for local testing purposes
        self.embedding_model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5", trust_remote_code=True
        )
        self.index_location = "../out/indexes"
        self.chroma_client = chromadb.PersistentClient(self.index_location)

    def query(self, index_name: str, query: str, k: int = 10):
        print(index_name)
        print(query)
        print(k)


if __name__ == "__main__":
    fire.Fire(Query)
