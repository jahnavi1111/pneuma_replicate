import json

import chromadb
import duckdb
import fire
import pandas as pd
from chromadb.db.base import UniqueConstraintError
from sentence_transformers import SentenceTransformer


class IndexGenerator:
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

    def generate_index(self, index_name: str, table_ids: list | tuple):
        if isinstance(table_ids, str):
            table_ids = (table_ids,)

        documents = []
        embeddings = []
        metadatas = []
        ids = []

        for table_id in table_ids:
            contexts = self.connection.sql(
                f"""SELECT id, context FROM table_contexts
                WHERE table_id='{table_id}'"""
            ).fetchall()

            summaries = self.connection.sql(
                f"""SELECT id, summary FROM table_summaries
                WHERE table_id='{table_id}'"""
            ).fetchall()

            for entry in contexts + summaries:
                entry_id = entry[0]
                content = json.loads(entry[1])
                payload = content["payload"]

                embedding = self.embedding_model.encode(payload)

                documents.append(payload)
                embeddings.append(embedding.tolist())
                metadatas.append({"table": table_id})
                ids.append(str(entry_id))

        try:
            chroma_collection = self.chroma_client.create_collection(
                name=index_name, metadata={"hnsw:space": "cosine"}
            )
        except UniqueConstraintError:
            return f"Index with name {index_name} already exists."

        chroma_collection.add(
            documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

        # If we decide to use DuckDB's vector store, we don't need to store
        # this data.
        index_id = self.connection.sql(
            f"""INSERT INTO indexes (name, location)
            VALUES ('{index_name}', '{self.index_location}')
            RETURNING id"""
        ).fetchone()[0]

        insert_df = pd.DataFrame.from_dict(
            {
                "index_id": [index_id] * len(table_ids),
                "table_id": table_ids,
            }
        )

        # So we know which tables are included in this index.
        self.connection.sql(
            """INSERT INTO index_table_mappings (index_id, table_id)
            SELECT * FROM insert_df""",
        )

        return f"Index named {index_name} with id {index_id} has been created with {len(insert_df)} tables."


if __name__ == "__main__":
    fire.Fire(IndexGenerator)
