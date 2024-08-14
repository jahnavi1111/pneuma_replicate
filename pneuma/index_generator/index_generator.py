import json
import os
import sys
from pathlib import Path

import chromadb
import duckdb
import fire
import pandas as pd
from chromadb.db.base import UniqueConstraintError
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.response import Response, ResponseStatus


class IndexGenerator:
    def __init__(
        self,
        db_path: str = os.path.expanduser("~/Documents/Pneuma/out/storage.db"),
        index_location: str = os.path.expanduser("~/Documents/Pneuma/out/indexes"),
    ):
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

    def generate_index(self, index_name: str, table_ids: list | tuple = None) -> str:
        if table_ids is None:
            print("No table ids provided. Generating index for all tables...")
            table_ids = [
                entry[0]
                for entry in self.connection.sql(
                    "SELECT id FROM table_status"
                ).fetchall()
            ]
        elif isinstance(table_ids, str):
            table_ids = (table_ids,)

        print(f"Generating index for {len(table_ids)} tables...")

        documents = []
        embeddings = []
        metadatas = []
        ids = []

        for table_id in table_ids:
            print(f"Processing table {table_id}...")
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

        if len(documents) == 0:
            return Response(
                status=ResponseStatus.ERROR,
                message="No context and summary entries found for the given table ids.",
            ).to_json()

        try:
            chroma_collection = self.chroma_client.create_collection(
                name=index_name, metadata={"hnsw:space": "cosine"}
            )
        except UniqueConstraintError:
            return Response(
                status=ResponseStatus.ERROR,
                message=f"Index named {index_name} already exists.",
            ).to_json()

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

        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"Index named {index_name} with id {index_id} has been created with {len(insert_df)} tables.",
        ).to_json()


if __name__ == "__main__":
    fire.Fire(IndexGenerator)
