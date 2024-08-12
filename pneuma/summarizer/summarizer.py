import json
import math
import random
import sys
from pathlib import Path

import duckdb
import fire
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
# from summarizer.pipeline_initializer import initialize_pipeline
# from summarizer.prompting_interface import prompt_pipeline
from utils.response import Response, ResponseStatus
from utils.table_status import TableStatus


class Summarizer:
    def __init__(self, db_path: str, hf_token: str = ""):
        self.db_path = db_path
        self.connection = duckdb.connect(db_path)

        # self.pipe = initialize_pipeline(
        #     "meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16, hf_token
        # )

        # Use small model for local testing
        # self.pipe = initialize_pipeline("TinyLlama/TinyLlama_v1.1", torch.bfloat16)

    def summarize(self, table_id: str = None):
        if table_id is None or table_id == "":
            print("Generating summaries for all unsummarized tables...")
            table_ids = [
                entry[0]
                for entry in self.connection.sql(
                    f"""SELECT id FROM table_status
                    WHERE status = '{TableStatus.REGISTERED}'"""
                ).fetchall()
            ]
            print(f"Found {len(table_ids)} unsummarized tables.")
        else:
            table_ids = [table_id]

        all_summary_ids = []
        for table_id in table_ids:
            print(f"Summarizing table with ID: {table_id}")
            all_summary_ids.extend(self.__summarize_table_by_id(table_id))

        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"Total of {len(all_summary_ids)} summaries has been added "
            f"with IDs: {', '.join([str(i[0]) for i in all_summary_ids])}.\n",
        ).to_json()

    def purge_tables(self):
        # drop summarized tables
        summarized_table_ids = [
            entry[0]
            for entry in self.connection.sql(
                f"SELECT id FROM table_status WHERE status = '{TableStatus.SUMMARIZED}'"
            ).fetchall()
        ]

        for table_id in summarized_table_ids:
            print(f"Dropping table with ID: {table_id}")
            self.connection.sql(f'DROP TABLE "{table_id}"')
            self.connection.sql(
                f"""UPDATE table_status
                SET status = '{TableStatus.DELETED}'
                WHERE id = '{table_id}'"""
            )

        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"Total of {len(summarized_table_ids)} tables have been purged.\n",
        ).to_json()

    def __summarize_table_by_id(self, table_id: str):
        status = self.connection.sql(
            f"SELECT status FROM table_status WHERE id = '{table_id}'"
        ).fetchone()[0]
        if status == str(TableStatus.SUMMARIZED) or status == str(TableStatus.DELETED):
            print(f"Table with ID {table_id} has already been summarized.")
            return []

        table_df = self.connection.sql(f"SELECT * FROM '{table_id}'").to_df()

        summaries = self.produce_summaries(table_df)

        insert_df = pd.DataFrame.from_dict(
            {
                "table_id": [table_id] * len(summaries),
                "summary": [
                    json.dumps({"payload": summary.strip()}) for summary in summaries
                ],
            }
        )

        summary_ids = self.connection.sql(
            """INSERT INTO table_summaries (table_id, summary)
            SELECT * FROM insert_df
            RETURNING id"""
        ).fetchall()

        self.connection.sql(
            f"""UPDATE table_status
            SET status = '{TableStatus.SUMMARIZED}'
            WHERE id = '{table_id}'"""
        )

        return summary_ids

    def produce_summaries(
        self,
        df: pd.DataFrame,
    ):
        return [" | ".join(df.columns)]


if __name__ == "__main__":
    fire.Fire(Summarizer)
