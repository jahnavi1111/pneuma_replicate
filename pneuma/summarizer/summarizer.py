import json
import sys
from pathlib import Path

import duckdb
import fire
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import os

from utils.pipeline_initializer import initialize_pipeline
from utils.prompting_interface import prompt_pipeline
from utils.response import Response, ResponseStatus
from utils.table_status import TableStatus


class Summarizer:
    def __init__(
        self,
        db_path: str = os.path.expanduser("~/Documents/Pneuma/out/storage.db"),
        hf_token: str = "",
    ):
        self.db_path = db_path
        self.connection = duckdb.connect(db_path)

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

    def summarize(self, table_id: str = None) -> str:
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

    def purge_tables(self) -> str:
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

    def __summarize_table_by_id(self, table_id: str) -> list[str]:
        status = self.connection.sql(
            f"SELECT status FROM table_status WHERE id = '{table_id}'"
        ).fetchone()[0]
        if status == str(TableStatus.SUMMARIZED) or status == str(TableStatus.DELETED):
            print(f"Table with ID {table_id} has already been summarized.")
            return []

        table_df = self.connection.sql(f"SELECT * FROM '{table_id}'").to_df()

        summaries = self.__produce_summaries(table_df)

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

    def __produce_summaries(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        summaries = []
        summaries.extend(self.__generate_column_summary(df))
        summaries.extend(self.__generate_descriptions(df))
        return summaries

    def __generate_column_summary(self, df: pd.DataFrame) -> list[str]:
        return [" | ".join(df.columns)]

    def __generate_descriptions(self, df: pd.DataFrame) -> list[str]:
        summaries = []
        cols = df.columns
        conversations = []
        for col in cols:
            prompt = self.__get_col_description_prompt(" | ".join(cols), col)
            conversations.append([{"role": "user", "content": prompt}])

        for i in tqdm(range(0, len(conversations), 3)):
            outputs = prompt_pipeline(
                self.pipe,
                conversations[i : i + 3],
                batch_size=3,
                context_length=8192,
                max_new_tokens=400,
                temperature=None,
                top_p=None,
            )
            for output in outputs:
                summary = output[-1]["content"]
                summaries.append(summary)
        return summaries

    def __get_col_description_prompt(self, columns: str, column: str):
        return f"""A table has the following columns:
/*
{columns}
*/
Describe briefly what the {column} column represents. If not possible, simply state "No description.\""""


if __name__ == "__main__":
    fire.Fire(Summarizer)
