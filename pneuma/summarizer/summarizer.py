import json
import logging
import os
import sys
from pathlib import Path

import duckdb
import fire
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logging_config import configure_logging
from utils.pipeline_initializer import initialize_pipeline
from utils.prompting_interface import prompt_pipeline
from utils.response import Response, ResponseStatus
from utils.summary_types import SummaryType
from utils.table_status import TableStatus

configure_logging()
logger = logging.getLogger("Summarizer")


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
        logger.info("TEST")
        if table_id is None or table_id == "":
            logger.info("Generating summaries for all unsummarized tables...")
            table_ids = [
                entry[0]
                for entry in self.connection.sql(
                    f"""SELECT id FROM table_status
                    WHERE status = '{TableStatus.REGISTERED}'"""
                ).fetchall()
            ]
            logger.info("Found %d unsummarized tables.", len(table_ids))
        else:
            table_ids = [table_id]

        all_summary_ids = []
        for table_id in table_ids:
            logger.info("Summarizing table with ID: %s", table_id)
            all_summary_ids.extend(self.__summarize_table_by_id(table_id))

        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"Total of {len(all_summary_ids)} summaries has been added "
            f"with IDs: {', '.join([str(summary_id) for summary_id in all_summary_ids])}.\n",
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
            logger.info("Dropping table with ID: %s", table_id)
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
            logger.warning("Table with ID %s has already been summarized.", table_id)
            return []

        table_df = self.connection.sql(f"SELECT * FROM '{table_id}'").to_df()

        standard_summary = self.__generate_column_summary(table_df)
        narration_summary = self.__generate_column_description(table_df)

        summary_ids = []

        summary_ids.append(
            self.connection.sql(
                f"""INSERT INTO table_summaries (table_id, summary, summary_type)
                VALUES ('{table_id}', '{json.dumps(standard_summary)}', '{SummaryType.STANDARD}')
                RETURNING id"""
            ).fetchone()[0]
        )

        summary_ids.append(
            self.connection.sql(
                f"""INSERT INTO table_summaries (table_id, summary, summary_type)
                VALUES ('{table_id}', '{json.dumps(narration_summary)}', '{SummaryType.NARRATION}')
                RETURNING id"""
            ).fetchone()[0]
        )

        self.connection.sql(
            f"""UPDATE table_status
            SET status = '{TableStatus.SUMMARIZED}'
            WHERE id = '{table_id}'"""
        )

        return summary_ids

    def __generate_column_summary(self, df: pd.DataFrame) -> str:
        return " | ".join(df.columns).strip()

    def __generate_column_description(self, df: pd.DataFrame) -> list[str]:
        # Used for quick local testing
        # return " description | ".join(df.columns).strip() + " description"
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

        # The summaries generated are summaries for each column. We want each document
        # to be a long string of all the column summaries.
        return " | ".join(summaries).strip()

    def __get_col_description_prompt(self, columns: str, column: str):
        return f"""A table has the following columns:
/*
{columns}
*/
Describe briefly what the {column} column represents. If not possible, simply state "No description.\""""


if __name__ == "__main__":
    fire.Fire(Summarizer)
