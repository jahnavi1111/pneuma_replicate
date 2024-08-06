import json
import math
import random
import sys
from pathlib import Path

import duckdb
import fire
import pandas as pd
import torch

# This is needed when importing this module from pneuma.py in a directory above
sys.path.append("../")

sys.path.append("../..")
# Repackage this when the module is ready
from benchmark_generator.context.utils.pipeline_initializer import initialize_pipeline
from benchmark_generator.context.utils.prompting_interface import prompt_pipeline

sys.path.append(str(Path(__file__).resolve().parent.parent))
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
        self.pipe = initialize_pipeline("TinyLlama/TinyLlama_v1.1", torch.bfloat16)

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

    def __summarize_table_by_id(self, table_id: str):
        status = self.connection.sql(
            f"SELECT status FROM table_status WHERE id = '{table_id}'"
        ).fetchone()[0]
        if status == str(TableStatus.SUMMARIZED) or status == str(TableStatus.DELETED):
            print(f"Table with ID {table_id} has already been summarized.")
            return []

        table_df = self.connection.sql(f"SELECT * FROM '{table_id}'").to_df()

        # summaries = self.produce_summaries(table_df)
        summaries = [
            "This summary is 'generated' one",
            "This is the second 'generated' summary",
        ]

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
        row_summaries_percentage=0.05,
    ):
        all_summaries: list[str] = []
        print("Start summarizing table")

        print("Summarizing some rows")
        result_df: pd.DataFrame = self.remove_similar_rows(df, threshold=0.9)
        sampled_df = result_df.sample(
            math.ceil(row_summaries_percentage * len(result_df)), random_state=42
        ).reset_index(drop=True)
        row_summaries = self.get_row_summaries(sampled_df)

        print("Summarizing the overall table")
        table_summary = self.get_table_summary(row_summaries)

        num_cols, cat_cols = self.get_categorical_numerical_cols(df)

        print("Summarizing the numerical cols")
        num_cols_summaries = self.get_num_columns_summaries(table_summary, df, num_cols)

        print("Summarizing the categorical cols")
        cat_cols_summaries = self.get_cat_columns_summaries(table_summary, df, cat_cols)

        all_summaries = (
            row_summaries + [table_summary] + num_cols_summaries + cat_cols_summaries
        )
        return all_summaries

    def get_cat_columns_summaries(
        self,
        table_summary: str,
        df: pd.DataFrame,
        cat_cols: list[str],
        show_unique_cat_threshold=10,
    ):
        cat_cols_summaries: list[str] = []
        for cat_col in cat_cols:
            print(f"==> Col {cat_col}", flush=True)
            col_stats = "; ".join(
                [f"{index}: {value}" for index, value in df[cat_col].describe().items()]
            )

            if len(df[cat_col].unique()) <= show_unique_cat_threshold:
                # Show unique values as well if less than the threshold
                col_stats += f"; categories: {df[cat_col].unique()}"

            prompt = self.get_col_summary_prompt(table_summary, cat_col, col_stats)
            cat_col_summary = prompt_pipeline(
                self.pipe,
                [{"role": "user", "content": prompt}],
                temperature=None,
                top_p=None,
                max_new_tokens=200,
            )[-1]["content"]
            cat_cols_summaries.append(cat_col_summary)
        return cat_cols_summaries

    def get_num_columns_summaries(
        self,
        table_summary,
        df: pd.DataFrame,
        num_cols: list[str],
    ):
        num_cols_summaries: list[str] = []
        for num_col in num_cols:
            print(f"==> Col {num_col}")
            col_stats = "; ".join(
                [f"{index}: {value}" for index, value in df[num_col].describe().items()]
            )
            prompt = self.get_col_summary_prompt(table_summary, num_col, col_stats)
            num_col_summary = prompt_pipeline(
                self.pipe,
                [{"role": "user", "content": prompt}],
                temperature=None,
                top_p=None,
                max_new_tokens=200,
            )[-1]["content"]
            num_cols_summaries.append(num_col_summary)
        return num_cols_summaries

    def get_table_summary(self, row_summaries: list[str]) -> str:
        """Summarize overall meaning of a table"""
        random.seed(42)
        sample_size = min(3, len(row_summaries))
        selected_summaries = random.sample(row_summaries, sample_size)

        summary_prompt = self.get_table_summary_prompt(selected_summaries)
        table_summary = prompt_pipeline(
            self.pipe,
            [{"role": "user", "content": summary_prompt}],
            temperature=None,
            top_p=None,
            max_new_tokens=150,
        )[-1]["content"]
        return table_summary

    def get_row_summaries(self, sampled_df: pd.DataFrame):
        row_summaries: list[str] = []
        for i in range(len(sampled_df)):
            print(f"Summarizing row {i} of sampled_df")
            row_info = self.get_row_info_from_df(sampled_df, i)
            prompt = self.get_row_summary_prompt(row_info)
            row_summary = prompt_pipeline(
                self.pipe,
                [{"role": "user", "content": prompt}],
                temperature=None,
                top_p=None,
                max_new_tokens=400,
            )[-1]["content"]
            row_summaries.append(row_summary)
        return row_summaries

    ### HELPER FUNCTIONS

    def row_similarity(self, row1: pd.Series, row2: pd.Series):
        """Compute how many columns are the same between two rows"""
        similarity = (row1 == row2).mean()
        return similarity

    def remove_similar_rows(self, df: pd.DataFrame, threshold=0.9):
        """Remove rows that have a similarity greater than or equal to the threshold"""
        to_drop = set()  # Set of indices to drop
        for i in range(len(df)):
            if i in to_drop:
                continue
            for j in range(i + 1, len(df)):
                if j in to_drop:
                    continue
                if self.row_similarity(df.iloc[i], df.iloc[j]) >= threshold:
                    to_drop.add(j)
        return df.drop(to_drop).reset_index(drop=True)

    def get_categorical_numerical_cols(self, df: pd.DataFrame, s=5):
        """
        Return two lists: numerical and categorical columns

        Side effect: int columns that are actually categorical will be converted to object data type
        """
        to_be_checked: list[str] = []
        num_cols: list[str] = []
        cat_cols: list[str] = []
        for col in df.columns:
            if df[col].dtype == "int64":
                # Check whether an integer column is actually categorical
                to_be_checked.append(col)
            elif df[col].dtype == "float64":
                num_cols.append(col)
            else:
                cat_cols.append(col)

        for col in to_be_checked:
            if s <= len(df):
                col_stats = f"{list(df[col])} ({len(df[col].unique())}/{len(df[col])} unique values)"
            else:
                col_stats = f"{list(df[col].sample(s, random_state=42))} ({len(df[col].unique())}/{len(df[col])} unique values)"
            prompt = self.get_dtype_check_prompt(col, col_stats)
            dtype_ans = prompt_pipeline(
                self.pipe,
                [{"role": "user", "content": prompt}],
                temperature=None,
                top_p=None,
                max_new_tokens=5,
            )[-1]["content"]
            if dtype_ans.strip().lower().startswith(
                "yes"
            ) or dtype_ans.strip().lower().startswith("**yes"):
                cat_cols.append(col)
                df[col] = df[col].astype(object)
        return (num_cols, cat_cols)

    ### PROMPTS

    def get_row_info_from_df(self, df: pd.DataFrame, row_idx=0):
        col = "col: " + " | ".join(df.columns)
        row = "row: " + " | ".join(df.iloc[row_idx].astype(str).str.strip())
        return col + "\n" + row

    def get_table_summary_prompt(self, selected_summaries: list[str]):
        return f"""Given these pieces of information regarding some row(s) of a dataset:
    /*
    {"; ".join(selected_summaries)}
    */
    Guess reasonably what this dataset is about. Respond briefly."""

    def get_dtype_check_prompt(self, col_name: str, stats: str):
        """Get prompt to check if an integer column is actually ID/categorical or not"""
        return f"""Do you think a column named {col_name}, which has values such as {stats}, is an identifier or categorical column? Begin your argument with yes/no."""

    def get_col_summary_prompt(self, dataset_info: str, col_name: str, col_stats: str):
        return f"""Given the following description of a dataset and statistics about column {col_name} of the dataset, generate a short paragraph about the column statistics while considering the description:
    Dataset description = "{dataset_info}"
    Column statistics = "{col_stats}"""

    def get_row_summary_prompt(self, row_info: str):
        return f"""Given this row of a dataset:
    /*
    {row_info}
    */
    Summarize it comprehensively into a single paragraph without adding any external information."""


if __name__ == "__main__":
    fire.Fire(Summarizer)
