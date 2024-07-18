import sys

import duckdb
import fire

sys.path.append("..")
from utils.table_status import TableStatus


class Registration:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = duckdb.connect(db_path)

    def setup(self):
        self.connection.sql("INSTALL httpfs")
        self.connection.sql("LOAD httpfs")

        self.connection.sql(
            """CREATE TABLE IF NOT EXISTS table_status (
                id VARCHAR PRIMARY KEY,
                table_name VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                time_created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                creator VARCHAR NOT NULL,
                hash VARCHAR NOT NULL,
                )
            """
        )

        # Arbitrary auto-incrementing id for contexts and summaries.
        # Change to "CREATE IF NOT EXISTS" on production.
        self.connection.sql("CREATE OR REPLACE SEQUENCE id_seq START 1")

        # DuckDB does not support "ON DELETE CASCADE" so be careful with deletions.
        self.connection.sql(
            """CREATE OR REPLACE TABLE table_contexts (
                id INTEGER DEFAULT nextval('id_seq') PRIMARY KEY,
                table_id VARCHAR NOT NULL REFERENCES table_status(id),
                context STRUCT(payload VARCHAR),
                )
            """
        )

        # DuckDB does not support "ON DELETE CASCADE" so be careful with deletions.
        self.connection.sql(
            """CREATE OR REPLACE TABLE table_summaries (
                id INTEGER DEFAULT nextval('id_seq') PRIMARY KEY,
                table_id VARCHAR NOT NULL REFERENCES table_status(id),
                summary STRUCT (payload VARCHAR),
                )
            """
        )

    def read_file(self, path: str, creator: str, file_type: str = "csv"):
        if file_type == "csv":
            name = path.split("/")[-1][:-4]
            table = self.connection.sql(
                f"""SELECT *
                    FROM read_csv(
                        '{path}',
                        auto_detect=True,
                        header=True
                    )"""
            )
            table_hash = self.connection.sql(
                f"""SELECT md5(string_agg(tbl::text, ''))
                FROM read_csv(
                    '{path}',
                    auto_detect=True,
                    header=True
                ) AS tbl"""
            ).fetchone()[0]
        elif file_type == "parquet":
            name = path.split("/")[-1][:-8]
            table = self.connection.sql(
                f"""SELECT *
                FROM read_parquet(
                    '{path}'
                )"""
            )
            table_hash = self.connection.sql(
                f"""SELECT md5(string_agg(tbl::text, ''))
                FROM read_parquet(
                    '{path}'
                ) AS tbl"""
            ).fetchone()[0]
        else:
            return "Invalid file type. Please use 'csv' or 'parquet'."

        # Check if table with the same hash already exist
        if self.connection.sql(
            f"SELECT * FROM table_status WHERE hash = '{table_hash}'"
        ).fetchone():
            return "This table already exists in the database."

        self.connection.register(path, table)

        self.connection.sql(
            f"""INSERT INTO table_status (id, table_name, status, creator, hash)
            VALUES ('{path}', '{name}', '{TableStatus.REGISTERED}', '{creator}', '{table_hash}')"""
        )

        return f"Table with ID: {path} has been added to the database."

    def add_context(self, table_id: str, context_path: str):
        with open(context_path, "r") as f:
            context = f.read()

        context_id = self.connection.sql(
            f"""INSERT INTO table_contexts (table_id, context)
            VALUES ( '{table_id}', {{'payload': '{context}'}} )
            RETURNING id"""
        ).fetchone()[0]

        return f"Context ID: {context_id}"

    def add_summary(self, table_id: str, summary_path: str):
        with open(summary_path, "r") as f:
            summary = f.read()

        summary_id = self.connection.sql(
            f"""INSERT INTO table_summaries (table_id, summary)
            VALUES ( '{table_id}', {{'payload': '{summary}'}} )
            RETURNING id"""
        ).fetchone()[0]

        return f"Summary ID: {summary_id}"


if __name__ == "__main__":
    fire.Fire(Registration)
