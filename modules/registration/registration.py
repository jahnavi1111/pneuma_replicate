import json
import os
import sys
from pathlib import Path

import duckdb
import fire

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.table_status import TableStatus


class Registration:
    def __init__(self, db_path: str):
        os.makedirs("../out", exist_ok=True)
        self.db_path = db_path
        self.connection = duckdb.connect(db_path)

    def setup(self):
        self.connection.execute("INSTALL httpfs")
        self.connection.execute("LOAD httpfs")

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
        self.connection.sql("CREATE SEQUENCE IF NOT EXISTS id_seq START 1")

        # DuckDB does not support "ON DELETE CASCADE" so be careful with deletions.
        self.connection.sql(
            """CREATE TABLE IF NOT EXISTS table_contexts (
                id INTEGER DEFAULT nextval('id_seq') PRIMARY KEY,
                table_id VARCHAR NOT NULL REFERENCES table_status(id),
                context JSON NOT NULL
                )
            """
        )

        # DuckDB does not support "ON DELETE CASCADE" so be careful with deletions.
        self.connection.sql(
            """CREATE TABLE IF NOT EXISTS table_summaries (
                id INTEGER DEFAULT nextval('id_seq') PRIMARY KEY,
                table_id VARCHAR NOT NULL REFERENCES table_status(id),
                summary JSON NOT NULL
                )
            """
        )

        self.connection.sql(
            """CREATE TABLE IF NOT EXISTS indexes (
                id INTEGER default nextval('id_seq') PRIMARY KEY,
                name VARCHAR NOT NULL,
                location VARCHAR NOT NULL,
                )
            """
        )

        self.connection.sql(
            """CREATE TABLE IF NOT EXISTS index_table_mappings (
                index_id INTEGER NOT NULL REFERENCES indexes(id),
                table_id VARCHAR NOT NULL REFERENCES table_status(id),
                PRIMARY KEY (index_id, table_id)
                )
            """
        )

        # TODO: Adjust the response column to the actual response type.
        self.connection.sql(
            """CREATE TABLE IF NOT EXISTS query_history (
                time TIMESTAMP DEFAULT CURRENT_TIMESTAMP PRIMARY KEY,
                table_id VARCHAR NOT NULL REFERENCES table_status(id), 
                query VARCHAR NOT NULL,
                response VARCHAR NOT NULL,
                querant VARCHAR NOT NULL
                )
            """
        )

    def read_table(
        self,
        path: str,
        creator: str,
        source: str = "file",
        s3_region: str = None,
        s3_access_key: str = None,
        s3_secret_access_key: str = None,
    ):
        if source == "s3":
            self.connection.execute(
                f"""
            SET s3_region='{s3_region}';
            SET s3_access_key_id='{s3_access_key}';
            SET s3_secret_access_key='{s3_secret_access_key}';
            """
            )
        elif source != "file":
            return "Invalid source. Please use 'file' or 's3'."

        # Index -1 to get the file extension, then slice [1:] to remove the dot.
        file_type = os.path.splitext(path)[-1][1:]

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

        # The double quote is necessary to consider the path, which may contain
        # full stop that may mess with schema as a single string. Having single quote
        # inside breaks the query, so having the double quote INSIDE the single quote
        # is the only way to make it work.
        table.create(f'"{path}"')

        self.connection.sql(
            f"""INSERT INTO table_status (id, table_name, status, creator, hash)
            VALUES ('{path}', '{name}', '{TableStatus.REGISTERED}', '{creator}', '{table_hash}')"""
        )

        return f"Table with ID: {path} has been added to the database."

    def add_context(self, table_id: str, context_path: str):
        with open(context_path, "r") as f:
            context = f.read()

        context_dict = {
            "payload": context.strip(),
        }

        context_id = self.connection.sql(
            f"""INSERT INTO table_contexts (table_id, context)
            VALUES ( '{table_id}', '{json.dumps(context_dict)}' )
            RETURNING id"""
        ).fetchone()[0]

        return f"Context ID: {context_id}"

    def add_summary(self, table_id: str, summary_path: str):
        with open(summary_path, "r") as f:
            summary = f.read()

        summary_dict = {
            "payload": summary.strip(),
        }

        summary_id = self.connection.sql(
            f"""INSERT INTO table_summaries (table_id, summary)
            VALUES ( '{table_id}', '{json.dumps(summary_dict)}' )
            RETURNING id"""
        ).fetchone()[0]

        return f"Summary ID: {summary_id}"


if __name__ == "__main__":
    fire.Fire(Registration)
