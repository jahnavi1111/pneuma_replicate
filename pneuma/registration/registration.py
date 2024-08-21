import json
import logging
import os
import sys
from pathlib import Path

import duckdb
import fire
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logging_config import configure_logging
from utils.response import Response, ResponseStatus
from utils.summary_types import SummaryType
from utils.table_status import TableStatus

configure_logging()
logger = logging.getLogger("Registration")


class Registration:
    def __init__(
        self, db_path: str = os.path.expanduser("~/Documents/Pneuma/out/storage.db")
    ):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.connection = duckdb.connect(db_path)

    def setup(self) -> str:
        try:
            self.connection.execute("INSTALL httpfs")
            self.connection.execute("LOAD httpfs")
            logger.info("HTTPFS installed and loaded.")

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
            logger.info("Table status table created.")

            # Arbitrary auto-incrementing id for contexts and summaries.
            # Change to "CREATE IF NOT EXISTS" on production.
            self.connection.sql("CREATE SEQUENCE IF NOT EXISTS id_seq START 1")
            logger.info("ID sequence created.")

            # DuckDB does not support "ON DELETE CASCADE" so be careful with deletions.
            self.connection.sql(
                """CREATE TABLE IF NOT EXISTS table_contexts (
                    id INTEGER DEFAULT nextval('id_seq') PRIMARY KEY,
                    table_id VARCHAR NOT NULL REFERENCES table_status(id),
                    context JSON NOT NULL
                    )
                """
            )
            logger.info("Table contexts table created.")

            # DuckDB does not support "ON DELETE CASCADE" so be careful with deletions.
            self.connection.sql(
                """CREATE TABLE IF NOT EXISTS table_summaries (
                    id INTEGER DEFAULT nextval('id_seq') PRIMARY KEY,
                    table_id VARCHAR NOT NULL REFERENCES table_status(id),
                    summary JSON NOT NULL,
                    summary_type VARCHAR NOT NULL,
                    )
                """
            )
            logger.info("Table summaries table created.")

            self.connection.sql(
                """CREATE TABLE IF NOT EXISTS indexes (
                    id INTEGER default nextval('id_seq') PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    location VARCHAR NOT NULL,
                    )
                """
            )
            logger.info("Indexes table created.")

            self.connection.sql(
                """CREATE TABLE IF NOT EXISTS index_table_mappings (
                    index_id INTEGER NOT NULL REFERENCES indexes(id),
                    table_id VARCHAR NOT NULL REFERENCES table_status(id),
                    PRIMARY KEY (index_id, table_id)
                    )
                """
            )
            logger.info("Index table mappings table created.")

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
            logger.info("Query history table created.")

            return Response(
                status=ResponseStatus.SUCCESS,
                message="Database Initialized.",
            ).to_json()
        except Exception as e:
            return Response(
                status=ResponseStatus.ERROR,
                message=f"Error initializing database: {e}",
            ).to_json()

    def add_table(
        self,
        path: str,
        creator: str,
        source: str = "file",
        s3_region: str = None,
        s3_access_key: str = None,
        s3_secret_access_key: str = None,
    ) -> str:
        if source not in ["file", "s3"]:
            return "Invalid source. Please use 'file' or 's3'."

        if source == "s3":
            self.connection.execute(
                f"""
            SET s3_region='{s3_region}';
            SET s3_access_key_id='{s3_access_key}';
            SET s3_secret_access_key='{s3_secret_access_key}';
            """
            )

        if os.path.isfile(path):
            return self.__read_table_file(path, creator).to_json()
        if os.path.isdir(path):
            return self.__read_table_folder(path, creator).to_json()

        return Response(
            status=ResponseStatus.ERROR,
            message=f"Invalid path: {path}",
        ).to_json()

    def add_metadata(
        self, metadata_path: str, metadata_type: str = "", table_id: str = ""
    ) -> str:
        if os.path.isfile(metadata_path):
            return self.__read_metadata_file(
                metadata_path, metadata_type, table_id
            ).to_json()
        if os.path.isdir(metadata_path):
            return self.__read_metadata_folder(
                metadata_path, metadata_type, table_id
            ).to_json()

        return Response(
            status=ResponseStatus.ERROR,
            message=f"Invalid path: {metadata_path}",
        ).to_json()

    def __read_table_file(
        self,
        path: str,
        creator: str,
    ) -> Response:
        # Index -1 to get the file extension, then slice [1:] to remove the dot.
        file_type = os.path.splitext(path)[-1][1:]

        if file_type not in ["csv", "parquet"]:
            return Response(
                status=ResponseStatus.ERROR,
                message="Invalid file type. Please use 'csv' or 'parquet'.",
            )

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

        # Check if table with the same hash already exist
        if self.connection.sql(
            f"SELECT * FROM table_status WHERE hash = '{table_hash}'"
        ).fetchone():
            return Response(
                status=ResponseStatus.ERROR,
                message="This table already exists in the database.",
            )

        # Check if table with the same ID already exists.
        # This means the same table with updated data is being registered.
        if self.connection.sql(
            f"SELECT * FROM table_status WHERE id = '{path}'"
        ).fetchone():
            # TODO
            pass

        # The double quote is necessary to consider the path, which may contain
        # full stop that may mess with schema as a single string. Having single quote
        # inside breaks the query, so having the double quote INSIDE the single quote
        # is the only way to make it work.
        table.create(f'"{path}"')

        self.connection.sql(
            f"""INSERT INTO table_status (id, table_name, status, creator, hash)
            VALUES ('{path}', '{name}', '{TableStatus.REGISTERED}', '{creator}', '{table_hash}')"""
        )
        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"Table with ID: {path} has been added to the database.",
        )

    def __read_table_folder(self, folder_path: str, creator: str) -> Response:
        logger.info("Reading folder %s...", folder_path)
        paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        file_count = 0
        for path in paths:
            logger.info("Processing %s...", path)

            # If the path is a folder, recursively read the folder.
            if os.path.isdir(path):
                logger.info(self.__read_table_folder(path, creator).message)
                continue

            response = self.__read_table_file(path, creator)
            logger.info(
                "Processing table %s %s: %s",
                path,
                response.status.value,
                response.message,
            )
            file_count += 1

        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"{file_count} files in folder {folder_path} has been processed.",
        )

    def __insert_metadata(
        self, metadata_type: str, metadata_content: str, table_id: str
    ) -> Response:
        payload = {
            "payload": metadata_content.strip(),
        }

        if metadata_type == "context":
            metadata_id = self.connection.sql(
                f"""INSERT INTO table_contexts (table_id, context)
                VALUES ('{table_id}', '{json.dumps(payload)}')
                RETURNING id"""
            ).fetchone()[0]
        elif metadata_type == "summary":
            metadata_id = self.connection.sql(
                f"""INSERT INTO table_summaries (table_id, summary, summary_type)
                VALUES ('{table_id}', '{json.dumps(payload)}', '{SummaryType.USER_GENERATED}')
                RETURNING id"""
            ).fetchone()[0]

        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"{metadata_type.capitalize()} ID: {metadata_id}",
        )

    def __read_metadata_file(
        self, metadata_path: str, metadata_type: str, table_id: str
    ) -> Response:
        # Index -1 to get the file extension, then slice [1:] to remove the dot.
        file_type = os.path.splitext(metadata_path)[-1][1:]

        if file_type not in ["txt", "csv"]:
            return Response(
                status=ResponseStatus.ERROR,
                message="Invalid file type. Please use 'txt' or 'csv'.",
            )

        if file_type == "txt":
            with open(metadata_path, "r") as f:
                metadata_content = f.read()

            return self.__insert_metadata(metadata_type, metadata_content, table_id)

        if file_type == "csv":
            metadata_df = pd.read_csv(metadata_path)
            # iterate over rows with iterrows()
            for index, row in metadata_df.iterrows():
                table_id = row["table_id"]
                metadata_type = row["metadata_type"]
                metadata_content = row["value"]
                logger.info(
                    self.__insert_metadata(
                        metadata_type, metadata_content, table_id
                    ).message
                )
            return Response(
                status=ResponseStatus.SUCCESS,
                message=f"{len(metadata_df)} metadata entries has been added.",
            )

        return None

    def __read_metadata_folder(
        self, metadata_path: str, metadata_type: str, table_id: str
    ) -> Response:
        logger.info("Reading metadata folder %s...", metadata_path)
        paths = [os.path.join(metadata_path, f) for f in os.listdir(metadata_path)]
        file_count = 0
        for path in paths:
            logger.info("Processing %s...", path)

            # If the path is a folder, recursively read the folder.
            if os.path.isdir(path):
                logger.info(
                    self.__read_metadata_folder(path, metadata_type, table_id).message
                )
                continue

            response = self.__read_metadata_file(path, metadata_type, table_id)
            logger.info(
                "Processing metadata %s %s: %s",
                path,
                response.status.value,
                response.message,
            )
            file_count += 1

        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"{file_count} files in folder {metadata_path} has been processed.",
        )


if __name__ == "__main__":
    fire.Fire(Registration)
