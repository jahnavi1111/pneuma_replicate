import duckdb
import fire


class Registration:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = duckdb.connect(db_path)

    def setup(self):
        # We would probably want to use "CREATE IF NOT EXISTS", but we use "CREATE OR REPLACE"
        # for now so changes while developing are immediately reflected in the database.
        self.connection.sql(
            """CREATE OR REPLACE TABLE table_status (
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
                context VARCHAR,
                )
            """
        )

        # DuckDB does not support "ON DELETE CASCADE" so be careful with deletions.
        self.connection.sql(
            """CREATE OR REPLACE TABLE table_summaries (
                id INTEGER DEFAULT nextval('id_seq') PRIMARY KEY,
                table_id VARCHAR NOT NULL REFERENCES table_status(id),
                summary VARCHAR,
                )
            """
        )

    def read_csv(self, csv_path: str):
        name = csv_path.split("/")[-1][:-4]
        table = self.connection.sql(
            f"""SELECT *
                FROM read_csv(
                '{csv_path}',
                auto_detect=True,
                header=True
            )"""
        )

        table_hash = self.connection.sql(
            f"""SELECT md5(string_agg(tbl::text, ''))s
                FROM read_csv(
                '{csv_path}',
                auto_detect=True,
                header=True
            ) AS tbl"""
        ).fetchone()[0]

        # Check if table with the same hash already exist
        if self.connection.sql(
            f"SELECT * FROM table_status WHERE hash = '{table_hash}'"
        ).fetchone():
            return "This table already exists in the database."

        self.connection.register(csv_path, table)

        self.connection.sql(
            f"""INSERT INTO table_status (id, table_name, status, creator, hash)
            VALUES ('{csv_path}', '{name}', 'ready', 'fake creator', '{table_hash}')"""
        )

        return f"Table with ID: {csv_path} has been added to the database."

    def add_context(self, table_id: str, context_path: str):
        with open(context_path, "r") as f:
            context = f.read()

        context_id = self.connection.sql(
            f"""INSERT INTO table_contexts (table_id, context)
            VALUES ('{table_id}', '{context}')
            RETURNING id"""
        ).fetchone()[0]

        return f"Context ID: {context_id}"

    def add_summary(self, table_id: str, summary_path: str):
        with open(summary_path, "r") as f:
            summary = f.read()

        summary_id = self.connection.sql(
            f"""INSERT INTO table_summaries (table_id, summary)
            VALUES ('{table_id}', '{summary}')
            RETURNING id"""
        ).fetchone()[0]

        return f"Summary ID: {summary_id}"


if __name__ == "__main__":
    fire.Fire(Registration)
