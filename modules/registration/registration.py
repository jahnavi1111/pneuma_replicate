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
                context_count INTEGER NOT NULL DEFAULT 0 CHECK (context_count >= 0),
                summary_count INTEGER NOT NULL DEFAULT 0 CHECK (summary_count >= 0),
                status VARCHAR,
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
                table_name VARCHAR,
                context VARCHAR,
                )
            """
        )

        # DuckDB does not support "ON DELETE CASCADE" so be careful with deletions.
        self.connection.sql(
            """CREATE OR REPLACE TABLE table_summaries (
                id INTEGER DEFAULT nextval('id_seq') PRIMARY KEY,
                table_id VARCHAR NOT NULL REFERENCES table_status(id),
                table_name VARCHAR,
                summaries VARCHAR,
                )
            """
        )

    def read_csv(self, csv_path: str):
        name = csv_path.split("/")[-1][:-4]
        self.connection.sql(
            f"""CREATE TABLE {name} AS
            FROM read_csv(
                '{csv_path}',
                auto_detect=True,
                header=True
            )"""
        )

        self.connection.sql(
            f"""INSERT INTO table_status (id, table_name, status)
            VALUES ('{csv_path}', '{name}', 'ready')"""
        )

        return f"Table ID: {csv_path}"

    def add_context(self, table_id: str, context_path: str):
        with open(context_path, "r") as f:
            context = f.read()

        table_name = self.connection.sql(
            f"SELECT table_name FROM table_status WHERE id = '{table_id}'"
        ).fetchone()[0]

        context_id = self.connection.sql(
            f"""INSERT INTO table_contexts (table_id, table_name, context)
            VALUES ('{table_id}', '{table_name}', '{context}')
            RETURNING id"""
        ).fetchone()[0]

        self.connection.sql(
            f"""UPDATE table_status
            SET context_count = context_count + 1
            WHERE id = '{table_id}'"""
        )

        return f"Context ID: {context_id}"


if __name__ == "__main__":
    fire.Fire(Registration)
