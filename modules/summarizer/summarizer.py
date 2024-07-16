import duckdb
import fire


class Summarizer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = duckdb.connect(db_path)

    def summarize(self, table_id: str):
        # get the table as df
        table_df = self.connection.sql(f"SELECT * FROM '{table_id}'").to_df()

        # DUMMY SUMMARIZATION
        summary = table_df.describe().to_string()

        summary_id = self.connection.sql(
            f"""INSERT INTO table_summaries (table_id, summary)
            VALUES ('{table_id}', '{summary}')
            RETURNING id"""
        ).fetchone()[0]

        self.connection.sql(
            f"""UPDATE table_status
            SET summary_count = summary_count + 1
            WHERE id = '{table_id}'"""
        )

        return f"Summary ID: {summary_id}"


if __name__ == "__main__":
    fire.Fire(Summarizer)
