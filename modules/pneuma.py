import duckdb
from index_generator.index_generator import IndexGenerator
from query.query import Query
from registration.registration import Registration
from summarizer.summarizer import Summarizer


class Pneuma:
    def __init__(self, hf_token: str = ""):
        self.out_path = "out"
        self.db_path = "out/storage.db"
        self.index_location = "out/indexes"

        self.registration = Registration(self.db_path, self.out_path)
        self.summarizer = Summarizer(self.db_path, hf_token)
        self.index_generator = IndexGenerator(self.db_path, self.index_location)
        self.query = Query(self.db_path, self.index_location)

        self.connection = duckdb.connect(self.db_path)

        print(self.registration.setup())

    def add_table(
        self,
        path: str,
        creator: str,
        source: str = "file",
        s3_region: str = None,
        s3_access_key: str = None,
        s3_secret_access_key: str = None,
    ):
        return self.registration.add_table(
            path, creator, source, s3_region, s3_access_key, s3_secret_access_key
        )

    def add_context(self, table_id: str, context_path: str):
        return self.registration.add_metadata(context_path, "context", table_id)

    def add_summary(self, table_id: str, summary_path: str):
        return self.registration.add_metadata(summary_path, "summary", table_id)

    def summarize(self, table_id: str = ""):
        return self.summarizer.summarize(table_id)

    def generate_index(self, index_name: str, table_ids: list | tuple):
        return self.index_generator.generate_index(index_name, table_ids)

    def query_index(self, index_name: str, query: str, k: int = 10):
        return self.query.query(index_name, query, k)


def main():
    pneuma = Pneuma()
    print("Hello world")


if __name__ == "__main__":
    main()
