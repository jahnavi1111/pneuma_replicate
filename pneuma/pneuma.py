import os

import duckdb
import fire
from index_generator.index_generator import IndexGenerator
from query.query import Query
from registration.registration import Registration
from summarizer.summarizer import Summarizer


class Pneuma:
    def __init__(
        self,
        out_path: str = os.path.expanduser("~/Documents/Pneuma/out"),
        hf_token: str = "",
    ):
        os.makedirs(out_path, exist_ok=True)
        self.out_path = out_path
        self.db_path = out_path + "/storage.db"
        self.index_location = out_path + "/indexes"

        self.registration = Registration(self.db_path)
        self.summarizer = Summarizer(self.db_path, hf_token)
        self.index_generator = IndexGenerator(self.db_path, self.index_location)
        self.query = Query(self.db_path, self.index_location)

        self.connection = duckdb.connect(self.db_path)

        print(self.registration.setup())

    def add_tables(
        self,
        path: str,
        creator: str,
        source: str = "file",
        s3_region: str = None,
        s3_access_key: str = None,
        s3_secret_access_key: str = None,
    ) -> str:
        return self.registration.add_tables(
            path, creator, source, s3_region, s3_access_key, s3_secret_access_key
        )

    def add_metadata(
        self,
        metadata_path: str = "",
        metadata_type: str = "",
        table_id: str = "",
    ) -> str:
        return self.registration.add_metadata(metadata_path, metadata_type, table_id)

    def summarize(self, table_id: str = "") -> str:
        return self.summarizer.summarize(table_id)

    def purge_tables(self) -> str:
        return self.summarizer.purge_tables()

    def generate_index(self, index_name: str, table_ids: list | tuple = None) -> str:
        return self.index_generator.generate_index(index_name, table_ids)

    def query_index(self, index_name: str, query: str, k: int = 10) -> str:
        return self.query.query(index_name, query, k)

    def sanity_check(self) -> str:
        return "This works!"


def main():
    print("Hello From Pneuma Main")
    fire.Fire(Pneuma)


if __name__ == "__main__":
    main()
