import duckdb
import fire


class IndexGenerator:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = duckdb.connect(db_path)

    def generate_index(self, index_name: str, table_ids: list | tuple):
        print(type(table_ids))
        print(table_ids)
        print(table_ids[0])


if __name__ == "__main__":
    fire.Fire(IndexGenerator)
