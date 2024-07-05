import duckdb
import fire


class Registration:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def read_csv(self, csv_path: str):
        name = csv_path.split("/")[-1][:-4]
        print(name)
        with duckdb.connect(self.db_path) as con:
            con.sql(
                f"""CREATE TABLE {name} AS
                FROM read_csv(
                    '{csv_path}',
                    auto_detect=True,
                    header=True
                )"""
            )
            print(con.sql("SHOW TABLES"))
            print(con.sql("SELECT * FROM BX1_chicago"))


if __name__ == "__main__":
    fire.Fire(Registration)
