from elasticsearch import Elasticsearch
import elasticsearch as es
import pandas as pd
import os
# from config import API_KEY

def load_benchmark(bench_name: str):
    if bench_name == "BX1":
        benchmark = pd.read_csv("benchmarks/BX1_chicago.csv")
    elif bench_name == "BX2":
        benchmark = pd.read_csv("benchmarks/BX2_chicago.csv")
    return benchmark

def index_a_table(client, table_id, file_path):
    # Read the entire CSV file content
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        # Create a document containing the file content as a single string
        document = {
            "table_id": table_id,
            "content": file_content
        }

        # Index the document
        client.index(index='table', id=table_id, body=document)
        print(f"table {table_id} indexed successfully.")
    else:
        print("File not found.")

def index_all(dir_path):
    for file in os.listdir(dir_path):
        if file.endswith(".csv"):
            table_id = file.split(".")[0]
            file_path = os.path.join(dir_path, file)
            index_a_table(client, table_id, file_path)



if __name__ == "__main__":
    client = Elasticsearch(
        "https://localhost:9200",
        api_key=API_KEY,  # Adjust
        ca_certs="http_ca.crt",
    )

    index_all("chicago_open_data")

    