import pandas as pd
from elasticsearch import Elasticsearch
import elasticsearch as es
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from config import API_KEY
import time
from tqdm import tqdm

def get_stopwords_removed_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.casefold() not in stop_words]
    return ' '.join(filtered_text)

def get_tables_ranks(hits):
    # Get the tables and the corresponding ranks (based on their relevance scores).
    # E.g., [(table_1, 1), (table_2, 1), (table_3, 2), ...]
    if hits is None or len(hits) == 0:
        return []
    rank = 1
    prev_score = hits[0]['_score']
    tables_ranks = []
    for hit in hits:
        if hit['_score'] < prev_score:
            rank += 1
        tables_ranks.append((hit['_source']['table_id'], rank))
        prev_score = hit['_score']
    return tables_ranks

def evaluate_index(client, benchmark: pd.DataFrame, remove_stopwords=False):
    accuracy_sum = 0
    precision_at_1_sum = 0
    reciprocal_rank_sum = 0
    for i in tqdm(range(benchmark.shape[0])):
        expected_table = benchmark["table"][i][-9:]
        question = benchmark["question"][i]
        if remove_stopwords:
            question = get_stopwords_removed_text(question)
        search_query = {
            "query": {
                "match": {
                    "content": question
                }
            }
        }
        result = client.search(index="table", body=search_query).body
        tables_ranks = get_tables_ranks(result["hits"]["hits"])

        # Measure the performance
        for j, (table, rank) in enumerate(tables_ranks):
            if table == expected_table:
                accuracy_sum += 1
                if rank == 1:
                    precision_at_1_sum += 1
                reciprocal_rank_sum += (1 / (j + 1))
                break
    return {
        "accuracy": round(accuracy_sum / benchmark.shape[0], 2),
        "Mean Precision@1": round(precision_at_1_sum / benchmark.shape[0], 2),
        "MRR": round(reciprocal_rank_sum / benchmark.shape[0], 2),
    }

if __name__ == "__main__":
    client = Elasticsearch(
        "https://localhost:9200",
        api_key=API_KEY,  # Adjust
        ca_certs="http_ca.crt",
    )
    start = time.time()
    results = evaluate_index(client, pd.read_csv("benchmarks/BX2_chicago.csv"), remove_stopwords=True)
    print(results)
    print(f"Execution time: {time.time() - start} seconds.")
    