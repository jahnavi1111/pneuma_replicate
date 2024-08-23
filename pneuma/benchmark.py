import json
from time import time

from pneuma import Pneuma


def main():
    out_path = "out_benchmark/"
    pneuma = Pneuma(out_path=out_path)

    results = {}

    # Read Tables
    start_time = time()
    response = pneuma.add_tables("sample_data/csv", "benchmarking")
    response = json.loads(response)
    print(response)
    end_time = time()
    print(
        f"Time to read {response['data']['file_count']} tables: {end_time - start_time} seconds"
    )
    results["read_tables"] = {
        "file_count": response["data"]["file_count"],
        "time": end_time - start_time,
    }

    # Add Contexts
    start_time = time()
    response = pneuma.add_metadata("sample_data/metadata.csv")
    response = json.loads(response)
    print(response)
    end_time = time()
    print(
        f"Time to add {response['data']['file_count']} contexts: {end_time - start_time} seconds"
    )
    results["add_contexts"] = {
        "file_count": response["data"]["file_count"],
        "time": end_time - start_time,
    }

    # Summarize
    start_time = time()
    response = pneuma.summarize()
    response = json.loads(response)
    print(response)
    end_time = time()
    print(
        f"Time to summarize {len(response['data']['table_ids'])} tables: {end_time - start_time} seconds"
    )
    results["summarize"] = {
        "table_count": len(response["data"]["table_ids"]),
        "time": end_time - start_time,
    }

    # Generate Index
    start_time = time()
    response = pneuma.generate_index("benchmark_index")
    response = json.loads(response)
    print(response)
    end_time = time()
    print(
        f"Time to generate index with {len(response['data']['table_ids'])} tables: {end_time - start_time} seconds"
    )
    results["generate_index"] = {
        "table_count": len(response["data"]["table_ids"]),
        "time": end_time - start_time,
    }

    # Query Index
    start_time = time()
    response = pneuma.query_index("benchmark_index", "Why was the dataset created?", 3)
    response = json.loads(response)
    print(response)
    end_time = time()
    print(f"Time to query index: {end_time - start_time} seconds")
    results["query_index"] = {"time": end_time - start_time}

    # Write results to file
    with open(f"{out_path}/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
