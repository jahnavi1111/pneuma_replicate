# Pneuma-Retriever

To run the experiment, do the following steps:

1. Download the necessary datasets, summaries, models, and benchmarks in the `pneuma_summarizer/summaries`, `data_src`, and `models`.

2. Install the requirements with the following script:
```bash
pip install -r requirements.txt
pip install -r ../benchmark_generator/context/requirements.txt
```

3. Index the summaries using any of the scenarios, all of which corresponds to files with this name format: `index_vector_[].py`.

4. Produce embeddings for the summaries with `produce_question_embeddings.py`.

5. Run the benchmark with `nohup python -u pneuma_retriever_benchmarking.py >> pneuma_retriever_benchmarking.out &`. Do not forget to adjust the name of the index (look for the `TODO` comment in this file).
