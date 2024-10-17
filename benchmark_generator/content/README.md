## 1. Setup
Python 3.8 or higher.
```
pip install -r ../requirements.txt 
```

## 2. Import data
If you want to download data, go to section 3, otherwise,

Create a folder named `data` side by side with the repo folder `table_ingestion` (by default)

Create a folder for each dataset in `data`, e.g. `nyc_open`

Create a folder named `tables_csv` in each dataset folder. 

```
    data
        nyc_open
            tables_csv
```
Put csv files in `tables_csv`

Run
```
./import_tables.sh <dataset>
```
To import data

## 3. Download data
We prepared `chembl_10K` (Capital K) (https://www.ebi.ac.uk/chembl/)
```
./get_data.sh chembl_10K
```

## 4. Generate questions
4.1) Set OpenAI key
```
export OPENAI_API_KEY=<key>
```
4.2) Run
```
./gen_questions.sh <dataset> <number of questions>
```
Questions are output to `./output/<dataset>/questions.jsonl`

Prompt logs are in file `prompt/log/<dataset>/log_...`. You can copy the Table Caption and Table Data parts in the file to ChatGPT and then ask ChatGPT to format it.

```
Table Caption:
...
Table Data:
...
```
## 5. Annotate answer tables
To be continued