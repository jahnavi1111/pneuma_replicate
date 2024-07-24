# Pneuma Architecture

Pneuma is built in several modules, each of which can be run independently. More details can be found [here](https://docs.google.com/document/d/16MsdIs80NssVtIhMq4r0RxXSpTKts_1MyyU2gf6ncpc).

## Table of Contents
1. [Getting Started](#getting-started)
2. [Registration Module](#registration-module)
    1. [Setup](#setup)
    2. [Read Table](#read-table)
    3. [Add Context](#add-context)
    4. [Add Summary](#add-summary)
3. [Summarizer Module](#summarizer-module)
    1. [Summarize](#summarize)
4. [Index Generation Module](#index-generation-module)
5. [Query Module](#query-module)

## Getting Started
This setup guide is written on a Windows environment with Python version 3.10.6.

For the summarizer module, you may want to access private/gated models such as `Meta-Llama-3-8B-Instruct`, as used in the paper. To do this, create a user access token in HuggingFace, then login using the following commands:
```shell
pip install -U "huggingface_hub[cli]"
huggingface-cli login [your_token]
```

Alternatively, you can pass the token directly when calling `initialize_pipeline` (`benchmark_generator/context/utils/pipeline_initializer.py`).

Clone the repository.
```shell
git clone https://github.com/TheDataStation/Pneuma
cd modules
```

(Recommended but not required) Create a virtual environment.

```
python -m venv venv
env\Scripts\activate.bat
```

Install required Python modules.

```
pip install -r requirements.txt
```

## Registration Module
This module is used to load data from various sources and context into DuckDB. Transformations, such as sorting rows, filtering out repeated values, etc. will also be done.

### Setup 
**Usage**: 
```shell
registration.py setup --db_path=PATH/TO/DATABASE_NAME.db
```

**Description**: Initializes the database schema. Creates a FILE_NAME.db file in the specified path.

**Example Usage**: 
```shell
registration.py setup --db_path=../storage.db
```

### Read Table
**Usage**: 
```shell
registration.py read_table --db_path=PATH/TO/DATABASE_NAME.db [OPTION]... PATH_TO_FILE.(csv/parquet) CREATOR_NAME
```

**Description**: Reads a table, formatted in CSV or PARQUET, from the local filesystem or an online storage bucket. If a file in a storage bucket is public, it can be read like a local file. The path of the table will be used as the ID.

- PATH_TO_FILE can be a path in the local filesystem or a bucket URI
- CREATOR_NAME is the name of the person who runs this command (TODO: Authenticate automatically)

**Options**:
- --s3_region=REGION

    Region of the s3 bucket.

-  --s3_access_key=AWS_ACCESS_KEY

    AWS access key ID

- --s3_secret_access_key=AWS_SECRET_ACCESS_KEY

    AWS secret access key ID

**Examples Usage**: 
```shell
registration.py read_table --db_path=../storage.db ../sample_data/5cq6-qygt.csv david csv
```

### Add Context
**Usage**: 
```shell
registration.py add_context --db_path=PATH/TO/DATABASE_NAME.db TABLE_ID PATH_TO_FILE.txt
```

**Description**: Creates a context entry for the specified table.

**Example Usage**: 
```shell
registration.py add_context --db_path=../storage.db ../sample_data/5cq6-qygt.csv ../sample_data/sample_context.txt
```

### Add Summary
**Usage**: 
```shell
registration.py add_summary --db_path=PATH/TO/DATABASE_NAME.db TABLE_ID PATH_TO_FILE.txt
```

**Description**: Creates a context entry for the specified table.

**Example Usage**: 
```shell
registration.py add_context --db_path=../storage.db ../sample_data/5cq6-qygt.csv ../sample_data/sample_summary.txt
```

## Summarizer Module
The summarized module generates content summaries of registered tables, which will then be stored in an index. These summaries will be useful for answering users' content-related questions

### Summarize
**Usage**: 
```shell
summarizer.py summarize --db_path=PATH/TO/DATABASE_NAME.db TABLE_ID
```

**Description**: Generates summary entries for the specified table.

**Example Usage**: 
```shell
summarizer.py summarize --db_path=PATH/TO/DATABASE_NAME.db ../sample_data/5cq6-qygt.csv
```

## Index Generation Module
Text

## Query Module
Text