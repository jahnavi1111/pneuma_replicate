# Pneuma Architecture

Pneuma is built in several modules, each of which can be run independently. More details can be found [here](https://docs.google.com/document/d/16MsdIs80NssVtIhMq4r0RxXSpTKts_1MyyU2gf6ncpc).

## Setup
This setup guide is written on a Windows environment with Python version 3.10.6

Clonethe repository.
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
**Usage**: `registration.py setup --db_path=PATH/TO/DATABASE_NAME.db`

**Description**: Initializes the database schema. Creates a FILE_NAME.db file in the specified path.

**Example Usage**: `registration.py setup --db_path=../storage.db`

### Read Table
**Usage**: `registration.py read_table --db_path=PATH/TO/DATABASE_NAME.db [OPTION]... PATH_TO_FILE.(csv/parquet) CREATOR_NAME FILE_TYPE`

**Description**: Reads a table, formatted in CSV or PARQUET, from the local filesystem or an online storage bucket. If a file in a storage bucket is public, it can be read like a local file.

- PATH_TO_FILE can be a path in the local filesystem or a bucket URI
- CREATOR_NAME is the name of the person who runs this command (TODO: Authenticate automatically)
- FILE_TYPE is either **csv** or **parquet**.

**Options**:
- --s3_region=REGION

    Region of the s3 bucket.

-  --s3_access_key=AWS_ACCESS_KEY

    AWS access key ID

- --s3_secret_access_key=AWS_SECRET_ACCESS_KEY

    AWS secret access key ID

**Examples Usage**: `registration.py read_table --db_path=../storage.db ../sample_data/5cq6-qygt.csv david csv`

### Add Context
**Usage**: `registration.py add_context --db_path=PATH/TO/DATABASE_NAME.db TABLE_ID PATH_TO_FILE.txt`

**Description**: Creates a context entry for the specified table.

**Example Usage**: `registration.py add_context --db_path=../storage.db ../sample_data/5cq6-qygt.csv ../sample_data/sample_context.txt`

### Add Summary
**Usage**: `registration.py add_summary --db_path=PATH/TO/DATABASE_NAME.db TABLE_ID PATH_TO_FILE.txt`

**Description**: Creates a context entry for the specified table.

**Example Usage**: `registration.py add_context --db_path=../storage.db ../sample_data/5cq6-qygt.csv ../sample_data/sample_summary.txt`
