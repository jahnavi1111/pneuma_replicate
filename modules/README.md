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