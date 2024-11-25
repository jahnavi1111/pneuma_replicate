import os
import tarfile
import urllib.request
from tqdm import tqdm


script_dir = os.path.dirname(os.path.abspath(__file__))


def extract_tar(file_path, extract_path="."):
    with tarfile.open(file_path, "r") as tar:
        tar.extractall(path=extract_path)
        print(f"Extracted all files to '{extract_path}'")


tables = [
    "pneuma_public_bi.tar",
    "pneuma_chicago_10K.tar",
    "pneuma_chembl_10K.tar",
    "pneuma_fetaqa.tar",
    "pneuma_adventure_works.tar",
]

for table in tqdm(tables):
    urllib.request.urlretrieve(
        f"https://storage.googleapis.com/pneuma_open/{table}",
        filename=os.path.join(script_dir, table),
    )
    extract_tar(os.path.join(script_dir, table), script_dir)
