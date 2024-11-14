import urllib.request
import tarfile


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

for table in tables:
    urllib.request.urlretrieve(
        f"https://storage.googleapis.com/pneuma_open/{table}",
        filename=table,
    )
    extract_tar(table)
