import os
import tarfile
import urllib.request
from tqdm import tqdm


script_dir = os.path.dirname(os.path.abspath(__file__))


def post_process_dataset(dataset_path: str):
    for table in os.listdir(dataset_path):
        os.rename(
            f"{dataset_path}/{table}", f"{dataset_path}/{table.split('_SEP_')[1]}"
        )
    print("Processed the dataset")
