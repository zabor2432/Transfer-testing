import argparse
import os
from anyio import Path
from kaggle.api.kaggle_api_extended import KaggleApi

parser = argparse.ArgumentParser(description="Fetch Kaggle datasets")
parser.add_argument(
    "--dataset",
    default="akash2sharma/tiny-imagenet",
    type=str,
    help="dataset name",
)


def fetch_kaggle_dataset(
    dataset_name: str, path: str = "data", unzip: bool = True
):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_name, path=path, unzip=unzip)


def main():
    args = parser.parse_args()
    data_folder = Path(os.getenv("DATA_FOLDER", "data"))
    fetch_kaggle_dataset(args.dataset, path=data_folder, unzip=True)


if __name__ == "__main__":
    main()
