from kaggle.api.kaggle_api_extended import KaggleApi


if __name__ == "__main__":
    # TODO: argparse for different datasets
    api = KaggleApi()
    api.authenticate()

    api.competition_download_file("tiny-imagenet", path="./data")
