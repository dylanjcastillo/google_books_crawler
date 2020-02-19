import logging
import os

logging.basicConfig(
    filename="books_crawler.log",
    filemode="w",
    format="[%(levelname)s] %(name)s %(asctime)s %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("chardet.charsetprober").disabled = True  # Generates too many logs


def download_data(username, key, dataset, download_path):
    """Download dataset using Kaggle's API

    Parameters
    ----------
    username
        Kaggle Username to download data
    key
        Kaggle's API Key
    dataset
        Name of dataset to download
    download_path
        Path where data will be downloaded
    """
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    import kaggle  # Fails if imported at the top of the file

    kaggle.api.dataset_download_files(dataset, path=download_path, unzip=True)
