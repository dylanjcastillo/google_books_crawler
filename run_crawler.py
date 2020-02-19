import argparse
import asyncio
import configparser
import os
import shutil
from pathlib import Path

from books_crawler import BooksCrawler
from utils import download_data, logger

ROOT_PATH = Path(os.path.abspath(""))
DATA_PATH = ROOT_PATH / "data"
INPUT_PATH = DATA_PATH / "input"
INPUT_DATA = INPUT_PATH / "books.csv"
TMP_PATH = DATA_PATH / "tmp"
OUTPUT_PATH = DATA_PATH / "output"
OUTPUT_DATA = OUTPUT_PATH / "books_output.csv"

config = configparser.ConfigParser()
try:
    config.read_file(open(ROOT_PATH / "config.ini"))
except Exception as err:
    logger.exception("You need to create a config.ini file before executing!")
    raise

GOOGLE_BOOKS_API = config.get("google_books_api", "url")
GOOGLE_BOOKS_KEY = config.get("google_books_api", "key")
MAX_RESULTS_PER_QUERY = config.getint("google_books_api", "max_results_per_query")
MAX_CONCURRENCY = config.getint("google_books_api", "max_concurrency")
LANGUAGE = config.get("google_books_api", "language")
KAGGLE_USER = config.get("kaggle", "username")
KAGGLE_KEY = config.get("kaggle", "key")
KAGGLE_DATASET = config.get("kaggle", "dataset")


async def execute_crawler():
    """Initialize and execute crawler"""
    crawler = BooksCrawler(
        input_file=INPUT_DATA,
        tmp_dir=TMP_PATH,
        output_file=OUTPUT_DATA,
        api_url=GOOGLE_BOOKS_API,
        api_key=GOOGLE_BOOKS_KEY,
        max_results_per_query=MAX_RESULTS_PER_QUERY,
        max_concurrency=MAX_CONCURRENCY,
        language=LANGUAGE,
    )
    await crawler.fetch_all_books()
    crawler.write_output()


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="books-crawler", description="Crawl books' metadata using Google Books API"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clears cache from previous executions",
    )
    args = parser.parse_args()
    return args


def main(args):
    INPUT_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.mkdir(exist_ok=True)
    if args.clear_cache:
        try:
            shutil.rmtree(TMP_PATH)
            logger.info("Cache was cleared!")
        except Exception as err:
            logger.info("There is no cache to clear!")
    TMP_PATH.mkdir(exist_ok=True)

    if not INPUT_DATA.exists() or args.clear_cache:
        try:
            download_data(
                username=KAGGLE_USER,
                key=KAGGLE_KEY,
                dataset=KAGGLE_DATASET,
                download_path=INPUT_PATH,
            )
        except Exception as err:
            logger.exception("Failed to download the data!")
            raise
    asyncio.run(execute_crawler())


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
