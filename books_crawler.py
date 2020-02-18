import asyncio
import shutil
import argparse
import configparser
import logging
import os
from pathlib import Path

import aiohttp
import pandas as pd
from aiohttp import ClientSession

ROOT_PATH = Path(os.path.abspath(""))
DATA_PATH = ROOT_PATH / "data"
INPUT_PATH = DATA_PATH / "input"
INPUT_DATA = INPUT_PATH / "books.csv"
TMP_PATH = DATA_PATH / "tmp"
OUTPUT_PATH = DATA_PATH / "output"
OUTPUT_DATA = OUTPUT_PATH / "books_output.csv"

logging.basicConfig(
    filename="books_crawler.log",
    filemode="w",
    format="[%(levelname)s] %(name)s %(asctime)s %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("books_crawler")
logging.getLogger("chardet.charsetprober").disabled = True

config = configparser.ConfigParser()
try:
    config.read_file(ROOT_PATH / "config.ini")
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


class BooksCrawler:
    """Crawler for books' metadata using asyncio Google Books API

    Attributes
    ----------
    columns_output
        Dictionary with expected columns and types
    columns_output_names
        Names of columns for the expected output
    """

    columns_output = {
        "isbn10": str,
        "isbn13": str,
        "title": str,
        "subtitle": str,
        "authors": str,
        "categories": str,
        "thumbnail": str,
        "description": str,
        "published_year": str,
    }
    columns_output_names = [*columns_output]

    def __init__(
        self,
        input_file,
        tmp_dir,
        output_file,
        api_url,
        api_key,
        max_results_per_query,
        max_concurrency,
        language,
    ):
        """Initialize crawler with required parameters

        Parameters
        ----------
        input_file
            Path of input file
        tmp_dir
            Directory where temporary data will be downloaded
        output_file
            Path of output file
        api_url
            URL of Google Books API used for downloading data
        api_key
            Key of for API authentication
        max_results_per_query
            Maximum number of results that can be retrieved at a time
        max_concurrency
            Maximum number of open connections
        language
            Language used for restricting results when downloading data
        """

        self.input_file = input_file
        self.tmp_dir = tmp_dir
        self.output_file = output_file
        self.api_url = api_url
        self.api_key = api_key
        self.max_results_per_query = max_results_per_query
        self.max_concurrency = max_concurrency
        self.language = language
        self.books_df = self.read_input(self.input_file)
        self.list_isbn = self.books_df["isbn13"].tolist()
        self.session = ClientSession()

    @staticmethod
    def read_input(input_data):
        """Read input data and do required pre-processing

        Parameters
        ----------
        input_data
            Path from where input data will be read

        Returns
        -------
        Pandas dataframe with input data
        """
        try:
            books_df = pd.read_csv(input_data, error_bad_lines=False)
            books_df = books_df.query("language_code.str.startswith('en')")
            books_df = books_df.rename(columns={"# num_pages": "num_pages"})
            numeric_cols = ["num_pages", "ratings_count", "text_reviews_count"]
            books_df[numeric_cols] = books_df[numeric_cols].apply(
                pd.to_numeric, errors="coerce", axis=1
            )
            books_df["isbn13"] = books_df["isbn13"].astype(
                str
            )  # Assure isbn13 is read as str
        except Exception:
            logger.exception("Was unable to read the data!")
            raise
        return books_df

    def concatenate_tmp_files(self):
        """Concatenate temporary files fetched from the API to generate output

        Returns
        -------
        Pandas dataframe with all the results gotten from the API
        """
        csv_files = [
            filename for filename in self.tmp_dir.iterdir() if filename.suffix == ".csv"
        ]
        frames_list = []
        logger.info(f"Merging previously downloaded files")
        try:
            for filename in csv_files:
                tmp_df = pd.read_csv(
                    filename,
                    index_col=None,
                    header=0,
                    na_values="None",
                    keep_default_na=True,
                    dtype=self.columns_output,
                )
                frames_list.append(tmp_df)
            concat_df = pd.concat(frames_list, axis=0, ignore_index=True)
        except Exception:
            logger.exception(f"An error occured while merging the files!")
            raise
        logger.info(f"Successfully merged the files!")
        return concat_df

    def write_output(self):
        """Write resulting dataframe in disk"""
        output_df = self.concatenate_tmp_files()
        output_df["nulls"] = output_df.isnull().sum(axis=1)  # Calculate nulls per row
        reduced_df = (
            output_df.query("~isbn13.isna()")
            .sort_values("nulls")
            .groupby("isbn13")
            .first()
            .reset_index()
            .drop(["nulls"], axis=1)
        )  # For duplicated rows, get the one with the fewest nulls
        result_df = pd.merge(
            reduced_df,
            self.books_df[["isbn13", "average_rating", "num_pages", "ratings_count"]],
            how="left",
            on="isbn13",
        )
        result_df.sort_values("ratings_count", ascending=False)
        result_df.to_csv(self.output_file, index=False)
        logger.info(
            f"Resulting dataframe has been saved in the following location: {self.output_file}"
        )
        logger.info(f"Shape of resulting dataframe: {result_df.shape}")

    def get_queries(self):
        """Generates the queries that will be used to download data from the API

        Returns
        -------
        Index and query to be executed
        """
        number_of_queries = len(self.list_isbn) // self.max_results_per_query
        for i in range(number_of_queries):
            start = i * self.max_results_per_query
            end = self.max_results_per_query * (1 + i)
            yield (
                i,
                self.api_url
                + "?q=isbn:"
                + "+OR+isbn:".join(self.list_isbn[start:end])
                + f"&maxResults={self.max_results_per_query}&langRestrict={self.language}",
            )

    async def fetch_all_books(self):
        """Creates asyncio's tasks for downloading data
        Returns
        -------

        """
        tasks = []
        for idx, query in self.get_queries():
            output_path = self.tmp_dir / f"_part{idx:04d}.csv"
            if output_path.is_file():
                logger.info(f"{output_path} Already downloaded. Will skip it.")
            else:
                task = asyncio.create_task(
                    self.restricted_fetch_and_write(
                        query=query, output_path=output_path
                    )
                )
                tasks.append(task)
        await asyncio.gather(*tasks)
        await self.session.close()

    async def restricted_fetch_and_write(self, query, output_path):
        """Limit how many tasks can be executed in parallel (to match API's restrictions)

        Parameters
        ----------
        query
            Query that will be executed
        output_path
            Output path where result from query will be downloaded

        Returns
        -------
        Function that executes API request
        """
        sem = asyncio.Semaphore(self.max_concurrency)
        async with sem:
            return await self.fetch_and_write(query, output_path)

    async def fetch_and_write(self, query, output_path):
        """Executes request and writes results in temporary directory

        Parameters
        ----------
        query
            Query that will be executed
        output_path
            Output path where result from query will be downloaded
        """
        response = await self.parse_response(query)
        response_df = pd.DataFrame(response, columns=self.columns_output_names)
        response_df.to_csv(output_path, index=False)
        logger.info(f"Wrote results: {output_path}")

    async def parse_response(self, query):
        """Handle error and parse response from API

        Parameters
        ----------
        query
            Query that will be executed
        Returns
        -------
        List of books and their metadata extracted from the API response
        """
        books = []
        try:
            response = await self.get_books_metadata(query)
        except (aiohttp.ClientError, aiohttp.http_exceptions.HttpProcessingError) as e:
            status = getattr(e, "status", None)
            message = getattr(e, "message", None)
            logger.error(f"aiohttp exception for {query} [{status}]:{message}")
            return books
        except KeyError:
            logger.error(f"No available data for: {query}")
            return books
        except Exception as non_exp_err:
            logger.exception(
                f"Non-expected exception occured for {query}:  {getattr(non_exp_err, '__dict__', {})}"
            )
            return books
        else:
            for item in response:
                books.append(self.extract_fields_from_response(item))
            return books

    async def get_books_metadata(self, query):
        """Make request to the API to get books's metadata

        Parameters
        ----------
        query
            Query that will be executed

        Returns
        -------
        List of items obtained from the request's reponse
        """
        response = await self.session.request(
            method="GET", url=query, params={"key": self.api_key}
        )
        response.raise_for_status()
        logger.info("Got response [%s] for URL: %s", response.status, query)
        response_json = await response.json()
        items = response_json.get("items", {})
        return items

    @staticmethod
    def extract_fields_from_response(item):
        """Extract relevant fields from the request's response

        Parameters
        ----------
        item
            Dictionary with the metadata obtained for each book
        Returns
        -------
        Tuple with relevant information extracted from the response

        """
        volume_info = item.get("volumeInfo", None)
        isbn_10 = None
        isbn_13 = None
        title = volume_info.get("title", None)
        subtitle = volume_info.get("subtitle", None)
        authors_list = [author for author in volume_info.get("authors", [])]
        categories_list = [category for category in volume_info.get("categories", [])]
        thumbnail = volume_info.get("imageLinks", {}).get("thumbnail", None)
        description = volume_info.get("description", None)
        published_date = volume_info.get("publishedDate", None)

        for idx in volume_info.get("industryIdentifiers", {}):
            type_idx = idx.get("type", None)
            value_idx = idx.get("identifier", None)
            if type_idx == "ISBN_10":
                isbn_10 = value_idx
            elif type_idx == "ISBN_13":
                isbn_13 = value_idx

        authors = ";".join(authors_list) if authors_list else None
        categories = ";".join(categories_list) if categories_list else None
        published_year = (
            published_date[:4]
            if published_date and published_date[:4].isdigit()
            else None
        )

        return (
            isbn_10,
            isbn_13,
            title,
            subtitle,
            authors,
            categories,
            thumbnail,
            description,
            published_year,
        )


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

    download_data(
        username=KAGGLE_USER,
        key=KAGGLE_KEY,
        dataset=KAGGLE_DATASET,
        download_path=INPUT_PATH,
    )
    asyncio.run(execute_crawler())


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
