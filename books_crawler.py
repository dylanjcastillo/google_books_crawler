import asyncio
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
INPUT_DATA = DATA_PATH / "input" / "books.csv"
TMP_DATA_PATH = DATA_PATH / "tmp"
OUTPUT_DATA = DATA_PATH / "output" / "book_processed_data.csv"
OUTPUT_FULL_DATA = DATA_PATH / "output" / "books_data.csv"

config = configparser.ConfigParser()
config.read(ROOT_PATH / "config.ini")


COLUMNS_OUTPUT = {
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


GOOGLE_BOOKS_API = config.get("google_books_api", "url")
GOOGLE_BOOKS_KEY = config.get("google_books_api", "key")
MAX_RESULTS_PER_QUERY = config.getint("google_books_api", "max_results_per_query")
MAX_CONCURRENCY = config.getint("google_books_api", "max_concurrency")
LANGUAGE = config.get("google_books_api", "language")
KAGGLE_USER = config.get("kaggle", "username")
KAGGLE_KEY = config.get("kaggle", "key")
KAGGLE_DATASET = config.get("kaggle", "dataset")

logging.basicConfig(
    filename="books_crawler.log",
    filemode="w",
    format="[%(levelname)s] %(name)s %(asctime)s %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("books_crawler")
logging.getLogger("chardet.charsetprober").disabled = True

sem = asyncio.Semaphore(MAX_CONCURRENCY)


async def get_and_write(session, query, output_path) -> None:
    response = await parse_response(session=session, query=query)
    response_df = pd.DataFrame(response, columns=COLUMNS_OUTPUT.keys())
    response_df.to_csv(output_path, index=False)
    logger.info(f"Wrote results: {output_path}")


def get_queries(list_isbn, max_n=40):
    number_of_queries = len(list_isbn) // max_n
    for i in range(number_of_queries):
        start = i * max_n
        end = max_n * (1 + i)
        yield (
            i,
            GOOGLE_BOOKS_API
            + "?q=isbn:"
            + "+OR+isbn:".join(list_isbn[start:end])
            + f"&maxResults={max_n}&langRestrict={LANGUAGE}",
        )


async def create_coroutines(list_isbn) -> None:
    async with ClientSession() as session:
        tasks = []
        for idx, query in get_queries(list_isbn, max_n=MAX_RESULTS_PER_QUERY):
            output_path = TMP_DATA_PATH / f"_part{idx:04d}.csv"
            if output_path.is_file():
                logger.info(f"{output_path} Already downloaded. Will skip it.")
            else:
                task = asyncio.create_task(
                    restricted_safe_and_write(
                        session=session, query=query, output_path=output_path
                    )
                )
                tasks.append(task)
        await asyncio.gather(*tasks)


async def restricted_safe_and_write(session, query, output_path):
    async with sem:
        return await get_and_write(session, query, output_path)


async def get_books_metadata(url: str, session: ClientSession):
    response = await session.request(
        method="GET", url=url, params={"key": GOOGLE_BOOKS_KEY}
    )
    response.raise_for_status()
    logger.info("Got response [%s] for URL: %s", response.status, url)
    response_json = await response.json()
    items = response_json.get("items", {})
    return items


def extract_fields_from_response(item):

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
        published_date[:4] if published_date and published_date[:4].isdigit() else None
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


async def parse_response(session: ClientSession, query):
    books = []
    try:
        response = await get_books_metadata(query, session)
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
            books.append(extract_fields_from_response(item))
        return books


def concatenate_temp_files():
    csv_files = [
        filename for filename in TMP_DATA_PATH.iterdir() if filename.suffix == ".csv"
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
                dtype=COLUMNS_OUTPUT,
            )
            frames_list.append(tmp_df)
        concat_df = pd.concat(frames_list, axis=0, ignore_index=True)
    except Exception:
        logger.exception(f"An error occured while merging the files!")
        raise
    logger.info(f"Successfully merged the files!")
    return concat_df


def generate_output_dataframe(books_df):
    output_df = concatenate_temp_files()
    output_df["nulls"] = output_df.isnull().sum(axis=1)  # Calculate nulls per row
    # For repeated rows, get the one with the fewest nulls
    reduced_df = (
        output_df.query("~isbn13.isna()")
        .sort_values("nulls")
        .groupby("isbn13")
        .first()
        .reset_index()
        .drop(["nulls"], axis=1)
    )
    result_df = pd.merge(
        reduced_df,
        books_df[["isbn13", "average_rating", "num_pages", "ratings_count"]],
        how="left",
        on="isbn13",
    )
    result_df.sort_values("ratings_count", ascending=False)
    result_df.to_csv(OUTPUT_FULL_DATA, index=False)
    logger.info(
        f"Resulting dataframe has been saved in the following location: {OUTPUT_FULL_DATA}"
    )
    logger.info(f"Shape of resulting dataframe: {result_df.shape}")


def download_data():
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USER
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY
    import kaggle  # Fails if imported at the top of the file

    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=INPUT_PATH, unzip=True)


def read_input_data():
    try:
        books_df = pd.read_csv(INPUT_DATA, error_bad_lines=False)
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
    return books_df, books_df["isbn13"].tolist()


def run_concurrently(list_isbn):
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(create_coroutines(list_isbn=list_isbn))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


if __name__ == "__main__":
    download_data()
    books_df, list_isbn = read_input_data()
    run_concurrently(list_isbn)
    generate_output_dataframe(books_df)

# 1. Download data
# 2. Read data
# 3. Start and run asyncio event loop
# 4. Merge files
# 5. Generate full dataframe
