import asyncio
import configparser
import logging
import os
import time
from pathlib import Path
import sys

import aiohttp
import pandas as pd
from aiohttp import ClientSession
from requests.exceptions import HTTPError


ROOT_PATH = Path(os.path.abspath("")).parent
DATA_PATH = Path(os.path.abspath("")).parent / "data"
INPUT_DATA = DATA_PATH / "raw" / "goodreads.csv"
TEMP_DATA_PATH = DATA_PATH / "tmp"
OUTPUT_DATA = DATA_PATH / "processed" / "book_processed_data.csv"

config = configparser.ConfigParser()
config.read(ROOT_PATH / "config.ini")

COLUMNS = {
    "isbn10": str,
    "isbn13": str,
    "title": str,
    "subtitle": str,
    "authors": str,
    "categories": str,
    "thumbnail": str,
    "description": str,
    "published_year": int,
}


GOOGLE_BOOKS_API = config["google_books_api"]["url"]
GOOGLE_BOOKS_KEY = config["google_books_api"]["key"]

logging.basicConfig(
    filename="books_crawler.log",
    filemode="w",
    format="[%(levelname)s] %(name)s %(asctime)s %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("books_crawler")
logging.getLogger("chardet.charsetprober").disabled = True


def write_to_csv(response, output_path) -> None:
    df_response = pd.DataFrame([response], columns=COLUMNS.keys())
    df_response = df_response.astype(COLUMNS)
    df_response.to_csv(output_path, index=False)


async def write_one(session, isbn10, isbn13) -> None:
    """Write the found HREFs from `url` to `file`."""
    response = await parse_response(session=session, isbn10=isbn10, isbn13=isbn13)
    output_path = TEMP_DATA_PATH / (isbn10 + "_" + isbn13 + ".csv")
    err_path = TEMP_DATA_PATH / (isbn10 + "_" + isbn13 + "_not_available")

    if response:
        write_to_csv(response, output_path)
        logger.info(f"Wrote results for ISBNs: {isbn10, isbn13}")
    else:
        with open(err_path, "w") as f:
            pass
        logger.warning(f"Skipped ISBNs: {isbn10, isbn13}")


async def bulk_crawl_and_write(isbn_data) -> None:
    """Crawl & write concurrently to `file` for multiple `urls`."""
    sem = asyncio.Semaphore(10)
    async with ClientSession() as session:
        tasks = []
        for _, row in isbn_data.iterrows():
            isbn10 = row["isbn"]
            isbn13 = row["isbn13"]
            output_path = TEMP_DATA_PATH / (isbn10 + "_" + isbn13 + ".csv")
            err_path = TEMP_DATA_PATH / (isbn10 + "_" + isbn13 + "_not_available")

            if output_path.is_file():
                logger.info(f"{isbn10, isbn13} already downloaded")
            elif err_path.is_file():
                logger.warning(
                    f"Previous attempts at downloading data for {isbn10, isbn13} failed. Skipping for now."
                )
            else:
                task = asyncio.ensure_future(
                    write_one(session=session, isbn10=row["isbn"], isbn13=row["isbn13"])
                )
                tasks.append(task)
        await asyncio.gather(*tasks)


async def get_info_from_api(url: str, session: ClientSession):
    """Get book information from specified API"""
    response = await session.request(
        method="GET", url=url, params={"key": GOOGLE_BOOKS_KEY}
    )
    response.raise_for_status()
    logger.info("Got response [%s] for URL: %s", response.status, url)
    response_json = await response.json()
    return response_json["items"][0]["volumeInfo"]


async def get_response_object(
    session: ClientSession, isbn10: str = None, isbn13: str = None
):

    if not isbn10 and not isbn13:
        raise Exception("Missing values for ISBN")

    if isbn13:
        try:
            response_isbn = await get_info_from_api(GOOGLE_BOOKS_API + isbn13, session)
            return response_isbn
        except KeyError as e:
            logger.error(f"No info available for ISBN13: {isbn13}")

    if isbn10:
        logger.info(f"Will try using ISBN10: {isbn10}")
        response_isbn = await get_info_from_api(GOOGLE_BOOKS_API + isbn10, session)
        return response_isbn


async def parse_response(
    session: ClientSession, isbn10: str = None, isbn13: str = None
):
    """Extract books information using Google Books API"""

    found = set()

    try:
        response_fields = await get_response_object(
            session=session, isbn10=isbn10, isbn13=isbn13
        )

    except (aiohttp.ClientError, aiohttp.http_exceptions.HttpProcessingError) as e:

        status = getattr(e, "status", None)
        message = getattr(e, 'message', None)
        logger.error(
            f"aiohttp exception for {isbn10, isbn13} [{status}]:{message}"
        )

        if status == "403":
            time.sleep(5)
            response_fields = await get_response_object(
                session=session, isbn10=isbn10, isbn13=isbn13
            )

        return found

    except KeyError:
        logger.error(f"No available data for: {isbn10, isbn13}")
        return found

    except Exception as non_exp_err:
        logger.exception(
            f"Non-expected exception occured for {isbn10, isbn13} :  {getattr(non_exp_err, '__dict__', {})}"
        )
        return found

    else:

        title = response_fields.get("title", None)
        subtitle = response_fields.get("subtitle", None)
        authors = ";".join([author for author in response_fields.get("authors", [])])
        categories = ";".join(
            [category for category in response_fields.get("categories", [])]
        )
        thumbnail = response_fields.get("imageLinks", {}).get("thumbnail", None)
        description = response_fields.get("description", None)
        try:
            published_year = response_fields.get("publishedDate", None)[:4]
        except TypeError:
            published_year = None

        found = (
            isbn10,
            isbn13,
            title,
            subtitle,
            authors,
            categories,
            thumbnail,
            description,
            published_year,
        )
        return found


if __name__ == "__main__":
    df_books = pd.read_csv(INPUT_DATA).query("language_code.str.startswith('en')")
    df_books = df_books.head(10)
    asyncio.run(bulk_crawl_and_write(isbn_data=df_books))
