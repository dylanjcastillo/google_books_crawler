import asyncio
import configparser
import logging
import os
from pathlib import Path

import aiohttp
import pandas as pd
from aiohttp import ClientSession

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


GOOGLE_BOOKS_API = config.get("google_books_api", "url")
GOOGLE_BOOKS_KEY = config.get("google_books_api", "key")
MAX_RESULTS_PER_QUERY = config.getint("google_books_api", "max_results_per_query")
MAX_CONCURRENCY = config.getint("google_books_api", "max_concurrency")

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


def write_to_csv(response, output_path) -> None:
    df_response = pd.DataFrame(response, columns=COLUMNS.keys())
    df_response = df_response.astype(COLUMNS)
    df_response.to_csv(output_path, index=False)


async def get_and_write(session, query, output_path) -> None:
    response = await parse_response(session=session, query=query)
    write_to_csv(response, output_path)
    logger.info(f"Wrote results for query: {query}")


def get_query(list_isbn, max_n=40):
    for i in range(df_books.shape[0] // max_n):
        start = i * max_n
        end = max_n * (1 + i)
        yield GOOGLE_BOOKS_API + "+OR+isbn:".join(
            list_isbn[start:end]
        ) + "&maxResults=40"


async def create_coroutines(list_isbn) -> None:
    async with ClientSession() as session:
        tasks = []
        for idx, query in enumerate(get_query(list_isbn, max_n=MAX_RESULTS_PER_QUERY)):
            output_path = TEMP_DATA_PATH / f"_part{idx:04d}.csv"
            if output_path.is_file():
                logger.info(f"{output_path} Already downloaded. Will skip it.")
            else:
                task = asyncio.create_task(
                    safe_get_and_write(
                        session=session, query=query, output_path=output_path
                    )
                )
                tasks.append(task)
        await asyncio.gather(*tasks)


async def safe_get_and_write(session, query, output_path):
    async with sem:
        return await get_and_write(session, query, output_path)


async def get_info_from_api(url: str, session: ClientSession):
    response = await session.request(
        method="GET", url=url, params={"key": GOOGLE_BOOKS_KEY}
    )
    response.raise_for_status()
    logger.info("Got response [%s] for URL: %s", response.status, url)
    response_json = await response.json()
    items = response_json.get("items", {})
    return items


async def parse_response(session: ClientSession, query):

    books = []

    try:
        response = await get_info_from_api(query, session)

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
            books.append(
                [
                    None,
                    None,
                    item.get("title", None),
                    item.get("subtitle", None),
                    ";".join([author for author in item.get("authors", [None])]),
                    ";".join([category for category in item.get("categories", [None])]),
                    item.get("imageLinks", {}).get("thumbnail", None),
                    item.get("description", None),
                    item.get("publishedDate", "0000")[:4],
                ]
            )

        return books


if __name__ == "__main__":
    df_books = pd.read_csv(INPUT_DATA).query("language_code.str.startswith('en')")
    list_isbn = df_books["isbn13"][:40]
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(create_coroutines(list_isbn=list_isbn))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
