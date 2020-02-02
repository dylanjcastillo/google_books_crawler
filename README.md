# Books Crawler
This repository contains the code for a Python-based crawler that gets metadata from books using the Google Books API. It leverages asyncio, aiohttp for executing concurrent requests.

## How to use

1. Clone repository
2. Install dependencies
3. Execute crawler from the command line

## Limitations

- Some responses result in multiple items for the same isbn. Around 5% of the results are duplicated, so that should be considered when using this dataset.
- The intial good books dataset contains ~12k books, however when using the ISBN for many of them, they don't return any values. It is possible to add different fields for doing the requests, so it should be possible to improve the number of matches (however that would require some manual checking of results)
