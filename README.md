![](crawler_cover_new.jpg)
# Google Books API Crawler
![Python](https://img.shields.io/badge/Python-v3.7.1-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue) [![Twitter](https://img.shields.io/twitter/url/https/twitter.com/_dylancastillo.svg?style=social&label=Follow%20%40_dylancastillo)](https://twitter.com/_dylancastillo)

This repository contains the code for a Python-based crawler that gets metadata from books using the Google Books API. Provided with a list of ISBNs, it leverages asyncio and aiohttp for executing concurrent requests to download the metadata associated with those ISBNs. 

## How to use

1. Clone repository: ```git clone https://github.com/dylanjcastillo/books_crawler.git```
2. Install dependencies:
```conda env create -f environment.yml```
3. Go to the [GCP Console and generate new API key](https://console.cloud.google.com/apis/credentials)
4. Go to your Kaggle's account and generate an API token
5. Fill the required fields in the *config_example.ini* file and save it as *config.ini*
5. Execute crawler from the command line: 
    - Regular execution: ```python run_crawler.py```
    - Clearing cache execution: ```python run_crawler.py --clear-cache```
    - Get help on how to run the crawler: ```python run_crawler.py --help```

## Limitations

- This is not meant to crawl all the available books in the Google Books API. You need to provide a list of valid ISBNs (as in the Goodreads-books dataset) and the crawler will download the metadata associated with those books.
- The intial Goodreads-books dataset contains ~12k books, however many of them do not return valid responses. It is possible to add different fields for doing the requests, so it should be possible to improve the number of matches. However, this is not yet implemented.
- To avoid having timeouts on the request, you need to generate an API key in the Google Clould Platform Console. But don't worry it's **completely free**!
- If you don't want to manually download the data, you'll need to generate a Token in Kaggle. [It is free and very easy to do!](https://adityashrm21.github.io/Setting-Up-Kaggle/)

## License

This project is licensed under the terms of the MIT license.
