# Books Crawler
This repository contains the code for a Python-based crawler that gets metadata from books using the Google Books API. It leverages asyncio and aiohttp for executing concurrent requests.

## How to use

1. Clone repository: ```git clone https://github.com/dylanjcastillo/books_crawler.git```
2. Install dependencies:
```conda env create -f environment.yml```
3. Go to the [GCP Console and generate new API key](https://console.cloud.google.com/apis/credentials)
4. Go to your Kaggle's account and generate an API token
5. Fill the required fields in the *config_example.ini* file and save it as *config.ini*
5. Go to the Execute crawler from the command line: ```python books_crawler.py```

## Limitations

- To avoid having timeouts on the request, you need to generate an API key in the Google Clould Platform Console. But don't worry it's **completely free**!
- If you don't want to manually download the data, you'll need to generate a Token in Kaggle. [It is free and very easy to do!] (https://adityashrm21.github.io/Setting-Up-Kaggle/)
- The intial good books dataset contains ~12k books, however when using the ISBN for many of them, they don't return any values. It is possible to add different fields for doing the requests, so it should be possible to improve the number of matches (however that would require some manual checking of results)
