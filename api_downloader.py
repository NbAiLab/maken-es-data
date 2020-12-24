#!/usr/bin/env python
import argparse
import json
import logging
import os
import time
import sys
from functools import reduce
from pathlib import Path
from typing import Tuple
from urllib import parse

import requests
from requests.adapters import HTTPAdapter, TimeoutSauce
from tqdm import tqdm
from urllib3.util import Retry

BASE_URL = os.environ.get("BASAE_URL", "https://api.nb.no")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d - %H:%M:%S"
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(console)


def get_http(retries: int=10):
    """
    Creates a HTTP/S session with a Retry adapter
    """
    adapter = HTTPAdapter(max_retries=Retry(
        total=retries,
    ))
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)
    return http


def parse_page(
    http: requests.Session, url: str, data: dict=None, timeout: int=5 * 60
) -> dict:
    """
    Retrive (using GET), parse, and return the JSON of a URL
    """
    page_res = http.get(url, params=data, timeout=timeout, headers={
        "Content-Type": "application/json",
    })
    if data:
        logger.info(f"Request to: {page_res.url}")
    if page_res.ok:
        return page_res.json()
    else:
        return {}


def process_page(
    page: dict, output: str, criteria: list, overwrite: bool=False
) -> bool:
    """
    Given a dict with the response of a API page, parse it to extract
    the date for each item into the right folder
    """
    embedded = page.get("_embedded", {})
    items = embedded.get("items", {})
    for item in items:
        item_id = item["id"]
        try:
            out = reduce(dict.__getitem__, criteria, item).lower()
        except KeyError:
            out = "unknown"
        dest = Path(os.path.join(output, out))
        dest.mkdir(parents=True, exist_ok=True)
        json_dest = dest / f"{item_id}.json"
        if not overwrite and json_dest.exists():
            continue
        with open(json_dest, "w") as file:
            json.dump(item, file)
    return bool(items)


def main(args: argparse.ArgumentParser) -> None:
    data = {
        "filter": args.filter,
        "size": args.size,
        "scroll": "true" if args.scroll else "false",
        "profile": args.profile,
    }
    if args.page:
        data.update({"page": args.page})
    url = f"{BASE_URL}{args.endpoint}"
    logger.info(
        f"Preparing to harvest {url} with "
        + (f", ".join(f"{k}={v}" for k, v in data.items() if v))
    )
    logger.info(
        f"Requests done with timeout of {args.timeout} seconds "
        f"and {args.retries} retries"
    )
    logger.info(f"Writing to {args.output}")
    logger.info(f"Splitting JSONs on {args.criteria}")
    next_url = url
    http = get_http(retries=args.retries)
    page_dict = parse_page(http, next_url, data=data, timeout=args.timeout)
    progress = tqdm(range(
        page_dict["page"]["totalPages"]),
        total=page_dict["page"]["totalPages"],
        desc="Pages",
    )
    criteria = args.criteria.split(".")
    overwrite = args.overwrite
    delay = float(args.delay)
    for page_num in progress:
        if not process_page(page_dict, args.output, criteria, overwrite):
            logger.info(f"Found page with no items: {next_url}")
            if args.log:
                with open(f"page_{page_num}.log", "w") as log_file:
                    log_file.write(str(page_dict))
        if page_num < int(page_dict["page"]["number"]):
            continue
        time.sleep(delay)
        if "next" in page_dict.get("_links", {}):
            next_url = page_dict["_links"]["next"]["href"]
            page_dict = parse_page(http, next_url)
        else:
            logger.info("No 'next' link to follow. Terminating...")
            break


if __name__ == "__main__":
    # https://api.nb.no/catalog/v1/items?filter=mediatype:bilder&filter=contentClasses:JP2&scroll=false&size=25&page=0
    parser = argparse.ArgumentParser(description=f""
    f"Downloader of JSON records of the catalog in {BASE_URL}. It uses the "
    f"/catalog/v1/items endpoint to perform requests by default. Each record "
    f"will be stored as an independent JSON file named as their identifier"
    f"", epilog=f"""Example usage:
    {__file__} -f mediatype:bilder contentClasses:JP2 --size 15
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--output-folder', '-o', dest="output", default="./out",
        help="Path to output folder. Defaults to ./out")
    parser.add_argument('--endpoint', '-e', default="/catalog/v1/items",
        help="Endpoint to use in the requests. Defaults to 'catalog/v1/items'")
    parser.add_argument('--filter', '-f', nargs='*',
        help="Filters to add to the query")
    parser.add_argument('--scroll', action='store_true',
        help="Whether or not enable scrolling (resource intensive). "
             "Defaults to false")
    parser.add_argument('--profile', default='wwwnbno',
        help="Profile. Defaults to 'wwwnbno'")
    parser.add_argument('--page', '-p', default=0,
        help="Start at a specific page. Defaults to 0",
        type=lambda x: max(1, min(int(x), 25)))
    parser.add_argument('--delay', '-d', default=0.1,
        help="Number of seconds to space out requests. Defaults to 0.1",
        type=float)
    parser.add_argument('--timeout', '-t', default=5 * 60,
        help="Number of seconds to wait for a response. Defaults to 300",
        type=int)
    parser.add_argument('--retries', '-r', default=10,
        help="Number of retries when retrieven a URL. Defaults to 10",
        type=int)
    parser.add_argument('--size', '-s', default=25,
        help="Number of results per page. Defaults to 25 [1-100]",
        type=lambda x: max(1, min(int(x), 100)))
    parser.add_argument('--criteria',
        default="accessInfo.accessAllowedFrom",
        help="JSON field key to use as criteria to split records in the "
             "output folder. Defaults to 'accessInfo.accessAllowedFrom'")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing JSON files. "
             "Defaults to false")
    parser.add_argument('--log', '-l', default=False, action='store_true',
        help="Log to disk failing results pages. If True, files named "
             "./page_<num>.log will be created. Defaults to false")
    args = parser.parse_args()
    main(args)
