#!/usr/bin/env python
import json
import logging
import sys
# import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from typing import List, NoReturn, Optional

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def get_logger() -> logging.Logger:
    """
    Get a logger
    """
    logger = logging.getLogger(__name__)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d - %H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def get_http(retries: int=10) -> requests.Session:
    """
    Create a HTTP/S session with a Retry adapter
    """
    adapter = HTTPAdapter(max_retries=Retry(
        total=retries,
    ))
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)
    return http


def save_vector(
    vector: np.ndarray,
    vector_dest: Path,
    vector_format: str,
    record: Optional[dict]=None,
    keys: Optional[List[str]]=None,
) -> NoReturn:
    """
    Save a vector in either binary (npy) or plain text (vct) format in
    vector_dest. If format is json, the record keys will also be saved
    """
    if keys is None:
        keys = ('id', 'filename', 'access', 'title', 'creators', 'subject')
    if vector_format == "npy":
        np.save(vector_dest, vector)
    elif vector_format  == "json":
        vector_data = {key: record[key] for key in keys if key in record}
        vector_data["vector"] = vector.tolist()
        vector_dest.write_text(json.dumps(vector_data))
    else:
        np.savetxt(vector_dest, vector)
