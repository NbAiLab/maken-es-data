#!/usr/bin/env python
import argparse
import json
import logging
import os
import sys
import urllib3; urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from pathlib import Path

import numpy as np
from tqdm import tqdm

from elasticsearch import Elasticsearch, helpers


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d - %H:%M:%S"
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(console)


def load_vector(vector_file, vector_format):
    if vector_format in ("npy", "numpy"):
        vector = np.load(vector_file).tolist()
    else:  # vct, txt
        with vector_file.open() as f:
            vector = np.array(f.read().splitlines(), dtype="float").tolist()
    return vector


class Elastic:

    def __init__(
        self, host, user=None, password=None,
        timeout=60, max_retries=200, retry_on_timeout=True
    ):
        # host = f"{scheme}://{host}:{port}"
        connection_kwargs = dict(
            hosts=host, timeout=timeout,
            max_retries=max_retries, retry_on_timeout=retry_on_timeout,
            verify_certs=False,
        )
        if user is not None and password is not None:
            connection_kwargs.update({"http_auth": (user, password)})
        self._conn = Elasticsearch(**connection_kwargs)
        self._items = []

        if not self._conn.ping():
            raise ConnectionError(f"Unable to connect to {host}")

    def create_index(self, index_name, mapping=None):
        if self._conn.indices.exists(index_name):
            return False
        response = self._conn.indices.create(index=index_name, ignore=400, body=mapping)
        if "error" in response:
            logger.error(f"Unable to create index: {str(response['error'])}")
            return False
        return True

    def drop_index(self, index_name):
        response = self._conn.indices.delete(index=index_name, ignore=[400, 404])
        if "error" in response:
            logger.error(f"Unable to drop index: {str(response['error'])}")
            return False
        return True

    def create_index_mapping(self, index_name, mapping_file):
        settings_file = open(mapping_file, "r")
        settings = settings_file.read()
        return self.create_index(index_name, settings)

    def commit_vectors(self, bar=None):
        msg = "Inserting bulk"
        if bar:
            bar.set_description(msg)
        else:
            logger.info(msg)
        self.bulk_insert()

    def bulk_insert(self):
        helpers.bulk(self._conn, self._items)
        self._items = []

    def add(self, index_name, data, data_type=None):
        self._items.append({
            "_index": index_name,
            "_type": data_type,
            "_source": data,
        })

    def insert_vector_file_vct(self, vector_file, index_name, bar=None):
        filename = vector_file.name.split("/")[-1]
        msg = "Processing file: {}".format(filename)
        if bar:
            bar.set_description(msg)
        else:
            print(msg)
        with vector_file.open() as f:
            data = {
                "filename": filename.rsplit(".", 1)[0],
                "vector": np.array(f.read().splitlines(), dtype="float").tolist()
            }
        self.add(index_name, data)

    def insert_vector_file(self, record_file, vector_file, index_name, vector_format="npy", bar=None):
        filename = record_file.name.split("/")[-1]
        msg = "Processing file: {}".format(filename)
        if bar:
            bar.set_description(msg)
        else:
            print(msg)
        try:
            vector = load_vector(vector_file, vector_format)
        except FileNotFoundError:
            logger.warning(f"Vector file {vector_file} not found")
            return
        with record_file.open() as file:
            record_json = json.load(file)
            urn = (
                record_json['metadata']['identifiers']['urn']
                .replace("URN:NBN:no-nb_", "")
            )
            subjects = record_json['metadata'].get("subject", {}).get("topics", [])
            subjects += record_json['metadata'].get("subjectName", [])
            data = {
                'id' : record_json['id'],
                'filename': urn,
                'access': record_json['accessInfo']['accessAllowedFrom'].lower(),
                'title': record_json['metadata'].get("title", ""),
                'creators': record_json['metadata'].get("creators", ""),
                'subject': subjects,
                'vector': vector
            }
        self.add(index_name, data)

    def insert_vector_file_json(self, vector_file, index_name, bar=None):
        """Single JSON files containing both record metadata and the corresponding vector"""
        filename = record_file.name.split("/")[-1]
        msg = "Processing file: {}".format(filename)
        if bar:
            bar.set_description(msg)
        else:
            print(msg)
        try:
            with vector_file.open() as file:
                data = json.load(file)
        except FileNotFoundError:
            logger.warning(f"Vector file {vector_file} not found")
            return
        self.add(index_name, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f""
    f"Insert documents into a Elasticsearch instance. The index is specified "
    f"by the mapping-file, the documents are composed using the records in "
    f"record-dir and the vectors in vectors-dir"
    f"", epilog=f"""Example usage:
    {__file__} http://192.0.0.10:9200 index.knn ./records/ ./vectors/ --vectors_format json
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('host',
        help='Elasticsearch server host (including scheme and port)')
    # parser.add_argument('port', help='Elasticsearch server port', type=int)
    parser.add_argument('--user', help='Username for basic auth. Environment variable ES_USER')
    parser.add_argument('--password', help='Password for basic auth. Environment variable ES_PASSWORD')
    parser.add_argument('index',
        metavar='index',
        help='Path to the index mapping file in JSON. The file name without '
             'the extension will be used as the new index name. '
             'If not a file, the string will be used as the index'
    )
    parser.add_argument('records_dir',
        metavar='records-dir', help='Directory with the records files')
    parser.add_argument('records_glob', default="**/*",
        metavar='records-glob', help='Glob for the directory with the records files')
    parser.add_argument('vectors_dir',
        metavar='vectors-dir', help='Directory with the vectors files')
    parser.add_argument('--vectors_format',
        metavar='vectors-format', default="npy",
        help='File format of the vectors files. Either json, npy, or vct. If json, records_dir is ignored.',
    )
    parser.add_argument('--vectors_suffix',
        metavar='vectors-suffix', default="",
        help='Filename suffix of the vectors files',
    )
    parser.add_argument('--bulk_size',
        metavar='bulk-size', default=1000, type=int,
        help='Number of documents to insert in bulk mode. Defaults to 1000'
    )
    parser.add_argument('--recreate_index', action='store_true',
        help='Drop and create the index if exists. Defaults to False'
    )
    parser.add_argument('--skip',
        metavar='skip', default=0, type=int,
        help='Skip a number of iterations. Defaults to 0'
    )
    args = parser.parse_args()

    # Recreate index
    client = Elastic(
        args.host,
        user=args.user or os.environ.get("ES_USER"),
        password=args.password or os.environ.get("ES_PASSWORD"),
    )
    if Path(args.index).exists():
        index_name = args.index.rsplit(".", 1)[0]
        if not client.create_index_mapping(index_name, args.index):
            if args.recreate_index:
                client.drop_index(index_name)
                client.create_index_mapping(index_name, args.index)
    elif args.index:
        index_name = args.index
    # Populate the index
    total = len(os.listdir(args.records_dir))
    records_path = Path(args.records_dir)
    vectors_path = Path(args.vectors_dir)
    bar = tqdm(records_path.rglob(args.records_glob), total=total)
    file_count = 0
    for iteration, record_file in enumerate(bar):
        if (not record_file.is_file()
            or record_file.suffix.lower() != ".json"
            or iteration < args.skip):
            continue
        record_path = record_file.relative_to(records_path).parent
        vector_file = (vectors_path
            / record_path
            / f"{record_file.stem}{args.vectors_suffix}.{args.vectors_format}"
        )
        if args.vectors_format in ("vct", "txt", "plain", "text", "npy", "numpy"):
            client.insert_vector_file(
                record_file, vector_file, index_name, args.vectors_format, bar=bar
            )
        elif args.vectors_format == "json":
            client.insert_vector_file_json(
                vector_file, index_name, bar=bar
            )
        else:
            logger.warning("Invalid vector format")
            exit(1)
        file_count += 1
        if file_count >= args.bulk_size:
            client.commit_vectors(bar=bar)
            file_count = 0
    # Commit remaining vectors
    client.commit_vectors()
