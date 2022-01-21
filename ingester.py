#!/usr/bin/env python
import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from elasticsearchHandler import elasticSearchHandler


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d - %H:%M:%S"
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(console)

global_elastic_handler = None

def drop_index(args, idx):
    global global_elastic_handler
    global_elastic_handler.dropIndex(idx)


def create_data_structure(args, idx, mapping_file):
    global global_elastic_handler
    settings_file = open(mapping_file, "r")
    settings = settings_file.read()
    return global_elastic_handler.createIndex(idx, settings)


def commit_vectors(bar=None):
    global global_elastic_handler
    msg = "Inserting bulk"
    if bar:
        bar.set_description(msg)
    else:
        print(msg)
    global_elastic_handler.bulkInsert()


def insert_vector_file_vct(vector_file, index_name, bar=None):
    global global_elastic_handler
    elastic_handler = global_elastic_handler
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
    elastic_handler.addToBulk(
        index_name, None, data
    )


def load_vector(vector_file, vector_format):
    if vector_format in ("npy", "numpy"):
        vector = np.load(vector_file).tolist()
    else:  # vct, txt
        with vector_file.open() as f:
            vector = np.array(f.read().splitlines(), dtype="float").tolist()
    return vector


def insert_vector_file(record_file, vector_file, index_name, vector_format="npy", bar=None):
    global global_elastic_handler
    elastic_handler = global_elastic_handler
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
    elastic_handler.addToBulk(
        index_name, None, data
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f""
    f"Insert documents into a Elasticsearch instance. The index is specified "
    f"by the mapping-file, the documents are composed using the records in "
    f"record-dir and the vectors in vectors-dir"
    f"", epilog=f"""Example usage:
    {__file__} 192.0.0.10 9200 user pass index.knn ./records/ ./vectors/ npy "" 500 false 0
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('server',
        help='IP address or logicalname for Elasticsearch server')
    parser.add_argument('port', help='Elasticsearch server port', type=int)
    parser.add_argument('user', help='Username for basic auth')
    parser.add_argument('password', help='Password for basic auth')
    parser.add_argument('mapping_file',
        metavar='mapping-file',
        help='Path the index mapping file in JSON. The file name without '
             'the extension will be used as the new index name. '
             'If not a file, the string will be used as the index'
    )
    parser.add_argument('records_dir',
        metavar='records-dir', help='Directory with the records files')
    parser.add_argument('vectors_dir',
        metavar='vectors-dir', help='Directory with the vectors files')
    parser.add_argument('vectors_format',
        metavar='vectors-format', default="npy",
        help='File format of the vectors files. Either npy or vct',
    )
    parser.add_argument('vectors_suffix',
        metavar='vectors-suffix', default="",
        help='Filename suffix of the vectors files',
    )
    parser.add_argument('bulk_size',
        metavar='bulk-size', default=500, type=int,
        help='Number of documents to insert in bulk mode. Defaults to 500'
    )
    parser.add_argument('recreate_index',
        metavar='recreate-index', default=False, type=lambda x: str(x).lower() in {'true', 't', '1', 'yes', 'y'},
        help='Drop and create the index if exists. Defaults to False'
    )
    parser.add_argument('skip',
        metavar='skip', default=0, type=int,
        help='Skip a number of iterations. Defaults to 0'
    )
    args = parser.parse_args()

    # Recreate index
    global_elastic_handler = elasticSearchHandler(
        args.server, args.port, args.user, args.password
    )
    if Path(args.mapping_file).exists():
        index_name = args.mapping_file.rsplit(".", 1)[0]
        if not create_data_structure(args, index_name, args.mapping_file):
            if args.recreate_index:
                drop_index(args, index_name)
                create_data_structure(args, index_name, args.mapping_file)
    elif args.mapping_file:
        index_name = args.mapping_file
    # Populate the index
    total = len(os.listdir(args.records_dir))
    records_path = Path(args.records_dir)
    vectors_path = Path(args.vectors_dir)
    bar = tqdm(records_path.rglob("./*.json"), total=total)
    file_count = 0
    for iteration, record_file in enumerate(bar):
        if iteration < args.skip:
            continue
        vector_file = record_file.name.rsplit(".json", 1)[0]
        vector_file = (
            vectors_path
            / f"{vector_file}{args.vectors_suffix}.{args.vectors_format}"
        )
        if args.vectors_format in ("vct", "txt", "plain", "text", "npy", "numpy"):
            insert_vector_file(
                record_file, vector_file, index_name, args.vectors_format, bar=bar
            )
        else:
            logger.warning("Invalid vector format")
            exit(1)
        file_count += 1
        if file_count >= args.bulk_size:
            commit_vectors(bar=bar)
            file_count = 0
    # Commit remaining vectors
    commit_vectors()
