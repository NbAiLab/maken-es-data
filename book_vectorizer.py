#!/usr/bin/env python
import argparse
import gc
import io
import json
import math
from optparse import Option
import os
import re
import xml.etree.ElementTree as ET
# import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Iterator, List, NoReturn, Optional, Tuple, Union

import numpy as np
import requests
from joblib import Parallel, delayed
from joblib import Memory
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec as Model
from huggingface_hub import snapshot_download

from utils import get_http, get_logger, save_vector

location = os.environ.get("CACHE_DIR", ".cache")
memory = Memory(location, verbose=0)
URN_DATE_RE = re.compile(r"_\d{4}\d{2}\d{2}")


@memory.cache
def get_model() -> Model:
    """Get a pre-trained model to run inference with"""
    return Model.load(
        (Path(snapshot_download("NbAiLab/maken-books", cache_dir=location)) / "model.bin").as_posix(),
        mmap="r"
    )


def preprocess_book(book: str) -> Union[str, List[str]]:
    words = [c for c in re.split(r"\W+", book) if len(c) > 0]
    return words


def model_predict(
    model: Model,
    tokens: List[List[str]],
    on_batches: bool=False
) -> np.ndarray:
    if on_batches:
        raise NotImplementedError("Model does not support batched prediction")
    else:
        # vectors = model.predict(
        #     (tf.expand_dims(t, 0) for t in tensors),
        #     use_multiprocessing=True,
        #     workers=os.cpu_count(),
        # )
        vectors = []
        for token_list in tokens:
            vectors.append(model.infer_vector(doc_words=token_list))
    return np.array(vectors)


def get_text_from_alto(
    raw: bytes,
    url: Optional[str]=None,
    alto_textline_xpath: Optional[str]=None,
    alto_string_threshold: Optional[float]=None,
) -> str:
    """ Convert ALTO xml file to element tree and return text """
    alto = io.BytesIO(raw)
    try:
        xml = ET.parse(alto)
    except ET.ParseError as e:
        print(f"Parser Error in file '{alto}': {e}")
    # Register ALTO namespaces
    # https://www.loc.gov/standards/alto/ | https://github.com/altoxml
    namespace = {
        'alto-1':      'http://schema.ccs-gmbh.com/metae/alto.xsd',
        'alto-1.2':    'http://schema.ccs-gmbh.com/metae/alto-1-2.xsd',
        'nb-alto-1.2': 'http://intern.nb.no/xml/schema/alto-1-2.xsd',
        'alto-1.4':    'http://schema.ccs-gmbh.com/metae/alto-1-4.xsd',
        'alto-2':      'http://www.loc.gov/standards/alto/ns-v2#',
        'alto-3':      'http://www.loc.gov/standards/alto/ns-v3#',
        'alto-4':      'http://www.loc.gov/standards/alto/ns-v4#',
    }
    logger = get_logger()
    # Extract namespace from document root
    if 'http://' in str(xml.getroot().tag.split('}')[0].strip('{')):
        xmlns = xml.getroot().tag.split('}')[0].strip('{')
    else:
        try:
            ns = xml.getroot().attrib
            xmlns = str(ns).split(' ')[1].strip('}').strip("'")
        except IndexError:
            logger.info(f"ALTO error '{str(url)}': no namespace declaration found.")
            xmlns = "no_namespace_found"
    text = ""
    if xmlns in namespace.values():
        if alto_textline_xpath is None:
            alto_textline_xpath = ".//Page//PrintSpace//TextLine"
        if alto_string_threshold is None:
            alto_string_threshold = 0.0
        for lines in xml.iterfind(alto_textline_xpath):
            text += "\n"
            for line in lines.findall("String"):
                confidence = float(line.attrib.get("WC", 0.0))
                if confidence >= alto_string_threshold:
                    # Handling hyphenated words <HYP>
                    subs_type = line.attrib.get("SUBS_TYPE")
                    if subs_type == "HypPart1":
                        text += line.attrib.get("SUBS_CONTENT") + " "
                    elif not subs_type:
                        text += line.attrib.get("CONTENT") + " "
    else:
        logger.info(f"ALTO error '{url}': namespace {xmlns} is not registered.")
    return text


def get_records(
    path: Path,
    glob: Optional[str]="",
    batch: Optional[int]=None,
    step: int=0
) -> Iterator[Union[dict, List[dict]]]:
    """
    Iterate over files in filepath one by one or in batches. The initial
    step number of steps will be ignored
    """
    logger = get_logger()
    records = []
    # ALTO files are not yet listed in the record metadata
    altos_url = "https://api.nb.no/catalog/v1/metadata/{record_id}/altos/"
    filepath = path.rglob(args.records_glob)
    for index, file in enumerate(filepath):
        if batch and math.ceil(index / batch) >= step:
            if not file.is_file() or file.suffix.lower() != ".json":
                continue
            try:
                record_json = json.loads(file.read_text())
                urn = (
                    record_json['metadata']['identifiers']['urn']
                    .replace("URN:NBN:no-nb_", "")
                )
                access = record_json['accessInfo'].get('accessAllowedFrom', '')
                subjects = (
                    record_json['metadata'].get("subject", {}).get("topics", [])
                ) + record_json['metadata'].get("subjectName", [])
                record = {
                    'id' : record_json['id'],
                    'filename': urn,
                    'access': access,
                    'title': record_json['metadata'].get("title", ""),
                    'creators': record_json['metadata'].get("creators", ""),
                    'subject': subjects,
                    'altos': altos_url.format(record_id=record_json['id']),
                    'path': file.relative_to(path).parent
                }
            except KeyError as err:
                logger.info(f"File {file.name} key error: {str(err)}")
                record = {}
        else:
            record = {}
        if batch is None:
            yield record
        else:
            records.append(record)
        if batch and len(records) == batch:
            yield records
            records = []


def download_book(
    http: requests.Session,
    altos: str,
    filename: str,
) -> Optional[str]:
    logger = get_logger()
    try:
        response = http.get(altos)
        response_ok = response.ok
        response_message = f"status {response.status_code}"
    except requests.exceptions.ConnectionError as error:
        response_ok = False
        response_message = str(error)
    if not response_ok:
        logger.info(
            f"ALTOs for {filename} ({altos}) could not be retrieved "
            f"({response_message})"
        )
        return
    pages = []
    record_altos = response.json()["_links"]["alto"]
    record_altos_bar = tqdm(
        record_altos, position=1, leave=False, desc="- Pages"
    )
    for record_alto in record_altos_bar:
        record_alto_response = http.get(record_alto["href"])
        if record_alto_response.ok:
            pages.append(
                get_text_from_alto(
                    record_alto_response.content, url=record_alto["href"]
                )
            )
        elif record_alto_response.status_code == 401:
            # If a page gets unauthroized, then the entire books is
            pages = []
            break
    book = "\n".join(pages)
    return book


def search_book(
    filename: str, paths: List[Path], extension: str="txt"
) -> Optional[str]:
    for path in paths:
        urn_date_match = URN_DATE_RE.search(filename)
        if urn_date_match:
            urn_date = urn_date_match.group()
            book_file = (
                path
                / urn_date[1:5] / urn_date[5:7] / urn_date[7:9]
                / f"{filename}.{extension}"
            )
            if book_file.exists():
                return book_file.read_text()
    # # Run expensive searchs if not found
    # for path in paths:
    #     for book_file in path.rglob("*"):
    #         if book_file.stem == filename:
    #             return book_file.read_text()


def get_book(
    http: requests.Session,
    altos: str,
    book_dest: Path,
    download: bool,
    filename: str,
    local_paths: Optional[List[str]]=None,
) -> Optional[str]:
    """
    Read an image from image_dest and if it doesn't exist, will try to
    download it base don the iiif URL (unless download is set to False).
    If local_paths are passed in, filename will be searched in them before
    downloading it.
    """
    logger = get_logger()
    book = None
    if book_dest.exists():
        try:
            book = book_dest.read_text()
        except (OSError):
            logger.info(f"Failed to load book {book_dest.name}")
    if download and book is None:
        if local_paths:
            try:
                book = search_book(
                    filename, [Path(path) for path in local_paths]
                )
            except (OSError):
                logger.info(f"Failed to load book {book_dest.name}")
        if book is None:
            book = download_book(http, altos, filename)
        if book:
            book_dest.write_text(book)
    return book


def get_vectors(
    records: Union[dict, Tuple[dict], List[dict]],
    output: Union[Path, str],
    objects: Optional[Union[Path, str]]=None,
    vector_format: str="npy",
    vector_suffix: str="",
    overwrite: bool=False,
    download: bool=False,
    local_paths: Optional[List[str]]=None,
    model: Optional[Model]=None,
    on_batches: bool=True,
    # n_jobs=None,
    bar: Optional[tqdm]=None,
    skip: bool=False,
    inference: bool=True,
    retries: int=10,
) -> NoReturn:
    """
    For each record in records, extract the corresponding image when
    appropriate and turn it into a vector
    """
    logger = get_logger()
    if skip:
        return
    if inference:
        model = get_model() if model is None else model  # useful to parallelize
    if not isinstance(records, (tuple, list)):
        records = [records]
    tokens = []
    vector_dests = []
    http = get_http(retries=retries)
    for record in records:
        try:
            record_id = record["id"]
            filename = record["filename"]
            path = record["path"]
            altos = record["altos"]
        except KeyError as err:
            logger.info(f"Skipping record {str(record)} ({str(err)})")
            continue
        dest = Path(output) / path
        dest.mkdir(parents=True, exist_ok=True)
        vector_dest = dest / f"{record_id}{vector_suffix}.{vector_format}"
        if not overwrite and vector_dest.exists():
            continue
        if objects:
            book_dest = Path(objects) / path / f"{record_id}.txt"
        else:
            book_dest = dest / f"{record_id}.txt"
        book = get_book(http, altos, book_dest, download, filename, local_paths)
        if book is None:
            continue
        if inference:
            book_tokens = preprocess_book(book)
            tokens.append(book_tokens)
            vector_dests.append(vector_dest)
    if inference and vector_dests:
        vectors = model_predict(model, tokens, on_batches)
        for vector_dest, vector in zip(vector_dests, vectors):
            save_vector(
                vector.squeeze(), vector_dest, vector_format, record=record,
            )
            if bar is not None:
                bar.set_description(
                    f"Saving vector {vector_dest.name} ({vector.shape})"
                )
    del tokens
    del vector_dests
    gc.collect()


def main(args: argparse.ArgumentParser) -> NoReturn:
    """Main script"""
    logger = get_logger()
    logger.info(f"Reading records: {args.records_dir}/{args.records_glob}")
    logger.info(f"Writing output: {args.output_dir}/{args.records_glob}{args.vectors_suffix}.{args.vectors_format}")
    if args.objects_dir:
        logger.info(f"Reading objects: {args.objects_dir}/{args.records_glob}.txt")
    logger.info(f"Vectors will {'' if args.overwrite else 'NOT '}be overwritten")
    logger.info(f"Books will {'' if args.download else 'NOT '}be downloaded if missing")
    logger.info(f"Inference will {'' if args.inference else 'NOT '}be run")
    logger.info(f"Processing batches of {args.batch} files")
    logger.info(f"Starting from iteration step {args.step}")

    logger.info("Calculating number of records...")
    path = Path(args.records_dir).rglob(args.records_glob)
    total = len(list(None for p in path if p.is_file() and p.suffix.lower() == ".json"))
    logger.info(f"Found {total} record files ({math.ceil(total / args.batch)} batches)")
    # path = Path(args.records_dir).rglob(args.records_glob)

    logger.info(f"Running using {'all' if args.n_jobs < 0 else args.n_jobs} processes")
    records = get_records(
        Path(args.records_dir),
        glob=args.records_glob,
        batch=args.batch,
        step=args.step
    )
    bar = tqdm(
        records,
        desc="Books",
        total=math.ceil(total / args.batch),
        position=0,
    )
    if args.inference:
        model = get_model() if args.n_jobs == 1 else None
    else:
        model = None
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(get_vectors)(
            records,
            output=args.output_dir,
            objects=args.objects_dir,
            vector_format=args.vectors_format,
            vector_suffix=args.vectors_suffix,
            overwrite=args.overwrite,
            download=args.download,
            local_paths=[p.strip() for p in args.search_local_paths.split(",")],
            model=model,
            on_batches=args.batch > 1 or args.n_jobs < 0 or args.n_jobs > 1,
            bar=bar if args.n_jobs == 1 else None,
            skip=step < args.step,
            inference=args.inference,
            retries=args.retries,
        )
        for step, records in enumerate(bar))


if __name__ == '__main__':
    yesno = lambda x: str(x).lower() in {'true', 't', '1', 'yes', 'y'}
    parser = argparse.ArgumentParser(description=f""
    f"Transforms records of books extrated using api.nb.no into vectors "
    f"(embeddings). "
    f"Using the records JSON files in records-dir, iterate over them in "
    f"batches, download the corresponding ALTO files if available, "
    f"and turn them into vectors using a custom Dov2Vec. The resulting vectors "
    f"(and downloaded books) will be stored in output-dir. "
    f"Note that output-dir will replicate the structure in records-dir, thus "
    f"specifying a proper records-glob is mandatory. If objects-dir is also"
    f"passed in, it is expected its structure is the same than records-dir"
    f"", epilog=f"""Example usage:
    {__file__} ./records "*" ./vectors --vectors_format txt --n_jobs -1 -b 100
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('records_dir',
        metavar='records-dir', help='Directory with the records files')
    parser.add_argument('records_glob', default="**/*",
        metavar='records-glob', help='Glob for the directory with the records files')
    parser.add_argument('output_dir',
        metavar='output-dir', help='Directory to store vectors/objects files')
    parser.add_argument('--vectors_format',
        default="npy",
        help='File format of the vectors files. Either npy, txt, json.',
    )
    parser.add_argument('--vectors_suffix',
        default="",
        help='Filename suffix of the vectors files',
    )
    parser.add_argument('--objects_dir',
        default="", help='Directory to read objects files (txt)')
    parser.add_argument('--no_inference', dest='inference',
        action='store_false',
        help='Disable model inference and vector generation. Defaults to False'
    )
    parser.add_argument('--overwrite', '-o',
        default=False, action='store_true',
        help='Overwrite vectors. Defaults to False'
    )
    parser.add_argument('--no_download', dest='download',
        action='store_false',
        help='Disable downloading books if missing. Defaults to False'
    )
    parser.add_argument('--n_jobs', '-j', dest='n_jobs',
        default=1, type=int,
        help='Number of multiprocessing jobs. Defaults to 1 (-1 for max)',
    )
    parser.add_argument('--batch', '-b',
        default=1, type=int,
        help='Number of inputs per batch for prediction. Defaults to 1',
    )
    parser.add_argument('--step', '-s',
        default=0, type=int,
        help='Iteration step to start the process. Defaults to 0',
    )
    parser.add_argument('--retries', '-r', default=10,
        help="Number of retries when retrieving an ALTO book. Defaults to 10",
        type=int)
    parser.add_argument('--search_local_paths', default="",
        help="Search for the book URN in local paths (comma separated)",
        type=str)
    args = parser.parse_args()
    main(args)
