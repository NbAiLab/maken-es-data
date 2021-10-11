#!/usr/bin/env python
import argparse
import gc
import io
import json
import logging
import math
import os
import sys
# import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Iterator, List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, TimeoutSauce
from joblib import Parallel, delayed
from joblib import Memory
from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm import tqdm
from urllib3.util import Retry

import tensorflow as tf
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import InceptionV3, imagenet_utils
from keras.models import Model
from keras.preprocessing.image import save_img, img_to_array, array_to_img
from keras_preprocessing.image import load_img


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ImageFile.LOAD_TRUNCATED_IMAGES = True
location = os.environ.get("CACHE_DIR", ".cache")
memory = Memory(location, verbose=0)


def get_logger() -> logging.Logger:
    """
    Get a logger
    """
    logger = logging.getLogger(__name__)
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


@memory.cache
def get_model() -> Model:
    """Get a pre-trained model to run inference with"""
    base = InceptionV3(include_top=True, weights='imagenet')
    return Model(inputs=base.input, outputs=base.get_layer('avg_pool').output)


def get_records(
    filepath: Path,
    batch: Optional[int]=None,
    step: int=0
) -> Iterator[Union[dict, List[dict]]]:
    """
    Iterate over files in filepath one by one or in batches. The initial
    step number of steps will be ignored
    """
    logger = get_logger()
    records = []
    for index, file in enumerate(filepath):
        if batch and math.ceil(index / batch) >= step:
            if not file.is_file():
                continue
            with file.open() as record_file:
                record_json = json.load(record_file)
                try:
                    iiif = record_json['_links']['thumbnail_custom']['href']
                except KeyError:
                    logger.info(f"File {file.name} does not have IIIF link")
                    continue
                    iiif = ""
                urn = (
                    record_json['metadata']['identifiers']['urn']
                    .replace("URN:NBN:no-nb_", "")
                )
                access = record_json['accessInfo']['accessAllowedFrom'].lower()
                record = {
                    'id' : record_json['id'],
                    'filename': urn,
                    'access': access,
                    'iiif': iiif,
                }
        else:
            record = {}
        if batch is None:
            yield record
        else:
            records.append(record)
        if batch and len(records) == batch:
            yield records
            records = []


def get_image(
    iiif: str,
    image_dest: Path,
    download: bool,
    filename: str,
) -> Optional[Image.Image]:
    """
    Read an image from image_dest and if it doesn't exist, will try to
    download it base don the iiif URL (unless download is set to False).
    The filename is only use for logging purposes
    """
    logger = get_logger()
    image = None
    if image_dest.exists():
        try:
            image = Image.open(image_dest).convert("RGB")
        except (UnidentifiedImageError, OSError):
            logger.info(f"Failed to load image {image_dest.name}")
    if download and image is None:
        logger.info(f"Downloading image for {filename}")
        response = requests.get(iiif.format(width=600, height=600))
        if not response.ok:
            response = requests.get(iiif.format(width=300, height=300))
        try:
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception:
            logger.info(
                f"Image {filename} could not be retrieved "
                f"(status {response.status_code})"
            )
            return
        image.save(image_dest, format='JPEG', subsampling=0, quality=100)
    return image


def save_vector(
    vector: np.ndarray,
    vector_dest: Path,
    vector_suffix: str
) -> NoReturn:
    """
    Save a vector in either binary (npy) or plain text (vct) format in
    vector_dest
    """
    if vector_suffix == "npy":
        np.save(vector_dest, vector.squeeze())
    else:
        np.savetxt(vector_dest, vector.squeeze())


def get_vectors(
    records: Union[dict, Tuple[dict], List[dict]],
    output: Union[Path, str],
    vector_format: str="npy",
    vector_suffix: str="",
    overwrite: bool=False,
    download: bool=False,
    model: Optional[Model]=None,
    on_batches: bool=True,
    # n_jobs=None,
    bar: Optional[tqdm]=None,
    skip: bool=False,
) -> NoReturn:
    """
    For each record in records, extract the corresponding image when
    appropriate and turn it into a vector
    """
    logger = get_logger()
    if skip:
        return
    model = get_model() if model is None else model  # useful to parallelize
    if not isinstance(records, (tuple, list)):
        records = [records]
    tensors = []
    vector_dests = []
    for record in records:
        try:
            record_id = record["id"]
            filename = record["filename"]
            access = record["access"]
            iiif = record["iiif"]
        except KeyError as err:
            logger.info(f"Skipping record {str(record)} ({str(err)})")
            continue
        dest = Path(os.path.join(output, access))
        dest.mkdir(parents=True, exist_ok=True)
        vector_dest = dest / f"{record_id}{vector_suffix}.{vector_format}"
        if not overwrite and vector_dest.exists():
            continue
        image_dest = dest / f"{record_id}.jpg"
        image = get_image(iiif, image_dest, download, filename)
        if image is None:
            continue
        tensor = preprocess_input(img_to_array(image.resize((299, 299))))
        tensors.append(tensor)
        vector_dests.append(vector_dest)
        image.close()
    if vector_dests:
        if on_batches:
            vectors = model.predict_on_batch(tf.stack(tensors))
        else:
            vectors = model.predict(
                (tf.expand_dims(t, 0) for t in tensors),
                use_multiprocessing=True,
                workers=os.cpu_count(),
            )
        for vector_dest, vector in zip(vector_dests, vectors):
            save_vector(vector.squeeze(), vector_dest, vector_suffix)
            if bar is not None:
                bar.set_description(
                    f"Saving vector {vector_dest.name} ({vector.shape})"
                )
    del tensors
    del vector_dests
    gc.collect()


def main(args: argparse.ArgumentParser) -> NoReturn:
    """Main script"""
    logger = get_logger()
    logger.info(f"Reading records: {args.records_dir}/{args.records_glob}")
    logger.info(f"Writing vectors: {args.vectors_dir}/{args.records_glob}{args.vectors_suffix}.{args.vectors_format}")
    logger.info(f"Vectors will {'' if args.overwrite else 'NOT'} be overwritten")
    logger.info(f"Images will {'' if args.download else 'NOT'} be downloaded if missing")
    logger.info(f"Processing batches of {args.batch} files")
    logger.info(f"Starting from iteration step {args.step}")

    logger.info("Calculating number of records...")
    path = Path(args.records_dir).rglob(args.records_glob)
    total = len(list(None for p in path if p.is_file()))
    logger.info(f"Found {total} record files ({math.ceil(total / args.batch)}) batches")
    path = Path(args.records_dir).rglob(args.records_glob)

    logger.info(f"Running using {'all' if args.n_jobs < 0 else args.n_jobs} processes")
    bar = tqdm(
        get_records(path, batch=args.batch, step=args.step),
        total=math.ceil(total / args.batch),
    )
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(get_vectors)(
            records,
            output=args.vectors_dir,
            vector_format=args.vectors_format,
            vector_suffix=args.vectors_suffix,
            overwrite=args.overwrite,
            download=args.download,
            model=get_model() if args.n_jobs == 1 else None,
            on_batches=args.batch > 1 or args.n_jobs < 0 or args.n_jobs > 1,
            bar=bar if args.n_jobs == 1 else None,
            skip=step < args.step,
        )
        for step, records in enumerate(bar))


if __name__ == '__main__':
    yesno = lambda x: str(x).lower() in {'true', 't', '1', 'yes', 'y'}
    parser = argparse.ArgumentParser(description=f""
    f"Transforms records of images extrated using api.nb.no into vectors "
    f"(embeddings). "
    f"Using the records JSON files in records-dir, iterate over them in "
    f"batches, download the corresponding IIIF images if available, "
    f"and turn them into vectors using Inception V3. The resulting vectors "
    f"(and downloaded images) will be stored in vectors-dir. "
    f"Note that vectors-dir will replicate the structure in records-dir, thus"
    f"specifying a proper records-glob is mandatory"
    f"", epilog=f"""Example usage:
    {__file__} ./records "*" ./vectors npy "_inception" false true -1 100 0
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('records_dir',
        metavar='records-dir', help='Directory with the records files')
    parser.add_argument('records_glob', default="**/*",
        metavar='records-glob', help='Glob for the directory with the records files')
    parser.add_argument('vectors_dir',
        metavar='vectors-dir', help='Directory to store vectors files')
    parser.add_argument('vectors_format',
        metavar='vectors-format', default="npy",
        help='File format of the vectors files. Either npy or vct (plain text)',
    )
    parser.add_argument('vectors_suffix',
        metavar='vectors-suffix', default="",
        help='Filename suffix of the vectors files',
    )
    parser.add_argument('overwrite',
        metavar='vectors-overwrite', default=False, type=yesno,
        help='Overwrite vectors. Defaults to False'
    )
    parser.add_argument('download',
        metavar='download-images', default=False, type=yesno,
        help='Download images if missing. Defaults to False'
    )
    parser.add_argument('n_jobs',
        metavar='n-jobs', default=1, type=int,
        help='Number of multiprocessing jobs. Defaults to 1 (-1 for max)',
    )
    parser.add_argument('batch',
        metavar='batch', default=1, type=int,
        help='Number of inputs per batch for prediction. Defaults to 1',
    )
    parser.add_argument('step',
        metavar='step', default=0, type=int,
        help='Iteration step to start the process. Defaults to 0',
    )
    args = parser.parse_args()
    main(args)

