#!/usr/bin/env python
import argparse
from asyncio.log import logger
import gc
import io
import json
import math
import os
# import warnings; warnings.filterwarnings("ignore", message=".*tensorflow.*")
from pathlib import Path
from typing import Iterator, List, NoReturn, Optional, Tuple, Union

import requests
from joblib import Parallel, delayed
from joblib import Memory
from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array

from utils import get_http, get_logger, save_vector

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ImageFile.LOAD_TRUNCATED_IMAGES = True
location = os.environ.get("CACHE_DIR", ".cache")
memory = Memory(location, verbose=0)


@memory.cache
def get_model() -> Model:
    """Get a pre-trained model to run inference with"""
    base = InceptionV3(include_top=True, weights='imagenet')
    return Model(inputs=base.input, outputs=base.get_layer('avg_pool').output)


def preprocess_image(image: Image.Image)-> tf.Tensor:
    return preprocess_input(img_to_array(image.resize((299, 299))))


def model_predict(
    model: Model,
    tensors: List[tf.Tensor],
    on_batches: bool=False
) -> tf.Tensor:
    if on_batches:
        vectors = model.predict_on_batch(tf.stack(tensors))
    else:
        vectors = model.predict(
            (tf.expand_dims(t, 0) for t in tensors),
            use_multiprocessing=True,
            workers=os.cpu_count(),
        )
    return vectors


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
    filepath = path.rglob(args.records_glob)
    for index, file in enumerate(filepath):
        if batch and math.ceil(index / batch) >= step:
            if not file.is_file() or file.suffix.lower() != ".json":
                continue
            with file.open() as record_file:
                record_json = json.load(record_file)
                try:
                    iiif = record_json['_links']['thumbnail_custom']['href']
                except KeyError:
                    logger.info(f"File {file.name} does not have IIIF link")
                    iiif = ""
                    continue
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
                    'iiif': iiif,
                    'title': record_json['metadata'].get("title", ""),
                    'creators': record_json['metadata'].get("creators", ""),
                    'subject': subjects,
                    'path': file.relative_to(path).parent
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


def search_image(filename: str, paths: List[Path]) -> Optional[str]:
    access_levels = (
        "",
        "everywhere", "library", "nb", "norway",
        "EVERYWHERE", "LIBRARY", "NB", "NORWAY",
    )
    for access in access_levels:
        for path in paths:
            image_file = path / access / filename
            if image_file.exists():
                return Image.open(
                    io.BytesIO(image_file.read_bytes())
                ).convert("RGB")
    # # Run expensive searchs if not found
    # for path in paths:
    #     for image_file in path.rglob("*"):
    #         if image_file.name == filename:
    #             return Image.open(image_file.read_bytes()).convert("RGB")


def download_image(
    http: requests.Session,
    iiif: str,
) -> Optional[Image.Image]:
    image = None
    response = http.get(iiif.format(width=600, height=600))
    if not response.ok:
        response = http.get(iiif.format(width=300, height=300))
    if response.ok:
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return image


def get_image(
    http: requests.Session,
    iiif: str,
    image_dest: Path,
    download: bool,
    filename: str,
    local_paths: Optional[List[str]]=None,
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
        if local_paths:
            try:
                image = search_image(
                    image_dest.name, [Path(path) for path in local_paths],
                )
            except (Exception):
                logger.info(f"Failed to load image {image_dest.name}")
        if image is None:
            image = download_image(http, iiif)
        if image:
            image.save(image_dest, format='JPEG', subsampling=0, quality=100)
    return image


def get_vectors(
    records: Union[dict, Tuple[dict], List[dict]],
    output: Union[Path, str],
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
    tensors = []
    vector_dests = []
    http = get_http(retries=retries)
    for record in records:
        try:
            record_id = record["id"]
            filename = record["filename"]
            path = record["path"]
            iiif = record["iiif"]
        except KeyError as err:
            logger.info(f"Skipping record {str(record)} ({str(err)})")
            continue
        dest = Path(output) / path
        dest.mkdir(parents=True, exist_ok=True)
        vector_dest = dest / f"{record_id}{vector_suffix}.{vector_format}"
        if not overwrite and vector_dest.exists():
            continue
        image_dest = dest / f"{record_id}.jpg"
        image = get_image(
            http, iiif, image_dest, download, filename, local_paths
        )
        if image is None:
            continue
        if inference:
            tensor = preprocess_image(image)
            tensors.append(tensor)
            vector_dests.append(vector_dest)
        image.close()
    if inference and vector_dests:
        vectors = model_predict(model, tensors, on_batches)
        for vector_dest, vector in zip(vector_dests, vectors):
            save_vector(
                vector.squeeze(), vector_dest, vector_format, record=record,
            )
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
    logger.info(f"Vectors will {'' if args.overwrite else 'NOT '}be overwritten")
    logger.info(f"Images will {'' if args.download_images else 'NOT '}be downloaded if missing")
    logger.info(f"Inference will {'' if args.inference else 'NOT '}be run")
    logger.info(f"Processing batches of {args.batch} files")
    logger.info(f"Starting from iteration step {args.step}")

    logger.info("Calculating number of records...")
    path = Path(args.records_dir).rglob(args.records_glob)
    total = len(list(None for p in path if p.is_file() and p.suffix.lower() == ".json"))
    logger.info(f"Found {total} record files ({math.ceil(total / args.batch)} batches)")

    logger.info(f"Running using {'all' if args.n_jobs < 0 else args.n_jobs} processes")
    records = get_records(
        Path(args.records_dir),
        glob=args.records_glob,
        batch=args.batch,
        step=args.step
    )
    bar = tqdm(
        records,
        desc="Images",
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
            output=args.vectors_dir,
            vector_format=args.vectors_format,
            vector_suffix=args.vectors_suffix,
            overwrite=args.overwrite,
            download=args.download_images,
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
    f"Transforms records of images extrated using api.nb.no into vectors "
    f"(embeddings). "
    f"Using the records JSON files in records-dir, iterate over them in "
    f"batches, download the corresponding IIIF images if available, "
    f"and turn them into vectors using Inception V3. The resulting vectors "
    f"(and downloaded images) will be stored in vectors-dir. "
    f"Note that vectors-dir will replicate the structure in records-dir, thus "
    f"specifying a proper records-glob is mandatory"
    f"", epilog=f"""Example usage:
    {__file__} ./records "*" ./vectors --vectors_format txt --n-jobs -1 -b 100
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('records_dir',
        metavar='records-dir', help='Directory with the records files')
    parser.add_argument('records_glob', default="**/*",
        metavar='records-glob', help='Glob for the directory with the records files')
    parser.add_argument('vectors_dir',
        metavar='vectors-dir', help='Directory to store vectors files')
    parser.add_argument('--vectors_format',
        default="npy",
        help='File format of the vectors files. Either npy or vct (plain text)',
    )
    parser.add_argument('--vectors_suffix',
        default="",
        help='Filename suffix of the vectors files',
    )
    parser.add_argument('--no_inference', dest='inference',
        action='store_false',
        help='Disable model inference and vector generation. Defaults to False'
    )
    parser.add_argument('--overwrite', '-o',
        default=False, action='store_true',
        help='Overwrite vectors. Defaults to False'
    )
    parser.add_argument('--no_download_images', dest='download_images',
        action='store_false',
        help='Disable downloading images if missing. Defaults to False'
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
        help="Number of retries when retrieving a IIIF image. Defaults to 10",
        type=int)
    parser.add_argument('--search_local_paths', default="",
        help="Search for the image ID in local paths (comma separated)",
        type=str)
    args = parser.parse_args()
    main(args)
