# Maken Elasticsearch Data Scripts

Scripts to download images and books from api.nb.no, convert them to vectors and ingest them into Elasticsearch

## API Downloader

Downloader of JSON records of the catalog in `https://api.nb.no`. It uses the `/catalog/v1/items` endpoint to perform requests by default. Each record will be stored as an independent JSON file named as their identifier.

Usage:

```bash
./api_downloader.py [-h] [--output-folder OUTPUT] [--endpoint ENDPOINT] [--filter [FILTER [FILTER ...]]] [--scroll] [--profile PROFILE] [--page PAGE] [--size SIZE] [--criteria CRITERIA]
```

Optional arguments:

- `-h`, `--help`. Shows help message.
- `--output-folder OUTPUT`, `-o OUTPUT`. Path to output folder. Defaults to `./out`.
- `--endpoint ENDPOINT`, `-e ENDPOINT`. Endpoint to use in the requests. Defaults to `catalog/v1/items`.
- `--filter [FILTER [FILTER ...]]`, `-f [FILTER [FILTER ...]]`. Filters to add to the query.
- `--scroll`. Whether or not enable scrolling (resource intensive). Defaults to false.
- `--profile PROFILE`. Profile. Defaults to `wwwnbno`.
- `--page PAGE`, `-p PAGE`. Start at a specific page. Defaults to 0.
- `--size SIZE`, `-s SIZE`. Number of results per page. Defaults to 25 [1-100].
- `--criteria CRITERIA`. JSON field key to use as criteria to split records in the output folder. Defaults to `accessInfo.accessAllowedFrom`.
- `--delay DELAY`, `-d DELAY`. Number of seconds to space out requests. Defaults to 1.0
- `--timeout TIMEOUT`, `-t TIMEOUT`. Number of seconds to wait for a response. Defaults to 300
- `--retries RETRIES`, `-r RETRIES`. Number of retries when retrieven a URL. Defaults to 10
- `--overwrite`. Overwrite existing JSON files. Defaults to false
- `--log`, `-l`. Log to disk failing results pages. When passed in, files named `./page_<num>.log` will be created. Defaults to false

Example usage:

```bash
./api_downloader.py -f mediatype:bilder -f contentClasses:JP2 --size 15
```

A few interesting filters:

- `mediatype:bilder` for images
- `mediatype:bøker` for books
- `contentClasses:JP2` for images in JPEG2000 format
- `digital:Ja` for images for which there is a digital version
- `firstDigitalContentTime:[20210922 TO 20310101]` for a time bounded query on digitization time

## Image Vectorizer

Transforms records of images extrated using api.nb.no into vectors (embeddings).
Using the records JSON files in records-dir, iterate over them in batches, download the corresponding IIIF images if available, and turn them into vectors using Inception V3. The resulting vectors (and downloaded images) will be stored in vectors-dir. Note that vectors-dir will replicate the structure in records-dir, thusspecifying a proper records-glob is mandatory

Usage:

```bash
./image_vectorizer.py [-h] records-dir records-glob vectors-dir vectors-format vectors-suffix vectors-overwrite download-images n-jobs batch step
```

Positional arguments:

- `records-dir`. Directory with the records files
- `records-glob`. Glob for the directory with the records files
- `vectors-dir`. Directory to store vectors files
- `vectors-format`. File format of the vectors files. Either npy or vct (plain text)
- `vectors-suffix`. Filename suffix of the vectors files
- `vectors-overwrite`. Overwrite vectors. Defaults to False
- `download-images`. Download images if missing. Defaults to False
- `n-jobs`. Number of multiprocessing jobs. Defaults to 1 (-1 for max)
- `batch`. Number of inputs per batch for prediction. Defaults to 1
- `step`. Iteration step to start the process. Defaults to 0

Optional arguments:

- `-h`, `--help`. Shows help message.

Example usage:

- Convert all image records in `./records` to vectors in plain text format, adding no suffix, not overwriting the existing ones, downloading images if nor present, using all cores avalialbe, and in batches of 100 images at a time.:

```bash
./image_vectorizer.py ./records "*" ./vectors vct "_inception" false true -1 100 0
```

- Convert image records in the subfolder `library` of `./data/records/images` into NumPy binary vectors with the suffix "_inceptionv3", without overwriting the existing ones nor downloading images not present, on 100 cores and batches of 1000 images:

```bash
python image_vectorizer.py ./data/records/images "library/*" ./data/vectors/images npy _inceptionv3 false false 100 1000 0
```

## Ingester

Insert documents into a Elasticsearch instance. The index is specified by the mapping-file, the documents are composed using the records in record-dir and the vectors in vectors-dir

Usage:

```bash
./ingester.py [-h] server port user password mapping-file records-dir vectors-dir vectors-format vectors-suffix bulk-size recreate-index skip
```

Positional arguments:

- `server`. IP address or logicalname for Elasticsearch server
- `port`. Elasticsearch server port
- `user`. Username for basic auth
- `password`. Password for basic auth
- `mapping-file`. Path the index mapping file in JSON. The file name without the extension will be used as the new index name. If not a file, the string will be used as the index
- `records-dir`. Directory with the records files
- `vectors-dir`. Directory with the vectors files
- `vectors-format`. File format of the vectors files. Either npy or vct
- `vectors-suffix`. Filename suffix of the vectors files
- `bulk-size`. Number of documents to insert in bulk mode. Defaults to 500
- `recreate-index`. Drop and create the index if exists. Defaults to False
- `skip`. Skip a number of iterations. Defaults to 0

Optional arguments:

- `-h`, `--help`. Shows help message.

Example usage:

- Insert documents into an existing index `index.knn` in batches of 500 documents using the records in `./records` and the vectors in NumPy binary format stored in `./vectors` with no suffix:

```bash
./ingester.py 192.0.0.10 9200 user pass index.knn ./records/ ./vectors/ npy "" 500 false 0
```

- Insert documents into a new index `restricted.knn` index in batches of 500 documents using the records in `./data/records/images/norway/` and the vectors ending in `_inceptionv3` in NumPy binary format stored in `./data/vectors/images/norway/`:

```bash
./ingester.py 192.0.0.10 9200 user pass restricted.knn.mapping ./data/records/images/norway/ ./data/vectors/images/norway/ npy _inceptionv3 500 false 0
```

### Circuit breaker issues

With around 1 million vectors with 2048 dimensions in 2 shards, the amount of memory need is around 18GB (1.1 * (4 * 2048 + 8 * 16) * 2000000), which might cause circuit breaker issues if the available RAM after the JVM is less than that and the default policy for the circuit breaker is the default 50%. As a work around (besides adding more RAM), is to disable the cuircuit breaker limit so the cache is removed and ingesting can continue.

- `GET _cluster/settings?include_defaults=true`

- `PUT _cluster/settings`

```json
{
    "persistent": {
        "knn.memory.circuit_breaker.limit": null
    }
}
```

- `PUT _cluster/settings`

```json
{
    "transient": {
        "knn.memory.circuit_breaker.limit": null
    }
}
```

### Warmup

After ingestion, the kNN indices can be warmed up. More on [Performance tuning](https://opensearch.org/docs/latest/search-plugins/knn/performance-tuning/).

- `GET _plugins/_knn/warmup/images.knn,books.knn`

## Pipeline

An example pipeline could download all image records, get their IIIF images, convert them into vectors, and insert them into a Elasticsearch instance.

```bash
./api_downloader.py -f mediatype:bilder contentClasses:JP2 --size 100 --scroll --delay 0.0 --log -o ./records/images

./image_vectorizer.py ./records/images "**/*" ./vectors/images npy _inceptionv3 false true 100 1000 0

./ingester.py 192.0.0.10 9200 user pass images.knn.mapping ./records/images/everywhere/ ./data/vectors/images/everywhere/ npy _inceptionv3 500 false 0
./ingester.py 192.0.0.10 9200 user pass images.knn.mapping ./records/images/nb/ ./data/vectors/images/nb/ npy _inceptionv3 500 false 0
./ingester.py 192.0.0.10 9200 user pass images.knn.mapping ./records/images/library/ ./data/vectors/images/library/ npy _inceptionv3 500 false 0
```
