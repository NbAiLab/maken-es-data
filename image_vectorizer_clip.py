import csv
import json
import os
import time
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing.dummy import Pool
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


def compute_image_embeddings(list_of_images):
    return model.get_image_features(
        **processor(images=list_of_images, return_tensors="pt", padding=True)
    )


def load_image(path, same_height=False):
    im = Image.open(path)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    if same_height:
        ratio = 224 / im.size[1]
    else:
        ratio = 224 / min(im.size)
    return im.resize((int(im.size[0] * ratio), int(im.size[1] * ratio)))


def main():
    embeddings = None
    rows = [["id", "label", "thumbnail"]]
    total = sum(1 for _ in Path("./vectors_20211011").rglob("**/*.jpg"))
    images_path = Path("./vectors_20211011").rglob("**/*.jpg")
    for i, image_path in enumerate(tqdm(images_path, total=total)):
        embedding = compute_image_embeddings(
            [load_image(image_path)]
        ).detach().numpy()[0]
        if embeddings is None:
            embeddings = embedding
        else:
            embeddings = np.vstack([embeddings, embedding])
        filename = image_path.as_posix()
        record_path = (filename
            .replace("vectors", "records")
            .replace(".jpg", ".json")
        )
        with open(record_path) as record_file:
            record = json.load(record_file)
        rows.append([
            image_path.stem,
            record["metadata"]["title"],
            record["_links"]["thumbnail_large"]["href"]
        ])
        if i and i % 1000 == 0:
            with open("clip.csv", "w") as clip_file:
                writer = csv.writer(clip_file)
                for row in tqdm(rows, desc="Writing rows and embeddings"):
                    writer.writerow(row)
            np.save("clip.npy", embeddings)

    with open("clip.csv", "w") as clip_file:
        writer = csv.writer(clip_file)
        for row in tqdm(rows, desc="Writing rows and embeddings"):
            writer.writerow(row)
    np.save("clip.npy", embeddings)


if __name__ == "__main__":
    main()
