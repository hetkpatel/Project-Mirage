import torch
from PIL import Image
from os import path, makedirs, cpu_count
from mimetypes import guess_type
from pqdm.processes import pqdm
from uuid import uuid4
from json import load, dump
from pillow_heif import register_heif_opener

import embedding_models.ResNet50_Embedding as ie

register_heif_opener()
model = ie.ResNet50_ImageEmbedder()


def _embed(session, id, file):
    model.eval()
    with torch.no_grad():
        # Transform images into tensors
        t = ie.get_transforms()(Image.open(file))
        # Create embedding vector
        vector = torch.squeeze(model(t.unsqueeze(0)))
        # Save vector in tmp session folder
        torch.save(vector, f"./.tmp/{session}/i/vectors/{id}.pt")


def process(session, input):
    try:
        if not path.exists(f"./.tmp/{session}/i/vectors"):
            makedirs(f"./.tmp/{session}/i/vectors")

        pqdm(
            [
                [session, _save_id_to_file(session, f), f]
                for f in input
                if _validate_source(f)
            ],
            _embed,
            n_jobs=cpu_count() * 2,
            argument_type="args",
        )

    except Exception as e:
        raise e


def _validate_source(source):
    try:
        return guess_type(source)[0].startswith("image/")
    except:
        return False


def _save_id_to_file(session, filename):
    # check if embedding map file exists, if not create it
    if not path.exists(f"./.tmp/{session}/i/embedding_map.json"):
        with open(f"./.tmp/{session}/i/embedding_map.json", "w") as f:
            dump({}, f)

    with open(f"./.tmp/{session}/i/embedding_map.json", "r") as j:
        data = load(j)
        id = uuid4().hex
        data[id] = filename
        with open(f"./.tmp/{session}/i/embedding_map.json", "w") as f:
            dump(data, f)

    return id
