from os import walk, path, cpu_count
import pandas as pd
import pyiqa
from mimetypes import guess_type
from pillow_heif import register_heif_opener
from pqdm.processes import pqdm

register_heif_opener()


def _check_quality(f):
    topiq_iaa = pyiqa.create_metric("topiq_iaa")
    try:
        return (f, topiq_iaa(f).item())
    except AssertionError:
        return (f, pd.NA)


def process(session):
    image_quality_df = pd.DataFrame(columns=["group", "image_name", "image_quality"])

    try:
        results = pqdm(
            [
                path.join(root, f)
                for root, _, files in walk(f"./output/{session}/images")
                for f in files
                if _validate_source(path.join(root, f))
            ],
            _check_quality,
            n_jobs=cpu_count() * 2,
        )

        for r in results:
            image_quality_df.loc[len(image_quality_df.index)] = [
                r[0].split("/")[-2],
                r[0].split("/")[-1],
                r[1],
            ]

    except Exception as e:
        raise e

    # save dataframe to excel
    image_quality_df.to_excel(
        f"./output/{session}/images/image_quality.xlsx", index=False
    )


def _validate_source(source):
    try:
        return guess_type(source)[0].startswith("image/")
    except:
        return False
