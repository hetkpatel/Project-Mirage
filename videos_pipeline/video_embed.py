import torch
import cv2
from PIL import Image
from os import path, makedirs
from mimetypes import guess_type
from multiprocessing import Manager, cpu_count
from concurrent.futures import ProcessPoolExecutor
from uuid import uuid4
from json import load, dump

import embedding_models.ResNet50_Embedding as ie

from rich import progress as P


def _embed(session, id, file, progress, task_id):
    model = ie.ResNet50_ImageEmbedder()
    model.eval()
    with torch.no_grad():
        # Create video feed from file
        video = cv2.VideoCapture(file)
        for n in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
            # Get frame
            success, frame = video.read()
            if success:
                # Transform images into tensors
                t = ie.get_transforms()(Image.fromarray(frame))
                # Create embedding vector
                vector = torch.squeeze(model(t.unsqueeze(0)))
                # Save vector in tmp session folder
                if not path.exists(f"./.tmp/{session}/v/vectors/{id}"):
                    makedirs(f"./.tmp/{session}/v/vectors/{id}")
                torch.save(vector, f"./.tmp/{session}/v/vectors/{id}/{uuid4().hex}.pt")

            # Update progress
            progress[task_id] = {
                "progress": n + 1,
                "total": int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
            }

        # Release video memory
        video.release()


def process(session, input):
    try:
        if not path.exists(f"./.tmp/{session}/v/vectors"):
            makedirs(f"./.tmp/{session}/v/vectors")

        with P.Progress(
            "[progress.description]{task.description}",
            P.BarColumn(bar_width=None),
            P.TaskProgressColumn(show_speed=True),
            P.MofNCompleteColumn(),
            P.TimeRemainingColumn(),
            expand=True,
        ) as progress:
            futures = []  # keep track of the jobs
            with Manager() as manager:
                # this is the key - we share some state between our
                # main process and our worker functions
                _progress = manager.dict()
                overall_progress_task = progress.add_task("[green]All jobs progress:")

                with ProcessPoolExecutor(max_workers=cpu_count() * 2) as executor:
                    for file in [f for f in input if _validate_source(f)]:
                        task_id = progress.add_task(
                            f"Processing {path.basename(file)}...", visible=True
                        )
                        id = uuid4().hex
                        # Save id and file path in tmp session folder in json file
                        _save_id_to_file(session, id, path.abspath(file))
                        futures.append(
                            executor.submit(
                                _embed,  # function
                                session,  # session
                                id,  # id
                                file,  # file
                                _progress,
                                task_id,
                            )
                        )

                    # monitor the progress:
                    while (
                        n_finished := sum([future.done() for future in futures])
                    ) < len(futures):
                        progress.update(
                            overall_progress_task,
                            completed=n_finished,
                            total=len(futures),
                        )
                        for task_id, update_data in _progress.items():
                            latest = update_data["progress"]
                            total = update_data["total"]
                            # update the progress bar for this task:
                            progress.update(
                                task_id,
                                completed=latest,
                                total=total,
                                visible=latest < total,
                            )

                    progress.update(overall_progress_task, advance=1)

                    # raise any errors:
                    for future in futures:
                        future.result()

    except Exception as e:
        raise e


def _validate_source(source):
    try:
        return guess_type(source)[0].startswith("video/")
    except:
        return False


def _save_id_to_file(session, id, filename):
    # check if embedding map file exists, if not create it
    if not path.exists(f"./.tmp/{session}/v/embedding_map.json"):
        with open(f"./.tmp/{session}/v/embedding_map.json", "w") as f:
            dump({}, f)

    with open(f"./.tmp/{session}/v/embedding_map.json", "r") as j:
        data = load(j)
        data[id] = filename
        with open(f"./.tmp/{session}/v/embedding_map.json", "w") as f:
            dump(data, f)
