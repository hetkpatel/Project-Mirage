from torch import load as tload
import pyiqa
from torch.nn.functional import cosine_similarity as cos
from tqdm import tqdm
from json import dump, load
from os import path, walk

# .832104802131654 - resnet50
THRESHOLD = 0.832104802131654


def process(session):
    clusters = {}

    files = [
        path.join(root, f)
        for root, _, files in walk(f"./.tmp/{session}/vectors/")
        for f in files
    ]

    def calculate_cosine_delta(target, batch) -> dict[str, float]:
        result = {target: 1.0}
        target_tensor = tload(target).unsqueeze(dim=0)

        for k in batch:
            if k != target:
                cos_sim = cos(target_tensor, tload(k).unsqueeze(dim=0))[0].item()
                if cos_sim >= THRESHOLD:
                    result[k] = cos_sim

        return result

    for file in tqdm(desc="Finding duplicates", iterable=files):
        similar_images = calculate_cosine_delta(file, files)
        clusters[file] = list(similar_images.keys())

    def get_cluster(node) -> list:
        cluster = set()
        stack = [node]

        while stack:
            current_node = stack.pop()
            cluster.add(current_node)

            for neighbor in clusters[current_node]:
                if neighbor not in cluster:
                    stack.append(neighbor)

        return list(cluster)

    groups = {}
    checked = set()
    for k, _ in clusters.items():
        if k not in checked:
            cluster = get_cluster(k)
            groups[len(groups)] = cluster
            checked.update(cluster)

    print(groups)

    # save images to the .tmp/session folder in their respective groups
    similarity_results = {}
    with open(f"./.tmp/{session}/embedding_map.json", "r") as em:
        embedding_map = load(em)
        topiq_iaa = pyiqa.create_metric("topiq_iaa")
        for _, similarImageEmbedIds in groups.items():
            imageList = []
            # Group similar images with real path
            for embedId in similarImageEmbedIds:
                id = path.basename(embedId).split(".")[0]
                org_path = embedding_map[id]
                imageList.append(org_path)

            # Find best quality image from group
            best_quality_image = ""
            best_quality = 0
            for image in imageList:
                quality = topiq_iaa(image).item()
                if quality > best_quality:
                    best_quality_image = image
                    best_quality = quality

            similarity_results[best_quality_image] = imageList

    with open(f"./.tmp/{session}/similarity_results.json", "w") as f:
        dump(similarity_results, f)
