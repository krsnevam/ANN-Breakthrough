"""Download PlantVillage and create tomato-only, leakage-conscious splits."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


DATASET_ID = "BrandonFors/Plant-Diseases-PlantVillage-Dataset"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "plantvillage"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "plantvillage_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_rows(dataset, row_indices, split_name, class_names, label_names):
    counts = Counter()
    for row_index in tqdm(row_indices, desc=f"Writing {split_name}"):
        row = dataset[int(row_index)]
        class_name = label_names[int(row["label"])]
        if class_name not in class_names:
            continue

        destination = OUTPUT_DIR / split_name / class_name
        destination.mkdir(parents=True, exist_ok=True)
        image = row["image"].convert("RGB")
        image.save(destination / f"{split_name}_{row_index:06d}.jpg", quality=95)
        counts[class_name] += 1
    return counts


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.rglob("*.jpg")):
        raise FileExistsError(
            f"Prepared images already exist in {OUTPUT_DIR}. Remove that generated "
            "folder before preparing the data again."
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {DATASET_ID}; the two parquet splits are about 500 MB total.")
    train_source = load_dataset(DATASET_ID, split="train", cache_dir=str(CACHE_DIR))
    test_source = load_dataset(DATASET_ID, split="test", cache_dir=str(CACHE_DIR))
    label_names = train_source.features["label"].names
    tomato_names = [name for name in label_names if name.startswith("Tomato___")]
    if len(tomato_names) != 10:
        raise ValueError(f"Expected 10 tomato classes, found {len(tomato_names)}: {tomato_names}")

    tomato_ids = {index for index, name in enumerate(label_names) if name in tomato_names}
    indices_by_class = defaultdict(list)
    for row_index, label in enumerate(train_source["label"]):
        if int(label) in tomato_ids:
            indices_by_class[int(label)].append(row_index)

    rng = random.Random(args.seed)
    train_indices = []
    validation_indices = []
    for class_id, class_indices in indices_by_class.items():
        rng.shuffle(class_indices)
        validation_count = max(1, round(len(class_indices) * args.validation_fraction))
        validation_indices.extend(class_indices[:validation_count])
        train_indices.extend(class_indices[validation_count:])
    test_indices = [
        row_index
        for row_index, label in enumerate(test_source["label"])
        if int(label) in tomato_ids
    ]

    counts = {}
    counts["train"] = save_rows(
        train_source, train_indices, "train", set(tomato_names), label_names
    )
    counts["validation"] = save_rows(
        train_source, validation_indices, "validation", set(tomato_names), label_names
    )
    counts["test"] = save_rows(
        test_source, test_indices, "test", set(tomato_names), label_names
    )

    manifest = {
        "dataset": DATASET_ID,
        "source": "https://huggingface.co/datasets/BrandonFors/Plant-Diseases-PlantVillage-Dataset",
        "seed": args.seed,
        "validation_fraction_of_source_train": args.validation_fraction,
        "class_names": tomato_names,
        "counts": {split: dict(split_counts) for split, split_counts in counts.items()},
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared images in {OUTPUT_DIR}")
    for split_name, split_counts in counts.items():
        print(f"{split_name}: {sum(split_counts.values()):,} images")


if __name__ == "__main__":
    main()