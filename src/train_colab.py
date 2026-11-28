"""Prepare PlantVillage tomato images and train MobileNetV2 in Colab."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from IPython.display import display
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V2_Weights
from tqdm.auto import tqdm


DATASET_ID = "BrandonFors/Plant-Diseases-PlantVillage-Dataset"
DATA_DIR = Path("/content/plantvillage")
OUTPUT_DIR = Path("/content/drive/MyDrive/ANN-bessel-training")
GRAPH_DIR = OUTPUT_DIR / "graphs"
SEED = 42


def prepare_data() -> list[str]:
    train_images = DATA_DIR / "train"
    if any(train_images.rglob("*.jpg")):
        print("Using prepared PlantVillage images in the current runtime.")
        return sorted(path.name for path in train_images.iterdir() if path.is_dir())

    print(f"Loading {DATASET_ID} from Hugging Face.", flush=True)
    train_source = load_dataset(DATASET_ID, split="train", cache_dir="/content/hf-cache")
    test_source = load_dataset(DATASET_ID, split="test", cache_dir="/content/hf-cache")
    label_names = train_source.features["label"].names
    class_names = [name for name in label_names if name.startswith("Tomato___")]
    if len(class_names) != 10:
        raise ValueError(f"Expected 10 tomato classes, found {len(class_names)}")
    tomato_ids = {index for index, name in enumerate(label_names) if name in class_names}

    by_class = defaultdict(list)
    for index, label in enumerate(train_source["label"]):
        if int(label) in tomato_ids:
            by_class[int(label)].append(index)

    rng = random.Random(SEED)
    train_ids, validation_ids = [], []
    for indices in by_class.values():
        rng.shuffle(indices)
        validation_count = max(1, round(len(indices) * 0.1))
        validation_ids.extend(indices[:validation_count])
        train_ids.extend(indices[validation_count:])
    test_ids = [
        index for index, label in enumerate(test_source["label"])
        if int(label) in tomato_ids
    ]

    def write_split(source, indices, split_name):
        for index in tqdm(indices, desc=f"Preparing {split_name}"):
            row = source[int(index)]
            class_name = label_names[int(row["label"])]
            destination = DATA_DIR / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)
            row["image"].convert("RGB").save(
                destination / f"{split_name}_{index:06d}.jpg", quality=95
            )

    write_split(train_source, train_ids, "train")
    write_split(train_source, validation_ids, "validation")
    write_split(test_source, test_ids, "test")
    return sorted(class_names)


def save_progress_graph(history, display_handle):
    epoch_numbers = range(1, len(history) + 1)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    series = [
        ("Loss", "train_loss", "validation_loss"),
        ("Accuracy", "train_accuracy", "validation_accuracy"),
        ("Error rate", "train_error", "validation_error"),
    ]
    for axis, (title, train_key, validation_key) in zip(axes, series):
        axis.plot(epoch_numbers, [row[train_key] for row in history], label="Train")
        axis.plot(
            epoch_numbers,
            [row[validation_key] for row in history],
            label="Validation",
        )
        if title == "Accuracy":
            axis.axhline(0.95, color="gray", linestyle="--", label="95% target")
        axis.set(title=title, xlabel="Completed epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"Training progress after epoch {len(history)}")
    figure.tight_layout()
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(GRAPH_DIR / "training_progress.png", dpi=150)
    figure.savefig(GRAPH_DIR / f"epoch_{len(history):02d}.png", dpi=150)
    if display_handle[0] is None:
        display_handle[0] = display(figure, display_id=True)
    else:
        display_handle[0].update(figure)
    plt.close(figure)


def run_epoch(model, loader, criterion, device, optimizer=None, freeze_features=False):
    training = optimizer is not None
    model.train(training)
    if training and freeze_features:
        model.features.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / total
    return total_loss / total, accuracy


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Training device:",
        torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        flush=True,
    )

    class_names = prepare_data()
    weights = MobileNet_V2_Weights.DEFAULT
    mean, std = weights.transforms().mean, weights.transforms().std
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = weights.transforms()
    train_data = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    validation_data = datasets.ImageFolder(
        DATA_DIR / "validation", transform=eval_transform
    )
    test_data = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)
    if train_data.classes != validation_data.classes or train_data.classes != test_data.classes:
        raise ValueError("Train, validation, and test classes do not match.")

    loader_args = {
        "batch_size": 32,
        "num_workers": 2,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_args)
    test_loader = DataLoader(test_data, shuffle=False, **loader_args)
    print(
        f"Images: train={len(train_data):,}, validation={len(validation_data):,}, "
        f"test={len(test_data):,}",
        flush=True,
    )

    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_accuracy, best_state = -1.0, None
    display_handle = [None]
    log_path = OUTPUT_DIR / "training_log.csv"
    fields = [
        "phase", "epoch", "train_loss", "train_accuracy", "train_error",
        "validation_loss", "validation_accuracy", "validation_error",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        csv.DictWriter(log_file, fieldnames=fields).writeheader()

    def train_phase(phase, epochs, learning_rate, freeze_features):
        nonlocal best_accuracy, best_state
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = run_epoch(
                model, train_loader, criterion, device, optimizer, freeze_features
            )
            validation_loss, validation_accuracy = run_epoch(
                model, validation_loader, criterion, device
            )
            row = {
                "phase": phase,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_error": 1 - train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "validation_error": 1 - validation_accuracy,
            }
            history.append(row)
            with log_path.open("a", newline="", encoding="utf-8") as log_file:
                csv.DictWriter(log_file, fieldnames=fields).writerow(row)
            print(
                f"{phase} {epoch}/{epochs}: train_loss={train_loss:.4f}, "
                f"train_error={1-train_accuracy:.2%}, "
                f"val_loss={validation_loss:.4f}, val_acc={validation_accuracy:.4f}, "
                f"val_error={1-validation_accuracy:.2%}",
                flush=True,
            )
            save_progress_graph(history, display_handle)
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "class_names": class_names,
                        "best_validation_accuracy": best_accuracy,
                    },
                    OUTPUT_DIR / "mobilenetv2_tomato_best.pt",
                )

    for parameter in model.features.parameters():
        parameter.requires_grad = False
    train_phase("classifier", 8, 1e-3, True)
    for parameter in model.features[-2:].parameters():
        parameter.requires_grad = True
    train_phase("fine_tune", 5, 1e-5, False)
    model.load_state_dict(best_state)

    model.eval()
    true_labels, predicted_labels = [], []
    total_test_loss = 0.0
    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Final held-out test"):
            labels = labels.to(device)
            logits = model(images.to(device, non_blocking=True))
            total_test_loss += criterion(logits, labels).item() * labels.size(0)
            predicted_labels.extend(logits.argmax(dim=1).cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    test_accuracy = sum(
        truth == prediction
        for truth, prediction in zip(true_labels, predicted_labels)
    ) / len(true_labels)
    metrics = {
        "test_accuracy": test_accuracy,
        "test_error": 1 - test_accuracy,
        "test_macro_f1": f1_score(true_labels, predicted_labels, average="macro"),
        "test_loss": total_test_loss / len(true_labels),
        "target_accuracy": 0.95,
        "target_met": test_accuracy >= 0.95,
        "class_names": class_names,
        "test_examples": len(true_labels),
    }
    (OUTPUT_DIR / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(
        classification_report(true_labels, predicted_labels, target_names=class_names),
        encoding="utf-8",
    )
    matrix = confusion_matrix(true_labels, predicted_labels)
    figure, axis = plt.subplots(figsize=(12, 10))
    axis.imshow(matrix, cmap="Greens")
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=[name.replace("Tomato___", "") for name in class_names],
        yticklabels=[name.replace("Tomato___", "") for name in class_names],
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Held-out test confusion matrix (accuracy {test_accuracy:.3f})",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved outputs in {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()