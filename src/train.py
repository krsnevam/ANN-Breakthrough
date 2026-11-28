"""Train and evaluate a MobileNetV2 tomato disease classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V2_Weights
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "plantvillage"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_PATH = OUTPUT_DIR / "training_log.csv"
GRAPH_DIR = OUTPUT_DIR / "graphs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(batch_size: int, workers: int, weights):
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
    validation_data = datasets.ImageFolder(DATA_DIR / "validation", transform=eval_transform)
    test_data = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)
    if train_data.classes != validation_data.classes or train_data.classes != test_data.classes:
        raise ValueError("Train, validation, and test folders must have identical class folders.")

    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)
    return train_loader, validation_loader, test_loader, train_data.classes


def run_epoch(model, loader, criterion, device, optimizer=None, freeze_features=False):
    is_training = optimizer is not None
    model.train(is_training)
    if freeze_features:
        model.features.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader, leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        progress.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / total, correct / total


def append_epoch_log(row):
    fieldnames = [
        "phase",
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_error",
        "validation_loss",
        "validation_accuracy",
        "validation_error",
    ]
    with LOG_PATH.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        if log_file.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def save_training_graph(history, global_epoch):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    epoch_numbers = range(1, len(history) + 1)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    metrics = [
        ("loss", "Loss", "train_loss", "validation_loss"),
        ("accuracy", "Accuracy", "train_accuracy", "validation_accuracy"),
        ("error", "Error rate", "train_error", "validation_error"),
    ]
    for axis, (filename, title, train_key, validation_key) in zip(axes, metrics):
        axis.plot(epoch_numbers, [row[train_key] for row in history], label="Train")
        axis.plot(
            epoch_numbers,
            [row[validation_key] for row in history],
            label="Validation",
        )
        if filename == "accuracy":
            axis.axhline(0.95, color="gray", linestyle="--", label="95% target")
        axis.set(title=title, xlabel="Completed epoch")
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle(f"Training progress after epoch {global_epoch}")
    figure.tight_layout()
    figure.savefig(GRAPH_DIR / "training_progress.png", dpi=150)
    figure.savefig(GRAPH_DIR / f"epoch_{global_epoch:02d}.png", dpi=150)
    plt.close(figure)


def train_phase(
    model,
    train_loader,
    validation_loader,
    device,
    epochs,
    learning_rate,
    phase_name,
    freeze_features,
    history,
    best_accuracy=-1.0,
    best_state=None,
):
    criterion = nn.CrossEntropyLoss()
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
            "phase": phase_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_error": 1.0 - train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_error": 1.0 - validation_accuracy,
        }
        history.append(row)
        append_epoch_log(row)
        save_training_graph(history, len(history))
        print(
            f"{phase_name} {epoch}/{epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
            f"train_error={1.0 - train_accuracy:.2%}, "
            f"val_loss={validation_loss:.4f}, val_acc={validation_accuracy:.4f}, "
            f"val_error={1.0 - validation_accuracy:.2%}",
            flush=True,
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_accuracy, best_state


def evaluate_test(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    true_labels = []
    predicted_labels = []
    model.eval()
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Final test evaluation"):
            logits = model(images.to(device, non_blocking=True))
            total_loss += criterion(logits, labels.to(device)).item() * labels.size(0)
            predicted_labels.extend(logits.argmax(dim=1).cpu().tolist())
            true_labels.extend(labels.tolist())
    accuracy = sum(
        truth == prediction for truth, prediction in zip(true_labels, predicted_labels)
    ) / len(true_labels)
    return total_loss / len(true_labels), accuracy, true_labels, predicted_labels


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.head_epochs < 1 or args.finetune_epochs < 0:
        raise ValueError("Batch size and head epochs must be positive; fine-tune epochs cannot be negative.")
    if not (DATA_DIR / "train").is_dir() or not (DATA_DIR / "validation").is_dir() or not (DATA_DIR / "test").is_dir():
        raise FileNotFoundError("Prepared data is missing. Run src/prepare_data.py first.")

    set_seed(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    weights = MobileNet_V2_Weights.DEFAULT
    train_loader, validation_loader, test_loader, class_names = make_loaders(
        args.batch_size, args.workers, weights
    )
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.to(device)

    for parameter in model.features.parameters():
        parameter.requires_grad = False
    history = []
    best_accuracy, best_state = train_phase(
        model, train_loader, validation_loader, device,
        args.head_epochs, 1e-3, "classifier", freeze_features=True,
        history=history,
    )

    if args.finetune_epochs:
        for parameter in model.features[-2:].parameters():
            parameter.requires_grad = True
        best_accuracy, best_state = train_phase(
            model,
            train_loader,
            validation_loader,
            device,
            args.finetune_epochs,
            1e-5,
            "fine_tune",
            freeze_features=False,
            history=history,
            best_accuracy=best_accuracy,
            best_state=best_state,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = MODEL_DIR / "mobilenetv2_tomato.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "class_names": class_names},
        checkpoint_path,
    )
    test_loss, test_accuracy, true_labels, predicted_labels = evaluate_test(
        model, test_loader, device
    )

    test_f1 = f1_score(true_labels, predicted_labels, average="macro")
    metrics = {
        "test_accuracy": test_accuracy,
        "test_error": 1.0 - test_accuracy,
        "test_macro_f1": test_f1,
        "test_loss": test_loss,
        "target_accuracy": 0.95,
        "target_met": test_accuracy >= 0.95,
        "class_names": class_names,
        "test_examples": len(true_labels),
    }
    (OUTPUT_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "classification_report.txt").write_text(
        classification_report(true_labels, predicted_labels, target_names=class_names),
        encoding="utf-8",
    )

    matrix = confusion_matrix(true_labels, predicted_labels)
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, cmap="Greens")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=[name.replace("Tomato___", "") for name in class_names],
        yticklabels=[name.replace("Tomato___", "") for name in class_names],
        xlabel="Predicted label",
        ylabel="True label",
        title=f"MobileNetV2 test confusion matrix (accuracy {test_accuracy:.3f})",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    print(f"Test accuracy: {test_accuracy:.4f} (target >= 0.9500)")
    print(f"Test error: {1.0 - test_accuracy:.2%}")
    print(f"Test macro F1: {test_f1:.4f}")
    print(f"Saved model: {checkpoint_path}")
    print(f"Saved evaluation artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()