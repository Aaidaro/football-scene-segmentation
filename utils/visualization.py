"""Plotting helpers for data analysis, training, and qualitative comparisons."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.data_loader import UNIFIED_CLASSES, UNIFIED_COLORS


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in UNIFIED_COLORS.items():
        colored[mask == class_idx] = color
    return colored


def denormalize_image(image_tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean_np = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_np = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    image = np.clip(image * std_np + mean_np, 0.0, 1.0)
    return image


def plot_class_distribution(
    train_counts: np.ndarray,
    val_counts: np.ndarray,
    output_path: str | Path,
    csv_path: str | Path | None = None,
) -> None:
    output_path = ensure_parent(output_path)

    x = np.arange(len(UNIFIED_CLASSES))
    width = 0.38

    train_pct = train_counts / max(train_counts.sum(), 1) * 100.0
    val_pct = val_counts / max(val_counts.sum(), 1) * 100.0

    fig, ax = plt.subplots(figsize=(12, 5))

    train_bars = ax.bar(
        x - width / 2,
        train_pct,
        width,
        label="Train",
    )

    val_bars = ax.bar(
        x + width / 2,
        val_pct,
        width,
        label="Validation",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(UNIFIED_CLASSES, rotation=35, ha="right")

    ax.set_ylabel("Pixel percentage (%)")
    ax.set_title("Class distribution after train/validation split")

    # Normal / linear scale
    ax.set_yscale("linear")

    ax.legend()

    # Show percentage on each bar
    ax.bar_label(
        train_bars,
        labels=[f"{pct:.2f}%\n({int(cnt):,})" for pct, cnt in zip(train_pct, train_counts)],
        padding=3,
        fontsize=8,
        rotation=90,
    )

    ax.bar_label(
        val_bars,
        labels=[f"{pct:.2f}%\n({int(cnt):,})" for pct, cnt in zip(val_pct, val_counts)],
        padding=3,
        fontsize=8,
        rotation=90,
    )

    # Add space above bars for labels
    max_pct = max(train_pct.max(), val_pct.max())
    ax.set_ylim(0, max_pct * 1.20 if max_pct > 0 else 1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    if csv_path is not None:
        csv_path = ensure_parent(csv_path)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "train_pixels", "val_pixels", "train_percent", "val_percent"])
            for i, name in enumerate(UNIFIED_CLASSES):
                writer.writerow([name, int(train_counts[i]), int(val_counts[i]), float(train_pct[i]), float(val_pct[i])])


def plot_loss_curves(history: Dict[str, List[float]], output_path: str | Path) -> None:
    output_path = ensure_parent(output_path)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_miou_curve(history: Dict[str, List[float]], output_path: str | Path) -> None:
    output_path = ensure_parent(output_path)
    epochs = np.arange(1, len(history["val_miou"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_miou"], label="Validation mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("Validation mIoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def save_prediction_grid(
    batch: dict,
    prediction_dict: Dict[str, torch.Tensor],
    output_path: str | Path,
    mean: Sequence[float],
    std: Sequence[float],
    max_items: int = 5,
    figure_title: str = "Qualitative Segmentation Comparison",
) -> None:
    """Save predictions with columns as samples and rows as Image | GT | model predictions."""
    output_path = ensure_parent(output_path)

    images = batch["image"]
    masks = batch["mask"]

    n = min(max_items, images.shape[0])

    row_titles = ["Image", "Ground truth"] + list(prediction_dict.keys())
    nrows = len(row_titles)
    ncols = n

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
    )

    fig.suptitle(figure_title, fontsize=18, fontweight="bold")

    for col in range(n):
        sample_title = f"Sample {col + 1}"

        # Row 0: original image
        axes[0, col].imshow(denormalize_image(images[col], mean, std))
        axes[0, col].set_title(f"{sample_title}\nImage", fontsize=11)
        axes[0, col].axis("off")

        # Row 1: ground-truth mask
        axes[1, col].imshow(colorize_mask(masks[col].detach().cpu().numpy()))
        axes[1, col].set_title(f"{sample_title}\nGround truth", fontsize=11)
        axes[1, col].axis("off")

        # Rows 2+: predictions from each model
        for row, (name, preds) in enumerate(prediction_dict.items(), start=2):
            axes[row, col].imshow(colorize_mask(preds[col].detach().cpu().numpy()))
            axes[row, col].set_title(f"{sample_title}\n{name}", fontsize=11)
            axes[row, col].axis("off")

    # Optional row labels on the left
    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(
            title,
            fontsize=12,
            fontweight="bold",
            rotation=90,
            labelpad=25,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=200)
    plt.close(fig)