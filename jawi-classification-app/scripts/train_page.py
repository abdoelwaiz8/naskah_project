"""
train_page.py
-------------
Training script untuk model klasifikasi halaman naskah penuh
(Arab Jawi vs Arab Asli) menggunakan ResNet34 fine-tuning.

Pipeline:
    1. Muat dataset dari page_dataset/train/ dan page_dataset/val/
    2. Fine-tune ResNet34 pretrained ImageNet
    3. Output: model/page_classifier.pth + model/page_class_names.json

Cara pakai:
    python scripts/train_page.py
    python scripts/train_page.py --epochs 40 --batch_size 16
    python scripts/train_page.py --lr 1e-4 --img_size 512

Rekomendasi dataset:
    - Minimal  : 50  gambar/kelas  (hasil cukup)
    - Disarankan: 150 gambar/kelas (hasil baik)
    - Ideal     : 300+ gambar/kelas (hasil sangat baik)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ---------------------------------------------------------------------------
# Konfigurasi path
# ---------------------------------------------------------------------------
ROOT_DIR         = Path(__file__).resolve().parent.parent
PAGE_DATASET_DIR = ROOT_DIR / "page_dataset"
MODEL_DIR        = ROOT_DIR / "model"
MODEL_PATH       = MODEL_DIR / "page_classifier.pth"
CLASS_NAMES_PATH = MODEL_DIR / "page_class_names.json"


# ---------------------------------------------------------------------------
# Default hyperparameter
# ---------------------------------------------------------------------------
DEFAULT_IMG_SIZE   = 384   # Lebih besar dari char model agar konteks halaman tertangkap
DEFAULT_EPOCHS     = 30
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR         = 3e-4
DEFAULT_NUM_WORKERS= 0     # 0 = aman di Windows


# ---------------------------------------------------------------------------
# Augmentasi data
# ---------------------------------------------------------------------------

def get_transforms(img_size: int) -> dict:
    """
    Mengembalikan dict transform untuk train dan val.

    Train: augmentasi agresif untuk mencegah overfitting pada dataset kecil.
    Val  : hanya resize + crop + normalize.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size + 64, img_size + 64)),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return {"train": train_tf, "val": val_tf}


# ---------------------------------------------------------------------------
# Builder model
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> nn.Module:
    """
    Fine-tune ResNet34:
    - Gunakan bobot ImageNet sebagai titik awal.
    - Ganti layer FC terakhir dengan 2-class output.
    - Bekukan semua layer kecuali layer4 dan FC (transfer learning parsial).
    """
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    # Bekukan semua layer
    for param in model.parameters():
        param.requires_grad = False

    # Buka layer4 dan FC agar bisa dilatih
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Ganti head
    in_features  = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    optimizer : optim.Optimizer,
    device    : torch.device,
) -> tuple[float, float]:
    """Satu epoch training. Kembalikan (loss, accuracy)."""
    model.train()
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    device    : torch.device,
) -> tuple[float, float]:
    """Evaluasi pada dataset validasi. Kembalikan (loss, accuracy)."""
    model.eval()
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs    = model(images)
        loss       = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    epochs     : int   = DEFAULT_EPOCHS,
    batch_size : int   = DEFAULT_BATCH_SIZE,
    lr         : float = DEFAULT_LR,
    img_size   : int   = DEFAULT_IMG_SIZE,
    num_workers: int   = DEFAULT_NUM_WORKERS,
) -> None:

    print("=" * 60)
    print("  Training: Page-Level Classifier (Arab Jawi vs Arab Asli)")
    print("=" * 60)

    # ── Validasi folder dataset ──────────────────────────────────────────
    for split in ("train", "val"):
        split_dir = PAGE_DATASET_DIR / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Folder dataset tidak ditemukan: {split_dir}\n"
                f"Jalankan terlebih dahulu:\n"
                f"  python scripts/split_page_dataset.py"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice  : {device}")
    print(f"Img size: {img_size}x{img_size}")
    print(f"Epochs  : {epochs}")
    print(f"Batch   : {batch_size}")
    print(f"LR      : {lr}")

    # ── Dataset ─────────────────────────────────────────────────────────
    tfs  = get_transforms(img_size)
    data = {
        split: datasets.ImageFolder(
            root      = str(PAGE_DATASET_DIR / split),
            transform = tfs[split],
        )
        for split in ("train", "val")
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = (device.type == "cuda"),
        )
        for split, ds in data.items()
    }

    class_names = data["train"].classes
    num_classes = len(class_names)
    print(f"\nKelas   : {class_names}")
    print(f"Train   : {len(data['train'])} gambar")
    print(f"Val     : {len(data['val'])} gambar")

    if len(data["train"]) < 10:
        print("\n[PERINGATAN] Dataset sangat kecil (<10 gambar).")
        print("Tambah lebih banyak gambar untuk hasil yang baik.")

    # ── Model ────────────────────────────────────────────────────────────
    model     = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Hanya optimalkan parameter yang requires_grad=True
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training ─────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_val_acc  = 0.0
    best_epoch    = 0

    print("\n" + "-" * 60)
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>9}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, loaders["val"], criterion, device
        )
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0
        marker     = " [BEST]" if val_acc > best_val_acc else ""

        print(
            f"{epoch:>6} {train_loss:>11.4f} {train_acc*100:>9.2f}%"
            f" {val_loss:>10.4f} {val_acc*100:>8.2f}%"
            f" {current_lr:>9.2e}"
            f"  ({elapsed:.1f}s){marker}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            # Simpan checkpoint terbaik
            torch.save({
                "epoch"          : epoch,
                "model_state_dict": model.state_dict(),
                "val_acc"        : val_acc,
                "class_names"    : class_names,
                "img_size"       : img_size,
            }, MODEL_PATH)

    # ── Simpan class names ───────────────────────────────────────────────
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    print("-" * 60)
    print(f"\n[SELESAI] Training selesai!")
    print(f"   Best val accuracy : {best_val_acc*100:.2f}% (epoch {best_epoch})")
    print(f"   Model disimpan    : {MODEL_PATH}")
    print(f"   Class names       : {CLASS_NAMES_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training model klasifikasi halaman naskah (Arab Jawi vs Arab Asli)."
    )
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--img_size",    type=int,   default=DEFAULT_IMG_SIZE)
    parser.add_argument("--num_workers", type=int,   default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()

    train(
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        img_size    = args.img_size,
        num_workers = args.num_workers,
    )


if __name__ == "__main__":
    main()
