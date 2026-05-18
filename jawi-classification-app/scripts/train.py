"""
train.py
--------
Melatih model ResNet34 pada dataset Jawi + Arab gabungan.
Dataset harus sudah di-split terlebih dahulu menggunakan split_dataset.py.

Cara menjalankan (dari root project):
    python scripts/train.py

Output:
    model/resnet34_jawi.pth   ← bobot model terbaik (berdasarkan val accuracy)
    model/class_names.json    ← diperbarui agar sinkron dengan ImageFolder
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ── Konfigurasi ─────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent.parent
DATASET_DIR      = BASE_DIR / "dataset"
MODEL_OUT        = BASE_DIR / "model" / "resnet34_jawi.pth"
CLASS_NAMES_FILE = BASE_DIR / "model" / "class_names.json"

BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 1e-3
NUM_WORKERS  = 2        # Turunkan ke 0 jika ada error DataLoader di Windows
IMAGE_SIZE   = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
# ────────────────────────────────────────────────────────────────────────────


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Mengembalikan transformasi augmentasi (train) dan evaluasi (val)."""
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_model(num_classes: int) -> nn.Module:
    """ResNet34 dengan pretrained ImageNet weights, head diganti sesuai kelas."""
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = running_correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss    += loss.item() * images.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return running_loss / n, running_correct / n


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = running_correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss    += loss.item() * images.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return running_loss / n, running_correct / n


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] : {device}")
    if device.type == "cuda":
        print(f"   GPU   : {torch.cuda.get_device_name(0)}")

    # ── Cek folder dataset ──────────────────────────────────────────
    for split in ("train", "val"):
        p = DATASET_DIR / split
        if not p.exists() or not any(p.iterdir()):
            print(
                f"\n❌ Folder '{p}' kosong atau tidak ada.\n"
                "   Jalankan dulu: python scripts/split_dataset.py"
            )
            return

    # ── Dataset & DataLoader ────────────────────────────────────────
    train_tf, val_tf = get_transforms()
    train_ds = datasets.ImageFolder(DATASET_DIR / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(DATASET_DIR / "val",   transform=val_tf)

    num_classes = len(train_ds.classes)
    print(f"[Info] Kelas  : {num_classes} total")
    print(f"   Arab   : {sum(1 for c in train_ds.classes if c.startswith('arab_'))}")
    print(f"   Jawi   : {sum(1 for c in train_ds.classes if c.startswith('jawi_'))}")
    print(f"   Train  : {len(train_ds):,} gambar")
    print(f"   Val    : {len(val_ds):,} gambar")

    # Simpan / perbarui class_names.json agar sinkron dengan ImageFolder
    CLASS_NAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as f:
        json.dump(train_ds.classes, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] class_names.json diperbarui: {CLASS_NAMES_FILE}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    # ── Model, Loss, Optimizer ──────────────────────────────────────
    model     = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ───────────────────────────────────────────────
    best_val_acc = 0.0
    print("\n>> Mulai training...\n")
    print(f"{'Epoch':>6}  {'TrainLoss':>9}  {'TrainAcc':>8}  {'ValLoss':>8}  {'ValAcc':>7}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        is_best = val_acc > best_val_acc
        marker  = " *" if is_best else ""
        print(
            f"{epoch:6d}  {train_loss:9.4f}  {train_acc*100:7.2f}%  "
            f"{val_loss:8.4f}  {val_acc*100:6.2f}%  {elapsed:5.1f}s{marker}"
        )

        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)

    # ── Ringkasan ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  [OK] Training selesai!")
    print(f"  Val accuracy terbaik : {best_val_acc * 100:.2f}%")
    print(f"  Model disimpan di    : {MODEL_OUT}")
    print(f"{'=' * 60}")
    print("\nLangkah berikutnya:")
    print("  streamlit run app/main.py")


if __name__ == "__main__":
    main()
