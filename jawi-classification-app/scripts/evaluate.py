"""
evaluate.py
# -*- coding: utf-8 -*-
-----------
Mengevaluasi performa model ResNet34 pada dataset validasi.

Menghasilkan:
  - Overall Accuracy & Loss
  - Per-class Precision, Recall, F1-Score
  - Top-5 kelas dengan performa terburuk
  - Confusion matrix (disimpan ke model/confusion_matrix.png)

Cara menjalankan (dari root project, pakai venv312):
    python scripts/evaluate.py
"""

import io
import json
import sys

# Force UTF-8 output agar karakter Unicode tidak error di Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ── Konfigurasi ─────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent.parent
DATASET_DIR      = BASE_DIR / "dataset"
MODEL_PATH       = BASE_DIR / "model" / "resnet34_jawi.pth"
CLASS_NAMES_FILE = BASE_DIR / "model" / "class_names.json"
OUTPUT_DIR       = BASE_DIR / "model"

BATCH_SIZE  = 32
NUM_WORKERS = 2
IMAGE_SIZE  = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
# ────────────────────────────────────────────────────────────────────────────


def check_requirements():
    """Cek keberadaan file & folder yang dibutuhkan."""
    errors = []
    if not MODEL_PATH.exists():
        errors.append(f"❌ Model tidak ditemukan: {MODEL_PATH}")
    if not CLASS_NAMES_FILE.exists():
        errors.append(f"❌ class_names.json tidak ditemukan: {CLASS_NAMES_FILE}")
    val_dir = DATASET_DIR / "val"
    if not val_dir.exists() or not any(val_dir.iterdir()):
        errors.append(f"❌ Folder validasi kosong/tidak ada: {val_dir}")
    if errors:
        print("\n".join(errors))
        sys.exit(1)


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_class_names() -> list[str]:
    with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def run_evaluation(model, loader, criterion, device, class_names):
    """Jalankan inferensi pada seluruh val set, kumpulkan prediksi & label."""
    model.eval()

    all_preds   = []
    all_labels  = []
    all_probs   = []
    running_loss = 0.0
    correct = 0

    print("  Mengevaluasi...", end="", flush=True)
    t0 = time.time()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        probs  = F.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        correct      += (preds == labels).sum().item()

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    elapsed = time.time() - t0
    n = len(loader.dataset)
    avg_loss = running_loss / n
    accuracy = correct / n

    print(f" selesai dalam {elapsed:.1f}s")
    return all_preds, all_labels, all_probs, avg_loss, accuracy


def compute_per_class_metrics(all_preds, all_labels, class_names):
    """Hitung TP, FP, FN per kelas → Precision, Recall, F1."""
    nc = len(class_names)
    tp = [0] * nc
    fp = [0] * nc
    fn = [0] * nc

    for pred, label in zip(all_preds, all_labels):
        if pred == label:
            tp[label] += 1
        else:
            fp[pred]   += 1
            fn[label]  += 1

    metrics = []
    for i, name in enumerate(class_names):
        prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        rec  = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        support = tp[i] + fn[i]
        metrics.append({
            "class": name, "precision": prec,
            "recall": rec, "f1": f1, "support": support,
            "correct": tp[i],
        })
    return metrics


def print_report(metrics, overall_acc, avg_loss, num_classes):
    """Cetak laporan performa ke terminal."""
    print("\n" + "=" * 72)
    print("  LAPORAN EVALUASI MODEL — ResNet34 Jawi/Arab")
    print("=" * 72)
    print(f"  Overall Accuracy : {overall_acc * 100:.2f}%")
    print(f"  Val Loss         : {avg_loss:.4f}")
    print(f"  Jumlah Kelas     : {num_classes}")
    print()

    # Header tabel
    header = f"  {'Kelas':<30} {'Precision':>9} {'Recall':>7} {'F1':>7} {'Support':>8} {'Benar':>7}"
    print(header)
    print("  " + "-" * 68)

    for m in sorted(metrics, key=lambda x: x["class"]):
        row = (
            f"  {m['class']:<30} "
            f"{m['precision']*100:>8.1f}% "
            f"{m['recall']*100:>6.1f}% "
            f"{m['f1']*100:>6.1f}% "
            f"{m['support']:>8} "
            f"{m['correct']:>7}"
        )
        print(row)

    print()
    # Macro averages
    macro_p  = sum(m["precision"] for m in metrics) / len(metrics)
    macro_r  = sum(m["recall"]    for m in metrics) / len(metrics)
    macro_f1 = sum(m["f1"]        for m in metrics) / len(metrics)
    total_support = sum(m["support"] for m in metrics)
    total_correct = sum(m["correct"] for m in metrics)

    print(f"  {'MACRO AVG':<30} {macro_p*100:>8.1f}% {macro_r*100:>6.1f}% {macro_f1*100:>6.1f}% {total_support:>8} {total_correct:>7}")
    print("=" * 72)


def print_worst_classes(metrics, top_n=5):
    """Tampilkan kelas dengan F1 terendah."""
    worst = sorted(metrics, key=lambda x: x["f1"])[:top_n]
    print(f"\n  ⚠️  Top-{top_n} Kelas dengan Performa Terendah (F1):")
    print(f"  {'Kelas':<30} {'F1':>7} {'Support':>8}")
    print("  " + "-" * 48)
    for m in worst:
        flag = " ← perlu lebih banyak data" if m["support"] < 20 else ""
        print(f"  {m['class']:<30} {m['f1']*100:>6.1f}%  {m['support']:>7}{flag}")


def try_save_confusion_matrix(all_labels, all_preds, class_names):
    """Simpan confusion matrix sebagai gambar (opsional, butuh matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        nc = len(class_names)
        cm = np.zeros((nc, nc), dtype=int)
        for t, p in zip(all_labels, all_preds):
            cm[t][p] += 1

        # Normalisasi per baris (recall view)
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm /= row_sums

        fig_size = max(12, nc // 2)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        short_names = [c.replace("arab_", "a:").replace("jawi_", "j:") for c in class_names]
        tick_marks = range(nc)
        ax.set_xticks(list(tick_marks))
        ax.set_yticks(list(tick_marks))
        ax.set_xticklabels(short_names, rotation=90, fontsize=7)
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_xlabel("Prediksi", fontsize=10)
        ax.set_ylabel("Label Sebenarnya", fontsize=10)
        ax.set_title("Confusion Matrix (Normalized per Baris)", fontsize=12)

        out_path = OUTPUT_DIR / "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"\n  [OK] Confusion matrix disimpan: {out_path}")
    except ImportError:
        print("\n  [INFO] matplotlib tidak tersedia -- confusion matrix dilewati.")
    except Exception as e:
        print(f"\n  [WARN] Gagal menyimpan confusion matrix: {e}")


def main():
    print("\n" + "=" * 46)
    print("    Evaluasi Model ResNet34 Jawi/Arab")
    print("=" * 46 + "\n")

    check_requirements()

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [Device] : {device}")
    if device.type == "cuda":
        print(f"  [GPU]    : {torch.cuda.get_device_name(0)}")

    # ── Class names & Dataset ────────────────────────────────────────────────
    class_names = load_class_names()
    num_classes = len(class_names)
    print(f"  [Kelas]  : {num_classes} total  "
          f"({sum(1 for c in class_names if c.startswith('arab_'))} Arab, "
          f"{sum(1 for c in class_names if c.startswith('jawi_'))} Jawi)\n")

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_ds = datasets.ImageFolder(DATASET_DIR / "val", transform=val_tf)
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    print(f"  [Val Set]: {len(val_ds):,} gambar dari {DATASET_DIR / 'val'}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"  [Model]  : Memuat {MODEL_PATH.name} ...", end="", flush=True)
    model = build_model(num_classes)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    print(" OK")

    criterion = nn.CrossEntropyLoss()

    # ── Evaluasi ─────────────────────────────────────────────────────────────
    all_preds, all_labels, all_probs, avg_loss, overall_acc = run_evaluation(
        model, val_loader, criterion, device, class_names
    )

    # ── Laporan ──────────────────────────────────────────────────────────────
    metrics = compute_per_class_metrics(all_preds, all_labels, class_names)
    print_report(metrics, overall_acc, avg_loss, num_classes)
    print_worst_classes(metrics, top_n=5)

    # ── Confusion matrix ─────────────────────────────────────────────────────
    try_save_confusion_matrix(all_labels, all_preds, class_names)

    print(f"\n  [DONE] Evaluasi selesai!\n")


if __name__ == "__main__":
    main()
