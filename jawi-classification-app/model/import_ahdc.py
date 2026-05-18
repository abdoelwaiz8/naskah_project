"""
convert_ahcd.py
---------------
Konversi AHCD1 dari format CSV ke folder per kelas.
Path sudah disesuaikan dengan lokasi file kamu.
"""

import csv
import numpy as np
from pathlib import Path
from PIL import Image

# ── Konfigurasi ────────────────────────────────────────────────────────
AHCD_DIR = Path(r"C:\Users\Waiz\Downloads\arab")

TRAIN_IMAGES_CSV = AHCD_DIR / "csvTrainImages 13440x1024.csv"
TRAIN_LABELS_CSV = AHCD_DIR / "csvTrainLabel 13440x1.csv"
TEST_IMAGES_CSV  = AHCD_DIR / "csvTestImages 3360x1024.csv"
TEST_LABELS_CSV  = AHCD_DIR / "csvTestLabel 3360x1.csv"

# Sesuaikan ini dengan path root project kamu
OUTPUT_DIR = Path(r"D:\Project Python\jawi-classification-app\raw_dataset\arab")

CLASS_NAMES = [
    "alef", "ba", "ta", "tha", "jeem",
    "ha", "kha", "dal", "thal", "ra",
    "zain", "seen", "sheen", "sad", "dad",
    "tah", "zah", "ain", "ghain", "fa",
    "qaf", "kaf", "lam", "meem", "noon",
    "ha2", "waw", "ya"
]
# ───────────────────────────────────────────────────────────────────────


def convert(images_csv: Path, labels_csv: Path, split_name: str):
    print(f"\n📂 Memproses: {split_name}")

    with open(labels_csv, "r") as f:
        labels = [int(row[0]) - 1 for row in csv.reader(f)]

    for name in CLASS_NAMES:
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)

    with open(images_csv, "r") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx % 500 == 0:
                print(f"  {idx}/{len(labels)} gambar diproses...")

            pixels  = np.array([int(p) for p in row], dtype=np.uint8)
            image   = pixels.reshape(32, 32)
            img_pil = Image.fromarray(image, mode="L").convert("RGB")
            img_pil = img_pil.resize((128, 128), Image.LANCZOS)

            class_name = CLASS_NAMES[labels[idx]]
            save_path  = OUTPUT_DIR / class_name / f"{split_name}_{idx:05d}.png"
            img_pil.save(save_path)

    print(f"  ✅ {len(labels)} gambar selesai")


def main():
    print("=" * 50)
    print("  Konversi AHCD1 CSV → Folder per Kelas")
    print("=" * 50)

    for f in [TRAIN_IMAGES_CSV, TRAIN_LABELS_CSV, TEST_IMAGES_CSV, TEST_LABELS_CSV]:
        if not f.exists():
            print(f"❌ File tidak ditemukan: {f}")
            return

    convert(TRAIN_IMAGES_CSV, TRAIN_LABELS_CSV, "train")
    convert(TEST_IMAGES_CSV,  TEST_LABELS_CSV,  "test")

    print("\n🎉 Selesai! Hasil tersimpan di:")
    print(f"   {OUTPUT_DIR}")
    total = 0
    for name in CLASS_NAMES:
        folder = OUTPUT_DIR / name
        if folder.exists():
            count = len(list(folder.glob("*.png")))
            total += count
            print(f"   arab/{name}/  → {count} gambar")
    print(f"\n   Total: {total} gambar")


if __name__ == "__main__":
    main()