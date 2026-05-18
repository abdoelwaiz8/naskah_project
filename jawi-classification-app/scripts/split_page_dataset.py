"""
split_page_dataset.py
---------------------
Membagi gambar halaman naskah mentah menjadi set train dan val.

Struktur input (raw_page_dataset/):
    raw_page_dataset/
    ├── arab_asli/          ← gambar halaman Al-Quran, kitab Arab
    │   ├── quran_001.jpg
    │   └── ...
    └── arab_jawi/          ← gambar halaman naskah Jawi Melayu
        ├── jawi_001.jpg
        └── ...

Struktur output (page_dataset/):
    page_dataset/
    ├── train/
    │   ├── arab_asli/
    │   └── arab_jawi/
    └── val/
        ├── arab_asli/
        └── arab_jawi/

Cara pakai:
    python scripts/split_page_dataset.py
    python scripts/split_page_dataset.py --val_ratio 0.25
    python scripts/split_page_dataset.py --seed 99
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Konfigurasi path
# ---------------------------------------------------------------------------
ROOT_DIR        = Path(__file__).resolve().parent.parent
RAW_PAGE_DIR    = ROOT_DIR / "raw_page_dataset"
PAGE_DATASET_DIR= ROOT_DIR / "page_dataset"

CLASSES         = ["arab_asli", "arab_jawi"]
IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def split_dataset(val_ratio: float = 0.20, seed: int = 42) -> None:
    """
    Membagi dataset mentah menjadi train/val.

    Args:
        val_ratio : Proporsi data validasi (default 0.20 = 20%).
        seed      : Random seed untuk reproducibility.
    """
    random.seed(seed)

    print("=" * 55)
    print("  Split Page Dataset")
    print("=" * 55)

    if not RAW_PAGE_DIR.exists():
        print(f"\n[ERROR] Folder tidak ditemukan: {RAW_PAGE_DIR}")
        print("Buat folder ini dan isi dengan gambar halaman naskah.")
        return

    for cls in CLASSES:
        src_dir = RAW_PAGE_DIR / cls
        if not src_dir.exists():
            print(f"\n[WARN] Folder kelas '{cls}' tidak ditemukan: {src_dir}")
            continue

        # Kumpulkan semua file gambar
        images = sorted([
            f for f in src_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ])

        if not images:
            print(f"\n[WARN] Tidak ada gambar di: {src_dir}")
            continue

        # Acak dan bagi
        random.shuffle(images)
        n_val   = max(1, int(len(images) * val_ratio))
        n_train = len(images) - n_val

        splits = {
            "train": images[:n_train],
            "val"  : images[n_train:],
        }

        print(f"\nKelas: {cls}")
        print(f"  Total : {len(images)} gambar")
        print(f"  Train : {n_train} gambar")
        print(f"  Val   : {n_val} gambar")

        for split_name, split_files in splits.items():
            dest_dir = PAGE_DATASET_DIR / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_files:
                dest = dest_dir / img_path.name
                shutil.copy2(img_path, dest)

        print(f"  [OK] Disalin ke {PAGE_DATASET_DIR / 'train' / cls}")
        print(f"  [OK] Disalin ke {PAGE_DATASET_DIR / 'val' / cls}")

    print("\n" + "=" * 55)
    print("  [SELESAI] Split selesai!")
    print(f"  Output: {PAGE_DATASET_DIR}")
    print("=" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Membagi dataset halaman naskah menjadi train/val."
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.20,
        help="Proporsi data validasi (default: 0.20)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()
    split_dataset(val_ratio=args.val_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
