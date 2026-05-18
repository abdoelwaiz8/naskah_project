"""
split_dataset.py
----------------
Menggabungkan dataset Arab (28 kelas) dan Jawi (22 kelas) dari raw_dataset/,
memberi prefix agar mudah dibedakan, lalu membagi menjadi train/val (80:20).

Cara menjalankan (dari root project):
    python scripts/split_dataset.py

Output:
    dataset/train/<arab_atau_jawi_kelas>/...
    dataset/val/<arab_atau_jawi_kelas>/...
    model/class_names.json  ← daftar kelas yang digunakan oleh inference.py
"""

import json
import random
import re
import shutil
from pathlib import Path

# ── Konfigurasi ─────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
TRAIN_RATIO  = 0.80          # 80% train, 20% val

BASE_DIR     = Path(__file__).resolve().parent.parent
RAW_ARAB     = BASE_DIR / "raw_dataset" / "arab"
RAW_JAWI     = BASE_DIR / "raw_dataset" / "jawi"
DATASET_DIR  = BASE_DIR / "dataset"
TRAIN_DIR    = DATASET_DIR / "train"
VAL_DIR      = DATASET_DIR / "val"
CLASS_NAMES_FILE = BASE_DIR / "model" / "class_names.json"

IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# ────────────────────────────────────────────────────────────────────────────


def jawi_folder_to_class(folder_name: str) -> str:
    """
    Konversi nama folder Jawi ke nama kelas dengan prefix 'jawi_'.
    Contoh: '1. Nya_Isolated' → 'jawi_nya_isolated'
             '10. Nga_Start'  → 'jawi_nga_start'
    """
    name = re.sub(r"^\d+\.\s*", "", folder_name)   # Hapus awalan "N. "
    name = name.strip().lower().replace(" ", "_")
    return f"jawi_{name}"


def collect_classes() -> dict[str, Path]:
    """
    Kumpulkan semua kelas beserta path folder sumbernya.
    Returns: {class_name: source_folder_path}
    """
    classes: dict[str, Path] = {}

    # ── Dataset Arab ──────────────────────────────────────
    if not RAW_ARAB.exists():
        print(f"  ⚠️  Folder Arab tidak ditemukan: {RAW_ARAB}")
    else:
        for folder in sorted(RAW_ARAB.iterdir()):
            if folder.is_dir():
                class_name = f"arab_{folder.name.lower()}"
                classes[class_name] = folder

    # ── Dataset Jawi ──────────────────────────────────────
    if not RAW_JAWI.exists():
        print(f"  ⚠️  Folder Jawi tidak ditemukan: {RAW_JAWI}")
    else:
        for folder in sorted(RAW_JAWI.iterdir()):
            if folder.is_dir():
                class_name = jawi_folder_to_class(folder.name)
                classes[class_name] = folder

    return classes


def split_and_copy(classes: dict[str, Path]) -> tuple[int, int]:
    """
    Acak, split, dan salin gambar ke dataset/train/ dan dataset/val/.
    Returns: (total_train, total_val)
    """
    random.seed(RANDOM_SEED)
    total_train = total_val = 0

    for class_name, src_folder in classes.items():
        images = [
            f for f in src_folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ]

        if not images:
            print(f"  ⚠️  {class_name:<35} → tidak ada gambar, dilewati.")
            continue

        random.shuffle(images)
        split_idx   = max(1, int(len(images) * TRAIN_RATIO))
        train_imgs  = images[:split_idx]
        val_imgs    = images[split_idx:]

        # Buat direktori tujuan
        (TRAIN_DIR / class_name).mkdir(parents=True, exist_ok=True)
        (VAL_DIR   / class_name).mkdir(parents=True, exist_ok=True)

        # Salin file
        for img in train_imgs:
            shutil.copy2(img, TRAIN_DIR / class_name / img.name)
        for img in val_imgs:
            shutil.copy2(img, VAL_DIR / class_name / img.name)

        print(
            f"  ✅ {class_name:<35} "
            f"train={len(train_imgs):4d}  val={len(val_imgs):4d}"
        )
        total_train += len(train_imgs)
        total_val   += len(val_imgs)

    return total_train, total_val


def main() -> None:
    print("=" * 65)
    print("  Split Dataset: Arab + Jawi  ->  train / val")
    print("=" * 65)

    # 1. Kumpulkan kelas
    print("\n📂 Mengumpulkan kelas dari raw_dataset/...")
    classes = collect_classes()

    if not classes:
        print("❌ Tidak ada kelas yang ditemukan. Periksa folder raw_dataset/.")
        return

    # Urutkan secara alfabetis (harus sama dengan urutan ImageFolder PyTorch)
    sorted_classes = {k: classes[k] for k in sorted(classes.keys())}

    n_arab = sum(1 for k in sorted_classes if k.startswith("arab_"))
    n_jawi = sum(1 for k in sorted_classes if k.startswith("jawi_"))
    print(f"   Arab : {n_arab} kelas")
    print(f"   Jawi : {n_jawi} kelas")
    print(f"   Total: {len(sorted_classes)} kelas")

    # 2. Simpan class_names.json (dibaca oleh inference.py)
    CLASS_NAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as f:
        json.dump(list(sorted_classes.keys()), f, ensure_ascii=False, indent=2)
    print(f"\n💾 Daftar kelas disimpan ke: {CLASS_NAMES_FILE}")

    # 3. Split & salin
    print("\n🔀 Membagi dan menyalin gambar...\n")
    total_train, total_val = split_and_copy(sorted_classes)

    print(f"\n{'=' * 65}")
    print(f"  [OK] Selesai!")
    print(f"  Total train : {total_train:,} gambar")
    print(f"  Total val   : {total_val:,} gambar")
    print(f"  Output      : {DATASET_DIR}")
    print(f"{'=' * 65}")
    print("\nLangkah berikutnya:")
    print("  python scripts/train.py")


if __name__ == "__main__":
    main()
