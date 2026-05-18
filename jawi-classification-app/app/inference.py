"""
inference.py
------------
Modul untuk memuat model ResNet34 terlatih dan melakukan inferensi
klasifikasi karakter Jawi/Arab dari gambar.

Nama kelas dibaca dari model/class_names.json (dihasilkan oleh
scripts/split_dataset.py atau scripts/train.py).
Jika file JSON belum ada, fallback ke daftar kelas default.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image

from utils import get_inference_transform


# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------

MODEL_PATH       = Path(__file__).resolve().parent.parent / "model" / "resnet34_jawi.pth"
CLASS_NAMES_FILE = Path(__file__).resolve().parent.parent / "model" / "class_names.json"

# ---------------------------------------------------------------------------
# Nama kelas: baca dari class_names.json jika ada,
# fallback ke daftar 22 kelas Arab default.
# class_names.json dibuat otomatis oleh scripts/split_dataset.py
# dan diperbarui oleh scripts/train.py.
# ---------------------------------------------------------------------------
_FALLBACK_CLASS_NAMES: list[str] = [
    "arab_alef", "arab_ba",   "arab_ta",    "arab_tha",   "arab_jeem",
    "arab_ha",   "arab_kha",  "arab_dal",   "arab_thal",  "arab_ra",
    "arab_zain", "arab_seen", "arab_sheen", "arab_sad",   "arab_dad",
    "arab_tah",  "arab_zah",  "arab_ain",   "arab_ghain", "arab_fa",
    "arab_qaf",  "arab_kaf",
]

def _load_class_names() -> list[str]:
    """Muat daftar kelas dari JSON; fallback ke daftar hardcoded."""
    if CLASS_NAMES_FILE.exists():
        with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return _FALLBACK_CLASS_NAMES

CLASS_NAMES: list[str] = _load_class_names()
NUM_CLASSES: int        = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Singleton – model dimuat sekali saja ke memori
# ---------------------------------------------------------------------------
_model: nn.Module | None = None
_device: torch.device | None = None


def _build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Membangun arsitektur ResNet34 dan memodifikasi lapisan FC
    agar sesuai dengan jumlah kelas dataset Jawi.

    Args:
        num_classes : Jumlah kelas output.

    Returns:
        nn.Module ResNet34 yang sudah dimodifikasi.
    """
    model = models.resnet34(weights=None)          # Jangan muat bobot ImageNet
    in_features = model.fc.in_features             # 512 pada ResNet34
    model.fc = nn.Linear(in_features, num_classes) # Ganti head
    return model


def load_model(
    model_path: Path | str = MODEL_PATH,
    num_classes: int = NUM_CLASSES,
    force_reload: bool = False,
) -> tuple[nn.Module, torch.device]:
    """
    Memuat model dari file `.pth` ke memori (singleton).
    Jika model sudah dimuat sebelumnya, fungsi ini akan mengembalikan
    instance yang sudah ada kecuali `force_reload=True`.

    Args:
        model_path   : Path ke file bobot `.pth`.
        num_classes  : Jumlah kelas output.
        force_reload : Paksa muat ulang model.

    Returns:
        Tuple (model, device)

    Raises:
        FileNotFoundError : Jika file `.pth` tidak ditemukan.
    """
    global _model, _device

    if _model is not None and not force_reload:
        return _model, _device

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"File bobot model tidak ditemukan di: {model_path}\n"
            "Pastikan Anda meletakkan file `resnet34_jawi.pth` di folder /model/"
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _build_model(num_classes)

    # Muat state dict; tangani file yang disimpan sebagai dict penuh atau state_dict langsung
    checkpoint = torch.load(model_path, map_location=_device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint  # Asumsikan langsung state_dict

    model.load_state_dict(state, strict=True)
    model.to(_device)
    model.eval()

    _model = model
    return _model, _device


def predict(
    image: Image.Image,
    model_path: Path | str = MODEL_PATH,
    num_classes: int = NUM_CLASSES,
    top_k: int = 3,
) -> dict:
    """
    Melakukan inferensi klasifikasi pada satu gambar PIL.

    Args:
        image       : Objek PIL.Image (mode RGB).
        model_path  : Path ke file bobot model.
        num_classes : Jumlah kelas output.
        top_k       : Jumlah prediksi teratas yang dikembalikan.

    Returns:
        Dict dengan struktur:
        {
            "predicted_class"  : str,    # Kelas dengan probabilitas tertinggi
            "confidence"       : float,  # Probabilitas kelas teratas (0.0–1.0)
            "top_k_predictions": [
                {"class": str, "confidence": float},
                ...
            ]
        }
    """
    model, device = load_model(model_path, num_classes)
    transform = get_inference_transform()

    # Preprocessing
    tensor = transform(image).unsqueeze(0).to(device)   # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)                           # (1, num_classes)
        probabilities = F.softmax(logits, dim=1)         # (1, num_classes)

    # Ambil top-k prediksi
    top_probs, top_indices = torch.topk(probabilities, k=min(top_k, num_classes), dim=1)

    top_probs   = top_probs.squeeze(0).cpu().tolist()
    top_indices = top_indices.squeeze(0).cpu().tolist()

    top_k_preds = [
        {"class": CLASS_NAMES[idx], "confidence": prob}
        for idx, prob in zip(top_indices, top_probs)
    ]

    return {
        "predicted_class"  : top_k_preds[0]["class"],
        "confidence"       : top_k_preds[0]["confidence"],
        "top_k_predictions": top_k_preds,
    }


def is_model_available(model_path: Path | str = MODEL_PATH) -> bool:
    """
    Mengecek apakah file bobot model tersedia.

    Returns:
        True jika file ada, False jika tidak.
    """
    return Path(model_path).exists()
