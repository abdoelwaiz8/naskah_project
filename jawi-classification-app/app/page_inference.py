"""
page_inference.py
-----------------
Modul inferensi untuk model klasifikasi halaman naskah penuh
(Arab Jawi vs Arab Asli) — model page-level (bukan per-karakter).

Model dimuat dari: model/page_classifier.pth
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
ROOT_DIR         = Path(__file__).resolve().parent.parent
MODEL_PATH       = ROOT_DIR / "model" / "page_classifier.pth"
CLASS_NAMES_PATH = ROOT_DIR / "model" / "page_class_names.json"

DEFAULT_IMG_SIZE = 384
DEFAULT_CLASSES  = ["arab_asli", "arab_jawi"]


# ---------------------------------------------------------------------------
# Singleton model
# ---------------------------------------------------------------------------
_page_model  : nn.Module | None    = None
_page_device : torch.device | None = None
_page_classes: list[str]           = DEFAULT_CLASSES
_page_img_size: int                = DEFAULT_IMG_SIZE


def is_page_model_available() -> bool:
    """Cek apakah file model page-level tersedia."""
    return MODEL_PATH.exists()


def _build_page_model(num_classes: int) -> nn.Module:
    """Bangun arsitektur ResNet34 dengan head yang sama seperti saat training."""
    model = models.resnet34(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_page_model(force_reload: bool = False) -> tuple[nn.Module, torch.device, list[str], int]:
    """
    Muat model page-level dari file .pth (singleton).

    Returns:
        Tuple (model, device, class_names, img_size)
    """
    global _page_model, _page_device, _page_classes, _page_img_size

    if _page_model is not None and not force_reload:
        return _page_model, _page_device, _page_classes, _page_img_size

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model page-level tidak ditemukan: {MODEL_PATH}\n"
            f"Latih terlebih dahulu dengan:\n"
            f"  python scripts/train_page.py"
        )

    _page_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=_page_device, weights_only=False)

    # Baca class names & img_size dari checkpoint atau fallback
    if isinstance(checkpoint, dict):
        _page_classes  = checkpoint.get("class_names", DEFAULT_CLASSES)
        _page_img_size = checkpoint.get("img_size",    DEFAULT_IMG_SIZE)
        state_dict     = checkpoint.get("model_state_dict", checkpoint)
    else:
        state_dict = checkpoint

    _page_model = _build_page_model(len(_page_classes))
    _page_model.load_state_dict(state_dict, strict=False)
    _page_model.to(_page_device)
    _page_model.eval()

    return _page_model, _page_device, _page_classes, _page_img_size


def predict_page(image: Image.Image) -> dict:
    """
    Klasifikasi halaman naskah penuh menggunakan model page-level.

    Args:
        image : PIL Image halaman naskah.

    Returns:
        {
            "script_type" : "Arab Jawi" | "Arab Asli",
            "confidence"  : float,
            "class_name"  : str,       # "arab_jawi" | "arab_asli"
            "probabilities": {class: float, ...}
        }
    """
    model, device, class_names, img_size = load_page_model()

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    tensor = tf(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    pred_idx   = int(torch.argmax(torch.tensor(probs)))
    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx]

    # Label ramah pengguna
    label_map = {
        "arab_asli": "Arab Asli",
        "arab_jawi": "Arab Jawi",
    }

    return {
        "script_type"   : label_map.get(pred_class, pred_class.replace("_", " ").title()),
        "confidence"    : confidence,
        "class_name"    : pred_class,
        "probabilities" : {cls: prob for cls, prob in zip(class_names, probs)},
    }
