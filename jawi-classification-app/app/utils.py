"""
utils.py
--------
Fungsi-fungsi pembantu (helper) untuk preprocessing gambar
dan utilitas umum lainnya.
"""

from __future__ import annotations

import io
import base64
from typing import Tuple

from PIL import Image
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Konstanta ImageNet (digunakan juga di inference.py agar sinkron)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

RESIZE_SIZE   = 256
CROP_SIZE     = 224


# ---------------------------------------------------------------------------
# Pemetaan nama kelas → karakter Arab/Jawi asli
# Huruf khusus Jawi: ڤ (pa), چ (ca/cha), ڠ (nga), ڽ (nya), ڬ (ga), ۋ (va)
# ---------------------------------------------------------------------------
CLASS_TO_ARABIC: dict[str, str] = {
    # Huruf Arab standar
    "arab_alef"  : "\u0627",   # ا Alef
    "arab_ba"    : "\u0628",   # ب Ba
    "arab_ta"    : "\u062a",   # ت Ta
    "arab_tha"   : "\u062b",   # ث Tha
    "arab_jeem"  : "\u062c",   # ج Jeem
    "arab_ha"    : "\u062d",   # ح Ha (kecil)
    "arab_ha2"   : "\u0647",   # ه Ha (besar)
    "arab_kha"   : "\u062e",   # خ Kha
    "arab_dal"   : "\u062f",   # د Dal
    "arab_thal"  : "\u0630",   # ذ Thal
    "arab_ra"    : "\u0631",   # ر Ra
    "arab_zain"  : "\u0632",   # ز Zain
    "arab_seen"  : "\u0633",   # س Seen
    "arab_sheen" : "\u0634",   # ش Sheen
    "arab_sad"   : "\u0635",   # ص Sad
    "arab_dad"   : "\u0636",   # ض Dad
    "arab_tah"   : "\u0637",   # ط Tah
    "arab_zah"   : "\u0638",   # ظ Zah
    "arab_ain"   : "\u0639",   # ع Ain
    "arab_ghain" : "\u063a",   # غ Ghain
    "arab_fa"    : "\u0641",   # ف Fa
    "arab_qaf"   : "\u0642",   # ق Qaf
    "arab_kaf"   : "\u0643",   # ك Kaf
    "arab_lam"   : "\u0644",   # ل Lam
    "arab_meem"  : "\u0645",   # م Meem
    "arab_noon"  : "\u0646",   # ن Noon
    "arab_waw"   : "\u0648",   # و Waw
    "arab_ya"    : "\u064a",   # ي Ya
    # Huruf khusus Jawi — Ca (چ)
    "jawi_ca_isolated" : "\u0686",
    "jawi_ca_start"    : "\u0686",
    "jawi_ca_middle"   : "\u0686",
    "jawi_ca_end"      : "\u0686",
    # Ga (ڬ)
    "jawi_ga_isolated" : "\u06ac",
    "jawi_ga_start"    : "\u06ac",
    "jawi_ga_middle"   : "\u06ac",
    "jawi_ga_end"      : "\u06ac",
    # Nga (ڠ)
    "jawi_nga_isolated": "\u06a0",
    "jawi_nga_start"   : "\u06a0",
    "jawi_nga_middle"  : "\u06a0",
    "jawi_nga_end"     : "\u06a0",
    # Nya (ڽ)
    "jawi_nya_isolated": "\u06bd",
    "jawi_nya_start"   : "\u06bd",
    "jawi_nya_middle"  : "\u06bd",
    "jawi_nya_end"     : "\u06bd",
    # Pa (ڤ)
    "jawi_pa_isolated" : "\u06a4",
    "jawi_pa_start"    : "\u06a4",
    "jawi_pa_middle"   : "\u06a4",
    "jawi_pa_end"      : "\u06a4",
    # Va (ۋ)
    "jawi_va_isolated" : "\u06cb",
    "jawi_va_start"    : "\u06cb",
    "jawi_va_middle"   : "\u06cb",
    "jawi_va_end"      : "\u06cb",
}


def get_arabic_char(class_name: str) -> str:
    """
    Mengembalikan karakter Arab/Jawi asli dari nama kelas model.

    Args:
        class_name : Nama kelas dari model (misal "arab_alef", "jawi_ca_isolated").

    Returns:
        Karakter Unicode Arab/Jawi, atau string kosong jika tidak ditemukan.
    """
    return CLASS_TO_ARABIC.get(class_name, "")


def get_display_label(class_name: str) -> str:
    """
    Mengembalikan label tampilan yang menggabungkan karakter Arab dan nama kelas.
    Contoh: "ا  (arab_alef)"

    Args:
        class_name : Nama kelas dari model.

    Returns:
        String label untuk ditampilkan di UI.
    """
    char = get_arabic_char(class_name)
    if char:
        return f"{char}  ({class_name})"
    return class_name


# ---------------------------------------------------------------------------
# Transform inferensi
# ---------------------------------------------------------------------------

def get_inference_transform() -> T.Compose:
    """
    Mengembalikan pipeline transformasi gambar yang digunakan saat inferensi.

    Langkah-langkah:
        1. Resize ke 256 px (sisi terpendek).
        2. CenterCrop ke 224 × 224 px.
        3. Konversi ke Tensor (normalisasi nilai ke [0, 1]).
        4. Normalisasi dengan statistik ImageNet.

    Returns:
        torchvision.transforms.Compose
    """
    return T.Compose([
        T.Resize(RESIZE_SIZE),
        T.CenterCrop(CROP_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Utilitas gambar
# ---------------------------------------------------------------------------

def pil_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Mengkonversi objek PIL.Image menjadi bytes."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Mengkode gambar PIL ke string base64 (berguna untuk embed di HTML)."""
    raw  = pil_to_bytes(image, fmt)
    b64  = base64.b64encode(raw).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def load_image_from_upload(uploaded_file) -> Image.Image:
    """
    Membaca file yang diunggah melalui st.file_uploader dan
    mengembalikan objek PIL.Image dalam mode RGB.
    """
    image = Image.open(uploaded_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def format_confidence(score: float) -> str:
    """Memformat skor kepercayaan sebagai persentase. Contoh: '93.45%'"""
    return f"{score * 100:.2f}%"


def get_confidence_color(score: float) -> str:
    """Mengembalikan warna CSS berdasarkan tingkat kepercayaan."""
    if score >= 0.85:
        return "#10b981"   # hijau (high confidence)
    elif score >= 0.60:
        return "#f59e0b"   # kuning (medium confidence)
    else:
        return "#ef4444"   # merah (low confidence)
