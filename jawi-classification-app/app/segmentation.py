"""
segmentation.py
---------------
Modul untuk segmentasi karakter dari gambar halaman naskah penuh.
Menggunakan OpenCV untuk mendeteksi kontur/blob karakter individual.

Pipeline:
    1. Konversi ke grayscale
    2. Gaussian blur untuk reduksi noise
    3. Adaptive thresholding (lebih baik untuk naskah dengan pencahayaan tidak merata)
    4. Morphological closing untuk menyambung komponen huruf yang terputus
    5. Deteksi kontur EXTERNAL
    6. Filter berdasarkan luas & aspect ratio
    7. Kembalikan patch karakter + bounding box
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Konstanta default
# ---------------------------------------------------------------------------
DEFAULT_MIN_AREA    = 300     # piksel² minimum agar dihitung sebagai karakter
DEFAULT_MAX_RATIO   = 0.04   # fraksi maks luas gambar untuk satu kontur
DEFAULT_PADDING     = 5      # piksel padding di sekitar bounding box
DEFAULT_MAX_CHARS   = 100    # maks karakter yang diproses (sample jika lebih)


# ---------------------------------------------------------------------------
# Konversi format
# ---------------------------------------------------------------------------

def _pil_to_cv(image: Image.Image) -> np.ndarray:
    """Konversi PIL Image RGB → numpy array BGR (format OpenCV)."""
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    """Konversi numpy array BGR → PIL Image RGB."""
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Fungsi utama
# ---------------------------------------------------------------------------

def segment_characters(
    image: Image.Image,
    min_area: int        = DEFAULT_MIN_AREA,
    max_area_ratio: float= DEFAULT_MAX_RATIO,
    padding: int         = DEFAULT_PADDING,
    max_chars: int       = DEFAULT_MAX_CHARS,
    use_adaptive: bool   = True,
) -> tuple[list[Image.Image], list[tuple[int, int, int, int]]]:
    """
    Segmentasi karakter dari gambar halaman naskah.

    Args:
        image          : PIL Image halaman naskah (mode apa pun).
        min_area       : Luas minimum kontur agar dianggap karakter (piksel²).
        max_area_ratio : Fraksi maks luas gambar untuk satu kontur.
        padding        : Piksel padding di sekitar bounding box karakter.
        max_chars      : Batas maksimum karakter yang dikembalikan.
        use_adaptive   : True = adaptive thresholding (lebih baik untuk naskah);
                         False = Otsu global.

    Returns:
        Tuple:
        - list[Image.Image]           : Patch karakter individual (mode RGB).
        - list[tuple[x, y, w, h]]    : Bounding box masing-masing karakter.
    """
    img_cv   = _pil_to_cv(image)
    h, w     = img_cv.shape[:2]
    max_area = h * w * max_area_ratio

    # ── 1. Grayscale ────────────────────────────────────────────────────────
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # ── 2. Gaussian blur ────────────────────────────────────────────────────
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # ── 3. Thresholding ─────────────────────────────────────────────────────
    if use_adaptive:
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21, C=10,
        )
    else:
        _, binary = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

    # ── 4. Morphological closing ─────────────────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # ── 5. Deteksi kontur ────────────────────────────────────────────────────
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # ── 6. Filter kontur ─────────────────────────────────────────────────────
    valid_bboxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)

        # Buang kontur dengan aspect ratio ekstrem (garis horizontal/vertikal)
        aspect = cw / (ch + 1e-6)
        if aspect > 10.0 or aspect < 0.08:
            continue

        valid_bboxes.append((x, y, cw, ch))

    # ── 7. Sample jika terlalu banyak karakter ───────────────────────────────
    if len(valid_bboxes) > max_chars:
        import random
        random.shuffle(valid_bboxes)
        valid_bboxes = valid_bboxes[:max_chars]

    # ── 8. Potong patch karakter ─────────────────────────────────────────────
    patches: list[Image.Image] = []
    final_bboxes: list[tuple[int, int, int, int]] = []

    for (x, y, cw, ch) in valid_bboxes:
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + cw + padding)
        y2 = min(h, y + ch + padding)

        patch = image.crop((x1, y1, x2, y2)).convert("RGB")
        patches.append(patch)
        final_bboxes.append((x1, y1, x2 - x1, y2 - y1))

    return patches, final_bboxes


def draw_annotated(
    image: Image.Image,
    bboxes: list[tuple[int, int, int, int]],
    classifications: list[str],
    confidence_threshold: float = 0.6,
) -> Image.Image:
    """
    Buat salinan gambar dengan bounding box monokrom berdasarkan hasil klasifikasi.

    Gaya Kotak:
        - Jawi-spesifik    : Garis hitam tebal solid (3px)
        - Arab standar     : Garis hitam tipis solid (1px)
        - Tidak dikenal/abu: Garis abu-abu putus-putus (1px)

    Args:
        image                : PIL Image asli.
        bboxes               : List bounding box (x, y, w, h).
        classifications      : List nama kelas per karakter (sama panjang dengan bboxes).
        confidence_threshold : Ambang confidence (tidak digunakan di sini, hanya untuk referensi).

    Returns:
        PIL Image dengan bounding box monokrom.
    """
    annotated = image.copy().convert("RGB")
    draw      = ImageDraw.Draw(annotated)

    JAWI_PREFIXES = {"jawi_ca", "jawi_ga", "jawi_nga", "jawi_nya", "jawi_pa", "jawi_va"}

    def draw_dashed_line(x1, y1, x2, y2, color, width=1, dash_len=4):
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length == 0:
            return
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        curr = 0.0
        while curr < length:
            end = min(curr + dash_len, length)
            draw.line([x1 + dx * curr, y1 + dy * curr, x1 + dx * end, y1 + dy * end], fill=color, width=width)
            curr += dash_len * 2

    def draw_dashed_rectangle(x1, y1, x2, y2, color, width=1, dash_len=4):
        draw_dashed_line(x1, y1, x2, y1, color, width, dash_len)
        draw_dashed_line(x2, y1, x2, y2, color, width, dash_len)
        draw_dashed_line(x2, y2, x1, y2, color, width, dash_len)
        draw_dashed_line(x1, y2, x1, y1, color, width, dash_len)

    for (x, y, bw, bh), cls in zip(bboxes, classifications):
        x1, y1 = x, y
        x2, y2 = x + bw, y + bh

        if cls and any(cls.startswith(p) for p in JAWI_PREFIXES):
            # Jawi - Hitam tebal solid
            draw.rectangle([x1, y1, x2, y2], outline="#000000", width=3)
        elif cls and cls.startswith("arab_"):
            # Arab standar - Hitam tipis solid
            draw.rectangle([x1, y1, x2, y2], outline="#111111", width=1)
        else:
            # Tidak dikenal / rendah - Abu putus-putus
            draw_dashed_rectangle(x1, y1, x2, y2, color="#888888", width=1, dash_len=4)

    return annotated

