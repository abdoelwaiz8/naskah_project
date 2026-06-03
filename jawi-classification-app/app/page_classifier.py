"""
page_classifier.py
------------------
Modul klasifikasi halaman naskah penuh (Arab Jawi vs Arab Asli).

Dua mode operasi (dipilih otomatis):
    ┌─────────────────────────────────────────────────────────────┐
    │ MODE 1 — Page-Level Model (model/page_classifier.pth ada)  │
    │   Akurat, tidak terpengaruh elemen dekoratif               │
    │   → satu forward pass langsung ke kesimpulan               │
    ├─────────────────────────────────────────────────────────────┤
    │ MODE 2 — Segmentasi + Voting (fallback jika model belum ada)│
    │   Bergantung pada kualitas segmentasi karakter             │
    │   → rentan false positive pada Al-Quran/ornamen            │
    └─────────────────────────────────────────────────────────────┘

Untuk mengaktifkan Mode 1 (direkomendasikan):
    1. Kumpulkan gambar halaman di raw_page_dataset/arab_asli/ dan /arab_jawi/
    2. python scripts/split_page_dataset.py
    3. python scripts/train_page.py
"""

from __future__ import annotations

from PIL import Image
import page_inference
import inference
from segmentation import segment_characters, draw_annotated
from utils import get_arabic_char


# ---------------------------------------------------------------------------
# Konstanta: prefiks kelas yang HANYA ada di Jawi (bukan Arab standar)
# ---------------------------------------------------------------------------
JAWI_SPECIFIC_PREFIXES: set[str] = {
    "jawi_ca",
    "jawi_ga",
    "jawi_nga",
    "jawi_nya",
    "jawi_pa",
    "jawi_va",
}


def _is_jawi_class(class_name: str) -> bool:
    """Kembalikan True jika kelas termasuk huruf Jawi-spesifik."""
    return any(class_name.startswith(p) for p in JAWI_SPECIFIC_PREFIXES)


# ---------------------------------------------------------------------------
# MODE 1: Page-level model
# ---------------------------------------------------------------------------

def _classify_page_model(image: Image.Image) -> dict:
    """
    Klasifikasi halaman menggunakan model page-level (page_classifier.pth).
    Satu forward pass → hasil langsung tanpa segmentasi.
    """
    result = page_inference.predict_page(image)

    return {
        "script_type"    : result["script_type"],
        "confidence"     : result["confidence"],
        "mode"           : "page_model",
        "total_chars"    : 0,
        "processed_chars": 0,
        "jawi_chars"     : 0,
        "arab_chars"     : 0,
        "jawi_ratio"     : 1.0 if result["class_name"] == "arab_jawi" else 0.0,
        "jawi_found"     : [],
        "annotated_image": image.copy(),
        "char_results"   : [],
        "probabilities"  : result.get("probabilities", {}),
    }


# ---------------------------------------------------------------------------
# MODE 2: Segmentasi + Voting (fallback)
# ---------------------------------------------------------------------------

def _classify_page_segmentation(
    image               : Image.Image,
    min_area            : int   = 300,
    max_area_ratio      : float = 0.04,
    confidence_threshold: float = 0.55,
    max_chars           : int   = 100,
    use_adaptive        : bool  = True,
    is_arab_asli_verified: bool = False,
) -> dict:
    """
    Klasifikasi halaman via segmentasi karakter + voting.
    Digunakan sebagai fallback jika model page-level belum tersedia.
    """
    patches, bboxes = segment_characters(
        image,
        min_area       = min_area,
        max_area_ratio = max_area_ratio,
        max_chars      = max_chars,
        use_adaptive   = use_adaptive,
    )

    total_chars = len(patches)

    if total_chars == 0:
        annotated = draw_annotated(image, [], [], confidence_threshold)
        return {
            "script_type"    : "Tidak Terdeteksi",
            "confidence"     : 0.0,
            "mode"           : "segmentation",
            "total_chars"    : 0,
            "processed_chars": 0,
            "jawi_chars"     : 0,
            "arab_chars"     : 0,
            "jawi_ratio"     : 0.0,
            "jawi_found"     : [],
            "annotated_image": annotated,
            "char_results"   : [],
            "probabilities"  : {},
        }

    jawi_chars    = 0
    arab_chars    = 0
    processed     = 0
    jawi_found_map: dict[str, str] = {}
    char_results  : list[dict]     = []
    classifications: list[str]     = [""] * total_chars

    for i, patch in enumerate(patches):
        try:
            result = inference.predict(patch, top_k=1)
            cls    = result["predicted_class"]
            conf   = result["confidence"]
            processed += 1

            is_jawi = _is_jawi_class(cls)
            if is_jawi and is_arab_asli_verified:
                is_jawi = False
                cls = "arab_converted"

            char_results.append({
                "class"     : cls,
                "confidence": conf,
                "is_jawi"   : is_jawi,
            })
            classifications[i] = cls

            if conf >= confidence_threshold:
                if is_jawi:
                    jawi_chars += 1
                    if cls not in jawi_found_map:
                        jawi_found_map[cls] = get_arabic_char(cls)
                else:
                    arab_chars += 1

        except Exception:
            char_results.append({"class": "", "confidence": 0.0, "is_jawi": False})

    annotated  = draw_annotated(image, bboxes, classifications, confidence_threshold)
    counted    = jawi_chars + arab_chars
    jawi_ratio = jawi_chars / max(counted, 1)

    if jawi_chars > 0:
        script_type = "Arab Jawi"
        confidence  = min(0.75 + jawi_ratio * 0.24, 0.99)
    elif processed == 0:
        script_type = "Tidak Terdeteksi"
        confidence  = 0.0
    else:
        script_type = "Arab Asli"
        arab_ratio  = arab_chars / max(counted, 1)
        confidence  = min(0.55 + arab_ratio * 0.44, 0.99)

    return {
        "script_type"    : script_type,
        "confidence"     : confidence,
        "mode"           : "segmentation",
        "total_chars"    : total_chars,
        "processed_chars": processed,
        "jawi_chars"     : jawi_chars,
        "arab_chars"     : arab_chars,
        "jawi_ratio"     : jawi_ratio,
        "jawi_found"     : [{"class": c, "char": ch} for c, ch in jawi_found_map.items()],
        "annotated_image": annotated,
        "char_results"   : char_results,
        "probabilities"  : {},
    }


# ---------------------------------------------------------------------------
# API Publik — auto-pilih mode
# ---------------------------------------------------------------------------

def classify_page(
    image               : Image.Image,
    min_area            : int   = 300,
    max_area_ratio      : float = 0.04,
    confidence_threshold: float = 0.55,
    max_chars           : int   = 100,
    use_adaptive        : bool  = True,
) -> dict:
    """
    Klasifikasi halaman naskah. Mode Hybrid (Otomatis):
    - Selalu melakukan segmentasi karakter untuk statistik dan anotasi visual.
    - Menggabungkan hasil dari segmentasi dengan model page-level (jika tersedia).
    """
    # 0. Cek prediksi awal dari model page-level untuk mendeteksi apakah halaman terverifikasi Arab Asli dengan tingkat kepercayaan tinggi.
    # Ini membantu mengabaikan false positives klasifikasi karakter Jawi akibat tanda harakat/tajwid pada Al-Quran.
    is_arab_asli_verified = False
    page_pred = None
    if page_inference.is_page_model_available():
        page_pred = page_inference.predict_page(image)
        if page_pred["script_type"] == "Arab Asli" and page_pred["confidence"] >= 0.85:
            is_arab_asli_verified = True

    # 1. Jalankan segmentasi (selalu) untuk visualisasi dan penjelasan statis
    result = _classify_page_segmentation(
        image,
        min_area             = min_area,
        max_area_ratio       = max_area_ratio,
        confidence_threshold = confidence_threshold,
        max_chars            = max_chars,
        use_adaptive         = use_adaptive,
        is_arab_asli_verified = is_arab_asli_verified,
    )

    # 2. Jika model page-level tersedia, gabungkan hasilnya secara Hybrid
    if page_pred is not None:
        page_type = page_pred["script_type"]
        page_conf = page_pred["confidence"]
        
        final_type = page_type
        final_conf = page_conf
        
        # Jika model page-level kurang yakin (< 85%), kita validasi dengan bukti segmentasi karakter
        if page_conf < 0.85:
            if result["jawi_chars"] > 0:
                final_type = "Arab Jawi"
                # Tingkatkan kepercayaan karena didukung bukti fisik huruf Jawi
                final_conf = max(page_conf, min(0.75 + result["jawi_ratio"] * 0.24, 0.99))
            elif result["processed_chars"] > 0 and result["jawi_chars"] == 0:
                final_type = "Arab Asli"
                # Tingkatkan kepercayaan karena tidak ada huruf Jawi spesifik yang ditemukan
                arab_ratio = result["arab_chars"] / max(result["processed_chars"], 1)
                final_conf = max(page_conf, min(0.60 + arab_ratio * 0.39, 0.99))
                
        # Timpa hasil prediksi murni segmentasi dengan hasil Hybrid
        result["script_type"]   = final_type
        result["confidence"]    = final_conf
        result["mode"]          = "hybrid"
        result["probabilities"] = page_pred.get("probabilities", {})

    return result
