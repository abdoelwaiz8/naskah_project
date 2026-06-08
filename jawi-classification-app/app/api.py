"""
api.py
------
FastAPI Backend untuk melayani klasifikasi naskah Jawi/Arab
dan menyediakan endpoint riwayat database.
"""

from __future__ import annotations

import io
import os
import base64
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(__file__))

import database
import inference
import page_classifier as pc
import page_inference
import gemini_service

app = FastAPI(
    title="Jawi Script Classifier API",
    description="API untuk klasifikasi naskah Arab Jawi vs Arab Asli menggunakan PyTorch & Gemini",
    version="1.0.0"
)

# CORS Middleware agar bisa diakses dari port lain jika dideploy terpisah
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi DB saat startup
@app.on_event("startup")
def on_startup():
    database.initialize_db()

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/model-status")
def model_status():
    return {
        "character_model_available": inference.is_model_available(),
        "page_model_available": page_inference.is_page_model_available()
    }

@app.post("/classify")
async def classify_manuscript(
    file: UploadFile = File(...),
    min_area: int = Form(300),
    confidence_threshold: float = Form(0.55),
    max_chars: int = Form(100),
    use_adaptive: bool = Form(True),
    api_key: str = Form(""),
):
    # Validasi model
    if not inference.is_model_available():
        raise HTTPException(
            status_code=503, 
            detail="File bobot model 'resnet34_jawi.pth' tidak ditemukan di folder /model/"
        )

    # Baca file gambar
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"File gambar tidak valid: {err}")

    # Lakukan klasifikasi
    try:
        # Panggil classify_page
        result = pc.classify_page(
            image,
            min_area=min_area,
            max_area_ratio=0.04,
            confidence_threshold=confidence_threshold,
            max_chars=max_chars,
            use_adaptive=use_adaptive,
        )

        # Integrasi Gemini jika API Key ada
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if key:
            try:
                context_result = gemini_service.identify_context(image, key)
                result["manuscript_context"] = context_result.get("kategori", "")
                result["manuscript_explanation"] = context_result.get("penjelasan", "")
            except Exception as gemini_err:
                result["manuscript_context"] = ""
                result["manuscript_explanation"] = f"Gagal mengidentifikasi konteks: {gemini_err}"
        else:
            result["manuscript_context"] = ""
            result["manuscript_explanation"] = ""

        # Konversi annotated_image ke base64
        annotated_pil = result["annotated_image"]
        buffered = io.BytesIO()
        annotated_pil.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Hapus objek PIL Image dari response dict agar JSON serializable
        del result["annotated_image"]
        result["annotated_image_base64"] = img_base64

        return result
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Gagal memproses gambar: {err}")

@app.get("/history")
def get_history():
    try:
        records = database.fetch_all_records()
        return [dict(r) for r in records]
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Gagal membaca database: {err}")

@app.post("/history")
def save_history(record: dict):
    try:
        timestamp_str = record.get("timestamp")
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
        
        record_id = database.insert_record(
            filename=record.get("filename", "unknown"),
            script_type=record.get("script_type", "Tidak Terdeteksi"),
            total_chars=record.get("total_chars", 0),
            jawi_chars=record.get("jawi_chars", 0),
            confidence_score=record.get("confidence_score", 0.0),
            timestamp=timestamp,
            manuscript_context=record.get("manuscript_context", ""),
        )
        return {"status": "success", "id": record_id}
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan ke database: {err}")

@app.delete("/history")
def clear_history():
    try:
        database.clear_all_records()
        return {"status": "success", "message": "Semua riwayat berhasil dihapus"}
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Gagal menghapus database: {err}")
