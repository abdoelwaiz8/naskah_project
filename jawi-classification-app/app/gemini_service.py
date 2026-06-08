"""
gemini_service.py
-----------------
Layanan identifikasi konteks naskah menggunakan Google Gemini API (google-genai SDK).
"""

import json
import os
import io
import PIL.Image
from google import genai
from google.genai import types


def identify_context(image: PIL.Image.Image, api_key: str = "") -> dict:
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("Gemini API Key tidak ditemukan.")

    client = genai.Client(api_key=key)

    # Konversi PIL Image ke bytes untuk dikirim ke API
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="JPEG")
    img_bytes = img_buffer.getvalue()

    prompt = (
        "Bertindaklah sebagai ahli filologi naskah Nusantara. "
        "Klasifikasikan naskah kuno pada gambar ke salah satu dari: 'Agama', 'Hikayat', atau 'Ilmu Pengetahuan' "
        "dan jelaskan alasannya (1-2 paragraf) dalam format JSON valid: "
        '{"kategori": "...", "penjelasan": "..."}'
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    try:
        data = json.loads(response.text)
        return {
            "kategori": data.get("kategori", "Agama"),
            "penjelasan": data.get("penjelasan", ""),
        }
    except Exception:
        return {
            "kategori": "Agama",
            "penjelasan": response.text,
        }
