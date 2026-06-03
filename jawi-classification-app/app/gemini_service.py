import google.generativeai as genai
import json
import os
import PIL.Image

def identify_context(image: PIL.Image.Image, api_key: str = "") -> dict:
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("Gemini API Key tidak ditemukan.")
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = (
        "Bertindaklah sebagai ahli filologi naskah Nusantara. "
        "Klasifikasikan naskah kuno pada gambar ke salah satu dari: 'Agama', 'Hikayat', atau 'Ilmu Pengetahuan' "
        "dan jelaskan alasannya (1-2 paragraf) dalam format JSON valid: "
        '{"kategori": "...", "penjelasan": "..."}'
    )
    response = model.generate_content(
        [prompt, image],
        generation_config={"response_mime_type": "application/json"}
    )
    try:
        data = json.loads(response.text)
        return {
            "kategori": data.get("kategori", "Agama"),
            "penjelasan": data.get("penjelasan", "")
        }
    except Exception:
        return {
            "kategori": "Agama",
            "penjelasan": response.text
        }
