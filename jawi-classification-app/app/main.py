"""
main.py
-------
Aplikasi Streamlit untuk identifikasi jenis naskah (Arab Jawi / Arab Asli)
menggunakan backend FastAPI (PyTorch ResNet34 & Gemini) dengan tampilan minimalis profesional.
"""

from __future__ import annotations
import sys
import os
import io
import base64
import requests
import pandas as pd
from PIL import Image
from datetime import datetime
import streamlit as st

# URL Backend FastAPI
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ===========================================================================
# Konfigurasi halaman
# ===========================================================================
st.set_page_config(
    page_title="Identifikasi Naskah Jawi",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# CSS (Minimalis Profesional - Abu-abu & Putih)
# ===========================================================================
def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Noto+Naskh+Arabic:wght@400;600;700&display=swap');

    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f9fafb;
        --border-color: #e5e7eb;
        --text-primary: #111827;
        --text-secondary: #4b5563;
        --text-muted: #9ca3af;
        --accent-color: #000000;
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Global Overrides */
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: var(--font-sans);
    }
    
    /* Enforce light background and modern styling on sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    /* Typography & Hierarchy — scoped to markdown/content only, NOT icons */
    h1, h2, h3, h4, h5, h6 {
        color: var(--accent-color) !important;
        font-family: var(--font-sans) !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }

    /* Only apply font to Streamlit's text content containers, not global spans */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] label {
        color: var(--text-secondary);
        font-family: var(--font-sans);
    }

    /* Sidebar text */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-secondary);
        font-family: var(--font-sans);
    }

    /* Slider and toggle labels */
    [data-testid="stWidgetLabel"] p {
        color: var(--text-secondary);
        font-family: var(--font-sans);
    }

    /* Hero Header */
    .hero-header { 
        text-align: left; 
        padding: 3rem 0 2rem; 
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 2.5rem;
        animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .hero-title {
        font-size: 2.75rem !important; 
        font-weight: 800 !important;
        color: var(--accent-color) !important;
        margin: 0 !important; 
        letter-spacing: -0.035em !important;
        line-height: 1.1 !important;
    }
    .hero-subtitle { 
        font-size: 1.05rem !important; 
        color: var(--text-secondary) !important; 
        margin-top: 0.75rem !important; 
        font-weight: 400 !important; 
        line-height: 1.6 !important;
        max-width: 750px;
    }

    /* Modern Minimal Card */
    .glass-card {
        background-color: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 12px; 
        padding: 1.75rem; 
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover { 
        border-color: var(--accent-color); 
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
    }

    /* Banner Hasil Klasifikasi (Dark Monochrome) */
    .script-banner {
        border-radius: 12px; 
        padding: 2.25rem 1.75rem;
        text-align: center; 
        border: 1px solid var(--accent-color);
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        color: #ffffff !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        margin-bottom: 1.5rem;
    }
    .script-label {
        font-size: 0.75rem; 
        font-weight: 600; 
        letter-spacing: 0.15em;
        text-transform: uppercase; 
        color: var(--text-muted) !important; 
        margin-bottom: 0.5rem;
    }
    .script-type {
        font-size: 2.25rem; 
        font-weight: 800; 
        line-height: 1.2; 
        margin: 0.5rem 0;
        color: #ffffff !important;
        letter-spacing: -0.025em;
    }
    .script-conf {
        font-size: 0.95rem; 
        color: #e5e7eb !important;
        line-height: 1.5;
        margin-top: 0.5rem;
    }
    .script-meta {
        margin-top: 1.25rem;
        font-size: 0.85rem;
        color: var(--text-muted) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 0.85rem;
    }
    .script-meta strong {
        color: #ffffff !important;
    }

    /* Stat Cards Row */
    .stat-row { 
        display: flex; 
        gap: 0.85rem; 
        margin: 1.5rem 0; 
        flex-wrap: wrap; 
    }
    .stat-card {
        flex: 1; 
        min-width: 100px;
        background-color: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 10px; 
        padding: 1.25rem 1rem; 
        text-align: center;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
        transition: all 0.25s ease;
    }
    .stat-card:hover {
        border-color: var(--accent-color);
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .stat-value { 
        font-size: 1.75rem; 
        font-weight: 800; 
        color: var(--accent-color) !important; 
        line-height: 1; 
    }
    .stat-label { 
        font-size: 0.65rem; 
        color: var(--text-secondary) !important; 
        margin-top: 0.5rem; 
        text-transform: uppercase; 
        letter-spacing: 0.05em; 
        font-weight: 600;
    }

    /* Jawi found container & pills */
    .jawi-pills { 
        display: flex; 
        flex-wrap: wrap; 
        gap: 0.75rem; 
        margin-top: 0.75rem; 
        margin-bottom: 1.5rem;
    }
    .jawi-pill-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        background-color: var(--bg-primary);
        border: 1px solid var(--border-color);
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        min-width: 55px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
        transition: all 0.2s ease;
    }
    .jawi-pill-container:hover {
        border-color: var(--accent-color);
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .jawi-pill {
        font-family: 'Noto Naskh Arabic', serif;
        font-size: 1.35rem; 
        color: var(--accent-color) !important; 
        font-weight: 700;
        line-height: 1.2;
    }
    .jawi-pill-label { 
        font-size: 0.65rem; 
        color: var(--text-secondary) !important; 
        margin-top: 0.15rem; 
        text-transform: capitalize;
    }

    /* Legend */
    .legend { 
        display: flex; 
        gap: 1.5rem; 
        font-size: 0.8rem; 
        color: var(--text-secondary) !important; 
        margin-top: 0.75rem; 
        margin-bottom: 1.5rem;
        flex-wrap: wrap; 
        padding: 0.85rem 1.25rem;
        background-color: var(--bg-secondary);
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .legend-line { 
        display: inline-block; 
        width: 20px; 
        height: 0px; 
        vertical-align: middle;
    }

    /* Modern Context Card (Gemini AI) */
    .context-card {
        background-color: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--accent-color);
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.02);
        animation: fadeIn 0.5s ease-out;
    }
    .context-label {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--text-secondary) !important;
        margin-bottom: 0.25rem;
    }
    .context-category {
        font-size: 1.3rem;
        font-weight: 800;
        color: var(--accent-color) !important;
        margin-bottom: 0.75rem;
        letter-spacing: -0.02em;
    }
    .context-explanation {
        font-size: 0.9rem;
        line-height: 1.6;
        color: var(--text-secondary) !important;
    }

    /* Placeholder Card for empty Uploads */
    .placeholder-card {
        text-align: center; 
        padding: 4.5rem 2rem; 
        border: 2px dashed var(--border-color); 
        border-radius: 12px;
        background-color: var(--bg-secondary);
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
    }
    .placeholder-card:hover {
        border-color: var(--accent-color);
        background-color: #f3f4f6;
    }
    .placeholder-text {
        color: var(--text-secondary) !important;
        font-size: 0.95rem;
        margin: 0;
        font-weight: 500;
    }

    /* Upload Area Styling — only style the outer container, not internal Streamlit elements */
    [data-testid="stFileUploader"] > section {
        background-color: var(--bg-secondary) !important;
        border: 1.5px dashed var(--border-color) !important;
        border-radius: 12px !important;
        transition: border-color 0.25s ease, background-color 0.25s ease !important;
    }
    [data-testid="stFileUploader"] > section:hover {
        border-color: var(--accent-color) !important;
        background-color: #f3f4f6 !important;
    }

    /* Buttons styling */
    .stButton > button, .stDownloadButton > button {
        width: 100%;
        background-color: var(--accent-color) !important;
        color: #ffffff !important; 
        border: 1px solid var(--accent-color) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 0.95rem !important; 
        font-weight: 600 !important;
        letter-spacing: 0.025em !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #1f2937 !important;
        border-color: #1f2937 !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
    }
    .stButton > button:active, .stDownloadButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Progress Bar */
    .stProgress > div > div > div {
        background-color: var(--accent-color) !important;
        border-radius: 4px !important;
    }
    .stProgress > div > div {
        background-color: #f3f4f6 !important;
        border-radius: 4px !important;
    }

    /* Metric Card Styling */
    [data-testid="stMetric"] {
        background-color: var(--bg-primary) !important;
        border-radius: 10px !important; 
        padding: 1.25rem !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--accent-color) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04) !important;
    }
    [data-testid="stMetricLabel"] p { 
        color: var(--text-secondary) !important; 
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="stMetricValue"] div { 
        color: var(--accent-color) !important; 
        font-weight: 800 !important; 
        font-size: 1.6rem !important;
        letter-spacing: -0.03em !important;
    }

    /* Alerts */
    .stAlert { 
        border-radius: 10px !important; 
        border: 1px solid var(--border-color) !important; 
        background-color: var(--bg-secondary) !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.02) !important;
    }
    .stAlert [data-testid="stNotificationContent"] {
        color: var(--text-primary) !important;
    }

    /* Sidebar Title & Elements */
    .sidebar-title {
        font-size: 0.75rem; 
        font-weight: 700; 
        text-transform: uppercase;
        letter-spacing: 0.1em; 
        color: var(--text-primary) !important; 
        margin-bottom: 0.8rem;
        margin-top: 1.2rem;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.4rem;
    }

    /* Dataframe Overrides */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* Global divider override */
    hr {
        margin: 2rem 0 !important;
        border: 0 !important;
        border-top: 1px solid var(--border-color) !important;
    }

    /* Keyframes Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)


# ===========================================================================
# Cek Ketersediaan Model dari Backend
# ===========================================================================
def check_model_availability() -> bool:
    try:
        response = requests.get(f"{BACKEND_URL}/model-status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("character_model_available", False)
    except Exception:
        pass
    return False


# ===========================================================================
# Render: Hero
# ===========================================================================
def render_hero() -> None:
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">Identifikasi Naskah Arab</h1>
        <p class="hero-subtitle">
            Unggah halaman naskah kuno untuk mengidentifikasi apakah ditulis dalam tulisan Arab Jawi atau Arab Asli.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# Render: Banner hasil
# ===========================================================================
def render_result_banner(result: dict) -> None:
    stype = result["script_type"]
    conf  = result["confidence"]

    if stype == "Arab Jawi":
        desc = "Naskah ini diidentifikasi menggunakan tulisan Arab Jawi"
    elif stype == "Arab Asli":
        desc = "Naskah ini diidentifikasi menggunakan tulisan Arab Asli"
    else:
        desc = "Karakter tidak dapat terdeteksi pada gambar ini"

    st.markdown(f"""
    <div class="script-banner">
        <div class="script-label">Hasil Klasifikasi Naskah</div>
        <div class="script-type">{stype}</div>
        <div class="script-conf">{desc}</div>
        <div class="script-meta">
            Tingkat Kepercayaan: <strong>{conf*100:.1f}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# Render: Statistik karakter
# ===========================================================================
def render_stats(result: dict) -> None:
    total     = result["total_chars"]
    jawi_c    = result["jawi_chars"]
    arab_c    = result["arab_chars"]
    jawi_pct  = f"{result['jawi_ratio']*100:.1f}%"

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-value">{total}</div>
            <div class="stat-label">Total Karakter</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{jawi_c}</div>
            <div class="stat-label">Huruf Jawi</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{arab_c}</div>
            <div class="stat-label">Huruf Arab</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{jawi_pct}</div>
            <div class="stat-label">Rasio Jawi</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# Render: Huruf Jawi yang ditemukan
# ===========================================================================
def render_jawi_found(jawi_found: list[dict]) -> None:
    if not jawi_found:
        return

    st.markdown("**Huruf Jawi Spesifik yang Ditemukan:**")
    pills_html = '<div class="jawi-pills">'
    for item in jawi_found:
        char = item["char"] or item["class"]
        cls  = item["class"]
        pills_html += f"""
        <div class="jawi-pill-container">
            <span class="jawi-pill">{char}</span>
            <span class="jawi-pill-label">{cls}</span>
        </div>"""
    pills_html += "</div>"
    st.markdown(pills_html, unsafe_allow_html=True)


# ===========================================================================
# Render: Informasi Konteks (Gemini AI)
# ===========================================================================
def render_context_card(category: str, explanation: str) -> None:
    if not category:
        return
    
    st.markdown(f"""
    <div class="context-card">
        <div class="context-label">Konteks Naskah (Gemini AI)</div>
        <div class="context-category">{category}</div>
        <div class="context-explanation">{explanation}</div>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# Render: Sidebar
# ===========================================================================
def render_sidebar() -> dict:
    params: dict = {}
    with st.sidebar:
        st.markdown("<div class='sidebar-title'>Status Model</div>", unsafe_allow_html=True)
        if check_model_availability():
            st.success("Model Tersedia (API Backend)")
        else:
            st.warning("Model Belum Tersedia di Backend")

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-title'>Parameter Analisis</div>", unsafe_allow_html=True)
        params["min_area"] = st.slider(
            "Ukuran Karakter Minimum (px²)",
            min_value=50, max_value=1000, value=300, step=50,
        )
        params["confidence_threshold"] = st.slider(
            "Ambang Kepercayaan Klasifikasi",
            min_value=0.30, max_value=0.90, value=0.55, step=0.05,
        )
        params["max_chars"] = st.slider(
            "Batas Karakter Maksimum",
            min_value=20, max_value=200, value=100, step=10,
        )
        params["use_adaptive"] = st.toggle(
            "Adaptive Thresholding",
            value=True,
        )

        params["api_key"] = os.environ.get("GEMINI_API_KEY", "")

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-title'>Riwayat Analisis</div>", unsafe_allow_html=True)
        try:
            response = requests.get(f"{BACKEND_URL}/history", timeout=5)
            records = response.json() if response.status_code == 200 else []
        except Exception:
            records = []

        if records:
            df = pd.DataFrame(records)
            df["confidence_score"] = df["confidence_score"].apply(lambda x: f"{x*100:.1f}%")
            df["manuscript_context"] = df["manuscript_context"].fillna("")
            df = df.rename(columns={
                "id"              : "ID",
                "filename"        : "File",
                "script_type"     : "Jenis Naskah",
                "confidence_score": "Kepercayaan",
                "timestamp"       : "Waktu",
                "manuscript_context": "Konteks",
            })
            
            # Format waktu
            df["Waktu"] = pd.to_datetime(df["Waktu"]).dt.strftime('%H:%M:%S %d/%m/%y')
            
            st.dataframe(df[["File","Jenis Naskah","Kepercayaan"]],
                         use_container_width=True, hide_index=True)

            col_dl, col_clr = st.columns(2)
            with col_dl:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Ekspor CSV", data=csv,
                                   file_name="riwayat_naskah.csv",
                                   mime="text/csv", use_container_width=True)
            with col_clr:
                if st.button("Hapus Semua", use_container_width=True):
                    try:
                        requests.delete(f"{BACKEND_URL}/history")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Gagal menghapus: {err}")
        else:
            st.markdown(
                "<p style='color:#000000;font-size:0.85rem;'>"
                "Belum ada riwayat analisis.</p>",
                unsafe_allow_html=True,
            )

    return params


# ===========================================================================
# Halaman Utama
# ===========================================================================
def main() -> None:
    inject_css()
    render_hero()
    params = render_sidebar()

    col_left, col_right = st.columns([1.1, 1], gap="large")

    # ── Kolom kiri: upload ────────────────────────────────────────────────
    with col_left:
        st.markdown("### Unggah Gambar Naskah")
        uploaded = st.file_uploader(
            label="Unggah berkas naskah",
            type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
            label_visibility="collapsed",
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image,
                     caption=f"Berkas: {uploaded.name}  ·  {image.size[0]}×{image.size[1]} px",
                     use_container_width=True)

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Nama File", uploaded.name[:18] + ("…" if len(uploaded.name) > 18 else ""))
            m2.metric("Ukuran Berkas", f"{uploaded.size / 1024:.1f} KB")
            m3.metric("Resolusi", f"{image.size[0]}×{image.size[1]}")

    # ── Kolom kanan: hasil ────────────────────────────────────────────────
    with col_right:
        st.markdown("### Hasil Identifikasi")

        if not uploaded:
            st.markdown("""
            <div class="placeholder-card">
                <p class="placeholder-text">Unggah naskah kuno untuk memulai analisis klasifikasi.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            identify_btn = st.button(
                "ANALISIS NASKAH", type="primary", use_container_width=True,
            )

            # Menampilkan hasil yang disimpan di session state (jika ada)
            if "page_result" in st.session_state and not identify_btn:
                res = st.session_state["page_result"]
                render_result_banner(res)
                render_stats(res)
                render_jawi_found(res["jawi_found"])
                if "manuscript_context" in res and res["manuscript_context"]:
                    render_context_card(res["manuscript_context"], res.get("manuscript_explanation", ""))

            if identify_btn:
                progress_bar = st.progress(0, text="Menghubungi API backend...")

                try:
                    # Persiapkan request file & parameter
                    buffered_img = io.BytesIO()
                    image.save(buffered_img, format="JPEG")
                    img_bytes = buffered_img.getvalue()

                    files = {"file": (uploaded.name, img_bytes, "image/jpeg")}
                    data = {
                        "min_area": str(params["min_area"]),
                        "confidence_threshold": str(params["confidence_threshold"]),
                        "max_chars": str(params["max_chars"]),
                        "use_adaptive": "true" if params["use_adaptive"] else "false",
                        "api_key": params["api_key"],
                    }

                    progress_bar.progress(30, text="Segmentasi & Klasifikasi karakter...")
                    
                    response = requests.post(f"{BACKEND_URL}/classify", files=files, data=data, timeout=120)
                    
                    if response.status_code != 200:
                        raise Exception(f"Backend API error: {response.text}")
                    
                    result = response.json()
                    
                    progress_bar.progress(85, text="Menyimpan hasil analisis...")

                    # Simpan ke DB riwayat lewat API
                    db_record = {
                        "filename": uploaded.name,
                        "script_type": result["script_type"],
                        "total_chars": result["total_chars"],
                        "jawi_chars": result["jawi_chars"],
                        "confidence_score": result["confidence"],
                        "timestamp": datetime.now().isoformat(),
                        "manuscript_context": result["manuscript_context"]
                    }
                    try:
                        requests.post(f"{BACKEND_URL}/history", json=db_record, timeout=5)
                    except Exception as db_err:
                        st.warning(f"Gagal mencatat riwayat: {db_err}")

                    # Simpan ke session state
                    st.session_state["page_result"] = result
                    
                    progress_bar.progress(100, text="Selesai!")
                    progress_bar.empty()

                    # Render hasil
                    render_result_banner(result)
                    render_stats(result)
                    render_jawi_found(result["jawi_found"])
                    if result.get("manuscript_context"):
                        render_context_card(result["manuscript_context"], result["manuscript_explanation"])

                    st.success("Analisis selesai! Hasil disimpan ke riwayat.")
                    st.rerun()

                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Terjadi kesalahan saat memproses: {e}")

    # ── Gambar anotasi (lebar penuh) ──────────────────────────────────────
    if "page_result" in st.session_state and uploaded:
        result = st.session_state["page_result"]
        if result.get("total_chars", 0) > 0 and "annotated_image_base64" in result:
            st.markdown("---")
            st.markdown("### Deteksi Karakter")
            st.markdown("""
            <div class="legend">
                <div class="legend-item">
                    <span class="legend-line" style="border-bottom: 3px solid #1a202c;"></span>
                    <span>Huruf Jawi-Spesifik (Tebal Solid)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-line" style="border-bottom: 1px solid #718096;"></span>
                    <span>Huruf Arab Standar (Tipis Solid)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-line" style="border-bottom: 1px dashed #a0aec0;"></span>
                    <span>Confidence Rendah (Putus-Putus)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Decode gambar base64
            img_bytes = base64.b64decode(result["annotated_image_base64"])
            annotated_img = Image.open(io.BytesIO(img_bytes))
            
            st.image(annotated_img,
                     caption="Deteksi bounding box karakter monokrom",
                     use_container_width=True)

    # ── Tabel riwayat lengkap (expandable) ──────────────────────────────────
    st.markdown("---")
    with st.expander("Tabel Riwayat Analisis Lengkap", expanded=False):
        try:
            response = requests.get(f"{BACKEND_URL}/history", timeout=5)
            records = response.json() if response.status_code == 200 else []
        except Exception:
            records = []

        if records:
            df = pd.DataFrame(records)
            df["confidence_score"] = df["confidence_score"].apply(lambda x: f"{x*100:.1f}%")
            df["manuscript_context"] = df["manuscript_context"].fillna("")
            df = df.rename(columns={
                "id": "ID",
                "filename": "File",
                "script_type": "Jenis Naskah",
                "total_chars": "Total Huruf",
                "jawi_chars": "Huruf Jawi",
                "confidence_score": "Kepercayaan",
                "timestamp": "Waktu",
                "manuscript_context": "Konteks"
            })
            
            # Format waktu
            df["Waktu"] = pd.to_datetime(df["Waktu"]).dt.strftime('%H:%M:%S %d/%m/%Y')
            
            st.dataframe(df[["ID", "File", "Jenis Naskah", "Konteks", "Total Huruf", "Huruf Jawi", "Kepercayaan", "Waktu"]], 
                         use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada riwayat analisis.")


if __name__ == "__main__":
    main()
