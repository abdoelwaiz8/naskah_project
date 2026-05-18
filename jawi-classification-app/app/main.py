"""
main.py
-------
Aplikasi Streamlit untuk identifikasi jenis naskah (Arab Jawi / Arab Asli)
dari gambar halaman naskah penuh menggunakan model ResNet34 PyTorch.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
from PIL import Image
from datetime import datetime

import database
import inference
import page_classifier as pc
from utils import load_image_from_upload, format_confidence, get_arabic_char

# ===========================================================================
# Konfigurasi halaman
# ===========================================================================
st.set_page_config(
    page_title="Identifikasi Naskah Jawi",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# CSS
# ===========================================================================
def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Noto+Naskh+Arabic:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Hero */
    .hero-header { text-align:center; padding:2rem 1rem 1rem; }
    .hero-title {
        font-size:2.6rem; font-weight:800;
        background: linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        background-clip:text; margin:0; line-height:1.2;
    }
    .hero-subtitle { font-size:1rem; color:rgba(255,255,255,0.5); margin-top:0.4rem; font-weight:300; }
    .badge {
        display:inline-block;
        background:linear-gradient(135deg,#7c3aed,#3b82f6);
        color:white; font-size:0.7rem; font-weight:600;
        letter-spacing:1.5px; padding:0.25rem 0.85rem;
        border-radius:9999px; margin-bottom:0.8rem; text-transform:uppercase;
    }

    /* Glass card */
    .glass-card {
        background:rgba(255,255,255,0.05);
        backdrop-filter:blur(16px);
        border:1px solid rgba(255,255,255,0.10);
        border-radius:20px; padding:1.6rem; margin-bottom:1rem;
        transition:border-color 0.3s;
    }
    .glass-card:hover { border-color:rgba(167,139,250,0.35); }

    /* Banner hasil */
    .script-banner {
        border-radius:20px; padding:2rem 1.5rem;
        text-align:center; border:2px solid;
        animation:fadeIn 0.5s ease;
    }
    .script-label {
        font-size:0.75rem; font-weight:600; letter-spacing:2.5px;
        text-transform:uppercase; opacity:0.7; margin-bottom:0.4rem;
    }
    .script-type {
        font-size:2.4rem; font-weight:800; line-height:1.2; margin:0.3rem 0;
    }
    .script-conf {
        font-size:0.9rem; margin-top:0.5rem; opacity:0.75;
    }

    /* Stat cards */
    .stat-row { display:flex; gap:0.8rem; margin:1rem 0; flex-wrap:wrap; }
    .stat-card {
        flex:1; min-width:80px;
        background:rgba(255,255,255,0.06);
        border:1px solid rgba(255,255,255,0.10);
        border-radius:14px; padding:0.9rem 1rem; text-align:center;
    }
    .stat-value { font-size:1.6rem; font-weight:700; color:#ffffff; line-height:1; }
    .stat-label { font-size:0.7rem; color:rgba(255,255,255,0.45); margin-top:0.3rem; text-transform:uppercase; letter-spacing:1px; }

    /* Jawi char pills */
    .jawi-pills { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.6rem; }
    .jawi-pill {
        background:rgba(16,185,129,0.15);
        border:1px solid rgba(16,185,129,0.4);
        border-radius:999px; padding:0.35rem 0.9rem;
        font-family:'Noto Naskh Arabic', serif;
        font-size:1.5rem; color:#10b981; direction:rtl;
    }
    .jawi-pill-label { font-size:0.65rem; color:rgba(255,255,255,0.4); display:block; text-align:center; margin-top:0.1rem; }

    /* Legend */
    .legend { display:flex; gap:1rem; font-size:0.75rem; color:rgba(255,255,255,0.5); margin-top:0.6rem; flex-wrap:wrap; }
    .legend-dot { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background:rgba(255,255,255,0.03) !important;
        border:2px dashed rgba(167,139,250,0.4) !important;
        border-radius:16px !important;
    }
    [data-testid="stFileUploader"]:hover { border-color:rgba(167,139,250,0.8) !important; }

    /* Buttons */
    .stButton > button {
        width:100%;
        background:linear-gradient(135deg,#7c3aed 0%,#3b82f6 100%) !important;
        color:white !important; border:none !important;
        border-radius:12px !important; padding:0.75rem 1.5rem !important;
        font-size:1rem !important; font-weight:600 !important;
        transition:all 0.25s ease !important;
        box-shadow:0 4px 20px rgba(124,58,237,0.35) !important;
    }
    .stButton > button:hover {
        transform:translateY(-2px) !important;
        box-shadow:0 8px 30px rgba(124,58,237,0.55) !important;
    }

    /* Progress */
    .stProgress > div > div > div {
        background:linear-gradient(90deg,#7c3aed,#3b82f6) !important;
        border-radius:9999px !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background:rgba(255,255,255,0.06) !important;
        border-radius:14px !important; padding:1rem !important;
        border:1px solid rgba(255,255,255,0.08) !important;
    }
    [data-testid="stMetricLabel"] { color:rgba(255,255,255,0.55) !important; }
    [data-testid="stMetricValue"] { color:#ffffff !important; font-weight:700 !important; }

    h1,h2,h3,h4,h5,h6,p,label,span { color:rgba(255,255,255,0.90) !important; }
    hr  { border-color:rgba(255,255,255,0.1) !important; }
    .stAlert { border-radius:12px !important; }

    .sidebar-title {
        font-size:0.72rem; font-weight:600; text-transform:uppercase;
        letter-spacing:1.5px; color:rgba(255,255,255,0.4) !important; margin-bottom:0.5rem;
    }

    @keyframes fadeIn {
        from { opacity:0; transform:translateY(10px); }
        to   { opacity:1; transform:translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)


# ===========================================================================
# Inisialisasi DB (cached)
# ===========================================================================
def init_database() -> None:
    database.initialize_db()


# ===========================================================================
# Render: Hero
# ===========================================================================
def render_hero() -> None:
    st.markdown("""
    <div class="hero-header">
        <div class="badge">📜 Computer Vision · Identifikasi Naskah</div>
        <h1 class="hero-title">Identifikasi Naskah Arab</h1>
        <p class="hero-subtitle">
            Unggah gambar halaman naskah — sistem akan menentukan apakah<br>
            naskah tersebut <strong>Arab Jawi</strong> atau <strong>Arab Asli</strong>
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
        color  = "#10b981"
        emoji  = "🟢"
        desc   = "Naskah ini menggunakan tulisan Arab Jawi"
    elif stype == "Arab Asli":
        color  = "#3b82f6"
        emoji  = "🔵"
        desc   = "Naskah ini menggunakan tulisan Arab Asli"
    else:
        color  = "#f59e0b"
        emoji  = "⚠️"
        desc   = "Karakter tidak dapat terdeteksi pada gambar ini"

    st.markdown(f"""
    <div class="script-banner glass-card"
         style="border-color:{color}; background:linear-gradient(135deg,{color}18,{color}08);">
        <div class="script-label">Hasil Identifikasi Naskah</div>
        <div style="font-size:2.5rem; margin:0.3rem 0;">{emoji}</div>
        <div class="script-type" style="color:{color};">{stype}</div>
        <div class="script-conf">{desc}</div>
        <div style="margin-top:0.8rem; font-size:0.85rem; color:rgba(255,255,255,0.6);">
            Tingkat Kepercayaan: <strong style="color:{color};">
            {format_confidence(conf)}</strong>
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
    processed = result["processed_chars"]
    jawi_pct  = f"{result['jawi_ratio']*100:.1f}%"

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-value">{total}</div>
            <div class="stat-label">Total Karakter Terdeteksi</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#10b981;">{jawi_c}</div>
            <div class="stat-label">Huruf Jawi Ditemukan</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#60a5fa;">{arab_c}</div>
            <div class="stat-label">Huruf Arab Standar</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#a78bfa;">{jawi_pct}</div>
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

    st.markdown("**✨ Huruf Jawi-Spesifik yang Ditemukan:**")
    pills_html = '<div class="jawi-pills">'
    for item in jawi_found:
        char = item["char"] or item["class"]
        cls  = item["class"]
        pills_html += f"""
        <div style="text-align:center;">
            <span class="jawi-pill">{char}</span>
            <span class="jawi-pill-label">{cls}</span>
        </div>"""
    pills_html += "</div>"
    st.markdown(pills_html, unsafe_allow_html=True)


# ===========================================================================
# Render: Sidebar
# ===========================================================================
def render_sidebar() -> dict:
    """Render sidebar dan kembalikan dict parameter segmentasi."""
    params: dict = {}
    with st.sidebar:
        # Status model
        st.markdown("<div class='sidebar-title'>⚙️ Status Model</div>", unsafe_allow_html=True)
        if inference.is_model_available():
            st.success("✅ Model tersedia", icon="🟢")
        else:
            st.warning("⚠️ Model belum ditemukan.\nLetakkan `resnet34_jawi.pth` di folder `/model/`.", icon="🟡")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Parameter segmentasi
        st.markdown("<div class='sidebar-title'>🔧 Parameter Segmentasi</div>", unsafe_allow_html=True)
        params["min_area"] = st.slider(
            "Ukuran Minimum Karakter (piksel²)",
            min_value=50, max_value=1000, value=300, step=50,
            help="Kontur lebih kecil dari nilai ini diabaikan (filter noise).",
        )
        params["confidence_threshold"] = st.slider(
            "Ambang Kepercayaan",
            min_value=0.30, max_value=0.90, value=0.55, step=0.05,
            help="Prediksi di bawah ambang ini tidak dihitung dalam voting.",
        )
        params["max_chars"] = st.slider(
            "Maks Karakter Diproses",
            min_value=20, max_value=200, value=100, step=10,
            help="Batasi jumlah karakter agar proses lebih cepat.",
        )
        params["use_adaptive"] = st.toggle(
            "Adaptive Thresholding",
            value=True,
            help="Direkomendasikan untuk naskah dengan pencahayaan tidak merata.",
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Riwayat
        st.markdown("<div class='sidebar-title'>📜 Riwayat Identifikasi</div>", unsafe_allow_html=True)
        records = database.fetch_all_records()
        if records:
            df = pd.DataFrame(records)
            df["confidence_score"] = df["confidence_score"].apply(lambda x: f"{x*100:.1f}%")
            df = df.rename(columns={
                "id"              : "ID",
                "filename"        : "File",
                "script_type"     : "Jenis Naskah",
                "total_chars"     : "Total Huruf",
                "jawi_chars"      : "Huruf Jawi",
                "confidence_score": "Kepercayaan",
                "timestamp"       : "Waktu",
            })
            st.dataframe(df[["ID","File","Jenis Naskah","Kepercayaan","Waktu"]],
                         use_container_width=True, hide_index=True)

            col_dl, col_clr = st.columns(2)
            with col_dl:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Export CSV", data=csv,
                                   file_name="riwayat_naskah.csv",
                                   mime="text/csv", use_container_width=True)
            with col_clr:
                if st.button("🗑️ Hapus Semua", use_container_width=True):
                    database.clear_all_records()
                    st.rerun()
        else:
            st.markdown(
                "<p style='color:rgba(255,255,255,0.35);font-size:0.85rem;'>"
                "Belum ada riwayat identifikasi.</p>",
                unsafe_allow_html=True,
            )

    return params


# ===========================================================================
# Halaman utama
# ===========================================================================
def main() -> None:
    inject_css()
    init_database()
    render_hero()
    params = render_sidebar()

    col_left, col_right = st.columns([1.1, 1], gap="large")

    # ── Kolom kiri: upload ────────────────────────────────────────────────
    with col_left:
        st.markdown("### 📤 Unggah Gambar Naskah")
        uploaded = st.file_uploader(
            label="Seret & lepas gambar naskah, atau klik untuk memilih",
            type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
            help="Format: JPG, PNG, WebP, TIFF",
            label_visibility="collapsed",
        )

        if uploaded:
            image = load_image_from_upload(uploaded)
            st.image(image,
                     caption=f"📁 {uploaded.name}  ·  {image.size[0]}×{image.size[1]} px",
                     use_container_width=True)

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Nama File", uploaded.name[:18] + ("…" if len(uploaded.name) > 18 else ""))
            m2.metric("Ukuran",    f"{uploaded.size / 1024:.1f} KB")
            m3.metric("Resolusi",  f"{image.size[0]}×{image.size[1]}")

    # ── Kolom kanan: hasil ────────────────────────────────────────────────
    with col_right:
        st.markdown("### 🔍 Hasil Identifikasi")

        if not uploaded:
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:3rem 1.5rem;">
                <div style="font-size:3.5rem;">📜</div>
                <p style="color:rgba(255,255,255,0.4);margin-top:0.8rem;">
                    Unggah gambar halaman naskah terlebih dahulu<br>
                    untuk memulai identifikasi.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            if not inference.is_model_available():
                st.error(
                    "❌ **File model tidak ditemukan!**\n\n"
                    "Letakkan `resnet34_jawi.pth` di dalam folder `/model/`.",
                    icon="🚫",
                )
            else:
                identify_btn = st.button(
                    "📜 Identifikasi Naskah", type="primary", use_container_width=True,
                )

                # Tampilkan hasil tersimpan di session state
                if "page_result" in st.session_state and not identify_btn:
                    render_result_banner(st.session_state["page_result"])
                    render_stats(st.session_state["page_result"])
                    render_jawi_found(st.session_state["page_result"]["jawi_found"])

                if identify_btn:
                    progress_bar = st.progress(0, text="Mempersiapkan analisis…")

                    try:
                        progress_bar.progress(10, text="Melakukan segmentasi karakter…")

                        result = pc.classify_page(
                            image,
                            min_area             = params["min_area"],
                            confidence_threshold = params["confidence_threshold"],
                            max_chars            = params["max_chars"],
                            use_adaptive         = params["use_adaptive"],
                        )

                        progress_bar.progress(90, text="Menghitung hasil…")
                        st.session_state["page_result"] = result

                        # Simpan ke database
                        database.insert_record(
                            filename         = uploaded.name,
                            script_type      = result["script_type"],
                            total_chars      = result["total_chars"],
                            jawi_chars       = result["jawi_chars"],
                            confidence_score = result["confidence"],
                            timestamp        = datetime.now(),
                        )

                        progress_bar.progress(100, text="Selesai!")
                        progress_bar.empty()

                        render_result_banner(result)
                        render_stats(result)
                        render_jawi_found(result["jawi_found"])

                        st.success("✅ Identifikasi selesai! Hasil disimpan ke riwayat.", icon="💾")

                    except Exception as e:
                        progress_bar.empty()
                        st.error(f"❌ Terjadi kesalahan: {e}")

    # ── Gambar anotasi (lebar penuh) ──────────────────────────────────────
    if "page_result" in st.session_state and uploaded:
        result = st.session_state["page_result"]
        if result["total_chars"] > 0:
            st.markdown("---")
            st.markdown("### 🖼️ Visualisasi Deteksi Karakter")
            st.markdown("""
            <div class="legend">
                <span><span class="legend-dot" style="background:#10b981;"></span>Huruf Jawi-spesifik</span>
                <span><span class="legend-dot" style="background:#60a5fa;"></span>Huruf Arab standar</span>
                <span><span class="legend-dot" style="background:#6b7280;"></span>Confidence rendah</span>
            </div>
            """, unsafe_allow_html=True)
            st.image(result["annotated_image"],
                     caption="Bounding box: hijau = Jawi, biru = Arab standar",
                     use_container_width=True)

    # ── Tabel riwayat (expandable) ────────────────────────────────────────
    st.markdown("---")
    with st.expander("📊 Tabel Riwayat Identifikasi Lengkap", expanded=False):
        records = database.fetch_all_records()
        if records:
            df = pd.DataFrame(records)
            df["confidence_score"] = df["confidence_score"].apply(lambda x: f"{x*100:.1f}%")
            df.columns = ["ID", "File", "Jenis Naskah", "Total Huruf",
                          "Huruf Jawi", "Kepercayaan", "Waktu"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("📭 Belum ada riwayat identifikasi.", icon="ℹ️")


if __name__ == "__main__":
    main()
