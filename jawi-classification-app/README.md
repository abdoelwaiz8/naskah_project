# 🔤 Jawi Script Classifier

Aplikasi web klasifikasi karakter Jawi dan Arab dari naskah kuno menggunakan **ResNet34 PyTorch** dan **Streamlit**.

---

## 📁 Struktur Proyek

```
jawi-classification-app/
├── app/
│   ├── main.py          # Frontend Streamlit (entry point)
│   ├── inference.py     # Logika pemuatan model & prediksi PyTorch
│   ├── database.py      # Koneksi SQLite & operasi CRUD
│   └── utils.py         # Fungsi-fungsi pembantu
├── model/
│   └── resnet34_jawi.pth  ← Letakkan file bobot di sini
├── data/
│   └── history.db         (auto-generated saat pertama kali dijalankan)
├── requirements.txt
└── README.md
```

---

## ⚙️ Persyaratan Sistem

- **Python** 3.10 atau lebih baru
- **pip** (package manager Python)
- **GPU NVIDIA** (opsional, untuk inferensi lebih cepat)

---

## 🚀 Cara Setup & Menjalankan

### 1. Clone atau buka folder proyek

```powershell
cd "d:\Project Python\jawi-classification-app"
```

### 2. Buat dan aktifkan virtual environment

```powershell
# Buat venv
python -m venv venv

# Aktifkan (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Aktifkan (Windows CMD)
venv\Scripts\activate.bat
```

### 3. Install dependensi

```powershell
pip install -r requirements.txt
```

> **Catatan GPU:** Jika Anda memiliki GPU NVIDIA dan ingin menggunakan CUDA, install PyTorch dengan CUDA support:
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Letakkan file model

Salin file bobot model Anda ke folder `/model/`:

```
jawi-classification-app/
└── model/
    └── resnet34_jawi.pth   ← file ini
```

### 5. Sesuaikan kelas output (jika perlu)

Buka `app/inference.py` dan perbarui variabel berikut sesuai dataset Anda:

```python
CLASS_NAMES: list[str] = [
    "alif", "ba", "ta", ...   # Sesuaikan dengan label kelas Anda
]
NUM_CLASSES: int = len(CLASS_NAMES)  # Otomatis terhitung
```

### 6. Jalankan aplikasi

```powershell
streamlit run app/main.py
```

Aplikasi akan terbuka otomatis di browser: **http://localhost:8501**

---

## 🎯 Fitur Aplikasi

| Fitur | Deskripsi |
|-------|-----------|
| 📤 **Upload Gambar** | Unggah JPG, PNG, atau WebP |
| 🔍 **Klasifikasi** | Prediksi kelas karakter Jawi/Arab |
| 📊 **Top-K Prediksi** | Tampil 3 prediksi teratas dengan skor |
| 💾 **Simpan Riwayat** | Hasil disimpan otomatis ke SQLite |
| 📜 **Lihat Riwayat** | Tabel riwayat di sidebar & expander |
| ⬇️ **Export CSV** | Download riwayat sebagai file CSV |
| 🗑️ **Reset History** | Hapus semua riwayat dengan satu klik |

---

## 🧠 Detail Model

- **Arsitektur**: ResNet34 (torchvision)
- **Input**: Gambar 224×224 px, 3 channel (RGB)
- **Preprocessing**:
  - Resize ke 256 px
  - CenterCrop ke 224×224 px
  - Normalisasi ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- **Output**: Probabilitas softmax untuk setiap kelas karakter

---

## 🗄️ Database

File database SQLite tersimpan di `data/history.db` dan dibuat otomatis.

**Schema tabel `classification_history`:**

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `id` | INTEGER | Primary Key (autoincrement) |
| `filename` | TEXT | Nama file gambar |
| `predicted_class` | TEXT | Kelas yang diprediksi |
| `confidence_score` | REAL | Skor kepercayaan (0.0–1.0) |
| `timestamp` | DATETIME | Waktu klasifikasi |

---

## 🛠️ Troubleshooting

**Model tidak ditemukan:**
```
FileNotFoundError: File bobot model tidak ditemukan di: .../model/resnet34_jawi.pth
```
→ Pastikan file `.pth` ada di folder `/model/`.

**Jumlah kelas tidak cocok (size mismatch):**
```
RuntimeError: Error(s) in loading state_dict...
```
→ Sesuaikan `NUM_CLASSES` dan `CLASS_NAMES` di `app/inference.py` dengan model Anda.

**Port sudah digunakan:**
```powershell
streamlit run app/main.py --server.port 8502
```

---

## 📝 Lisensi

Proyek ini bersifat open-source dan bebas digunakan untuk keperluan akademik dan penelitian.
