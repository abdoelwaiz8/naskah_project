"""
database.py
-----------
Modul untuk mengelola koneksi SQLite, pembuatan tabel, dan operasi CRUD.
Database file: /data/history.db

Schema v2 (page-level classification):
    - id               INTEGER PRIMARY KEY
    - filename         TEXT
    - script_type      TEXT    ("Arab Jawi" | "Arab Asli" | "Tidak Terdeteksi")
    - total_chars      INTEGER (jumlah karakter terdeteksi)
    - jawi_chars       INTEGER (jumlah karakter Jawi yang ditemukan)
    - confidence_score REAL
    - timestamp        DATETIME
"""

import sqlite3
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Path database
# ---------------------------------------------------------------------------
DB_DIR  = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "history.db"


# ---------------------------------------------------------------------------
# Koneksi
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """Membuat dan mengembalikan koneksi ke database SQLite."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Inisialisasi & migrasi
# ---------------------------------------------------------------------------

def initialize_db() -> None:
    """
    Membuat tabel `classification_history` jika belum ada (schema v2).
    Kemudian menjalankan migrasi untuk database lama (schema v1).
    """
    create_sql = """
    CREATE TABLE IF NOT EXISTS classification_history (
        id               INTEGER  PRIMARY KEY AUTOINCREMENT,
        filename         TEXT     NOT NULL,
        script_type      TEXT     NOT NULL DEFAULT '',
        total_chars      INTEGER  NOT NULL DEFAULT 0,
        jawi_chars       INTEGER  NOT NULL DEFAULT 0,
        confidence_score REAL     NOT NULL DEFAULT 0.0,
        timestamp        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """
    with _get_connection() as conn:
        conn.execute(create_sql)
        conn.commit()

    _migrate_db()


def _migrate_db() -> None:
    """
    Migrasi database v1 (predicted_class NOT NULL) ke schema v2.

    SQLite tidak mendukung ALTER COLUMN, sehingga migrasi dilakukan dengan:
        1. Deteksi kolom yang ada via PRAGMA.
        2. Jika schema lama (ada predicted_class, belum ada script_type):
           - Rename tabel lama ke _old
           - Buat tabel baru dengan schema v2
           - Salin data lama ke tabel baru (predicted_class → script_type)
           - Drop tabel lama
        3. Jika hanya kolom baru yang kurang, gunakan ALTER TABLE ADD COLUMN.
    """
    with _get_connection() as conn:
        cursor = conn.execute("PRAGMA table_info(classification_history)")
        columns = {row[1] for row in cursor.fetchall()}

        if "predicted_class" in columns and "script_type" not in columns:
            # ── Migrasi penuh: schema lama → schema v2 ────────────────────
            conn.executescript("""
                BEGIN;

                ALTER TABLE classification_history
                    RENAME TO classification_history_old;

                CREATE TABLE classification_history (
                    id               INTEGER  PRIMARY KEY AUTOINCREMENT,
                    filename         TEXT     NOT NULL,
                    script_type      TEXT     NOT NULL DEFAULT '',
                    total_chars      INTEGER  NOT NULL DEFAULT 0,
                    jawi_chars       INTEGER  NOT NULL DEFAULT 0,
                    confidence_score REAL     NOT NULL DEFAULT 0.0,
                    timestamp        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                INSERT INTO classification_history
                    (id, filename, script_type, total_chars, jawi_chars,
                     confidence_score, timestamp)
                SELECT
                    id,
                    filename,
                    COALESCE(predicted_class, ''),
                    0,
                    0,
                    COALESCE(confidence_score, 0.0),
                    timestamp
                FROM classification_history_old;

                DROP TABLE classification_history_old;

                COMMIT;
            """)

        else:
            # ── Tambah kolom yang belum ada (schema v2 parsial) ───────────
            for sql in [
                "ALTER TABLE classification_history ADD COLUMN script_type TEXT DEFAULT ''",
                "ALTER TABLE classification_history ADD COLUMN total_chars INTEGER DEFAULT 0",
                "ALTER TABLE classification_history ADD COLUMN jawi_chars  INTEGER DEFAULT 0",
            ]:
                try:
                    conn.execute(sql)
                except Exception:
                    pass  # Kolom sudah ada
            conn.commit()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def insert_record(
    filename        : str,
    script_type     : str,
    total_chars     : int,
    jawi_chars      : int,
    confidence_score: float,
    timestamp       : datetime | None = None,
) -> int:
    """
    Menyimpan satu record hasil klasifikasi halaman ke database.

    Returns:
        ID baris yang baru dimasukkan.
    """
    if timestamp is None:
        timestamp = datetime.now()

    sql = """
    INSERT INTO classification_history
        (filename, script_type, total_chars, jawi_chars, confidence_score, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    with _get_connection() as conn:
        cursor = conn.execute(
            sql,
            (filename, script_type, total_chars, jawi_chars, confidence_score, timestamp),
        )
        conn.commit()
        return cursor.lastrowid


def fetch_all_records() -> list[dict]:
    """
    Mengambil semua record dari tabel, diurutkan dari yang terbaru.

    Returns:
        List of dict, masing-masing merepresentasikan satu baris.
    """
    sql = """
    SELECT id, filename, script_type, total_chars, jawi_chars, confidence_score, timestamp
    FROM classification_history
    ORDER BY timestamp DESC
    """
    with _get_connection() as conn:
        rows = conn.execute(sql).fetchall()
        return [dict(row) for row in rows]


def delete_record(record_id: int) -> None:
    """Menghapus satu record berdasarkan ID."""
    with _get_connection() as conn:
        conn.execute(
            "DELETE FROM classification_history WHERE id = ?", (record_id,)
        )
        conn.commit()


def clear_all_records() -> None:
    """Menghapus semua record dari tabel (reset history)."""
    with _get_connection() as conn:
        conn.execute("DELETE FROM classification_history")
        conn.commit()
