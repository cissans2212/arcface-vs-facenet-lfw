"""
Konfigurasi Path & Dataset untuk Project Facial Recognition
Path dataset dikonfigurasi secara relatif (../dataset/ relatif terhadap folder facial_recognition/)
sehingga aman dan portabel tanpa hard-code path Windows spesifik.
"""

import os
from pathlib import Path

# Root folder project: facial_recognition/
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Path dataset bersama: '../dataset/' relatif terhadap folder project facial_recognition/
# Mendukung override lewat environment variable DATASET_DIR jika dibutuhkan di lingkungan berbeda
DEFAULT_DATASET_PATH = (PROJECT_DIR.parent / "dataset").resolve()
DATASET_DIR = Path(os.getenv("DATASET_DIR", str(DEFAULT_DATASET_PATH))).resolve()

# Subdirektori dataset LFW yang telah ada
DATABASE_DIR = DATASET_DIR / "database"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Subdirektori dataset kelas (data foto mahasiswa mandiri)
KELAS_DIR = DATASET_DIR / "kelas"

# Direktori internal project facial_recognition
SRC_DIR = PROJECT_DIR / "src"
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
RESULTS_DIR = PROJECT_DIR / "results"
DOCS_DIR = PROJECT_DIR / "docs"
APP_DIR = PROJECT_DIR / "app"

def get_dataset_status():
    """Mengembalikan status keberadaan dataset bersama."""
    return {
        "dataset_root": str(DATASET_DIR),
        "exists": DATASET_DIR.exists(),
        "database_exists": DATABASE_DIR.exists(),
        "train_exists": TRAIN_DIR.exists(),
        "test_exists": TEST_DIR.exists(),
        "kelas_exists": KELAS_DIR.exists(),
    }

if __name__ == "__main__":
    status = get_dataset_status()
    print("Dataset Status:")
    for key, val in status.items():
        print(f"  {key}: {val}")
