"""
Script Persiapan Dataset Kelas Mahasiswa untuk Face Verification.
Project: Face Verification (ArcFace vs FaceNet) - Dataset Kelas

Fitur:
1. Normalisasi dan ekstraksi gambar dari arsip ZIP di CV/dataset/kelas/ ke dataset/kelas/extracted/.
2. Pemotongan wajah terpusat (aligned face crops 250x250) ke dataset/kelas/cropped_faces/
   menggunakan RetinaFace (konsisten dengan format 250x250 LFW).
3. Pembentukan 200 pasangan verifikasi wajah (100 SAME, 100 DIFFERENT) dari 33 mahasiswa.
4. Penyimpanan pasangan ke dataset/processed/kelas_pairs.csv secara mandiri tanpa menimpa pairs.csv LFW.
"""

import itertools
import json
import os
import random
import re
import sys
import zipfile
import io
from pathlib import Path
from collections import defaultdict

# Konfigurasi encoding UTF-8 untuk output Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import pillow_heif
pillow_heif.register_heif_opener()

# DeepFace untuk deteksi wajah presisi
from deepface import DeepFace

# Path project & dataset
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATASET_DIR = Path(os.getenv("DATASET_DIR", PROJECT_DIR.parent / "dataset")).resolve()
KELAS_DIR = DATASET_DIR / "kelas"
EXTRACTED_DIR = KELAS_DIR / "extracted"
CROPPED_DIR = KELAS_DIR / "cropped_faces"
PROCESSED_DIR = PROJECT_DIR / "dataset" / "processed"
OUTPUT_PAIRS_CSV = PROCESSED_DIR / "kelas_pairs.csv"


def clean_identity_name(raw_name: str) -> str:
    """Membersihkan prefix NIM dan karakter spesial dari nama folder mahasiswa."""
    name = re.sub(r'^[A-Z0-9\._]+_', '', raw_name)
    name = name.strip().replace(' ', '_')
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    return name


def extract_zip_archives():
    """Mengekstrak dan menstandarisasi seluruh gambar dari zip ke format JPEG RGB."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    zip_files = sorted(list(KELAS_DIR.glob("*.zip")))
    if not zip_files:
        print("Tidak ada file ZIP di folder kelas.")
        return

    print(f"Mengekstrak {len(zip_files)} file ZIP dari {KELAS_DIR}...")
    identities_counts = defaultdict(int)

    for zp in zip_files:
        with zipfile.ZipFile(zp, 'r') as z:
            for info in z.infolist():
                if info.is_dir() or info.filename.lower().endswith(".docx"):
                    continue
                parts = info.filename.split('/')
                if len(parts) >= 3:
                    raw_id = parts[1]
                elif len(parts) == 2:
                    raw_id = parts[0]
                else:
                    continue

                clean_id = clean_identity_name(raw_id)
                if not clean_id or clean_id.lower() in ["dataset", "dataface"]:
                    continue

                try:
                    data = z.read(info.filename)
                    im = Image.open(io.BytesIO(data)).convert("RGB")
                    id_dir = EXTRACTED_DIR / clean_id
                    id_dir.mkdir(parents=True, exist_ok=True)

                    identities_counts[clean_id] += 1
                    out_name = f"{clean_id}_{identities_counts[clean_id]:04d}.jpg"
                    out_path = id_dir / out_name
                    if not out_path.exists():
                        im.save(out_path, "JPEG", quality=95)
                except Exception as e:
                    print(f"  [Peringatan] Gagal membaca {info.filename}: {e}")

    print(f"Ekstraksi selesai: {sum(identities_counts.values())} citra dari {len(identities_counts)} identitas.")


def crop_face_aligned(img_pil: Image.Image, target_size=(250, 250)) -> Image.Image:
    """
    Mendeteksi wajah menggunakan RetinaFace dan meng-crop area wajah (250x250).
    Jika deteksi gagal, melakukan center square crop terpusat.
    """
    # Resize sementara untuk deteksi cepat jika resolusi sangat besar (> 1280)
    w_orig, h_orig = img_pil.size
    max_dim = max(w_orig, h_orig)
    scale = 1.0
    if max_dim > 1280:
        scale = 1280.0 / max_dim
        detect_im = img_pil.resize((int(w_orig * scale), int(h_orig * scale)), Image.Resampling.BILINEAR)
    else:
        detect_im = img_pil

    arr = np.array(detect_im)
    try:
        faces = DeepFace.extract_faces(
            img_path=arr,
            detector_backend="retinaface",
            enforce_detection=False,
            align=True,
        )
        if faces and len(faces) > 0 and faces[0]["confidence"] > 0.5:
            face_arr = faces[0]["face"]
            if face_arr.dtype in [np.float32, np.float64]:
                face_arr = np.clip(face_arr * 255.0, 0, 255).astype(np.uint8)
            crop_im = Image.fromarray(face_arr).convert("RGB")
            return crop_im.resize(target_size, Image.Resampling.LANCZOS)
    except Exception:
        pass

    # Fallback jika deteksi wajah tidak menemukan area terpercaya: center-crop
    min_edge = min(w_orig, h_orig)
    left = (w_orig - min_edge) // 2
    top = (h_orig - min_edge) // 4  # sedikit ke atas agar fokus wajah
    right = left + min_edge
    bottom = top + min_edge
    if bottom > h_orig:
        bottom = h_orig
        top = bottom - min_edge
    crop_im = img_pil.crop((left, top, right, bottom))
    return crop_im.resize(target_size, Image.Resampling.LANCZOS)


def generate_cropped_faces():
    """Memotong wajah untuk seluruh citra mahasiswa ke dataset/kelas/cropped_faces/."""
    CROPPED_DIR.mkdir(parents=True, exist_ok=True)
    all_id_dirs = sorted([d for d in EXTRACTED_DIR.iterdir() if d.is_dir()])
    total_imgs = sum(len(list(d.glob("*.jpg"))) for d in all_id_dirs)
    print(f"\nMemproses pemotongan wajah terpusat (250x250) untuk {total_imgs} citra...")

    processed = 0
    for id_dir in all_id_dirs:
        out_id_dir = CROPPED_DIR / id_dir.name
        out_id_dir.mkdir(parents=True, exist_ok=True)
        for img_p in sorted(id_dir.glob("*.jpg")):
            out_p = out_id_dir / img_p.name
            if not out_p.exists():
                try:
                    with Image.open(img_p) as im:
                        cropped = crop_face_aligned(im, target_size=(250, 250))
                        cropped.save(out_p, "JPEG", quality=95)
                except Exception as e:
                    print(f"  Gagal memproses {img_p.name}: {e}")
            processed += 1
            if processed % 30 == 0 or processed == total_imgs:
                print(f"  [Progress] {processed}/{total_imgs} citra diproses...")

    print("Pemotongan wajah selesai!")


def generate_verification_pairs(total_pairs: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Membangun pasangan verifikasi wajah (SAME dan DIFFERENT) dari citra cropped_faces.
    - 50% SAME (label=1)
    - 50% DIFFERENT (label=0)
    """
    rng = random.Random(seed)
    all_id_dirs = sorted([d for d in CROPPED_DIR.iterdir() if d.is_dir()])

    # Kumpulkan daftar file relatif per identitas
    id_images = {}
    for id_dir in all_id_dirs:
        imgs = sorted([f"kelas/cropped_faces/{id_dir.name}/{p.name}" for p in id_dir.glob("*.jpg")])
        if len(imgs) >= 2:
            id_images[id_dir.name] = imgs

    identities = sorted(list(id_images.keys()))
    print(f"\nIdentitas valid dengan minimal 2 foto: {len(identities)}")

    target_same = total_pairs // 2
    target_diff = total_pairs - target_same

    # 1. Pasangan SAME
    same_pool = []
    for id_name in identities:
        pairs = list(itertools.combinations(id_images[id_name], 2))
        rng.shuffle(pairs)
        for p1, p2 in pairs:
            same_pool.append({
                "image1": p1,
                "image2": p2,
                "identity1": id_name,
                "identity2": id_name,
                "label": 1,
                "split": "kelas_test",
            })

    rng.shuffle(same_pool)
    chosen_same = same_pool[:target_same]

    # 2. Pasangan DIFFERENT
    diff_pool = []
    id_combos = list(itertools.combinations(identities, 2))
    rng.shuffle(id_combos)

    diff_pairs_set = set()
    attempts = 0
    while len(diff_pool) < target_diff and attempts < 5000:
        for id1, id2 in id_combos:
            if len(diff_pool) >= target_diff:
                break
            im1 = rng.choice(id_images[id1])
            im2 = rng.choice(id_images[id2])
            pair_key = tuple(sorted([im1, im2]))
            if pair_key not in diff_pairs_set:
                diff_pairs_set.add(pair_key)
                diff_pool.append({
                    "image1": im1,
                    "image2": im2,
                    "identity1": id1,
                    "identity2": id2,
                    "label": 0,
                    "split": "kelas_test",
                })
        attempts += 1

    all_pairs = chosen_same + diff_pool
    rng.shuffle(all_pairs)
    df = pd.DataFrame(all_pairs)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PAIRS_CSV, index=False)
    print(f"Berhasil membuat {len(df)} pasangan verifikasi:")
    print(f"  - Pasangan SAME (label=1)     : {len(chosen_same)}")
    print(f"  - Pasangan DIFFERENT (label=0): {len(diff_pool)}")
    print(f"  - Tersimpan di                : {OUTPUT_PAIRS_CSV}")

    return df


def main():
    print("=" * 60)
    print("INTEGRASI DATASET KELAS & PEMBUATAN PASANGAN VERIFIKASI")
    print("=" * 60)
    extract_zip_archives()
    generate_cropped_faces()
    generate_verification_pairs(total_pairs=200, seed=42)
    print("\n[SUKSES] Dataset kelas siap digunakan untuk evaluasi ArcFace & FaceNet.")


if __name__ == "__main__":
    main()
