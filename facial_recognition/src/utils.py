"""
Utility functions for dataset inspection, path resolution, and pair validation.
Project: Facial Recognition (ArcFace vs FaceNet) - LFW Dataset
"""

import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image


def get_project_dir() -> Path:
    """Mengembalikan path direktori root project facial_recognition."""
    return Path(__file__).resolve().parent.parent


def get_dataset_dir() -> Path:
    """
    Mengembalikan path absolut direktori dataset bersama (CV/dataset/).
    Menggunakan navigasi path relatif ../dataset/ dari root project facial_recognition
    dan mendukung override melalui environment variable DATASET_DIR.
    """
    project_dir = get_project_dir()
    default_dataset = (project_dir.parent / "dataset").resolve()
    env_dataset = os.getenv("DATASET_DIR")
    if env_dataset:
        return Path(env_dataset).resolve()
    return default_dataset


def inspect_raw_dataset(dataset_dir: Path) -> dict:
    """
    Melakukan inspeksi menyeluruh terhadap dataset LFW di folder dataset_dir.
    Memeriksa:
    - Jumlah identity/orang
    - Jumlah total gambar (dan rincian per subfolder train, test, database)
    - Distribusi jumlah gambar per identity
    - Format gambar
    - Ukuran resolusi gambar
    - Identity dengan jumlah gambar < 2
    - Pengecekan file gambar rusak (corrupted)
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Direktori dataset tidak ditemukan di: {dataset_dir}")

    subfolders = ["train", "test", "database"]
    folder_stats = {}
    all_identities = set()
    total_images_all = 0
    unique_identities_total_counts = defaultdict(int)
    formats = set()
    resolutions = set()
    corrupted_files = []

    for sub in subfolders:
        sub_path = dataset_dir / sub
        if not sub_path.exists():
            continue

        id_counts = {}
        for id_dir in sorted(sub_path.iterdir()):
            if not id_dir.is_dir():
                continue
            all_identities.add(id_dir.name)
            img_files = list(id_dir.glob("*.jpg")) + list(id_dir.glob("*.jpeg")) + list(id_dir.glob("*.png"))
            n_imgs = len(img_files)
            id_counts[id_dir.name] = n_imgs
            total_images_all += n_imgs

            if sub in ["train", "test"]:  # menghitung gambar unik non-duplikat database
                unique_identities_total_counts[id_dir.name] += n_imgs

            for img_p in img_files:
                try:
                    with Image.open(img_p) as im:
                        formats.add(im.format)
                        resolutions.add(im.size)
                        im.verify()
                except Exception as e:
                    corrupted_files.append((str(img_p.relative_to(dataset_dir)), str(e)))

        folder_stats[sub] = {
            "num_identities": len(id_counts),
            "total_images": sum(id_counts.values()),
            "id_counts": id_counts,
        }

    # Cek identity dengan jumlah gambar < 2 (berdasarkan train+test)
    ids_less_than_2 = [
        k for k, v in unique_identities_total_counts.items() if v < 2
    ]

    return {
        "dataset_dir": str(dataset_dir),
        "total_identities": len(all_identities),
        "total_images_raw": total_images_all,
        "total_unique_images": sum(unique_identities_total_counts.values()),
        "folder_stats": folder_stats,
        "unique_identities_counts": dict(unique_identities_total_counts),
        "formats": sorted(list(formats)),
        "resolutions": sorted(list(resolutions)),
        "identities_less_than_2": ids_less_than_2,
        "corrupted_files": corrupted_files,
        "corrupted_count": len(corrupted_files),
    }


def validate_pairs_csv(csv_path: Path, dataset_dir: Path) -> dict:
    """
    Validasi integritas file pairs.csv untuk Face Verification.
    Memeriksa:
    - Keberadaan file dan kolom-kolom wajib
    - Validitas path gambar dan kemampuan gambar untuk dibaca
    - Kebenaran label (label=1 jika id1==id2, label=0 jika id1!=id2)
    - Tidak ada self-pair (image1 == image2)
    - Tidak ada duplicate pair (unordered pair unik)
    - Keseimbangan jumlah SAME vs DIFFERENT
    - Keseimbangan jumlah development vs test
    - Data leakage: tidak ada irisan identity dan gambar antara development dan test.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"File pairs.csv tidak ditemukan di: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ["image1", "image2", "identity1", "identity2", "label", "split"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib hilang pada pairs.csv: {missing_cols}")

    total_pairs = len(df)
    dev_df = df[df["split"] == "development"]
    test_df = df[df["split"] == "test"]

    # 1. Keseimbangan pasangan
    dev_same = len(dev_df[dev_df["label"] == 1])
    dev_diff = len(dev_df[dev_df["label"] == 0])
    test_same = len(test_df[test_df["label"] == 1])
    test_diff = len(test_df[test_df["label"] == 0])

    # 2. Cek self-pair (image1 == image2)
    self_pairs = df[df["image1"] == df["image2"]]
    self_pair_count = len(self_pairs)

    # 3. Cek duplicate pair (tanpa memandang urutan image1 dan image2)
    unordered_pairs = [tuple(sorted([r["image1"], r["image2"]])) for _, r in df.iterrows()]
    unique_pair_count = len(set(unordered_pairs))
    duplicate_pair_count = total_pairs - unique_pair_count

    # 4. Cek kebenaran label
    label_same_errors = df[(df["label"] == 1) & (df["identity1"] != df["identity2"])]
    label_diff_errors = df[(df["label"] == 0) & (df["identity1"] == df["identity2"])]
    label_error_count = len(label_same_errors) + len(label_diff_errors)

    # 5. Cek keberadaan file gambar dan pembacaan
    all_referenced_images = set(df["image1"]).union(set(df["image2"]))
    missing_images = []
    unreadable_images = []

    for rel_path in all_referenced_images:
        full_p = dataset_dir / rel_path
        if not full_p.exists():
            missing_images.append(rel_path)
        else:
            try:
                with Image.open(full_p) as im:
                    im.verify()
            except Exception as e:
                unreadable_images.append((rel_path, str(e)))

    # 6. Cek Data Leakage (Identity & Image Disjointness)
    dev_identities = set(dev_df["identity1"]).union(set(dev_df["identity2"]))
    test_identities = set(test_df["identity1"]).union(set(test_df["identity2"]))
    identity_overlap = dev_identities.intersection(test_identities)

    dev_images = set(dev_df["image1"]).union(set(dev_df["image2"]))
    test_images = set(test_df["image1"]).union(set(test_df["image2"]))
    image_overlap = dev_images.intersection(test_images)

    is_valid = (
        self_pair_count == 0
        and duplicate_pair_count == 0
        and label_error_count == 0
        and len(missing_images) == 0
        and len(unreadable_images) == 0
        and len(identity_overlap) == 0
        and len(image_overlap) == 0
    )

    return {
        "is_valid": is_valid,
        "total_pairs": total_pairs,
        "development_pairs": len(dev_df),
        "test_pairs": len(test_df),
        "dev_same": dev_same,
        "dev_diff": dev_diff,
        "test_same": test_same,
        "test_diff": test_diff,
        "total_same": dev_same + test_same,
        "total_diff": dev_diff + test_diff,
        "self_pair_count": self_pair_count,
        "duplicate_pair_count": duplicate_pair_count,
        "label_error_count": label_error_count,
        "missing_images_count": len(missing_images),
        "unreadable_images_count": len(unreadable_images),
        "dev_identities_count": len(dev_identities),
        "test_identities_count": len(test_identities),
        "identity_overlap_count": len(identity_overlap),
        "image_overlap_count": len(image_overlap),
        "identity_overlap": sorted(list(identity_overlap)),
        "image_overlap": sorted(list(image_overlap)),
    }



def compute_cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """
    Menghitung Cosine Distance antara dua vektor embedding:
    distance = 1 - (u . v) / (||u|| * ||v||)
    Keterangan: Nilai yang lebih kecil berarti representasi wajah LEBIH MIRIP (sudut lebih kecil).
    """
    dot = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 1.0
    cosine_sim = dot / (norm_u * norm_v)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    return float(1.0 - cosine_sim)


def sanitize_path_to_filename(rel_path: str) -> str:
    """Mengubah path relatif seperti 'train/George_W_Bush/0001.jpg' menjadi nama file yang aman."""
    return rel_path.replace("/", "__").replace("\\", "__") + ".npy"


def find_optimal_threshold(distances: np.ndarray, labels: np.ndarray, criterion: str = "balanced_accuracy"):
    """
    Mencari threshold optimal HANYA pada development set.
    Mendukung kriteria: 'balanced_accuracy' (default, memaksimalkan 1 - HTER) atau 'f1'.
    Karena metric adalah cosine distance, prediksi = 1 jika distance <= threshold.
    """
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    candidate_thresholds = np.linspace(distances.min(), distances.max(), num=1000)

    best_thresh = candidate_thresholds[0]
    best_score = -1.0
    best_metrics = {}

    for t in candidate_thresholds:
        preds = (distances <= t).astype(int)
        acc = float(np.mean(preds == labels))
        prec = float(precision_score(labels, preds, zero_division=0))
        rec = float(recall_score(labels, preds, zero_division=0))
        f1 = float(f1_score(labels, preds, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        frr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        bacc = float((rec + (1.0 - far)) / 2.0)

        score = bacc if criterion == "balanced_accuracy" else f1

        if score > best_score:
            best_score = score
            best_thresh = float(t)
            best_metrics = {
                "threshold": round(best_thresh, 4),
                "accuracy": round(acc, 4),
                "balanced_accuracy": round(bacc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "far": round(far, 4),
                "frr": round(frr, 4),
                "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            }

    return best_thresh, best_metrics


def evaluate_split(distances: np.ndarray, labels: np.ndarray, threshold: float):
    """Mengevaluasi performa prediksi dengan threshold yang sudah ditentukan sebelumnya."""
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    preds = (distances <= threshold).astype(int)
    acc = float(np.mean(preds == labels))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    frr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return preds, {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "far": round(far, 4),
        "frr": round(frr, 4),
        "threshold": round(threshold, 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def compute_roc_and_eer(distances: np.ndarray, labels: np.ndarray):
    """
    Menghitung kurva ROC, AUC, dan Equal Error Rate (EER).
    Skor kemiripan = 1 - distance (nilai lebih tinggi = lebih yakin SAME).
    """
    from sklearn.metrics import auc, roc_curve

    scores = 1.0 - distances
    fpr, tpr, roc_threshs = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))

    # EER dihitung pada titik di mana FPR ~ FNR (FNR = 1 - TPR)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_thresh = float(roc_threshs[eer_idx])

    return fpr, tpr, roc_auc, eer, eer_thresh


def plot_roc_curve(fpr, tpr, roc_auc, eer, output_path: Path, model_name: str = "FaceNet"):
    """Membuat dan menyimpan visualisasi ROC Curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr, tpr, color="#059669", lw=2.5, label=f"{model_name} ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    plt.scatter([eer], [1.0 - eer], color="#DC2626", s=60, zorder=5, label=f"EER = {eer:.4f}")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (FAR)", fontsize=12)
    plt.ylabel("True Positive Rate (1 - FRR)", fontsize=12)
    plt.title(f"Receiver Operating Characteristic (ROC) - {model_name} on LFW Test Set", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    dataset_dir = get_dataset_dir()
    print("Running Inspection on:", dataset_dir)
    report = inspect_raw_dataset(dataset_dir)
    print(f"Identities: {report['total_identities']}")
    print(f"Total Unique Images: {report['total_unique_images']}")
    print(f"Corrupted: {report['corrupted_count']}")
