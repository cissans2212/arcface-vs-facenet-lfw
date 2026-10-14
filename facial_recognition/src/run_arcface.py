"""
Script Pengujian Face Verification menggunakan ArcFace (Tahap 2).
- Framework: DeepFace (Pretrained ArcFace)
- Dataset: LFW pairs dari dataset/processed/pairs.csv
- Detektor: 'skip' (citra LFW sudah terpusat/cropped, fokus murni pada facial recognition)
- Metrik jarak: Cosine Distance (nilai lebih kecil = lebih mirip)
- Threshold: Dioptimalkan HANYA pada development set (800 pasang), lalu dibekukan untuk test set (200 pasang)
- Evaluasi: Accuracy, Precision, Recall, F1, Confusion Matrix, FAR, FRR, ROC-AUC, EER
- Output: embeddings/, arcface_results.csv, metrics.json, roc_curve.png
"""

import json
import os
import sys
import time
from pathlib import Path

# Konfigurasi encoding UTF-8 untuk output terminal Windows
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend untuk plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve

from deepface import DeepFace


def get_paths():
    """Mengembalikan path relatif yang aman dan portabel."""
    src_dir = Path(__file__).resolve().parent
    project_dir = src_dir.parent
    dataset_dir = Path(os.getenv("DATASET_DIR", project_dir.parent / "dataset")).resolve()
    processed_dir = project_dir / "dataset" / "processed"
    pairs_csv = processed_dir / "pairs.csv"

    exp_arcface_dir = project_dir / "experiments" / "arcface"
    embeddings_dir = exp_arcface_dir / "embeddings"
    results_csv = exp_arcface_dir / "arcface_results.csv"
    metrics_json = exp_arcface_dir / "metrics.json"
    roc_png = exp_arcface_dir / "roc_curve.png"

    embeddings_dir.mkdir(parents=True, exist_ok=True)

    return {
        "project_dir": project_dir,
        "dataset_dir": dataset_dir,
        "pairs_csv": pairs_csv,
        "exp_arcface_dir": exp_arcface_dir,
        "embeddings_dir": embeddings_dir,
        "results_csv": results_csv,
        "metrics_json": metrics_json,
        "roc_png": roc_png,
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


def extract_or_load_embeddings(unique_img_rel_paths, dataset_dir: Path, embeddings_dir: Path, batch_size: int = 50):
    """
    Mengekstrak face embeddings menggunakan pretrained ArcFace dari DeepFace,
    atau memuat dari cache disk jika sudah pernah diekstraksi.
    """
    embeddings_dict = {}
    missing_paths = []

    # 1. Cek cache
    for rel_p in unique_img_rel_paths:
        cached_file = embeddings_dir / sanitize_path_to_filename(rel_p)
        if cached_file.exists():
            try:
                emb = np.load(cached_file)
                embeddings_dict[rel_p] = emb
            except Exception:
                missing_paths.append(rel_p)
        else:
            missing_paths.append(rel_p)

    if missing_paths:
        print(f"Mengekstrak {len(missing_paths)} embedding wajah baru menggunakan ArcFace (DeepFace)...")
        total_batches = (len(missing_paths) + batch_size - 1) // batch_size

        for i in range(0, len(missing_paths), batch_size):
            batch_rel = missing_paths[i : i + batch_size]
            batch_abs = [str(dataset_dir / p) for p in batch_rel]
            batch_num = i // batch_size + 1

            print(f"  [Batch {batch_num}/{total_batches}] Memproses {len(batch_abs)} gambar...")
            reps = DeepFace.represent(
                img_path=batch_abs,
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=False,
            )

            for rel_p, rep_item in zip(batch_rel, reps):
                # rep_item adalah list of dicts untuk tiap gambar
                emb = np.array(rep_item[0]["embedding"], dtype=np.float32)
                embeddings_dict[rel_p] = emb

                # Simpan ke cache disk
                save_path = embeddings_dir / sanitize_path_to_filename(rel_p)
                np.save(save_path, emb)
    else:
        print(f"Semua {len(unique_img_rel_paths)} embedding wajah berhasil dimuat dari cache disk.")

    return embeddings_dict


def find_optimal_threshold(distances: np.ndarray, labels: np.ndarray, criterion: str = "balanced_accuracy"):
    """
    Mencari threshold optimal HANYA pada development set.
    Mendukung kriteria: 'balanced_accuracy' (default, memaksimalkan 1 - HTER) atau 'f1'.
    Karena metric adalah cosine distance, prediksi = 1 jika distance <= threshold.
    """
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
    scores = 1.0 - distances
    fpr, tpr, roc_threshs = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))

    # EER dihitung pada titik di mana FPR ~ FNR (FNR = 1 - TPR)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_thresh = float(roc_threshs[eer_idx])

    return fpr, tpr, roc_auc, eer, eer_thresh


def plot_roc_curve(fpr, tpr, roc_auc, eer, output_path: Path):
    """Membuat dan menyimpan visualisasi ROC Curve."""
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"ArcFace ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    plt.scatter([eer], [1.0 - eer], color="#DC2626", s=60, zorder=5, label=f"EER = {eer:.4f}")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (FAR)", fontsize=12)
    plt.ylabel("True Positive Rate (1 - FRR)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) - ArcFace on LFW Test Set", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    start_time = time.time()
    paths = get_paths()

    print("=" * 65)
    print("TAHAP 2: IMPLEMENTASI DAN PENGUJIAN ARCFACE (DEEPFACE)")
    print("=" * 65)

    if not paths["pairs_csv"].exists():
        raise FileNotFoundError(f"File pairs.csv tidak ditemukan di: {paths['pairs_csv']}")

    # 1. Muat dataset pairs.csv
    df = pd.read_csv(paths["pairs_csv"])
    print(f"Total pasangan dimuat : {len(df)} pasang")
    print(f"  - Development set   : {len(df[df['split'] == 'development'])} pasang")
    print(f"  - Test set          : {len(df[df['split'] == 'test'])} pasang")

    unique_imgs = sorted(list(set(df["image1"]).union(set(df["image2"]))))
    print(f"Total citra unik     : {len(unique_imgs)} citra")

    # 2. Ekstrak atau muat face embeddings ArcFace
    t_emb_start = time.time()
    embeddings = extract_or_load_embeddings(
        unique_img_rel_paths=unique_imgs,
        dataset_dir=paths["dataset_dir"],
        embeddings_dir=paths["embeddings_dir"],
        batch_size=50,
    )
    t_emb_end = time.time()
    print(f"Waktu ekstraksi embedding: {t_emb_end - t_emb_start:.2f} detik")

    # 3. Hitung cosine distance untuk setiap pasangan
    print("\nMenghitung Cosine Distance untuk seluruh pasangan...")
    distances = []
    for _, row in df.iterrows():
        emb1 = embeddings[row["image1"]]
        emb2 = embeddings[row["image2"]]
        dist = compute_cosine_distance(emb1, emb2)
        distances.append(dist)

    df["distance"] = distances

    # 4. Tentukan threshold optimal HANYA pada development set
    dev_mask = df["split"] == "development"
    dev_dists = df.loc[dev_mask, "distance"].to_numpy()
    dev_labels = df.loc[dev_mask, "label"].to_numpy()

    print("\nMengoptimalkan threshold pada Development Set (800 pasang)...")
    optimal_thresh, dev_metrics = find_optimal_threshold(dev_dists, dev_labels)
    print(f"Threshold Optimal Ditemukan: {optimal_thresh:.4f}")
    print(f"  -> Development Accuracy  : {dev_metrics['accuracy'] * 100:.2f}%")
    print(f"  -> Development F1-score  : {dev_metrics['f1']:.4f}")
    print(f"  -> Development FAR       : {dev_metrics['far']:.4f}")
    print(f"  -> Development FRR       : {dev_metrics['frr']:.4f}")

    # 5. Evaluasi pada Test Set menggunakan threshold yang dibekukan
    print("\nMengevaluasi Test Set (200 pasang) dengan Threshold Tetap...")
    test_mask = df["split"] == "test"
    test_dists = df.loc[test_mask, "distance"].to_numpy()
    test_labels = df.loc[test_mask, "label"].to_numpy()

    test_preds, test_metrics = evaluate_split(test_dists, test_labels, optimal_thresh)
    dev_preds, _ = evaluate_split(dev_dists, dev_labels, optimal_thresh)

    # Isi kolom prediction pada DataFrame
    df.loc[dev_mask, "prediction"] = dev_preds
    df.loc[test_mask, "prediction"] = test_preds
    df["prediction"] = df["prediction"].astype(int)

    # 6. Hitung ROC, AUC, dan EER pada Test Set
    fpr, tpr, roc_auc, eer, eer_thresh = compute_roc_and_eer(test_dists, test_labels)
    plot_roc_curve(fpr, tpr, roc_auc, eer, paths["roc_png"])
    print(f"ROC Curve berhasil disimpan di: {paths['roc_png']}")

    # 7. Simpan arcface_results.csv
    cols_order = ["image1", "image2", "identity1", "identity2", "label", "distance", "prediction", "split"]
    df_results = df[cols_order]
    df_results.to_csv(paths["results_csv"], index=False)
    print(f"Hasil inferensi berhasil disimpan di: {paths['results_csv']}")

    # 8. Simpan metrics.json
    total_time = time.time() - start_time
    metrics_data = {
        "model": "ArcFace",
        "framework": "DeepFace",
        "distance_metric": "cosine",
        "distance_explanation": "Nilai cosine distance lebih kecil berarti citra wajah lebih mirip (angle lebih kecil).",
        "threshold": optimal_thresh,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "far": test_metrics["far"],
        "frr": test_metrics["frr"],
        "auc": round(roc_auc, 4),
        "eer": round(eer, 4),
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "total_pairs": len(df),
        "development_pairs": int(np.sum(dev_mask)),
        "test_pairs": int(np.sum(test_mask)),
        "development_metrics": dev_metrics,
        "elapsed_time_seconds": round(total_time, 2),
    }

    with open(paths["metrics_json"], "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metrics JSON berhasil disimpan di: {paths['metrics_json']}")

    # 9. Tampilkan Ringkasan Evaluasi Test
    print("\n" + "=" * 65)
    print("HASIL EVALUASI TEST SET (ARCFACE)")
    print("=" * 65)
    print(f"1. Status Eksekusi   : BERHASIL")
    print(f"2. Total Pasangan    : {len(df)} pasang ({len(unique_imgs)} citra unik)")
    print(f"3. Split Data        : Development: 800 | Test: 200")
    print(f"4. Metrik Jarak      : Cosine Distance (nilai lebih kecil = lebih mirip)")
    print(f"5. Threshold Dev     : {optimal_thresh:.4f}")
    print(f"6. Test Accuracy     : {test_metrics['accuracy'] * 100:.2f}%")
    print(f"7. Test Precision    : {test_metrics['precision']:.4f}")
    print(f"8. Test Recall       : {test_metrics['recall']:.4f}")
    print(f"9. Test F1-Score     : {test_metrics['f1']:.4f}")
    print(f"10. Test FAR         : {test_metrics['far']:.4f} ({test_metrics['far'] * 100:.2f}%)")
    print(f"11. Test FRR         : {test_metrics['frr']:.4f} ({test_metrics['frr'] * 100:.2f}%)")
    print(f"12. Test AUC         : {roc_auc:.4f}")
    print(f"    Test EER         : {eer:.4f} ({eer * 100:.2f}%)")
    print(f"    Confusion Matrix : TN={test_metrics['confusion_matrix']['TN']}, FP={test_metrics['confusion_matrix']['FP']}, FN={test_metrics['confusion_matrix']['FN']}, TP={test_metrics['confusion_matrix']['TP']}")
    print(f"    Waktu Eksekusi   : {total_time:.2f} detik")
    print("=" * 65)


if __name__ == "__main__":
    main()
