"""
Script Pengujian Face Verification menggunakan FaceNet (Tahap 3).
- Framework: DeepFace (Pretrained Facenet, 128-d embedding)
- Dataset: LFW pairs dari dataset/processed/pairs.csv (identik dengan ArcFace)
- Detektor: 'skip' (citra LFW sudah terpusat/cropped, konsisten dengan ArcFace)
- Metrik jarak: Cosine Distance (nilai lebih kecil = lebih mirip, identik dengan ArcFace)
- Threshold: Dioptimalkan HANYA pada development set (800 pasang), lalu dibekukan untuk test set (200 pasang)
- Evaluasi: Accuracy, Precision, Recall, F1, Confusion Matrix, FAR, FRR, ROC-AUC, EER
- Output: experiments/facenet/ (embeddings/, facenet_results.csv, metrics.json, roc_curve.png)
"""

import json
import os
import sys
import time
from pathlib import Path

# Konfigurasi encoding UTF-8 untuk output terminal Windows
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from deepface import DeepFace

from utils import (
    compute_cosine_distance,
    compute_roc_and_eer,
    evaluate_split,
    find_optimal_threshold,
    get_dataset_dir,
    get_project_dir,
    plot_roc_curve,
    sanitize_path_to_filename,
)


def get_paths():
    """Mengembalikan path direktori dan file yang portabel dan terisolasi untuk eksperimen FaceNet."""
    project_dir = get_project_dir()
    dataset_dir = get_dataset_dir()
    pairs_csv = project_dir / "dataset" / "processed" / "pairs.csv"

    exp_facenet_dir = project_dir / "experiments" / "facenet"
    embeddings_dir = exp_facenet_dir / "embeddings"
    results_csv = exp_facenet_dir / "facenet_results.csv"
    metrics_json = exp_facenet_dir / "metrics.json"
    roc_png = exp_facenet_dir / "roc_curve.png"

    embeddings_dir.mkdir(parents=True, exist_ok=True)

    return {
        "project_dir": project_dir,
        "dataset_dir": dataset_dir,
        "pairs_csv": pairs_csv,
        "exp_facenet_dir": exp_facenet_dir,
        "embeddings_dir": embeddings_dir,
        "results_csv": results_csv,
        "metrics_json": metrics_json,
        "roc_png": roc_png,
    }


def extract_or_load_embeddings(unique_img_rel_paths, dataset_dir: Path, embeddings_dir: Path, batch_size: int = 50):
    """
    Mengekstrak face embeddings menggunakan pretrained FaceNet (128-d) dari DeepFace,
    atau memuat dari cache disk jika sudah pernah diekstraksi.
    """
    embeddings_dict = {}
    missing_paths = []

    # 1. Cek cache lokal di folder embeddings FaceNet
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
        print(f"Mengekstrak {len(missing_paths)} embedding wajah baru menggunakan FaceNet (DeepFace)...")
        total_batches = (len(missing_paths) + batch_size - 1) // batch_size

        for i in range(0, len(missing_paths), batch_size):
            batch_rel = missing_paths[i : i + batch_size]
            batch_abs = [str(dataset_dir / p) for p in batch_rel]
            batch_num = i // batch_size + 1

            print(f"  [Batch {batch_num}/{total_batches}] Memproses {len(batch_abs)} gambar...")
            reps = DeepFace.represent(
                img_path=batch_abs,
                model_name="Facenet",
                detector_backend="skip",
                enforce_detection=False,
            )

            for rel_p, rep_item in zip(batch_rel, reps):
                emb = np.array(rep_item[0]["embedding"], dtype=np.float32)
                embeddings_dict[rel_p] = emb

                # Simpan ke cache disk
                save_path = embeddings_dir / sanitize_path_to_filename(rel_p)
                np.save(save_path, emb)
    else:
        print(f"Semua {len(unique_img_rel_paths)} embedding wajah FaceNet berhasil dimuat dari cache disk.")

    return embeddings_dict


def main():
    start_time = time.time()
    paths = get_paths()

    print("=" * 65)
    print("TAHAP 3: IMPLEMENTASI DAN PENGUJIAN FACENET (DEEPFACE)")
    print("=" * 65)

    if not paths["pairs_csv"].exists():
        raise FileNotFoundError(f"File pairs.csv tidak ditemukan di: {paths['pairs_csv']}")

    # 1. Muat dataset pairs.csv yang identik dengan ArcFace
    df = pd.read_csv(paths["pairs_csv"])
    print(f"Total pasangan dimuat : {len(df)} pasang")
    print(f"  - Development set   : {len(df[df['split'] == 'development'])} pasang")
    print(f"  - Test set          : {len(df[df['split'] == 'test'])} pasang")

    unique_imgs = sorted(list(set(df["image1"]).union(set(df["image2"]))))
    print(f"Total citra unik     : {len(unique_imgs)} citra")

    # 2. Ekstrak atau muat face embeddings FaceNet (128-d)
    t_emb_start = time.time()
    embeddings = extract_or_load_embeddings(
        unique_img_rel_paths=unique_imgs,
        dataset_dir=paths["dataset_dir"],
        embeddings_dir=paths["embeddings_dir"],
        batch_size=50,
    )
    t_emb_end = time.time()
    print(f"Waktu ekstraksi embedding FaceNet: {t_emb_end - t_emb_start:.2f} detik")

    # 3. Hitung cosine distance untuk setiap pasangan (sama dengan ArcFace)
    print("\nMenghitung Cosine Distance untuk seluruh pasangan (FaceNet)...")
    distances = []
    for _, row in df.iterrows():
        emb1 = embeddings[row["image1"]]
        emb2 = embeddings[row["image2"]]
        dist = compute_cosine_distance(emb1, emb2)
        distances.append(dist)

    df["distance"] = distances

    # 4. Tentukan threshold optimal HANYA pada development set (sama dengan ArcFace: balanced accuracy)
    dev_mask = df["split"] == "development"
    dev_dists = df.loc[dev_mask, "distance"].to_numpy()
    dev_labels = df.loc[dev_mask, "label"].to_numpy()

    print("\nMengoptimalkan threshold pada Development Set (800 pasang)...")
    optimal_thresh, dev_metrics = find_optimal_threshold(dev_dists, dev_labels, criterion="balanced_accuracy")
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
    plot_roc_curve(fpr, tpr, roc_auc, eer, paths["roc_png"], model_name="FaceNet")
    print(f"ROC Curve berhasil disimpan di: {paths['roc_png']}")

    # 7. Simpan facenet_results.csv
    cols_order = ["image1", "image2", "identity1", "identity2", "label", "distance", "prediction", "split"]
    df_results = df[cols_order]
    df_results.to_csv(paths["results_csv"], index=False)
    print(f"Hasil inferensi berhasil disimpan di: {paths['results_csv']}")

    # 8. Simpan metrics.json
    total_time = time.time() - start_time
    metrics_data = {
        "model": "Facenet",
        "framework": "DeepFace",
        "embedding_dimension": 128,
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
    print("HASIL EVALUASI TEST SET (FACENET)")
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
    print(f"13. Test EER         : {eer:.4f} ({eer * 100:.2f}%)")
    print(f"    Confusion Matrix : TN={test_metrics['confusion_matrix']['TN']}, FP={test_metrics['confusion_matrix']['FP']}, FN={test_metrics['confusion_matrix']['FN']}, TP={test_metrics['confusion_matrix']['TP']}")
    print(f"    Waktu Eksekusi   : {total_time:.2f} detik")
    print("=" * 65)


if __name__ == "__main__":
    main()
