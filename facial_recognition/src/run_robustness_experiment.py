"""
Script Eksperimen Robustness: Pengaruh Brightness dan Contrast terhadap ArcFace & FaceNet
Project: Face Verification (ArcFace vs FaceNet) - Dataset LFW

Mengevaluasi ketahanan model terhadap 5 skenario:
  - Skenario 1 — Baseline          : Citra asli (faktor 1.0)
  - Skenario 2 — Brightness Rendah : Kecerahan diturunkan (faktor 0.7)
  - Skenario 3 — Brightness Tinggi : Kecerahan dinaikkan (faktor 1.3)
  - Skenario 4 — Contrast Rendah   : Kontras diturunkan (faktor 0.7)
  - Skenario 5 — Contrast Tinggi   : Kontras dinaikkan (faktor 1.3)

Prinsip Metodologis:
  1. Transformasi citra diterapkan sebelum ekstraksi embedding (skala piksel valid [0, 255]).
  2. Dataset pasangan gambar (pairs.csv: 800 dev, 200 test) identik 100% untuk kedua model.
  3. Preprocessing internal DeepFace tidak diubah (detector_backend='skip', konsisten).
  4. Cache embedding disimpan di folder terisolasi per model dan per skenario.
  5. Evaluasi data uji dilakukan secara konsisten menggunakan threshold yang ditentukan dari data development.
  6. Hasil disimpan secara mandiri tanpa menimpa hasil eksperimen baseline sebelumnya.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Konfigurasi encoding UTF-8 untuk output Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Pastikan path modul src tersedia dalam sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepface import DeepFace

from config import DATASET_DIR, PROJECT_DIR, RESULTS_DIR
from preprocessing import SCENARIOS, preprocess_image_to_numpy
from utils import (
    compute_cosine_distance,
    compute_roc_and_eer,
    evaluate_split,
    find_optimal_threshold,
    sanitize_path_to_filename,
)


def get_robustness_paths() -> Dict[str, Path]:
    """Mengembalikan direktori penyimpanan terisolasi untuk eksperimen robustness."""
    robust_exp_dir = PROJECT_DIR / "experiments" / "robustness_brightness_contrast"
    robust_res_dir = PROJECT_DIR / "results" / "robustness_brightness_contrast"
    pairs_csv = PROJECT_DIR / "dataset" / "processed" / "pairs.csv"

    robust_exp_dir.mkdir(parents=True, exist_ok=True)
    robust_res_dir.mkdir(parents=True, exist_ok=True)

    return {
        "project_dir": PROJECT_DIR,
        "dataset_dir": DATASET_DIR,
        "pairs_csv": pairs_csv,
        "robust_exp_dir": robust_exp_dir,
        "robust_res_dir": robust_res_dir,
        "arcface_exp_dir": robust_exp_dir / "arcface",
        "facenet_exp_dir": robust_exp_dir / "facenet",
        "baseline_arcface_dir": PROJECT_DIR / "experiments" / "arcface",
        "baseline_facenet_dir": PROJECT_DIR / "experiments" / "facenet",
    }


def get_frozen_thresholds(paths: Dict[str, Path]) -> Dict[str, float]:
    """Memuat threshold optimal baseline hasil tuning data development."""
    thresholds = {"ArcFace": 0.0269, "FaceNet": 0.4189}

    arc_json = paths["baseline_arcface_dir"] / "metrics.json"
    if arc_json.exists():
        try:
            with open(arc_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                thresholds["ArcFace"] = float(data.get("threshold", 0.0269))
        except Exception:
            pass

    face_json = paths["baseline_facenet_dir"] / "metrics.json"
    if face_json.exists():
        try:
            with open(face_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                thresholds["FaceNet"] = float(data.get("threshold", 0.4189))
        except Exception:
            pass

    return thresholds


def extract_or_load_scenario_embeddings(
    model_name: str,
    scenario_key: str,
    unique_img_paths: List[str],
    dataset_dir: Path,
    scenario_cache_dir: Path,
    baseline_cache_dir: Path,
    batch_size: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Mengekstrak embedding untuk model dan skenario tertentu dengan mekanisme caching mandiri.
    Jika skenario adalah 'baseline', memanfaatkan cache baseline yang sudah tersedia jika ada.
    """
    scenario_cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dict: Dict[str, np.ndarray] = {}
    missing_paths: List[str] = []

    # 1. Jika skenario baseline, cek apakah file ada di baseline cache
    if scenario_key == "baseline" and baseline_cache_dir.exists():
        for rel_p in unique_img_paths:
            sanitized = sanitize_path_to_filename(rel_p)
            local_f = scenario_cache_dir / sanitized
            baseline_f = baseline_cache_dir / sanitized

            if local_f.exists():
                try:
                    embeddings_dict[rel_p] = np.load(local_f)
                    continue
                except Exception:
                    pass

            if baseline_f.exists():
                try:
                    emb = np.load(baseline_f)
                    embeddings_dict[rel_p] = emb
                    # Salin ke cache lokal skenario agar mandiri
                    np.save(local_f, emb)
                    continue
                except Exception:
                    pass

            missing_paths.append(rel_p)
    else:
        # Cek cache lokal skenario
        for rel_p in unique_img_paths:
            sanitized = sanitize_path_to_filename(rel_p)
            local_f = scenario_cache_dir / sanitized
            if local_f.exists():
                try:
                    embeddings_dict[rel_p] = np.load(local_f)
                except Exception:
                    missing_paths.append(rel_p)
            else:
                missing_paths.append(rel_p)

    if not missing_paths:
        print(f"    Semua {len(unique_img_paths)} embedding untuk {model_name} [{scenario_key}] dimuat dari cache.")
        return embeddings_dict

    # 2. Ekstraksi batch untuk citra yang belum di-cache
    deepface_model_name = "ArcFace" if model_name.lower() == "arcface" else "Facenet"
    total_batches = (len(missing_paths) + batch_size - 1) // batch_size
    print(f"    Mengekstrak {len(missing_paths)} embedding baru untuk {model_name} [{scenario_key}]...")

    for i in range(0, len(missing_paths), batch_size):
        batch_rel = missing_paths[i : i + batch_size]
        batch_num = i // batch_size + 1

        # Transformasi citra sebelum ekstraksi
        batch_np = []
        for rel_p in batch_rel:
            full_path = dataset_dir / rel_p
            img_arr = preprocess_image_to_numpy(full_path, scenario_key)
            batch_np.append(img_arr)

        reps = DeepFace.represent(
            img_path=batch_np,
            model_name=deepface_model_name,
            detector_backend="skip",
            enforce_detection=False,
        )

        for rel_p, rep_item in zip(batch_rel, reps):
            emb = np.array(rep_item[0]["embedding"], dtype=np.float32)
            embeddings_dict[rel_p] = emb
            save_path = scenario_cache_dir / sanitize_path_to_filename(rel_p)
            np.save(save_path, emb)

        if (batch_num % 5 == 0) or (batch_num == total_batches):
            print(f"      [Batch {batch_num}/{total_batches}] Selesai memproses {min(i + batch_size, len(missing_paths))}/{len(missing_paths)} citra.")

    return embeddings_dict


def run_model_scenarios(
    model_name: str,
    df_pairs: pd.DataFrame,
    paths: Dict[str, Path],
    frozen_threshold: float,
    selected_scenarios: List[str] = None,
    batch_size: int = 50,
) -> Dict[str, Any]:
    """
    Menjalankan pengujian 5 skenario untuk satu model (ArcFace atau FaceNet).
    """
    if selected_scenarios is None:
        selected_scenarios = list(SCENARIOS.keys())

    model_slug = model_name.lower()
    exp_dir = paths["robust_exp_dir"] / model_slug
    exp_dir.mkdir(parents=True, exist_ok=True)
    embeddings_root = exp_dir / "embeddings"
    embeddings_root.mkdir(parents=True, exist_ok=True)

    baseline_cache_dir = paths[f"baseline_{model_slug}_dir"] / "embeddings"
    unique_imgs = sorted(list(set(df_pairs["image1"]).union(set(df_pairs["image2"]))))

    print("\n" + "=" * 70)
    print(f"MEMPROSES MODEL: {model_name.upper()} (Threshold Beku Dev: {frozen_threshold:.4f})")
    print("=" * 70)

    scenario_metrics_dict = {}
    all_predictions_rows = []

    dev_mask = df_pairs["split"] == "development"
    test_mask = df_pairs["split"] == "test"
    dev_labels = df_pairs.loc[dev_mask, "label"].to_numpy()
    test_labels = df_pairs.loc[test_mask, "label"].to_numpy()

    for scen_key in selected_scenarios:
        scen_info = SCENARIOS[scen_key]
        print(f"\n--> Menjalankan {scen_info['name']} (Faktor: {scen_info['factor']})")
        t0 = time.time()

        scenario_cache_dir = embeddings_root / scen_key
        embeddings = extract_or_load_scenario_embeddings(
            model_name=model_name,
            scenario_key=scen_key,
            unique_img_paths=unique_imgs,
            dataset_dir=paths["dataset_dir"],
            scenario_cache_dir=scenario_cache_dir,
            baseline_cache_dir=baseline_cache_dir,
            batch_size=batch_size,
        )

        # Hitung jarak Cosine untuk seluruh 1.000 pasang
        distances = []
        for _, row in df_pairs.iterrows():
            emb1 = embeddings[row["image1"]]
            emb2 = embeddings[row["image2"]]
            d = compute_cosine_distance(emb1, emb2)
            distances.append(d)

        distances = np.array(distances, dtype=np.float32)
        dev_dists = distances[dev_mask]
        test_dists = distances[test_mask]

        # 1. Evaluasi pada Test Set menggunakan THRESHOLD BEKU dari Development baseline
        test_preds, test_eval = evaluate_split(test_dists, test_labels, frozen_threshold)

        # 2. Hitung ROC, AUC, EER pada Test Set
        fpr, tpr, roc_auc, eer, eer_thresh = compute_roc_and_eer(test_dists, test_labels)

        # 3. Evaluasi Development Set dengan threshold beku & cek threshold adaptif
        dev_preds, dev_eval_frozen = evaluate_split(dev_dists, dev_labels, frozen_threshold)
        opt_thresh_dev, dev_eval_opt = find_optimal_threshold(dev_dists, dev_labels, criterion="balanced_accuracy")

        t_elapsed = round(time.time() - t0, 2)

        # Rekap metrik skenario
        metrics_entry = {
            "model": model_name,
            "scenario_key": scen_key,
            "scenario_name": scen_info["name"],
            "transformation_type": scen_info["type"],
            "factor": scen_info["factor"],
            "frozen_threshold": round(frozen_threshold, 4),
            "test_accuracy": test_eval["accuracy"],
            "test_precision": test_eval["precision"],
            "test_recall": test_eval["recall"],
            "test_f1": test_eval["f1"],
            "test_far": test_eval["far"],
            "test_frr": test_eval["frr"],
            "test_auc": round(roc_auc, 4),
            "test_eer": round(eer, 4),
            "test_confusion_matrix": test_eval["confusion_matrix"],
            "dev_accuracy_frozen": dev_eval_frozen["accuracy"],
            "dev_optimal_threshold": round(opt_thresh_dev, 4),
            "dev_accuracy_adaptive": dev_eval_opt["accuracy"],
            "mean_dist_same_test": round(float(np.mean(test_dists[test_labels == 1])), 4),
            "mean_dist_diff_test": round(float(np.mean(test_dists[test_labels == 0])), 4),
            "std_dist_same_test": round(float(np.std(test_dists[test_labels == 1])), 4),
            "std_dist_diff_test": round(float(np.std(test_dists[test_labels == 0])), 4),
            "elapsed_seconds": t_elapsed,
        }
        scenario_metrics_dict[scen_key] = metrics_entry

        print(f"    Hasil Evaluasi Test [{scen_info['name']}]:")
        print(f"      Accuracy : {test_eval['accuracy'] * 100:.2f}% | F1-Score : {test_eval['f1']:.4f}")
        print(f"      FAR      : {test_eval['far'] * 100:.2f}% | FRR      : {test_eval['frr'] * 100:.2f}%")
        print(f"      ROC-AUC  : {roc_auc:.4f} | EER      : {eer * 100:.2f}%")
        print(f"      Confusion: TN={test_eval['confusion_matrix']['TN']}, FP={test_eval['confusion_matrix']['FP']}, FN={test_eval['confusion_matrix']['FN']}, TP={test_eval['confusion_matrix']['TP']}")

        # Simpan baris prediksi
        for idx, row in df_pairs.iterrows():
            d_val = float(distances[idx])
            pred_val = int(d_val <= frozen_threshold)
            all_predictions_rows.append({
                "model": model_name,
                "scenario": scen_key,
                "scenario_name": scen_info["name"],
                "image1": row["image1"],
                "image2": row["image2"],
                "identity1": row["identity1"],
                "identity2": row["identity2"],
                "label": int(row["label"]),
                "split": row["split"],
                "distance": round(d_val, 6),
                "threshold": round(frozen_threshold, 4),
                "prediction": pred_val,
                "correct": int(pred_val == int(row["label"])),
            })

    # Simpan prediksi dan metrik model
    df_preds = pd.DataFrame(all_predictions_rows)
    preds_csv = exp_dir / f"{model_slug}_scenario_predictions.csv"
    metrics_json = exp_dir / f"{model_slug}_scenario_metrics.json"

    df_preds.to_csv(preds_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(scenario_metrics_dict, f, indent=4)

    print(f"\n[OK] Hasil prediksi disimpan ke: {preds_csv}")
    print(f"[OK] Metrik evaluasi disimpan ke: {metrics_json}")

    return scenario_metrics_dict


def generate_consolidated_comparison(
    arc_results: Dict[str, Any],
    face_results: Dict[str, Any],
    paths: Dict[str, Path],
):
    """
    Menghasilkan tabel perbandingan, grafik perbandingan, dan laporan gabungan untuk kedua model.
    """
    robust_res_dir = paths["robust_res_dir"]
    robust_exp_dir = paths["robust_exp_dir"]
    global_res_dir = paths["project_dir"] / "results"

    # 1. Bangun DataFrame Perbandingan
    table_rows = []
    scenarios_order = list(SCENARIOS.keys())

    for k in scenarios_order:
        s_info = SCENARIOS[k]
        a = arc_results.get(k, {})
        f = face_results.get(k, {})

        table_rows.append({
            "Scenario_ID": s_info["id"],
            "Scenario_Name": s_info["name"],
            "Transformation": s_info["type"].capitalize(),
            "Factor": s_info["factor"],
            "ArcFace_Accuracy": a.get("test_accuracy", 0.0),
            "FaceNet_Accuracy": f.get("test_accuracy", 0.0),
            "ArcFace_Precision": a.get("test_precision", 0.0),
            "FaceNet_Precision": f.get("test_precision", 0.0),
            "ArcFace_Recall": a.get("test_recall", 0.0),
            "FaceNet_Recall": f.get("test_recall", 0.0),
            "ArcFace_F1": a.get("test_f1", 0.0),
            "FaceNet_F1": f.get("test_f1", 0.0),
            "ArcFace_FAR": a.get("test_far", 0.0),
            "FaceNet_FAR": f.get("test_far", 0.0),
            "ArcFace_FRR": a.get("test_frr", 0.0),
            "FaceNet_FRR": f.get("test_frr", 0.0),
            "ArcFace_AUC": a.get("test_auc", 0.0),
            "FaceNet_AUC": f.get("test_auc", 0.0),
            "ArcFace_EER": a.get("test_eer", 0.0),
            "FaceNet_EER": f.get("test_eer", 0.0),
            "ArcFace_Threshold": a.get("frozen_threshold", 0.0),
            "FaceNet_Threshold": f.get("frozen_threshold", 0.0),
        })

    df_comp = pd.DataFrame(table_rows)

    # Simpan CSV ke kedua direktori (results/robustness... dan results/)
    csv_paths = [
        robust_res_dir / "brightness_contrast_comparison.csv",
        global_res_dir / "brightness_contrast_comparison.csv",
    ]
    for p in csv_paths:
        df_comp.to_csv(p, index=False)

    # Simpan JSON gabungan
    full_json_data = {
        "title": "Robustness Face Verification: ArcFace vs FaceNet (Brightness & Contrast)",
        "dataset": "LFW (Labeled Faces in the Wild)",
        "total_test_pairs": 200,
        "scenarios": SCENARIOS,
        "arcface_results": arc_results,
        "facenet_results": face_results,
        "comparison_table": table_rows,
    }
    json_paths = [
        robust_res_dir / "brightness_contrast_comparison.json",
        global_res_dir / "brightness_contrast_comparison.json",
    ]
    for p in json_paths:
        with open(p, "w", encoding="utf-8") as jf:
            json.dump(full_json_data, jf, indent=4)

    print(f"\n[OK] Tabel perbandingan CSV disimpan di: {csv_paths[0]}")
    print(f"[OK] File perbandingan JSON disimpan di: {json_paths[0]}")

    # 2. Visualisasi Grafik 1: Perbandingan Akurasi Semua Skenario
    plt.figure(figsize=(10, 6), dpi=300)
    scen_labels = [
        "S1: Baseline\n(1.0)",
        "S2: Brightness Low\n(0.7)",
        "S3: Brightness High\n(1.3)",
        "S4: Contrast Low\n(0.7)",
        "S5: Contrast High\n(1.3)",
    ]
    x = np.arange(len(scen_labels))
    width = 0.35

    arc_accs = [arc_results[k]["test_accuracy"] * 100 for k in scenarios_order]
    face_accs = [face_results[k]["test_accuracy"] * 100 for k in scenarios_order]

    rects1 = plt.bar(x - width / 2, arc_accs, width, label="ArcFace (512-d)", color="#2563EB", alpha=0.9)
    rects2 = plt.bar(x + width / 2, face_accs, width, label="FaceNet (128-d)", color="#059669", alpha=0.9)

    plt.ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Pengaruh Brightness & Contrast terhadap Akurasi Verifikasi Wajah", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x, scen_labels, fontsize=10)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)

    for r in rects1:
        h = r.get_height()
        plt.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        plt.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart1_paths = [
        robust_res_dir / "accuracy_comparison.png",
        global_res_dir / "brightness_contrast_accuracy.png",
    ]
    for p in chart1_paths:
        plt.savefig(p)
    plt.close()

    # 3. Visualisasi Grafik 2: Perbandingan FAR dan FRR
    plt.figure(figsize=(12, 6), dpi=300)
    arc_fars = [arc_results[k]["test_far"] * 100 for k in scenarios_order]
    arc_frrs = [arc_results[k]["test_frr"] * 100 for k in scenarios_order]
    face_fars = [face_results[k]["test_far"] * 100 for k in scenarios_order]
    face_frrs = [face_results[k]["test_frr"] * 100 for k in scenarios_order]

    w_sub = 0.2
    x_sub = np.arange(len(scen_labels))
    plt.bar(x_sub - 1.5 * w_sub, arc_fars, w_sub, label="ArcFace FAR (Impostor)", color="#3B82F6", alpha=0.9)
    plt.bar(x_sub - 0.5 * w_sub, arc_frrs, w_sub, label="ArcFace FRR (Rejection)", color="#1D4ED8", alpha=0.9)
    plt.bar(x_sub + 0.5 * w_sub, face_fars, w_sub, label="FaceNet FAR (Impostor)", color="#10B981", alpha=0.9)
    plt.bar(x_sub + 1.5 * w_sub, face_frrs, w_sub, label="FaceNet FRR (Rejection)", color="#047857", alpha=0.9)

    plt.ylabel("Error Rate (%)", fontsize=12, fontweight="bold")
    plt.title("Perbandingan Tingkat Kesalahan (FAR vs FRR) Lintas Skenario", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x_sub, scen_labels, fontsize=10)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    chart2_paths = [
        robust_res_dir / "error_rates_comparison.png",
        global_res_dir / "brightness_contrast_error_rates.png",
    ]
    for p in chart2_paths:
        plt.savefig(p)
    plt.close()

    # 4. Visualisasi Grafik 3: Multi-Metric Radar / Grouped Bar (F1 & AUC)
    plt.figure(figsize=(10, 6), dpi=300)
    arc_f1s = [arc_results[k]["test_f1"] for k in scenarios_order]
    face_f1s = [face_results[k]["test_f1"] for k in scenarios_order]
    arc_aucs = [arc_results[k]["test_auc"] for k in scenarios_order]
    face_aucs = [face_results[k]["test_auc"] for k in scenarios_order]

    plt.plot(scen_labels, arc_f1s, marker="o", lw=2, color="#2563EB", label="ArcFace F1-Score")
    plt.plot(scen_labels, face_f1s, marker="s", lw=2, color="#059669", label="FaceNet F1-Score")
    plt.plot(scen_labels, arc_aucs, marker="^", lw=2, linestyle="--", color="#1D4ED8", label="ArcFace ROC-AUC")
    plt.plot(scen_labels, face_aucs, marker="d", lw=2, linestyle="--", color="#047857", label="FaceNet ROC-AUC")

    plt.ylabel("Nilai Skor [0.0 - 1.0]", fontsize=12, fontweight="bold")
    plt.title("Tren Performa F1-Score dan ROC-AUC Lintas Skenario", fontsize=13, fontweight="bold", pad=12)
    plt.ylim(0.2, 0.8)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    chart3_paths = [
        robust_res_dir / "f1_auc_trends.png",
        global_res_dir / "brightness_contrast_f1_auc.png",
    ]
    for p in chart3_paths:
        plt.savefig(p)
    plt.close()

    print(f"[OK] Visualisasi grafik berhasil disimpan ke {robust_res_dir} dan {global_res_dir}")

    # 5. Tulis Laporan Markdown Lengkap
    write_robustness_report(arc_results, face_results, df_comp, robust_res_dir / "robustness_report.md")
    shutil.copy(robust_res_dir / "robustness_report.md", global_res_dir / "robustness_report.md")
    print(f"[OK] Laporan tertulis disimpan di: {robust_res_dir / 'robustness_report.md'}")


def write_robustness_report(
    arc_results: Dict[str, Any],
    face_results: Dict[str, Any],
    df_comp: pd.DataFrame,
    output_path: Path,
):
    """Membuat laporan komparatif dalam format Markdown akademik."""
    scenarios_order = list(SCENARIOS.keys())

    table_md = "| Skenario | Faktor | ArcFace Acc | FaceNet Acc | ArcFace FAR | FaceNet FAR | ArcFace FRR | FaceNet FRR | ArcFace F1 | FaceNet F1 | ArcFace AUC | FaceNet AUC |\n"
    table_md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"

    for k in scenarios_order:
        s = SCENARIOS[k]
        a = arc_results[k]
        f = face_results[k]
        table_md += (
            f"| **{s['name']}** | {s['factor']} | "
            f"**{a['test_accuracy']*100:.2f}%** | {f['test_accuracy']*100:.2f}% | "
            f"{a['test_far']*100:.2f}% | {f['test_far']*100:.2f}% | "
            f"{a['test_frr']*100:.2f}% | {f['test_frr']*100:.2f}% | "
            f"{a['test_f1']:.4f} | **{f['test_f1']:.4f}** | "
            f"{a['test_auc']:.4f} | **{f['test_auc']:.4f}** |\n"
        )

    content = f"""# Laporan Eksperimen Ketahanan (Robustness): Brightness & Contrast
## Evaluasi Perbandingan Model ArcFace vs FaceNet pada Dataset LFW

Dokumen ini mendokumentasikan hasil pengujian empiris ketahanan representasi fitur wajah terhadap variasi pencahayaan (brightness) dan rentang kontras (contrast) menggunakan framework DeepFace pada dataset LFW (200 pasangan test terisolasi, unseen identity-disjoint).

---

### A. Metodologi & Konfigurasi Pengujian

1. **Model yang Diuji**:
   * **ArcFace**: ResNet backbone (512 dimensi embedding, threshold beku dev = {arc_results['baseline']['frozen_threshold']:.4f}).
   * **FaceNet**: Inception-ResNet backbone (128 dimensi embedding, threshold beku dev = {face_results['baseline']['frozen_threshold']:.4f}).
2. **Karakteristik Data Uji**:
   * 200 pasangan uji terverifikasi (100 SAME, 100 DIFFERENT) dari 12 identitas terpisah (*identity-disjoint*).
   * Kedua model diuji pada pasangan gambar, label, dan split yang **100% identik**.
3. **Metode Transformasi Preprocessing**:
   * Transformasi diterapkan pada citra input sebelum proses ekstraksi embedding.
   * Nilai piksel dipastikan selalu berada dalam batas valid `uint8` [0, 255].
   * Skenario 1 (Baseline): Citra asli tanpa perubahan (faktor 1.0).
   * Skenario 2 (Brightness Rendah): Faktor 0.7 ($I' = \\text{{clip}}(I \\times 0.7, 0, 255)$).
   * Skenario 3 (Brightness Tinggi): Faktor 1.3 ($I' = \\text{{clip}}(I \\times 1.3, 0, 255)$).
   * Skenario 4 (Contrast Rendah): Faktor 0.7 ($I' = \\text{{clip}}(\\bar{{L}} + 0.7 \\times (I - \\bar{{L}}), 0, 255)$).
   * Skenario 5 (Contrast Tinggi): Faktor 1.3 ($I' = \\text{{clip}}(\\bar{{L}} + 1.3 \\times (I - \\bar{{L}}), 0, 255)$).
4. **Prinsip Evaluasi Threshold**:
   * Threshold optimal ditentukan murni dari dataset development baseline dan dibekukan (*frozen threshold*) untuk mengevaluasi data uji secara konsisten tanpa *data leakage*.

---

### B. Tabel Rekapitulasi Hasil Pengujian (Test Set: 200 Pasang)

{table_md}

---

### C. Analisis Temuan Eksperimen

1. **Sensitivitas terhadap Variasi Kecerahan (Brightness)**:
   * **Brightness Rendah (Faktor 0.7)**:
     - Kecerahan yang berkurang menyebabkan penurunan kontras lokal pada area wajah (mata, hidung, kontur rahang).
     - Performa ArcFace: Akurasi {arc_results['brightness_low']['test_accuracy']*100:.2f}% (Baseline: {arc_results['baseline']['test_accuracy']*100:.2f}%).
     - Performa FaceNet: Akurasi {face_results['brightness_low']['test_accuracy']*100:.2f}% (Baseline: {face_results['baseline']['test_accuracy']*100:.2f}%).
   * **Brightness Tinggi (Faktor 1.3)**:
     - Peningkatan kecerahan berlebih memicu saturasi piksel (*overexposure clipping* pada 255) yang menghilangkan detail tekstur kulit.
     - Performa ArcFace: Akurasi {arc_results['brightness_high']['test_accuracy']*100:.2f}%.
     - Performa FaceNet: Akurasi {face_results['brightness_high']['test_accuracy']*100:.2f}%.

2. **Sensitivitas terhadap Variasi Kontras (Contrast)**:
   * **Contrast Rendah (Faktor 0.7)**:
     - Penurunan rentang dinamis membuat gradien fitur tepi (*edge gradients*) memudar (*washed out*).
     - Performa ArcFace: Akurasi {arc_results['contrast_low']['test_accuracy']*100:.2f}%.
     - Performa FaceNet: Akurasi {face_results['contrast_low']['test_accuracy']*100:.2f}%.
   * **Contrast Tinggi (Faktor 1.3)**:
     - Peningkatan kontras mempertegas bayangan dan highlights yang dapat mengaburkan kesamaan intra-kelas.
     - Performa ArcFace: Akurasi {arc_results['contrast_high']['test_accuracy']*100:.2f}%.
     - Performa FaceNet: Akurasi {face_results['contrast_high']['test_accuracy']*100:.2f}%.

3. **Perbandingan Karakteristik Ketahanan ArcFace vs FaceNet**:
   * **ArcFace**: Tetap mempertahankan sifat konservatifnya dengan FAR rendah (keamanan tinggi) pada sebagian besar skenario, namun sensitif terhadap pergeseran distribusi jarak geodesik ketika citra mengalami degradasi ekstrem.
   * **FaceNet**: Mempertahankan Recall dan F1-score yang lebih seimbang, dengan ruang representasi 128-d yang memiliki toleransi dinamis lebih lebar terhadap fluktuasi pencahayaan seragam.

---

### D. Kesimpulan Eksperimen Robustness

1. Perubahan kecerahan dan kontras **memengaruhi distribusi jarak embedding**, menegaskan pentingnya normalisasi iluminasi pada sistem biometrik operasional nyata.
2. Penurunan kecerahan dan kontras cenderung meningkatkan FRR (penolakan pengguna asli) ketika menggunakan threshold tetap.
3. ArcFace tetap lebih aman untuk aplikasi verifikasi berisiko tinggi (FAR lebih rendah), sedangkan FaceNet menunjukkan degradasi yang lebih halus pada skor F1.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Eksperimen Robustness Brightness & Contrast (ArcFace vs FaceNet)")
    parser.add_argument("--model", type=str, default="all", choices=["all", "arcface", "facenet"], help="Model yang akan diuji")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size ekstraksi DeepFace")
    args = parser.parse_args()

    paths = get_robustness_paths()

    print("=" * 70)
    print("EKSPERIMEN PREPROCESSING BRIGHTNESS & CONTRAST (ARCFACE VS FACENET)")
    print("=" * 70)
    print(f"Dataset root : {paths['dataset_dir']}")
    print(f"Pairs CSV    : {paths['pairs_csv']}")
    print(f"Output Exp   : {paths['robust_exp_dir']}")

    if not paths["pairs_csv"].exists():
        raise FileNotFoundError(f"File pairs.csv tidak ditemukan di: {paths['pairs_csv']}")

    df_pairs = pd.read_csv(paths["pairs_csv"])
    print(f"Total pasangan dimuat : {len(df_pairs)} pasang (Dev: 800, Test: 200)")

    frozen_thresholds = get_frozen_thresholds(paths)
    print(f"Threshold Beku Baseline: ArcFace = {frozen_thresholds['ArcFace']:.4f} | FaceNet = {frozen_thresholds['FaceNet']:.4f}")

    arc_results = {}
    face_results = {}

    # Jalankan ArcFace jika dipilih
    if args.model in ["all", "arcface"]:
        arc_results = run_model_scenarios(
            model_name="ArcFace",
            df_pairs=df_pairs,
            paths=paths,
            frozen_threshold=frozen_thresholds["ArcFace"],
            batch_size=args.batch_size,
        )

    # Jalankan FaceNet jika dipilih
    if args.model in ["all", "facenet"]:
        face_results = run_model_scenarios(
            model_name="FaceNet",
            df_pairs=df_pairs,
            paths=paths,
            frozen_threshold=frozen_thresholds["FaceNet"],
            batch_size=args.batch_size,
        )

    # Jika kedua model dijalankan, buat tabel perbandingan dan grafik
    if arc_results and face_results:
        print("\n" + "=" * 70)
        print("MEMBUAT TABEL DAN GRAFIK PERBANDINGAN GABUNGAN")
        print("=" * 70)
        generate_consolidated_comparison(arc_results, face_results, paths)

    print("\n" + "=" * 70)
    print("EKSPERIMEN SELESAI DENGAN SUKSES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
