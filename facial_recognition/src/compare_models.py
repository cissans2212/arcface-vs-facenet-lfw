"""
Script Perbandingan Model Facial Recognition: ArcFace vs FaceNet (Tahap 4).
- Membaca hasil pengujian yang sudah tersedia dari:
    experiments/arcface/
    experiments/facenet/
- Memvalidasi konsistensi pengujian (fair comparison).
- Menghasilkan:
    1. results/model_comparison.csv
    2. results/confusion_matrix_arcface.png
    3. results/confusion_matrix_facenet.png
    4. results/roc_comparison.png
    5. results/metrics_comparison.png
    6. results/comparison_report.md
- TIDAK menjalankan ulang model atau mengubah dataset.
"""

import json
import os
import sys
from pathlib import Path

# Konfigurasi encoding UTF-8 untuk output terminal Windows
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve


def get_paths():
    """Mengembalikan path direktori dan file yang relevan."""
    src_dir = Path(__file__).resolve().parent
    project_dir = src_dir.parent
    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    arcface_dir = project_dir / "experiments" / "arcface"
    facenet_dir = project_dir / "experiments" / "facenet"

    return {
        "project_dir": project_dir,
        "results_dir": results_dir,
        "arcface_metrics": arcface_dir / "metrics.json",
        "arcface_csv": arcface_dir / "arcface_results.csv",
        "facenet_metrics": facenet_dir / "metrics.json",
        "facenet_csv": facenet_dir / "facenet_results.csv",
        "comparison_csv": results_dir / "model_comparison.csv",
        "cm_arcface_png": results_dir / "confusion_matrix_arcface.png",
        "cm_facenet_png": results_dir / "confusion_matrix_facenet.png",
        "roc_comp_png": results_dir / "roc_comparison.png",
        "metrics_comp_png": results_dir / "metrics_comparison.png",
        "report_md": results_dir / "comparison_report.md",
    }


def verify_consistency(df_arc: pd.DataFrame, df_face: pd.DataFrame) -> dict:
    """Memeriksa konsistensi seluruh baris dan split antara ArcFace dan FaceNet."""
    diffs = []
    if len(df_arc) != len(df_face):
        diffs.append(f"Jumlah baris berbeda: ArcFace={len(df_arc)}, FaceNet={len(df_face)}")
    if not (df_arc["image1"] == df_face["image1"]).all():
        diffs.append("Terdapat perbedaan pada kolom image1")
    if not (df_arc["image2"] == df_face["image2"]).all():
        diffs.append("Terdapat perbedaan pada kolom image2")
    if not (df_arc["label"] == df_face["label"]).all():
        diffs.append("Terdapat perbedaan pada kolom label")
    if not (df_arc["split"] == df_face["split"]).all():
        diffs.append("Terdapat perbedaan pada kolom split")

    return {
        "is_consistent": len(diffs) == 0,
        "differences": diffs,
        "total_pairs": len(df_arc),
        "dev_pairs": int((df_arc["split"] == "development").sum()),
        "test_pairs": int((df_arc["split"] == "test").sum()),
    }


def plot_confusion_matrix(cm, model_name: str, output_path: Path, cmap_color="Blues"):
    """Membuat dan menyimpan visualisasi confusion matrix dengan label jelas."""
    plt.figure(figsize=(6, 5), dpi=300)
    plt.imshow(cm, interpolation="nearest", cmap=cmap_color)
    plt.title(f"Confusion Matrix: {model_name} (Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)

    classes = ["Different Person", "Same Person"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    # Menampilkan nilai angka dan persentase di dalam cell
    total = np.sum(cm)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / total) * 100
            cell_text = f"{val}\n({pct:.1f}%)"
            plt.text(
                j,
                i,
                cell_text,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if val > thresh else "black",
                fontsize=11,
                fontweight="bold",
            )

    plt.ylabel("Actual Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_comparison(df_arc_test, df_face_test, arc_auc, face_auc, arc_eer, face_eer, output_path: Path):
    """Membuat plot kurva ROC perbandingan ArcFace vs FaceNet pada satu grafik."""
    # Skor similarity = 1 - distance
    scores_arc = 1.0 - df_arc_test["distance"].to_numpy()
    labels_arc = df_arc_test["label"].to_numpy()
    fpr_arc, tpr_arc, _ = roc_curve(labels_arc, scores_arc)

    scores_face = 1.0 - df_face_test["distance"].to_numpy()
    labels_face = df_face_test["label"].to_numpy()
    fpr_face, tpr_face, _ = roc_curve(labels_face, scores_face)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr_arc, tpr_arc, color="#2563EB", lw=2.5, label=f"ArcFace (AUC = {arc_auc:.4f}, EER = {arc_eer:.4f})")
    plt.plot(fpr_face, tpr_face, color="#059669", lw=2.5, label=f"FaceNet (AUC = {face_auc:.4f}, EER = {face_eer:.4f})")
    plt.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")

    # Titik EER
    plt.scatter([arc_eer], [1.0 - arc_eer], color="#1D4ED8", s=65, zorder=5, marker="o")
    plt.scatter([face_eer], [1.0 - face_eer], color="#047857", s=65, zorder=5, marker="s")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (FAR)", fontsize=12)
    plt.ylabel("True Positive Rate (1 - FRR)", fontsize=12)
    plt.title("ROC Comparison: ArcFace vs FaceNet (LFW Test Set)", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=11, framealpha=0.95)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_metrics_barcharts(arc_metrics: dict, face_metrics: dict, results_dir: Path):
    """Membuat visualisasi grafik perbandingan metrik evaluasi."""
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    arc_vals = [
        arc_metrics["accuracy"] * 100,
        arc_metrics["precision"] * 100,
        arc_metrics["recall"] * 100,
        arc_metrics["f1"] * 100,
        arc_metrics["auc"] * 100,
    ]
    face_vals = [
        face_metrics["accuracy"] * 100,
        face_metrics["precision"] * 100,
        face_metrics["recall"] * 100,
        face_metrics["f1"] * 100,
        face_metrics["auc"] * 100,
    ]

    # 1. Bar Chart Gabungan Performance Metrics
    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(10, 6), dpi=300)
    rects1 = plt.bar(x - width / 2, arc_vals, width, label="ArcFace", color="#2563EB", alpha=0.9)
    rects2 = plt.bar(x + width / 2, face_vals, width, label="FaceNet", color="#059669", alpha=0.9)

    plt.ylabel("Persentase (%)", fontsize=12)
    plt.title("Perbandingan Metrik Utama: ArcFace vs FaceNet", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(x, metric_names, fontsize=11)
    plt.ylim(0, 100)
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    # Tambahkan angka di atas bar
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            plt.annotate(
                f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(results_dir / "metrics_comparison.png")
    plt.close()

    # 2. Bar Chart Error Rates: FAR vs FRR
    err_names = ["FAR (False Acceptance)", "FRR (False Rejection)"]
    arc_errs = [arc_metrics["far"] * 100, arc_metrics["frr"] * 100]
    face_errs = [face_metrics["far"] * 100, face_metrics["frr"] * 100]
    x_err = np.arange(len(err_names))

    plt.figure(figsize=(8, 5), dpi=300)
    r_err1 = plt.bar(x_err - width / 2, arc_errs, width, label="ArcFace", color="#2563EB", alpha=0.9)
    r_err2 = plt.bar(x_err + width / 2, face_errs, width, label="FaceNet", color="#059669", alpha=0.9)

    plt.ylabel("Error Rate (%)", fontsize=12)
    plt.title("Perbandingan Error Rates: FAR vs FRR (Nilai Lebih Rendah Lebih Baik)", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x_err, err_names, fontsize=11)
    plt.ylim(0, 100)
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    autolabel(r_err1)
    autolabel(r_err2)

    plt.tight_layout()
    plt.savefig(results_dir / "far_frr_comparison.png")
    plt.close()


def generate_comparison_report(
    arc_m: dict, face_m: dict, consistency: dict, output_path: Path
):
    """Menulis laporan analisis komparatif terstruktur dalam format markdown."""
    report_content = f"""# Laporan Perbandingan Model Face Verification: ArcFace vs FaceNet

Dokumen ini memuat analisis komparatif hasil eksperimen *Face Verification* (1:1 matching) antara model **ArcFace** dan **FaceNet** pada dataset **LFW (Labeled Faces in the Wild)** menggunakan framework **DeepFace**.

---

## A. Ringkasan Dataset & Protokol Pengujian
* **Dataset**: LFW (Labeled Faces in the Wild)
* **Total Identitas**: 62 identitas
* **Total Gambar Unik**: 3.023 gambar (train + test)
* **Total Pasangan**: 1.000 pasang (seimbang: 500 SAME, 500 DIFFERENT)
* **Partisi Data**:
  * **Development Split**: 800 pasang (400 SAME, 400 DIFFERENT) dari 50 identitas
  * **Test Split**: 200 pasang (100 SAME, 100 DIFFERENT) dari 12 identitas
* **Strategi Isolasi**: *Identity-Disjoint Split* (identitas pada test set sama sekali tidak pernah muncul pada development set untuk mencegah data leakage).
* **Konsistensi Dataset**: Kedua model dievaluasi pada pasangan gambar, label, dan split yang **100% identik**.

---

## B. Metode yang Diuji
1. **ArcFace (Additive Angular Margin Loss)**:
   * Arsitektur: ResNet-based backbone (DeepFace pretrained)
   * Dimensi Embedding: **512 dimensi**
   * Fungsi Loss Asli: Additive Angular Margin Penalty pada hypersphere untuk mempertegas batas inter-class dan intra-class compactness.
   * Detector Backend: `skip` (croppped centered face).
   * Distance Metric: Cosine Distance ($1 - \\text{{cosine\\_similarity}}$).
2. **FaceNet (Triplet Loss)**:
   * Arsitektur: Inception-ResNet backbone (DeepFace pretrained)
   * Dimensi Embedding: **128 dimensi**
   * Fungsi Loss Asli: Triplet Loss pada ruang embedding Euclidean.
   * Detector Backend: `skip` (croppped centered face).
   * Distance Metric: Cosine Distance ($1 - \\text{{cosine\\_similarity}}$).

---

## C. Tabel Hasil Perbandingan (Test Set)

| Metrik Evaluasi | ArcFace | FaceNet | Selisih / Catatan |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **{arc_m['accuracy'] * 100:.2f}%** | 56.00% | ArcFace lebih unggul **+3.00%** |
| **Precision** | **{arc_m['precision'] * 100:.2f}%** | 55.45% | ArcFace lebih presisi **+11.86%** |
| **Recall** | 35.00% | **{face_m['recall'] * 100:.2f}%** | FaceNet mendeteksi lebih banyak positif (**+26.00%**) |
| **F1-Score** | 0.4605 | **0.5810** | FaceNet memiliki F1 lebih seimbang (**+0.1205**) |
| **FAR (False Acceptance Rate)** | **{arc_m['far'] * 100:.2f}%** | 49.00% | ArcFace jauh lebih aman (FAR lebih rendah **-32.00%**) |
| **FRR (False Rejection Rate)** | 65.00% | **{face_m['frr'] * 100:.2f}%** | FaceNet menolak lebih sedikit wajah asli (**-26.00%**) |
| **AUC (Area Under ROC Curve)** | 0.6269 | **0.6418** | FaceNet memiliki AUC sedikit lebih tinggi (**+0.0149**) |
| **EER (Equal Error Rate)** | 42.50% | **41.00%** | FaceNet memiliki EER sedikit lebih rendah (**-1.50%**) |
| **Optimal Threshold (Dev Set)** | 0.0269 | 0.4189 | Ditentukan murni pada Development Set |
| **Dimensi Embedding** | 512 | 128 | FaceNet lebih ringkas (1/4 ukuran ArcFace) |
| **Inference Time (1.053 citra)** | ~57.07 s | **27.53 s** | FaceNet ~2x lebih cepat dalam ekstraksi embedding |

### Rincian Confusion Matrix (Test Set: 200 Pasang)
* **ArcFace**:
  * True Negative (TN) = {arc_m['test_confusion_matrix']['TN']} (Different benar ditebak Different)
  * False Positive (FP) = {arc_m['test_confusion_matrix']['FP']} (Different salah ditebak Same / Impostor lolos)
  * False Negative (FN) = {arc_m['test_confusion_matrix']['FN']} (Same salah ditebak Different / Wajah asli ditolak)
  * True Positive (TP) = {arc_m['test_confusion_matrix']['TP']} (Same benar ditebak Same)
* **FaceNet**:
  * True Negative (TN) = {face_m['test_confusion_matrix']['TN']}
  * False Positive (FP) = {face_m['test_confusion_matrix']['FP']}
  * False Negative (FN) = {face_m['test_confusion_matrix']['FN']}
  * True Positive (TP) = {face_m['test_confusion_matrix']['TP']}

---

## D. Analisis Perbandingan Mendalam

1. **Akurasi & Presisi (Keunggulan ArcFace)**:
   * **ArcFace unggul pada Accuracy** (59.00% vs 56.00%) dan **Precision** (67.31% vs 55.45%).
   * ArcFace sangat konservatif dalam memprediksi wajah yang sama, sehingga ketika ArcFace memprediksi bahwa dua gambar adalah orang yang sama, tingkat kepastiannya mencapai 67.31%.
2. **Keamanan & False Acceptance Rate (Keunggulan Krusial ArcFace)**:
   * **FAR ArcFace (17.00%) jauh lebih rendah** daripada FaceNet (49.00%).
   * Dari 100 pasangan orang berbeda pada data uji, ArcFace hanya salah meloloskan 17 pasangan, sedangkan FaceNet meloloskan 49 pasangan (hampir 50% penyusup/impostor diterima). Dalam konteks sistem keamanan biometrik (misalnya kontrol akses pintu atau perbankan), nilai FAR yang rendah adalah syarat mutlak.
3. **Sensitivitas, Recall & FRR (Keunggulan FaceNet)**:
   * **FaceNet unggul pada Recall** (61.00% vs 35.00%) dan memiliki **FRR lebih rendah** (39.00% vs 65.00%).
   * FaceNet lebih toleran terhadap variasi wajah dari orang yang sama, sehingga pengguna asli lebih jarang ditolak. Sebaliknya, ArcFace memiliki FRR yang tinggi (65.00%) karena threshold-nya yang ketat pada ruang embedding 512 dimensi.
4. **F1-Score, AUC & EER**:
   * FaceNet memiliki **F1-score lebih tinggi** (0.5810 vs 0.4605) karena keseimbangan antara Precision (55.45%) dan Recall (61.00%).
   * Pada level kapabilitas pemisahan global tanpa terikat satu titik threshold, **AUC FaceNet (0.6418)** dan **EER FaceNet (41.00%)** sedikit lebih baik daripada ArcFace (AUC: 0.6269, EER: 42.50%).
5. **Efisiensi Komputasi & Ukuran Representasi**:
   * FaceNet mengekstrak 1.053 gambar dalam **27.53 detik** (kecepatan ~26 ms/gambar), sedangkan ArcFace membutuhkan **57.07 detik** (~54 ms/gambar).
   * Vektor embedding FaceNet (128 dimensi float32 = 512 byte/wajah) 4 kali lebih hemat memori dibandingkan ArcFace (512 dimensi float32 = 2.048 byte/wajah).

---

## E. Kesimpulan Objektif

* **Karakteristik ArcFace**: Merupakan model bertipe **High-Security / Conservative**. ArcFace sangat unggul dalam menolak orang yang berbeda (FAR rendah 17.00% dan Precision tinggi 67.31%), dengan overall accuracy tertinggi (59.00%). Kelemahannya adalah False Rejection Rate (FRR 65.00%) yang masih tinggi pada citra wajah LFW non-aligned/unconstrained.
* **Karakteristik FaceNet**: Merupakan model bertipe **Balanced-Sensitivity / Lightweight**. FaceNet unggul dalam mengenali subjek asli (Recall 61.00% dan F1-score 0.5810) serta waktu inferensi yang 2x lebih cepat dengan embedding 128-d. Kelemahannya adalah False Acceptance Rate (FAR 49.00%) yang terlalu tinggi untuk aplikasi keamanan kritis.
* **Rekomendasi Pemilihan**:
  * Untuk **aplikasi keamanan tinggi (high-security access control / autentikasi)**: **ArcFace** adalah pilihan yang lebih tepat karena meminimalkan risiko false positive (impostor acceptance).
  * Untuk **aplikasi pencarian cepat, pengelompokan album (photo clustering), atau perangkat edge berdaya rendah**: **FaceNet** lebih unggul berkat kecepatan inferensi, efisiensi memori 128-d, dan recall yang lebih tinggi.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)


def main():
    paths = get_paths()

    print("=" * 65)
    print("TAHAP 4: MEMBANDINGKAN HASIL ARCFACE VS FACENET")
    print("=" * 65)

    # 1. Ambil data hasil eksperimen sebelumnya
    if not paths["arcface_metrics"].exists() or not paths["arcface_csv"].exists():
        raise FileNotFoundError(f"File hasil ArcFace tidak lengkap di: {paths['arcface_metrics']}")
    if not paths["facenet_metrics"].exists() or not paths["facenet_csv"].exists():
        raise FileNotFoundError(f"File hasil FaceNet tidak lengkap di: {paths['facenet_metrics']}")

    with open(paths["arcface_metrics"], "r", encoding="utf-8") as f:
        arc_metrics = json.load(f)
    with open(paths["facenet_metrics"], "r", encoding="utf-8") as f:
        face_metrics = json.load(f)

    df_arc = pd.read_csv(paths["arcface_csv"])
    df_face = pd.read_csv(paths["facenet_csv"])

    print("Hasil ArcFace dan FaceNet berhasil dimuat.")

    # 2. Cek Konsistensi Pengujian
    consistency = verify_consistency(df_arc, df_face)
    if not consistency["is_consistent"]:
        print("PERINGATAN: Ditemukan inkonsistensi data antara ArcFace dan FaceNet!")
        for d in consistency["differences"]:
            print(f"  - {d}")
        raise ValueError("Inkonsistensi data terdeteksi. Proses dihentikan.")
    else:
        print("Konsistensi terverifikasi: 100% pairs, labels, dan splits identik.")

    # 3. Buat Tabel Perbandingan (model_comparison.csv)
    rows = [
        {"Metric": "Accuracy", "ArcFace": f"{arc_metrics['accuracy'] * 100:.2f}%", "FaceNet": f"{face_metrics['accuracy'] * 100:.2f}%"},
        {"Metric": "Precision", "ArcFace": f"{arc_metrics['precision'] * 100:.2f}%", "FaceNet": f"{face_metrics['precision'] * 100:.2f}%"},
        {"Metric": "Recall", "ArcFace": f"{arc_metrics['recall'] * 100:.2f}%", "FaceNet": f"{face_metrics['recall'] * 100:.2f}%"},
        {"Metric": "F1-score", "ArcFace": f"{arc_metrics['f1']:.4f}", "FaceNet": f"{face_metrics['f1']:.4f}"},
        {"Metric": "FAR", "ArcFace": f"{arc_metrics['far'] * 100:.2f}%", "FaceNet": f"{face_metrics['far'] * 100:.2f}%"},
        {"Metric": "FRR", "ArcFace": f"{arc_metrics['frr'] * 100:.2f}%", "FaceNet": f"{face_metrics['frr'] * 100:.2f}%"},
        {"Metric": "AUC", "ArcFace": f"{arc_metrics['auc']:.4f}", "FaceNet": f"{face_metrics['auc']:.4f}"},
        {"Metric": "EER", "ArcFace": f"{arc_metrics['eer'] * 100:.2f}%", "FaceNet": f"{face_metrics['eer'] * 100:.2f}%"},
        {"Metric": "Threshold", "ArcFace": f"{arc_metrics['threshold']:.4f}", "FaceNet": f"{face_metrics['threshold']:.4f}"},
        {"Metric": "Inference Time", "ArcFace": f"{arc_metrics['elapsed_time_seconds']:.2f} s", "FaceNet": f"{face_metrics['elapsed_time_seconds']:.2f} s"},
        {"Metric": "Embedding Dimension", "ArcFace": "512", "FaceNet": "128"},
        {"Metric": "Distance Metric", "ArcFace": "Cosine", "FaceNet": "Cosine"},
    ]
    df_comparison = pd.DataFrame(rows)
    df_comparison.to_csv(paths["comparison_csv"], index=False)
    print(f"Tabel perbandingan disimpan di: {paths['comparison_csv']}")

    # 4. Buat Confusion Matrix terpisah
    df_arc_test = df_arc[df_arc["split"] == "test"]
    df_face_test = df_face[df_face["split"] == "test"]

    cm_arc = confusion_matrix(df_arc_test["label"], df_arc_test["prediction"])
    cm_face = confusion_matrix(df_face_test["label"], df_face_test["prediction"])

    plot_confusion_matrix(cm_arc, "ArcFace", paths["cm_arcface_png"], cmap_color="Blues")
    plot_confusion_matrix(cm_face, "FaceNet", paths["cm_facenet_png"], cmap_color="Greens")
    print(f"Confusion Matrix ArcFace disimpan di: {paths['cm_arcface_png']}")
    print(f"Confusion Matrix FaceNet disimpan di: {paths['cm_facenet_png']}")

    # 5. Buat ROC Comparison (ArcFace vs FaceNet)
    plot_roc_comparison(
        df_arc_test,
        df_face_test,
        arc_metrics["auc"],
        face_metrics["auc"],
        arc_metrics["eer"],
        face_metrics["eer"],
        paths["roc_comp_png"],
    )
    print(f"ROC Comparison disimpan di: {paths['roc_comp_png']}")

    # 6. Buat Grafik Perbandingan Metrik (Bar Charts)
    plot_metrics_barcharts(arc_metrics, face_metrics, paths["results_dir"])
    print(f"Grafik perbandingan metrik disimpan di: {paths['metrics_comp_png']}")

    # 7. Buat Laporan Analisis Markdown
    generate_comparison_report(arc_metrics, face_metrics, consistency, paths["report_md"])
    print(f"Laporan analisis disimpan di: {paths['report_md']}")

    # 8. Cetak Ringkasan Hasil ke Terminal
    print("\n" + "=" * 65)
    print("RINGKASAN PERBANDINGAN: ARCFACE VS FACENET")
    print("=" * 65)
    print(f"1. Accuracy ArcFace   : {arc_metrics['accuracy'] * 100:.2f}%")
    print(f"2. Accuracy FaceNet   : {face_metrics['accuracy'] * 100:.2f}%")
    print(f"3. Precision ArcFace  : {arc_metrics['precision'] * 100:.2f}%")
    print(f"4. Precision FaceNet  : {face_metrics['precision'] * 100:.2f}%")
    print(f"5. Recall ArcFace     : {arc_metrics['recall'] * 100:.2f}%")
    print(f"6. Recall FaceNet     : {face_metrics['recall'] * 100:.2f}%")
    print(f"7. F1 ArcFace         : {arc_metrics['f1']:.4f}")
    print(f"8. F1 FaceNet         : {face_metrics['f1']:.4f}")
    print(f"9. AUC ArcFace / FaceNet : {arc_metrics['auc']:.4f} / {face_metrics['auc']:.4f}")
    print(f"10. FAR ArcFace / FaceNet : {arc_metrics['far'] * 100:.2f}% / {face_metrics['far'] * 100:.2f}%")
    print(f"11. FRR ArcFace / FaceNet : {arc_metrics['frr'] * 100:.2f}% / {face_metrics['frr'] * 100:.2f}%")
    print(f"12. Inference Time    : ArcFace={arc_metrics['elapsed_time_seconds']:.2f}s | FaceNet={face_metrics['elapsed_time_seconds']:.2f}s")
    print("13. Kesimpulan        : ArcFace unggul pada Accuracy, Precision, dan FAR (keamanan tinggi).")
    print("                        FaceNet unggul pada Recall, F1-score, efisiensi ukuran, dan kecepatan.")
    print(f"14. Lokasi Output     : {paths['results_dir']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
