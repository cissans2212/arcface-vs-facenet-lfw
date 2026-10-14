# Laporan Perbandingan Model Face Verification: ArcFace vs FaceNet

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
   * Distance Metric: Cosine Distance ($1 - \text{cosine\_similarity}$).
2. **FaceNet (Triplet Loss)**:
   * Arsitektur: Inception-ResNet backbone (DeepFace pretrained)
   * Dimensi Embedding: **128 dimensi**
   * Fungsi Loss Asli: Triplet Loss pada ruang embedding Euclidean.
   * Detector Backend: `skip` (croppped centered face).
   * Distance Metric: Cosine Distance ($1 - \text{cosine\_similarity}$).

---

## C. Tabel Hasil Perbandingan (Test Set)

| Metrik Evaluasi | ArcFace | FaceNet | Selisih / Catatan |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **59.00%** | 56.00% | ArcFace lebih unggul **+3.00%** |
| **Precision** | **67.31%** | 55.45% | ArcFace lebih presisi **+11.86%** |
| **Recall** | 35.00% | **61.00%** | FaceNet mendeteksi lebih banyak positif (**+26.00%**) |
| **F1-Score** | 0.4605 | **0.5810** | FaceNet memiliki F1 lebih seimbang (**+0.1205**) |
| **FAR (False Acceptance Rate)** | **17.00%** | 49.00% | ArcFace jauh lebih aman (FAR lebih rendah **-32.00%**) |
| **FRR (False Rejection Rate)** | 65.00% | **39.00%** | FaceNet menolak lebih sedikit wajah asli (**-26.00%**) |
| **AUC (Area Under ROC Curve)** | 0.6269 | **0.6418** | FaceNet memiliki AUC sedikit lebih tinggi (**+0.0149**) |
| **EER (Equal Error Rate)** | 42.50% | **41.00%** | FaceNet memiliki EER sedikit lebih rendah (**-1.50%**) |
| **Optimal Threshold (Dev Set)** | 0.0269 | 0.4189 | Ditentukan murni pada Development Set |
| **Dimensi Embedding** | 512 | 128 | FaceNet lebih ringkas (1/4 ukuran ArcFace) |
| **Inference Time (1.053 citra)** | ~57.07 s | **27.53 s** | FaceNet ~2x lebih cepat dalam ekstraksi embedding |

### Rincian Confusion Matrix (Test Set: 200 Pasang)
* **ArcFace**:
  * True Negative (TN) = 83 (Different benar ditebak Different)
  * False Positive (FP) = 17 (Different salah ditebak Same / Impostor lolos)
  * False Negative (FN) = 65 (Same salah ditebak Different / Wajah asli ditolak)
  * True Positive (TP) = 35 (Same benar ditebak Same)
* **FaceNet**:
  * True Negative (TN) = 51
  * False Positive (FP) = 49
  * False Negative (FN) = 39
  * True Positive (TP) = 61

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
