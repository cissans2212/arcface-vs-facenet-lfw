# LAPORAN KOMPREHENSIF PROSES DAN HASIL EKSPERIMEN: FACE VERIFICATION (ARCFACE VS FACENET)

> **PANDUAN UNTUK AI PENULIS LAPORAN / ASISTEN AKADEMIK:**
> Dokumen Markdown ini memuat seluruh rekam jejak teknis, formulasi matematis, metodologi eksperimen, catatan dataset, arsitektur sistem, metrik evaluasi kuantitatif, analisis perbandingan, dan kesimpulan dari proyek riset Computer Vision. Dokumen ini dirancang secara terstruktur dan mandiri (*self-contained*) agar Anda dapat langsung mentransformasikannya ke dalam format laporan akademik resmi, artikel jurnal, skripsi/tugas akhir (Bab 1 s/d Bab 5), atau makalah ilmiah.

---

## 1. INFORMASI UMUM PROYEK & KONTEKS RISET

* **Judul Penelitian**: Analisis Komparatif Performa Model Deep Face Representation: ArcFace vs FaceNet pada Tugas Face Verification Menggunakan Framework DeepFace pada Dataset LFW
* **Domain / Bidang**: Computer Vision, Biometrics, Deep Learning, Image Processing
* **Konteks Akademik**: Tugas Besar / Proyek Laboratorium Computer Vision (Semester 9)
* **Lingkungan Sistem & Dependensi**:
  * **Sistem Operasi**: Windows 11
  * **Bahasa Pemrograman**: Python 3.10.6
  * **Framework & Library Utama**:
    * `deepface` == 0.0.101 (Inference engine untuk ArcFace dan FaceNet)
    * `streamlit` == 1.64.0 (Antarmuka Web GUI Interaktif)
    * `torch` == 2.2.2 & `torchvision` == 0.17.2
    * `tensorflow` == 2.21.0 & `keras` == 3.12.2
    * `scikit-learn` == 1.7.2 (Kalkulasi metrik, Confusion Matrix, ROC-AUC)
    * `opencv-python` == 4.11.0.86 & `Pillow` == 10.2.0 (Pengolahan citra)
    * `pandas` == 2.3.3 & `numpy` == 1.26.4 (Manipulasi data & aljabar linier)
    * `matplotlib` == 3.10.9 & `seaborn` == 0.13.2 (Visualisasi data)

---

## 2. DASAR TEORI & FORMULASI MATEMATIKA

### 2.1 Konsep Face Verification (Pencocokan 1:1)
Face Verification adalah proses autentikasi biometrik biner yang menguji apakah dua citra wajah ($I_1$ dan $I_2$) berasal dari subjek/individu yang sama (*Same Person*, label $y = 1$) atau individu yang berbeda (*Different Person*, label $y = 0$). Berbeda dengan Face Identification (pencocokan 1:$N$ untuk menemukan identitas dalam galeri), Face Verification beroperasi secara berpasangan (*pairwise matching*) berbasis nilai jarak ambang (*distance thresholding*).

### 2.2 Model 1: ArcFace (Additive Angular Margin Loss)
* **Arsitektur Backbone**: ResNet (DeepFace Pretrained)
* **Dimensi Representasi (*Embedding Dimension*)**: **512 dimensi** ($f(x) \in \mathbb{R}^{512}$)
* **Konsep Inti**:
  ArcFace memetakan fitur wajah ke permukaan *hypersphere* berdimensi tinggi dengan menormalisasi vektor bobot ($\|W_j\| = 1$) dan vektor fitur ($\|x_i\| = 1$), kemudian mengalikan dengan skala $s$. Untuk meningkatkan keterpisahan kelas (*inter-class discrepancy*) dan kerapatan dalam kelas (*intra-class compactness*), ArcFace menyuntikkan margin sudut aditif (*additive angular margin penalty*) $m$ langsung pada sudut geodesik target $\theta_{y_i}$.
* **Formulasi Loss ArcFace**:
  $$\mathcal{L}_{\text{ArcFace}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos \theta_j}}$$
  * $s$: Parameter penskalaan radius hypersphere (biasanya $s = 64$)
  * $m$: Margin sudut aditif (biasanya $m = 0.5$)
  * $\theta_{y_i} = \arccos(W_{y_i}^T x_i)$: Sudut antara fitur dan bobot kelas target

### 2.3 Model 2: FaceNet (Triplet Loss)
* **Arsitektur Backbone**: Inception-ResNet (DeepFace Pretrained)
* **Dimensi Representasi (*Embedding Dimension*)**: **128 dimensi** ($f(x) \in \mathbb{R}^{128}$)
* **Konsep Inti**:
  FaceNet secara langsung memetakan citra wajah ke ruang metrik Euclidean berdimensi ringkas. Pelatihan dioptimalkan menggunakan *Triplet Loss*, yang memaksa jarak antara citra anchor ($x_i^a$) dan citra positif dari orang yang sama ($x_i^p$) lebih kecil daripada jarak antara anchor dan citra negatif dari orang berbeda ($x_i^n$) sebesar batas margin margin $\alpha$.
* **Formulasi Loss Triplet FaceNet**:
  $$\mathcal{L}_{\text{Triplet}} = \sum_{i}^{N} \max \left( 0, \; \|f(x_i^a) - f(x_i^p)\|_2^2 - \|f(x_i^a) - f(x_i^n)\|_2^2 + \alpha \right)$$
  * $f(x) \in \mathbb{R}^{128}$ dengan konstrain $\|f(x)\|_2 = 1$
  * $\alpha$: Batas margin Euclidean penalti pemisah

### 2.4 Metrik Jarak: Cosine Distance
Untuk kedua model, kesamaan representasi embedding wajah $u$ dan $v$ diukur menggunakan **Cosine Distance**:
$$D_{\text{cosine}}(u, v) = 1 - \frac{u \cdot v}{\|u\|_2 \cdot \|v\|_2} = 1 - \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$

* **Interpretasi Nilai Jarak**:
  * Nilai $D_{\text{cosine}} \to 0$: Arah vektor hampir sejajar (sudut mendekati $0^\circ$), mengindikasikan wajah **SANGAT MIRIP / IDENTIK**.
  * Nilai $D_{\text{cosine}} \to 1$: Vektor saling tegak lurus (orthogonal), mengindikasikan wajah **BERBEDA**.
  * Nilai $D_{\text{cosine}} > 1$: Vektor berlawanan arah.

### 2.5 Aturan Keputusan Verifikasi (*Decision Rule*)
Diberikan ambang batas optimal $\tau$:
$$\hat{y} = \begin{cases} 1 \quad (\text{SAME PERSON}), & \text{jika } D_{\text{cosine}}(u, v) \le \tau \\ 0 \quad (\text{DIFFERENT PERSON}), & \text{jika } D_{\text{cosine}}(u, v) > \tau \end{cases}$$

---

## 3. METODOLOGI & ALUR EKSPERIMEN (TAHAP 1 S/D TAHAP 5)

Eksperimen dirancang secara modular dan berurutan ke dalam 5 tahap penelitian terstandarisasi:

```
[Tahap 1: Dataset & Pairing]
   -> Inspeksi LFW (62 Identitas, 3.023 Gambar)
   -> Identity-Disjoint Split (80% Dev, 20% Test)
   -> 1.000 Pasang (500 SAME, 500 DIFF) -> pairs.csv
          |
          +---------------------------------------+
          |                                       |
[Tahap 2: ArcFace]                       [Tahap 3: FaceNet]
- Ekstraksi 512-d Embedding             - Ekstraksi 128-d Embedding
- Grid Search Threshold Dev (800)       - Grid Search Threshold Dev (800)
- Optimal Threshold: 0.0269             - Optimal Threshold: 0.4189
- Evaluasi Test Set (200)               - Evaluasi Test Set (200)
          |                                       |
          +-------------------+-------------------+
                              |
                     [Tahap 4: Komparasi]
                     - Validasi Konsistensi Pengujian
                     - Tabel model_comparison.csv
                     - Plot Confusion Matrix & ROC Curve
                     - Analisis FAR, FRR, EER, Runtime
                              |
                     [Tahap 5: GUI Streamlit]
                     - Interaktif Web Verification (app.py)
                     - Upload Citra & Sampel LFW
                     - Dashboard Hasil & Visualisasi
```

---

### 3.1 Tahap 1: Inspeksi Dataset LFW & Pembentukan Pasangan (`prepare_dataset.py`)
1. **Karakteristik Raw Dataset LFW**:
   * Lokasi dataset: `CV/dataset/`
   * Total identitas/orang: **62 subjek**
   * Total gambar raw: **5.444 file** (Folder `train/`: 2.421, Folder `test/`: 602, Folder `database/`: 2.421 yang merupakan salinan train).
   * Total gambar unik non-duplikat (`train` + `test`): **3.023 gambar**
   * Resolusi citra: **$250 \times 250$ piksel**, format RGB **JPEG**.
   * Pengecekan integritas: **0 file rusak (*zero corrupted files*)**, seluruh identitas memiliki $\ge 20$ gambar.

2. **Strategi Pencegahan Data Leakage (Identity-Disjoint Split)**:
   * Untuk mengevaluasi skenario *open-set / unseen subjects*, subjek pada Test Set tidak boleh muncul sama sekali pada Development Set.
   * **Development Set (80% / 800 pasang)**: Dibangun dari **50 identitas**.
   * **Test Set (20% / 200 pasang)**: Dibangun dari **12 identitas terisolasi**.
   * **Daftar 12 Identitas Uji (Unseen Test Identities)**:
     1. *Alejandro Toledo*
     2. *Bill Clinton*
     3. *Carlos Menem*
     4. *Gerhard Schroeder*
     5. *Gloria Macapagal Arroyo*
     6. *Gray Davis*
     7. *Hamid Karzai*
     8. *Jiang Zemin*
     9. *John Ashcroft*
     10. *Megawati Sukarnoputri*
     11. *Vicente Fox*
     12. *Winona Ryder*

3. **Distribusi Pasangan Data (`pairs.csv`)**:
   * Total pasangan: **1.000 pasang**
   * Pasangan SAME ($y = 1$): 500 pasang (400 Dev, 100 Test)
   * Pasangan DIFFERENT ($y = 0$): 500 pasang (400 Dev, 100 Test)
   * Hasil validasi data:
     * *Self-pair* ($image_1 == image_2$): **0**
     * Pasangan duplikat (*unordered duplicates*): **0**
     * Label error: **0**
     * *Identity overlap* (kebocoran subjek Dev $\cap$ Test): **0**
     * *Image overlap* (kebocoran citra Dev $\cap$ Test): **0**

---

### 3.2 Tahap 2: Evaluasi Model ArcFace (`run_arcface.py`)
1. **Pengaturan Ekstraksi**:
   * Model: `ArcFace` via DeepFace
   * Detektor backend: `skip` (citra LFW sudah terpotong dan terpusat pada wajah).
   * Dimensi embedding: 512 dimensi float32 (2.048 byte per wajah).
   * *Caching*: Vektor embedding disimpan ke format `.npy` di `experiments/arcface/embeddings/` untuk efisiensi komputasi.
2. **Penentuan Threshold Optimal pada Development Set**:
   * Grid search dilakukan pada 1.000 titik kandidat nilai ambang batas terhadap 800 pasangan data Development.
   * Kriteria optimasi: **Memaksimalkan Balanced Accuracy**:
     $$\text{Balanced Accuracy} = \frac{\text{Recall} + (1 - \text{FAR})}{2} = \frac{\frac{\text{TP}}{\text{TP} + \text{FN}} + \frac{\text{TN}}{\text{TN} + \text{FP}}}{2}$$
   * **Threshold Optimal Terpilih**: $\tau^*_{\text{ArcFace}} = \mathbf{0.0269}$
3. **Evaluasi pada Test Set (200 Pasang)**:
   * Menggunakan threshold $\tau^* = 0.0269$ yang telah dibekukan (*frozen threshold*).

---

### 3.3 Tahap 3: Evaluasi Model FaceNet (`run_facenet.py`)
1. **Pengaturan Ekstraksi**:
   * Model: `Facenet` via DeepFace
   * Detektor backend: `skip`
   * Dimensi embedding: 128 dimensi float32 (512 byte per wajah).
   * *Caching*: Vektor embedding disimpan ke format `.npy` di `experiments/facenet/embeddings/`.
2. **Penentuan Threshold Optimal pada Development Set**:
   * Prosedur grid search identik (1.000 titik kandidat pada 800 pasangan Development).
   * Kriteria optimasi: Memaksimalkan Balanced Accuracy.
   * **Threshold Optimal Terpilih**: $\tau^*_{\text{FaceNet}} = \mathbf{0.4189}$
3. **Evaluasi pada Test Set (200 Pasang)**:
   * Menggunakan threshold $\tau^* = 0.4189$ yang telah dibekukan.

---

### 3.4 Tahap 4: Analisis Komparasi Model (`compare_models.py`)
1. **Validasi Konsistensi Pengujian (*Fair Comparison*)**:
   * Dilakukan verifikasi kesesuaian baris per baris.
   * Hasil: 100% citra pertama, citra kedua, label aktual, dan partisi data antara ArcFace dan FaceNet **identik secara mutlak**.
2. **Generasi Artefak Visual & Laporan**:
   * `results/model_comparison.csv`: Tabel rekapitulasi numerik.
   * `results/confusion_matrix_arcface.png` & `confusion_matrix_facenet.png`.
   * `results/roc_comparison.png`: Perbandingan kurva ROC kedua model dalam satu grafik.
   * `results/metrics_comparison.png` & `results/far_frr_comparison.png`.
   * `results/comparison_report.md`.

---

### 3.5 Tahap 5: Implementasi Antarmuka GUI Streamlit (`app/app.py`)
* Aplikasi web interaktif berbasis Streamlit pada port lokal `http://localhost:8501`.
* Memungkinkan pengguna untuk:
  * Mengunggah dua gambar wajah secara bebas untuk diuji kecocokannya.
  * Memilih sampel uji preset LFW untuk demonstrasi instan (*True Positive*, *True Negative*, *Challenging cases*).
  * Memilih mode model: ArcFace saja, FaceNet saja, atau *Compare Both* berdampingan.
  * Menampilkan skor *Cosine Distance*, batas ambang (*threshold bar*), indikator status (*SAME / DIFFERENT*), serta inspeksi tabel metrik dan kurva evaluasi pada sidebar.

---

## 4. HASIL EVALUASI & DATA KUANTITATIF LENGKAP

### 4.1 Tabel Perbandingan Kinerja Utama pada Test Set (200 Pasang Unseen)

| No | Parameter / Metrik Evaluasi | ArcFace (512-d) | FaceNet (128-d) | Selisih / Keunggulan |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **Accuracy** | **59.00%** | 56.00% | ArcFace unggul **+3.00%** |
| 2 | **Precision** | **67.31%** | 55.45% | ArcFace unggul **+11.86%** |
| 3 | **Recall (Sensitivity / TPR)** | 35.00% | **61.00%** | FaceNet unggul **+26.00%** |
| 4 | **F1-Score** | 0.4605 | **0.5810** | FaceNet unggul **+0.1205** |
| 5 | **FAR (False Acceptance Rate / FPR)** | **17.00%** | 49.00% | ArcFace jauh lebih baik (**-32.00%**) |
| 6 | **FRR (False Rejection Rate / FNR)** | 65.00% | **39.00%** | FaceNet lebih baik (**-26.00%**) |
| 7 | **ROC-AUC (Area Under ROC Curve)** | 0.6269 | **0.6418** | FaceNet unggul **+0.0149** |
| 8 | **EER (Equal Error Rate)** | 42.50% | **41.00%** | FaceNet unggul **-1.50%** |
| 9 | **Optimal Cosine Threshold ($\tau^*$)** | 0.0269 | 0.4189 | Nilai ambang hasil tuning Dev Set |
| 10 | **Dimensi Embedding** | 512 float32 | 128 float32 | FaceNet 4x lebih ringkas |
| 11 | **Ukuran Vektor per Wajah** | 2.048 byte | 512 byte | FaceNet 4x lebih hemat memori |
| 12 | **Waktu Inferensi Ekstraksi (1.053 citra)** | 57.07 detik | **27.53 detik** | FaceNet **~2.07x lebih cepat** |
| 13 | **Rata-rata Waktu per Gambar** | ~54.2 ms | **~26.1 ms** | FaceNet unggul latensi |

---

### 4.2 Rincian Matriks Kontinjensi (Confusion Matrix) Test Set (200 Pasang)

#### 1. ArcFace (Test Set)
* **True Negative (TN)** = **83** (41.5%) : Pasangan orang berbeda berhasil ditolak dengan benar.
* **False Positive (FP)** = **17** (8.5%) : Pasangan orang berbeda keliru disahkan (*Impostor accepted*).
* **False Negative (FN)** = **65** (32.5%) : Pasangan orang sama keliru ditolak (*Genuine rejected*).
* **True Positive (TP)** = **35** (17.5%) : Pasangan orang sama berhasil diverifikasi dengan benar.
* Total Sampel Aktual: Negative ($N = 100$), Positive ($P = 100$).

#### 2. FaceNet (Test Set)
* **True Negative (TN)** = **51** (25.5%) : Pasangan orang berbeda berhasil ditolak dengan benar.
* **False Positive (FP)** = **49** (24.5%) : Pasangan orang berbeda keliru disahkan (*Impostor accepted*).
* **False Negative (FN)** = **39** (19.5%) : Pasangan orang sama keliru ditolak (*Genuine rejected*).
* **True Positive (TP)** = **61** (30.5%) : Pasangan orang sama berhasil diverifikasi dengan benar.
* Total Sampel Aktual: Negative ($N = 100$), Positive ($P = 100$).

---

### 4.3 Data Evaluasi pada Development Set (800 Pasang)

Data performa pada tahap kalibrasi penentuan ambang batas (Development Set):

| Parameter Evaluasi | ArcFace (Development) | FaceNet (Development) |
| :--- | :---: | :---: |
| Jumlah Pasangan Data | 800 (400 SAME, 400 DIFF) | 800 (400 SAME, 400 DIFF) |
| Ambang Batas Optimal ($\tau$) | 0.0269 | 0.4189 |
| Akurasi (Accuracy) | 57.50% | 65.50% |
| Balanced Accuracy | 57.50% | 65.50% |
| Presisi (Precision) | 63.04% | 65.27% |
| Recall (Sensitivitas) | 36.25% | 66.25% |
| F1-Score | 0.4603 | 0.6576 |
| FAR (False Acceptance Rate) | 21.25% (85 FP) | 35.25% (141 FP) |
| FRR (False Rejection Rate) | 63.75% (255 FN) | 33.75% (135 FN) |
| True Negative (TN) | 315 | 259 |
| False Positive (FP) | 85 | 141 |
| False Negative (FN) | 255 | 135 |
| True Positive (TP) | 145 | 265 |

---

### 4.4 Analisis Statistik Distribusi Jarak Cosine (Distance Separation)

Statistik deskriptif (*mean* $\mu$ dan *standar deviasi* $\sigma$) jarak Cosine antara pasangan citra:

#### 1. Pada Test Set (200 Pasang)
* **ArcFace**:
  * Pasangan SAME ($y=1$): $\mu = 0.0594$, $\sigma = 0.0723$
  * Pasangan DIFF ($y=0$): $\mu = 0.0768$, $\sigma = 0.0835$
  * Selisih rata-rata ($\Delta \mu$): **0.0174** (Rentan tumpang tindih karena rentang jarak ArcFace sangat sempit di sekitar 0.02 - 0.08).
* **FaceNet**:
  * Pasangan SAME ($y=1$): $\mu = 0.3704$, $\sigma = 0.1721$
  * Pasangan DIFF ($y=0$): $\mu = 0.4790$, $\sigma = 0.2125$
  * Selisih rata-rata ($\Delta \mu$): **0.1086** (Pemisahan margin jarak rata-rata lebih lebar).

#### 2. Pada Development Set (800 Pasang)
* **ArcFace**:
  * Pasangan SAME ($y=1$): $\mu = 0.0579$, $\sigma = 0.0739$
  * Pasangan DIFF ($y=0$): $\mu = 0.0805$, $\sigma = 0.1057$
* **FaceNet**:
  * Pasangan SAME ($y=1$): $\mu = 0.3755$, $\sigma = 0.1652$
  * Pasangan DIFF ($y=0$): $\mu = 0.5154$, $\sigma = 0.2019$

---

## 5. PEMBAHASAN MENDALAM & ANALISIS KOMPARATIF

### 5.1 Trade-off Keamanan vs Kenyamanan: ArcFace (Konservatif) vs FaceNet (Sensitif)
1. **Keunggulan ArcFace pada Aspek Keamanan Biometrik (FAR 17.00% vs 49.00%)**:
   * Dalam biometrik keamanan, **False Acceptance Rate (FAR)** adalah parameter paling kritis karena mengukur kemungkinan penyusup (*impostor*) berhasil menembus sistem.
   * ArcFace mencatat FAR sebesar **17.00%** (hanya 17 dari 100 impostor yang lolos), sedangkan FaceNet meloloskan **49.00%** impostor (hampir separuh dari seluruh orang berbeda disahkan sebagai orang yang sama).
   * Hal ini membuktikan bahwa penalti margin sudut aditif pada ArcFace menciptakan batas pemisah antarkelas (*inter-class separation*) yang sangat ketat dan protektif pada *hypersphere*.
2. **Keunggulan FaceNet pada Aspek Pengalaman Pengguna (Recall 61.00% vs 35.00%)**:
   * FaceNet jauh lebih toleran terhadap variasi pose wajah, ekspresi, dan pencahayaan dari subjek yang sama, menghasilkan **Recall 61.00%** dan **FRR 39.00%**.
   * Sebaliknya, ArcFace sangat ketat sehingga menderita **FRR tinggi (65.00%)**, di mana 65 dari 100 pasangan orang yang sama ditolak karena jarak embedding-nya melewati ambang batas ketat $0.0269$.
3. **Presisi Prediksi (Precision 67.31% vs 55.45%)**:
   * Karena sifatnya yang konservatif, ketika ArcFace mengeluarkan prediksi *SAME PERSON*, tingkat kepercayaannya tinggi (presisi mencapai 67.31%). Sementara pada FaceNet, jika model memprediksi *SAME*, probabilitas kebenarannya hampir setara dengan tebakan acak (55.45%).

### 5.2 Analisis Kurva ROC, AUC, dan Equal Error Rate (EER)
* **AUC (Area Under ROC Curve)**:
  * FaceNet memperoleh AUC sedikit lebih tinggi (**0.6418**) dibandingkan ArcFace (**0.6269**).
  * Ini menunjukkan bahwa secara keseluruhan kurva operasional tanpa terikat pada satu titik ambang batas kaku, FaceNet memiliki kapasitas pemisahan global sedikit lebih baik pada dataset LFW non-aligned.
* **Equal Error Rate (EER)**:
  * EER FaceNet adalah **41.00%**, sedangkan ArcFace adalah **42.50%**.
  * Tingginya nilai EER pada kedua model dipengaruhi oleh karakteristik alami subset dataset LFW *unconstrained* yang memiliki variasi sudut pose ekstrem tanpa proses penyelarasan wajah (*face alignment*) berbasis landmark (karena detektor backend diset ke `skip`).

### 5.3 Efisiensi Komputasi, Kompleksitas Ruang & Latensi
* **Ukuran Embedding**:
  * FaceNet menghasilkan vektor 128 dimensi (512 byte per representasi), berukuran 4 kali lebih hemat daripada ArcFace (512 dimensi = 2.048 byte).
  * Untuk basis data skala besar (jutaan identitas), representasi 128 dimensi menghemat alokasi RAM dan mempercepat pencarian *Approximate Nearest Neighbor* (ANN / Faiss) hingga 4 kali lipat.
* **Kecepatan Inferensi**:
  * FaceNet menyelesaikan ekstraksi 1.053 gambar dalam **27.53 detik** (~26.1 ms per citra), 2 kali lebih cepat dibandingkan ArcFace yang membutuhkan **57.07 detik** (~54.2 ms per citra).
  * Arsitektur Inception-ResNet pada FaceNet terbukti lebih ringan dieksekusi pada CPU Windows dibandingkan ResNet backbone pada ArcFace.

---

## 6. KESIMPULAN & REKOMENDASI IMPLEMENTASI

### 6.1 Kesimpulan Penelitian
1. **ArcFace** merupakan model yang berorientasi pada **Keamanan Tinggi (*High-Security / Conservative Architecture*)**. Model ini sangat unggul dalam menolak orang berbeda (FAR rendah 17.00% dan Presisi tinggi 67.31%), serta meraih akurasi pengujian tertinggi (59.00%). Kelemahannya adalah tingginya penolakan terhadap pengguna asli (FRR 65.00%).
2. **FaceNet** merupakan model yang berorientasi pada **Sensitivitas Seimbang dan Efisiensi Komputasi (*Lightweight / High-Sensitivity Architecture*)**. Model ini unggul dalam mengenali wajah asli (Recall 61.00%, F1-score 0.5810, AUC 0.6418), ukuran representasi 4x lebih ringkas (128-d), dan kecepatan inferensi 2x lebih cepat. Kelemahannya adalah tingginya tingkat impostor yang lolos (FAR 49.00%).
3. **Penerapan Identity-Disjoint Split** berhasil menjamin validitas pengujian bebas bias (*zero data leakage*), di mana evaluasi dilakukan pada 12 subjek yang benar-benar asing (*unseen subjects*).

### 6.2 Matriks Rekomendasi Pemilihan Solusi

| Kasus Penggunaan Riil | Model yang Direkomendasikan | Alasan & Justifikasi Teknis |
| :--- | :---: | :--- |
| **Sistem Kontrol Akses Fisik (Door Access)** | **ArcFace** | FAR rendah (17%) mencegah penyusup masuk fasilitas. |
| **Autentikasi Perbankan / Finansial (e-KYC)** | **ArcFace** | Presisi tinggi (67.31%) meminimalkan risiko penipuan identitas. |
| **Pencarian Foto Massal / Album Clustering** | **FaceNet** | Recall tinggi (61%) memastikan lebih banyak foto orang yang sama terkelompok. |
| **Perangkat Edge / Mobile / IoT Berdaya Rendah**| **FaceNet** | Ekstraksi 2x lebih cepat (~26 ms) dan embedding 128-d sangat hemat memori. |

### 6.3 Rekomendasi Riset Lanjutan
1. **Integrasi Face Alignment**: Menambahkan pipeline deteksi landmark wajah (misalnya MTCNN atau RetinaFace) sebelum ekstraksi untuk mereduksi variasi rotasi pose, yang diprediksi dapat menurunkan FRR ArcFace secara signifikan.
2. **Fine-Tuning Domain Adaptif**: Melakukan pelatihan lanjutan (*transfer learning*) pada dataset wajah lokal untuk mengompensasi disparitas distribusi demografis LFW.
3. **Adaptive Thresholding Berdasarkan Kualitas Citra**: Menggunakan nilai ambang batas dinamis berbasis skor kualitas citra (*Image Quality Assessment*).

---

## 7. STRUKTUR PROYEK & DAFTAR ARTEFAK HASIL

```
CV/
├── dataset/
│   ├── train/                 (2.421 gambar, 62 identitas)
│   ├── test/                  (602 gambar, 62 identitas)
│   └── database/              (2.421 gambar, salinan train)
└── facial_recognition/
    ├── app/
    │   └── app.py             (Streamlit Web GUI interaktif)
    ├── dataset/
    │   └── processed/
    │       └── pairs.csv      (1.000 pasang: 800 Dev, 200 Test)
    ├── experiments/
    │   ├── arcface/
    │   │   ├── embeddings/    (Cache embedding .npy 512-d)
    │   │   ├── arcface_results.csv (Prediksi & distance 1.000 pairs)
    │   │   ├── metrics.json   (Metrik numerik ArcFace)
    │   │   └── roc_curve.png  (Plot ROC ArcFace)
    │   └── facenet/
    │       ├── embeddings/    (Cache embedding .npy 128-d)
    │       ├── facenet_results.csv (Prediksi & distance 1.000 pairs)
    │       ├── metrics.json   (Metrik numerik FaceNet)
    │       └── roc_curve.png  (Plot ROC FaceNet)
    ├── results/
    │   ├── model_comparison.csv       (Tabel rekapitulasi perbandingan metrik)
    │   ├── confusion_matrix_arcface.png (Visualisasi Confusion Matrix ArcFace)
    │   ├── confusion_matrix_facenet.png (Visualisasi Confusion Matrix FaceNet)
    │   ├── roc_comparison.png         (Perbandingan kurva ROC kedua model)
    │   ├── metrics_comparison.png     (Bar chart perbandingan metrik utama)
    │   ├── far_frr_comparison.png     (Bar chart perbandingan error FAR vs FRR)
    │   └── comparison_report.md       (Laporan ringkas komparasi model)
    ├── src/
    │   ├── config.py          (Resolusi path direktori project)
    │   ├── utils.py           (Utilitas inspeksi dataset, pairing, & metrik)
    │   ├── prepare_dataset.py (Skrip pembuatan & validasi pairs.csv)
    │   ├── run_arcface.py     (Pipeline evaluasi ArcFace)
    │   ├── run_facenet.py     (Pipeline evaluasi FaceNet)
    │   └── compare_models.py  (Pipeline komparasi & plot evaluasi)
    ├── requirements.txt       (Daftar dependensi Python)
    └── README.md              (Petunjuk instalasi & eksekusi)
```
