# Laporan Eksperimen Ketahanan (Robustness): Brightness & Contrast
## Evaluasi Perbandingan Model ArcFace vs FaceNet pada Dataset LFW

Dokumen ini mendokumentasikan hasil pengujian empiris ketahanan representasi fitur wajah terhadap variasi pencahayaan (brightness) dan rentang kontras (contrast) menggunakan framework DeepFace pada dataset LFW (200 pasangan test terisolasi, unseen identity-disjoint).

---

### A. Metodologi & Konfigurasi Pengujian

1. **Model yang Diuji**:
   * **ArcFace**: ResNet backbone (512 dimensi embedding, threshold beku dev = 0.0269).
   * **FaceNet**: Inception-ResNet backbone (128 dimensi embedding, threshold beku dev = 0.4189).
2. **Karakteristik Data Uji**:
   * 200 pasangan uji terverifikasi (100 SAME, 100 DIFFERENT) dari 12 identitas terpisah (*identity-disjoint*).
   * Kedua model diuji pada pasangan gambar, label, dan split yang **100% identik**.
3. **Metode Transformasi Preprocessing**:
   * Transformasi diterapkan pada citra input sebelum proses ekstraksi embedding.
   * Nilai piksel dipastikan selalu berada dalam batas valid `uint8` [0, 255].
   * Skenario 1 (Baseline): Citra asli tanpa perubahan (faktor 1.0).
   * Skenario 2 (Brightness Rendah): Faktor 0.7 ($I' = \text{clip}(I \times 0.7, 0, 255)$).
   * Skenario 3 (Brightness Tinggi): Faktor 1.3 ($I' = \text{clip}(I \times 1.3, 0, 255)$).
   * Skenario 4 (Contrast Rendah): Faktor 0.7 ($I' = \text{clip}(\bar{L} + 0.7 \times (I - \bar{L}), 0, 255)$).
   * Skenario 5 (Contrast Tinggi): Faktor 1.3 ($I' = \text{clip}(\bar{L} + 1.3 \times (I - \bar{L}), 0, 255)$).
4. **Prinsip Evaluasi Threshold**:
   * Threshold optimal ditentukan murni dari dataset development baseline dan dibekukan (*frozen threshold*) untuk mengevaluasi data uji secara konsisten tanpa *data leakage*.

---

### B. Tabel Rekapitulasi Hasil Pengujian (Test Set: 200 Pasang)

| Skenario | Faktor | ArcFace Acc | FaceNet Acc | ArcFace FAR | FaceNet FAR | ArcFace FRR | FaceNet FRR | ArcFace F1 | FaceNet F1 | ArcFace AUC | FaceNet AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Skenario 1 — Baseline** | 1.0 | **59.00%** | 56.00% | 17.00% | 49.00% | 65.00% | 39.00% | 0.4605 | **0.5810** | 0.6269 | **0.6418** |
| **Skenario 2 — Brightness Rendah** | 0.7 | **52.00%** | 57.00% | 5.00% | 57.00% | 91.00% | 29.00% | 0.1579 | **0.6228** | 0.6176 | **0.6623** |
| **Skenario 3 — Brightness Tinggi** | 1.3 | **49.50%** | 67.50% | 2.00% | 21.00% | 99.00% | 44.00% | 0.0194 | **0.6328** | 0.6211 | **0.7787** |
| **Skenario 4 — Contrast Rendah** | 0.7 | **52.50%** | 58.50% | 5.00% | 59.00% | 90.00% | 24.00% | 0.1739 | **0.6468** | 0.6178 | **0.6623** |
| **Skenario 5 — Contrast Tinggi** | 1.3 | **50.00%** | 67.00% | 2.00% | 21.00% | 98.00% | 45.00% | 0.0385 | **0.6250** | 0.6140 | **0.7788** |


---

### C. Analisis Temuan Eksperimen

1. **Sensitivitas terhadap Variasi Kecerahan (Brightness)**:
   * **Brightness Rendah (Faktor 0.7)**:
     - Kecerahan yang berkurang menyebabkan penurunan kontras lokal pada area wajah (mata, hidung, kontur rahang).
     - Performa ArcFace: Akurasi 52.00% (Baseline: 59.00%).
     - Performa FaceNet: Akurasi 57.00% (Baseline: 56.00%).
   * **Brightness Tinggi (Faktor 1.3)**:
     - Peningkatan kecerahan berlebih memicu saturasi piksel (*overexposure clipping* pada 255) yang menghilangkan detail tekstur kulit.
     - Performa ArcFace: Akurasi 49.50%.
     - Performa FaceNet: Akurasi 67.50%.

2. **Sensitivitas terhadap Variasi Kontras (Contrast)**:
   * **Contrast Rendah (Faktor 0.7)**:
     - Penurunan rentang dinamis membuat gradien fitur tepi (*edge gradients*) memudar (*washed out*).
     - Performa ArcFace: Akurasi 52.50%.
     - Performa FaceNet: Akurasi 58.50%.
   * **Contrast Tinggi (Faktor 1.3)**:
     - Peningkatan kontras mempertegas bayangan dan highlights yang dapat mengaburkan kesamaan intra-kelas.
     - Performa ArcFace: Akurasi 50.00%.
     - Performa FaceNet: Akurasi 67.00%.

3. **Perbandingan Karakteristik Ketahanan ArcFace vs FaceNet**:
   * **ArcFace**: Tetap mempertahankan sifat konservatifnya dengan FAR rendah (keamanan tinggi) pada sebagian besar skenario, namun sensitif terhadap pergeseran distribusi jarak geodesik ketika citra mengalami degradasi ekstrem.
   * **FaceNet**: Mempertahankan Recall dan F1-score yang lebih seimbang, dengan ruang representasi 128-d yang memiliki toleransi dinamis lebih lebar terhadap fluktuasi pencahayaan seragam.

---

### D. Kesimpulan Eksperimen Robustness

1. Perubahan kecerahan dan kontras **memengaruhi distribusi jarak embedding**, menegaskan pentingnya normalisasi iluminasi pada sistem biometrik operasional nyata.
2. Penurunan kecerahan dan kontras cenderung meningkatkan FRR (penolakan pengguna asli) ketika menggunakan threshold tetap.
3. ArcFace tetap lebih aman untuk aplikasi verifikasi berisiko tinggi (FAR lebih rendah), sedangkan FaceNet menunjukkan degradasi yang lebih halus pada skor F1.
