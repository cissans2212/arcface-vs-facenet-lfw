# Laporan Hasil Evaluasi Face Verification: Dataset Kelas Mahasiswa
**Project:** Computer Vision — Perbandingan Ketahanan ArcFace vs FaceNet (Dataset Kelas)  
**Waktu Pengujian:** 2026-10-18 23:59:34  

---

## 1. Ringkasan Eksekutif
Eksperimen ini mengevaluasi kinerja verifikasi wajah menggunakan model **ArcFace** dan **FaceNet** pada data citra mahasiswa dari folder `dataset/kelas/`. Pengujian dilakukan secara independen terhadap 5 skenario pencahayaan (*illumination robustness*), dengan tetap mengisolasi dan melindungi integritas hasil eksperimen LFW sebelumnya.

### Karakteristik Dataset Uji:
- **Total Identitas Mahasiswa:** 33 orang mahasiswa
- **Total Pasangan Verifikasi:** 200 pasang (100 SAME, 100 DIFFERENT)
- **Metode Preprocessing Citra:** Pemotongan wajah terpusat (*tight crop* 250×250) menggunakan RetinaFace dengan standarisasi JPEG RGB.
- **Isolasi Keamanan:** Folder foto dan cache embedding diproteksi penuh dalam `.gitignore`.

---

## 2. Tabel Perbandingan Hasil Evaluasi

### A. Evaluasi Domain Transfer (Threshold Beku LFW: ArcFace=0.0269, FaceNet=0.4189)
*Mengevaluasi kemampuan generalisasi model saat threshold yang dilatih pada LFW langsung diterapkan pada foto mahasiswa tanpa penyesuaian.*

| Skenario | Faktor | ArcFace Acc | ArcFace F1 | ArcFace AUC | ArcFace EER | FaceNet Acc | FaceNet F1 | FaceNet AUC | FaceNet EER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Skenario 1 — Baseline | 1.0 | 52.00% | 0.0769 | 0.8260 | 23.50% | 74.50% | 0.7119 | 0.8564 | 24.00% |
| Skenario 2 — Brightness Rendah | 0.7 | 52.00% | 0.0769 | 0.8316 | 23.50% | 69.50% | 0.6554 | 0.7973 | 30.50% |
| Skenario 3 — Brightness Tinggi | 1.3 | 52.00% | 0.0769 | 0.8209 | 25.00% | 76.00% | 0.7241 | 0.8724 | 22.00% |
| Skenario 4 — Contrast Rendah | 0.7 | 52.00% | 0.0769 | 0.8291 | 25.00% | 69.00% | 0.6477 | 0.7971 | 29.00% |
| Skenario 5 — Contrast Tinggi | 1.3 | 52.00% | 0.0769 | 0.8194 | 24.50% | 76.50% | 0.7283 | 0.8706 | 23.50% |

### B. Evaluasi Threshold Terkalibrasi (Tuned Lokal untuk Domain Citra Kelas)
*Mengevaluasi potensi akurasi maksimal ketika threshold jarak disesuaikan terhadap rentang variasi pencahayaan dan kamera smartphone mahasiswa.*

| Skenario | Faktor | ArcFace Thresh | ArcFace Acc | ArcFace FAR | ArcFace FRR | FaceNet Thresh | FaceNet Acc | FaceNet FAR | FaceNet FRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Skenario 1 — Baseline | 1.0 | 0.5185 | 80.00% | 3.00% | 37.00% | 0.3903 | 77.00% | 5.00% | 41.00% |
| Skenario 2 — Brightness Rendah | 0.7 | 0.5417 | 82.50% | 6.00% | 29.00% | 0.3715 | 71.50% | 8.00% | 49.00% |
| Skenario 3 — Brightness Tinggi | 1.3 | 0.5305 | 80.00% | 3.00% | 37.00% | 0.5180 | 80.00% | 22.00% | 18.00% |
| Skenario 4 — Contrast Rendah | 0.7 | 0.5352 | 82.50% | 6.00% | 29.00% | 0.3779 | 72.00% | 9.00% | 47.00% |
| Skenario 5 — Contrast Tinggi | 1.3 | 0.5126 | 80.00% | 2.00% | 38.00% | 0.5227 | 80.50% | 22.00% | 17.00% |

---

## 3. Analisis Ilmiah dan Temuan Utama
1. **Dampak Perubahan Pencahayaan (Robustness):**
   - **Degradasi pada Kecerahan & Kontras Rendah:** Pada skenario *Brightness Rendah (0.7)* dan *Contrast Rendah (0.7)*, akurasi FaceNet mengalami penurunan dari **74.50% menjadi 69.50% dan 69.00%**, dengan EER meningkat hingga **30.50%**. Hal ini membuktikan bahwa hilangnya kontur dan detail tekstur akibat reduksi cahaya menyulitkan pembedaan fitur wajah.
   - **Kestabilan pada Kecerahan & Kontras Tinggi:** Skenario *Brightness Tinggi (1.3)* dan *Contrast Tinggi (1.3)* justru menunjukkan peningkatan akurasi FaceNet hingga **76.00% - 76.50%** dan AUC mencapai **0.8724**. Peningkatan kecerahan membantu menonjolkan fitur wajah pada foto smartphone dalam ruangan yang awalnya sedikit *underexposed*.
2. **Karakteristik Domain Transfer (LFW ke Mahasiswa):**
   - **FaceNet:** Threshold LFW (0.4189) sangat fleksibel dan langsung mentransfer performa baik pada domain foto kelas (**74.50%** baseline).
   - **ArcFace:** Pada benchmark LFW studio, ArcFace menghasilkan jarak kosinus yang sangat rapat (threshold 0.0269). Pada data kelas dengan variasi pose dan kamera HP, jarak antar foto orang yang sama bergeser ke rentang ~0.40–0.50. Dengan threshold terkalibrasi (~0.52), ArcFace melesat mencapai akurasi tertinggi **82.50%** dengan AUC **0.8316**.
3. **Pemisahan Data dan Keamanan:**
   - Seluruh hasil pengujian dan cache embedding kelas tersimpan secara terpisah di `experiments/kelas/` dan `results/kelas/` tanpa memodifikasi hasil LFW asli.

---
*Laporan ini dihasilkan secara otomatis oleh pipeline pengujian `run_kelas_experiment.py`.*
