# Perbandingan Metode Deteksi Wajah: MTCNN vs RetinaFace

Dokumen ini memuat perbandingan teknis antara dua metode deteksi wajah (*face detection*) yang populer: **MTCNN (Multi-task Cascaded Convolutional Networks)** dan **RetinaFace (Single-stage Dense Face Localisation)**.

---

## 1. MTCNN (Multi-task Cascaded Convolutional Networks)

### Pengertian
MTCNN adalah kerangka kerja deteksi wajah berbasis *deep learning* yang mengintegrasikan deteksi wajah dan penyelarasan titik *landmark* wajah (*facial landmark alignment*) secara berjenjang (*cascaded*).

### Tujuan
Mendeteksi lokasi wajah (*bounding box*) dan 5 titik *landmark* utama wajah (mata kiri, mata kanan, hidung, sudut mulut kiri, dan sudut mulut kanan) pada citra secara *coarse-to-fine*.

### Tahapan Utama
MTCNN terdiri dari arsitektur 3 tahap berjenjang (*cascaded networks*):
1. **P-Net (Proposal Network)**:
   * Jaringan *fully convolutional* ringan untuk memindai citra pada berbagai skala (*image pyramid*).
   * Menghasilkan kandidat wilayah wajah (*candidate bounding box proposals*) dan vektor regresi kalibrasi secara cepat.
2. **R-Net (Refine Network)**:
   * Jaringan konvolusional untuk menyaring kandidat dari P-Net.
   * Menolak sebagian besar proposal *false positive*, melakukan kalibrasi *bounding box*, dan menerapkan *Non-Maximum Suppression* (NMS).
3. **O-Net (Output Network)**:
   * Jaringan yang lebih dalam untuk mengevaluasi kandidat wajah dengan akurasi lebih tinggi.
   * Menghasilkan posisi akhir *bounding box*, skor probabilitas wajah, dan 5 titik koordinat *facial landmarks*.

### Cara Kerja Sederhana
1. Citra masukan diubah ukurannya menjadi piramida citra (*image pyramid*) untuk menangani variasi skala wajah.
2. P-Net menyaring area potensial dan mengeliminasi area latar belakang non-wajah.
3. R-Net memverifikasi proposal yang lolos dan menyaring kandidat lebih ketat.
4. O-Net menentukan *bounding box* final dan menghitung 5 titik *landmark* wajah.

### Output
* Skor probabilitas wajah (*confidence score* $[0.0 - 1.0]$).
* Koordinat *bounding box* $[x_1, y_1, x_2, y_2]$.
* 5 titik *facial landmarks* (mata kanan, mata kiri, hidung, sudut mulut kanan, sudut mulut kiri).

### Kelebihan
* Ringan (*lightweight*) secara komputasi dan arsitektural.
* Kecepatan inferensi tinggi pada CPU karena ukuran model kecil.
* Mendukung deteksi multi-wajah secara terpadu dengan titik *landmark*.

### Keterbatasan
* Pendekatan berjenjang (*cascade*) rentan gagal jika wajah berukuran sangat kecil (*tiny faces*), mengalami oklusi berat, atau rotasi ekstrem pada tahap awal (P-Net).
* Sangat bergantung pada pembuatan piramida citra (*image pyramid*) yang menambah *overhead* pada resolusi tinggi.

---

## 2. RetinaFace (Single-stage Dense Face Localisation)

### Pengertian
RetinaFace adalah detektor wajah *single-stage* berbasis *dense anchor* yang memanfaatkan *Feature Pyramid Network* (FPN) dan *multi-task loss* untuk melokalisasi wajah pada berbagai skala secara simultan dalam satu kali inferensi.

### Tujuan
Mencapai akurasi deteksi wajah tingkat tinggi, khususnya untuk wajah dengan variasi skala ekstrem, pose sulit, pencahayaan minim, dan oklusi.

### Tahapan Utama
1. **Feature Pyramid Network (FPN) Backbone**:
   * Mengekstraksi fitur multiskala dari *backbone* (misal ResNet atau MobileNet).
   * Menggabungkan fitur tingkat rendah (*low-level spatial detail*) dengan fitur tingkat tinggi (*high-level semantic context*).
2. **Context Module (Single-Stage Dense Anchors)**:
   * Menangkap konteks spasial yang lebih luas di sekitar wajah menggunakan modul konvolusi independen.
   * Menempatkan kotak *anchor* padat (*dense anchors*) pada setiap tingkat piramida fitur (stride 8, 16, 32).
3. **Multi-task Loss Optimization**:
   * Melakukan pembelajaran bersama (*joint learning*) untuk 3 tugas sekaligus:
     * Klasifikasi wajah (*face classification loss*).
     * Regresi kotak pembatas (*box regression loss*).
     * Regresi 5 titik *facial landmarks* (*landmark regression loss*).
4. **Post-Processing (NMS)**:
   * Pengelompokan dan eliminasi deteksi tumpang-tindih menggunakan *Non-Maximum Suppression* (NMS).

### Cara Kerja Sederhana
1. Citra diproses langsung oleh jaringan *backbone* dan piramida fitur (FPN) tanpa memerlukan piramida citra manual.
2. Setiap tingkat fitur secara simultan memprediksi keberadaan wajah, offset *bounding box*, dan posisi *landmark*.
3. Fitur tingkat resolusi tinggi (stride 8) mendeteksi wajah kecil, sedangkan tingkat resolusi rendah (stride 32) mendeteksi wajah besar.
4. NMS diterapkan untuk memilih kotak deteksi terbaik dengan skor probabilitas tertinggi.

### Output
* Skor probabilitas wajah (*confidence score* $[0.0 - 1.0]$).
* Koordinat *bounding box* $[x_1, y_1, x_2, y_2]$.
* 5 titik *facial landmarks* (mata kanan, mata kiri, hidung, sudut mulut kanan, sudut mulut kiri).

### Kelebihan
* Akurasi sangat tinggi (*state-of-the-art*) dalam mendeteksi wajah kecil, oklusi, dan pose menyamping/ekstrem.
* Lokalisasi *bounding box* lebih presisi dan konsisten.
* Arsitektur *single-stage* berbasis FPN menghindari *error compounding* seperti pada sistem berjenjang (*cascaded*).

### Keterbatasan
* Membutuhkan daya komputasi lebih besar dibandingkan MTCNN.
* Waktu inferensi per citra lebih lama pada CPU tanpa akselerasi GPU.

---

## 3. Perbandingan MTCNN dan RetinaFace

| Parameter | MTCNN | RetinaFace |
|:---|:---|:---|
| **Pendekatan Deteksi** | *Multi-stage cascaded networks* (P-Net $\rightarrow$ R-Net $\rightarrow$ O-Net) | *Single-stage dense anchor* berbasis *Feature Pyramid Network* (FPN) |
| **Bounding Box** | Dihasilkan secara bertahap (*coarse-to-fine proposal & refinement*) | Dihasilkan langsung dari *dense anchors* di berbagai tingkat fitur FPN |
| **Facial Landmarks** | 5 titik *landmark* dihitung pada tahap akhir (O-Net) | 5 titik *landmark* dipelajari secara bersamaan (*multi-task loss*) |
| **Kecepatan** | Lebih cepat pada CPU (~0.05 – 0.08 detik/gambar) | Lebih lambat pada CPU (~0.30 – 0.40 detik/gambar) karena kompleksitas FPN |
| **Kemampuan Deteksi Wajah** | Baik untuk wajah frontal dan pencahayaan normal; akurasi menurun pada pose ekstrem/oklusi | Sangat tinggi, unggul pada sudut ekstrem (*profile face*), oklusi, dan wajah kecil |
| **Kondisi Penggunaan** | Sistem *real-time* berdaya rendah (CPU, mobile, embedded system) | Sistem yang memprioritaskan akurasi maksimal, rekognisi wajah kritis, dan kondisi liar (*in-the-wild*) |
