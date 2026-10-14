# Facial Recognition: ArcFace vs FaceNet (DeepFace)

## 1. Tujuan Project
Proyek ini bertujuan untuk membandingkan performa dua arsitektur *deep face representation* mutakhir, yaitu **ArcFace** (Additive Angular Margin Loss) dan **FaceNet** (Triplet Loss), menggunakan framework **DeepFace** untuk tugas **Face Verification** pada dataset **LFW (Labeled Faces in the Wild)**.

## 2. Dataset LFW
- **Sumber Data**: Subset LFW yang berada di root workspace `CV/dataset/`.
- **Karakteristik Dataset**:
  - Total identitas: **62 orang**
  - Total gambar raw: **5.444 file** (terdiri atas `train/`: 2.421, `test/`: 602, dan `database/`: 2.421 yang merupakan duplikat train)
  - Total gambar unik non-duplikat (`train` + `test`): **3.023 gambar**
  - Format & Resolusi: Semua berformat **JPEG** dengan ukuran seragam **250 × 250 piksel**
  - Tidak ada identitas dengan jumlah gambar < 2 (minimum 20 gambar per identitas pada train+test)
  - Tidak ditemukan file gambar yang rusak (*zero corrupted files*)
- **Prinsip Tunggal & Non-Duplikasi**: Dataset tidak disalin maupun dipindahkan. Project `facial_recognition/` mengakses dataset melalui path relatif (`../dataset/`).

## 3. ArcFace vs FaceNet
- **ArcFace**: Mengoptimalkan klasifikasi fitur dengan menambahkan margin sudut aditif (*additive angular margin penalty*) langsung pada fungsi loss berbasis *geodesic distance* pada hypersphere, menghasilkan margin inter-class yang tegas dan intra-class compactness yang tinggi.
- **FaceNet**: Memetakan citra wajah langsung ke representasi Euclidean berdimensi 128/512 (*embedding*) menggunakan arsitektur CNN dengan optimasi *Triplet Loss* (meminimalkan jarak anchor-positive dan memaksimalkan jarak anchor-negative).
- **Framework**: Keduanya akan dievaluasi melalui inferensi model pada framework **DeepFace**. *(Catatan: implementasi model belum dilakukan pada tahap ini).*

## 4. Konsep Face Verification
*Face Verification* adalah proses pencocokan 1:1 (*one-to-one matching*) untuk menentukan apakah dua citra wajah berasal dari individu yang sama (*Same Person*, label = 1) atau individu yang berbeda (*Different Person*, label = 0).
- Pasangan diuji dengan mengukur jarak fitur / kesamaan vektor embedding (*cosine distance* atau *Euclidean distance*).
- Keputusan diambil berdasarkan ambang batas kesamaan (*similarity threshold*).
- **Pencegahan Data Leakage**: Digunakan strategi **Identity-Disjoint Split**:
  - **Development (80% / 800 pasang)**: Dibangun dari 50 identitas.
  - **Test (20% / 200 pasang)**: Dibangun dari 12 identitas yang **sama sekali tidak pernah muncul** pada tahap development.
  - Hal ini menjamin evaluasi bersifat *unseen subjects* (open-set), sehingga penentuan threshold pada development tidak bias terhadap subjek pada data uji test.

## 5. Struktur `pairs.csv`
File metadata pasangan disimpan di `dataset/processed/pairs.csv` dengan kolom-kolom berikut:

| Kolom | Tipe | Penjelasan | Contoh |
| :--- | :--- | :--- | :--- |
| `image1` | string | Path relatif gambar pertama terhadap `CV/dataset/` | `train/George_W_Bush/George_W_Bush_0001.jpg` |
| `image2` | string | Path relatif gambar kedua terhadap `CV/dataset/` | `train/George_W_Bush/George_W_Bush_0002.jpg` |
| `identity1` | string | Nama identitas/orang pada gambar pertama | `George_W_Bush` |
| `identity2` | string | Nama identitas/orang pada gambar kedua | `George_W_Bush` |
| `label` | integer | Label verifikasi: `1` = SAME PERSON, `0` = DIFFERENT PERSON | `1` |
| `split` | string | Partisi subset data: `development` (800) atau `test` (200) | `development` |

### Ringkasan Pasangan Data
- **Total Pasangan**: 1.000 pasang
- **SAME (label = 1)**: 500 pasang (400 Development, 100 Test)
- **DIFFERENT (label = 0)**: 500 pasang (400 Development, 100 Test)
- **Self-pair (`image1 == image2`)**: 0 (tidak ada)
- **Duplicate pair**: 0 (seluruh 1.000 pasangan unik)

## 6. Cara Menjalankan Persiapan Dataset
Jalankan script pembuatan dan validasi dataset dari folder project `facial_recognition/`:

```bash
# Pindah ke folder project facial_recognition (jika belum)
cd "facial_recognition"

# Jalankan script inspeksi, pairing, dan validasi
python src/prepare_dataset.py
```


## 7. Cara Menjalankan Eksperimen & Evaluasi Model
Dari folder `CV/facial_recognition/`:

```bash
# 1. Menjalankan evaluasi ArcFace (Tahap 2)
python src/run_arcface.py

# 2. Menjalankan evaluasi FaceNet (Tahap 3)
python src/run_facenet.py

# 3. Menghasilkan tabel komparasi, confusion matrix, ROC comparison, dan report (Tahap 4)
python src/compare_models.py
```

## 8. Cara Menjalankan GUI Streamlit (Tahap 5)
Untuk menjalankan antarmuka grafis (GUI) verifikasi wajah interaktif:

```bash
# Jalankan aplikasi Streamlit
streamlit run app/app.py
```
Aplikasi akan membuka browser secara otomatis di `http://localhost:8501`.
- Anda dapat mengupload dua gambar (JPG, JPEG, PNG) atau memilih skenario sampel dataset.
- Pilih model: **ArcFace**, **FaceNet**, atau **Compare Both**.
- Klik **VERIFY FACE** untuk melihat hasil perbandingan distance, threshold, dan prediksi.
- Sidebar memuat tabel metrik evaluasi lengkap dari `results/model_comparison.csv` serta grafik kurva ROC.

