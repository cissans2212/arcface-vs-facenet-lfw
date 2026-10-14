import os
import sys
import time
import glob
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Setup path agar src/ bisa diimport langsung
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mtcnn_detector import MTCNNDetector
from retinaface_detector import RetinaFaceDetector

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Komparasi Deteksi Wajah: MTCNN vs RetinaFace",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache Resource untuk Model agar tidak reload berulang kali
@st.cache_resource
def load_detectors():
    mtcnn = MTCNNDetector(keep_all=True)
    rf = RetinaFaceDetector(threshold=0.9, allow_upscaling=False)
    return mtcnn, rf

# Helper function untuk menggambar deteksi MTCNN pada gambar RGB
def draw_mtcnn_vis(img_rgb, faces):
    vis = img_rgb.copy()
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        score = face['score']
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 102, 255), 2)
        label = f"MTCNN: {score:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 102, 255), 1, cv2.LINE_AA)
        for pt in face.get('landmarks', []):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (255, 204, 0), -1)
    return vis

# Helper function untuk menggambar deteksi RetinaFace pada gambar RGB
def draw_rf_vis(img_rgb, faces):
    vis = img_rgb.copy()
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        score = face['score']
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 204, 102), 2)
        label = f"RetinaFace: {score:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 204, 102), 1, cv2.LINE_AA)
        for pt in face.get('landmarks', []):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (255, 51, 51), -1)
    return vis

def main():
    st.title("👤 Perbandingan Metode Deteksi Wajah: MTCNN vs RetinaFace")
    st.markdown("Proyek Computer Vision — Evaluasi Performa Deteksi Wajah pada Dataset **LFW People**.")

    # Sidebar
    st.sidebar.header("⚙️ Navigasi & Pengaturan")
    menu = st.sidebar.radio(
        "Pilih Menu:",
        ["🔍 Uji Deteksi Interaktif", "📊 Hasil Eksperimen 602 Gambar", "📖 Penjelasan Teknis Metode"]
    )

    mtcnn_detector, rf_detector = load_detectors()

    # =========================================================================
    # MENU 1: UJI DETEKSI INTERAKTIF
    # =========================================================================
    if menu == "🔍 Uji Deteksi Interaktif":
        st.subheader("Eksperimen Deteksi Wajah Side-by-Side")

        col_input1, col_input2 = st.columns([1, 1])

        with col_input1:
            input_source = st.radio(
                "Sumber Citra:",
                ["Pilih dari Dataset Uji (dataset/test/)", "Unggah Gambar Sendiri"],
                horizontal=True
            )

        test_dir = os.path.join(CURRENT_DIR, "dataset", "test")
        input_image_path = None
        pil_image = None

        if input_source == "Pilih dari Dataset Uji (dataset/test/)":
            if os.path.exists(test_dir):
                identities = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
                selected_id = st.sidebar.selectbox("Pilih Identitas:", identities)

                id_folder = os.path.join(test_dir, selected_id)
                images = sorted(os.listdir(id_folder))
                selected_img_name = st.sidebar.selectbox("Pilih File Gambar:", images)

                input_image_path = os.path.join(id_folder, selected_img_name)
                pil_image = Image.open(input_image_path).convert("RGB")
            else:
                st.error("Folder dataset uji `dataset/test` tidak ditemukan.")
        else:
            uploaded_file = st.sidebar.file_uploader("Unggah file gambar (JPG/PNG):", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                pil_image = Image.open(uploaded_file).convert("RGB")
                temp_path = os.path.join(CURRENT_DIR, "temp_uploaded.jpg")
                pil_image.save(temp_path)
                input_image_path = temp_path

        if pil_image is not None and input_image_path is not None:
            img_np = np.array(pil_image)

            # Tombol Deteksi
            if st.button("🚀 Jalankan Deteksi (MTCNN & RetinaFace)", type="primary"):
                with st.spinner("Menjalankan inferensi pada kedua metode..."):
                    # MTCNN
                    mtcnn_detected, mtcnn_faces, mtcnn_time = mtcnn_detector.detect(input_image_path)
                    mtcnn_vis = draw_mtcnn_vis(img_np, mtcnn_faces)

                    # RetinaFace
                    rf_detected, rf_faces, rf_time = rf_detector.detect(input_image_path)
                    rf_vis = draw_rf_vis(img_np, rf_faces)

                # Tampilan 3 Kolom Gambar
                st.markdown("---")
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown("### 📷 Citra Asli")
                    st.image(pil_image, use_container_width=True, caption=f"Ukuran: {pil_image.size[0]}x{pil_image.size[1]} px")

                with c2:
                    st.markdown("### 🔵 MTCNN")
                    st.image(mtcnn_vis, use_container_width=True, caption="Bounding Box & 5 Landmarks (Kuning)")
                    status_text = "✅ Terdeteksi" if mtcnn_detected else "❌ Tidak Terdeteksi"
                    st.markdown(f"**Status**: {status_text}")
                    st.markdown(f"**Jumlah Wajah**: {len(mtcnn_faces)}")
                    st.markdown(f"**Waktu Inferensi**: `{mtcnn_time:.4f} detik`")
                    if len(mtcnn_faces) > 0:
                        st.markdown(f"**Confidence**: `{mtcnn_faces[0]['score']:.4f}`")
                        st.markdown(f"**Bounding Box**: `{mtcnn_faces[0]['bbox']}`")

                with c3:
                    st.markdown("### 🟢 RetinaFace")
                    st.image(rf_vis, use_container_width=True, caption="Bounding Box & 5 Landmarks (Merah)")
                    status_text = "✅ Terdeteksi" if rf_detected else "❌ Tidak Terdeteksi"
                    st.markdown(f"**Status**: {status_text}")
                    st.markdown(f"**Jumlah Wajah**: {len(rf_faces)}")
                    st.markdown(f"**Waktu Inferensi**: `{rf_time:.4f} detik`")
                    if len(rf_faces) > 0:
                        st.markdown(f"**Confidence**: `{rf_faces[0]['score']:.4f}`")
                        st.markdown(f"**Bounding Box**: `{rf_faces[0]['bbox']}`")

                # Metrik Komparasi
                st.markdown("---")
                st.markdown("#### ⚡ Komparasi Langsung")
                m1, m2, m3 = st.columns(3)
                m1.metric("Waktu MTCNN", f"{mtcnn_time:.4f} s")
                m2.metric("Waktu RetinaFace", f"{rf_time:.4f} s")
                if mtcnn_time > 0 and rf_time > 0:
                    diff_ratio = rf_time / mtcnn_time
                    m3.metric("Rasio Kecepatan", f"{diff_ratio:.2f}x", "MTCNN Lebih Cepat" if diff_ratio > 1 else "RetinaFace Lebih Cepat")

    # =========================================================================
    # MENU 2: HASIL EKSPERIMEN 602 GAMBAR
    # =========================================================================
    elif menu == "📊 Hasil Eksperimen 602 Gambar":
        st.subheader("Hasil Evaluasi Eksperimen Komparasi Lengkap")

        csv_path = os.path.join(CURRENT_DIR, "results", "comparison.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            total_test = len(df)
            mtcnn_ok = df['mtcnn_detected'].sum()
            rf_ok = df['retinaface_detected'].sum()
            mtcnn_avg_t = df['mtcnn_time_sec'].mean()
            rf_avg_t = df['retinaface_time_sec'].mean()

            # Kartu Metrik
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Gambar Uji", f"{total_test} Citra")
            k2.metric("Akurasi MTCNN", f"{(mtcnn_ok/total_test)*100:.1f}%", f"{mtcnn_ok}/{total_test}")
            k3.metric("Akurasi RetinaFace", f"{(rf_ok/total_test)*100:.1f}%", f"{rf_ok}/{total_test}")
            k4.metric("Rata-rata Waktu (MTCNN vs RF)", f"{mtcnn_avg_t:.4f}s / {rf_avg_t:.4f}s")

            st.markdown("---")
            st.markdown("#### 📋 Tabel Rekapitulasi Kinerja")
            rekap = pd.DataFrame([
                {
                    "Metode": "MTCNN",
                    "Total Gambar": total_test,
                    "Berhasil Deteksi": mtcnn_ok,
                    "Gagal Deteksi": total_test - mtcnn_ok,
                    "Tingkat Keberhasilan": f"{(mtcnn_ok/total_test)*100:.1f}%",
                    "Rata-rata Waktu (detik)": round(mtcnn_avg_t, 4)
                },
                {
                    "Metode": "RetinaFace",
                    "Total Gambar": total_test,
                    "Berhasil Deteksi": rf_ok,
                    "Gagal Deteksi": total_test - rf_ok,
                    "Tingkat Keberhasilan": f"{(rf_ok/total_test)*100:.1f}%",
                    "Rata-rata Waktu (detik)": round(rf_avg_t, 4)
                }
            ])
            st.table(rekap)

            # Filter data interaktif
            st.markdown("---")
            st.markdown("#### 🔎 Eksplorasi Data Per Gambar")
            all_ids = ["Semua Identitas"] + sorted(df['identity'].unique().tolist())
            selected_filter = st.selectbox("Filter berdasarkan Identitas:", all_ids)

            if selected_filter != "Semua Identitas":
                filtered_df = df[df['identity'] == selected_filter]
            else:
                filtered_df = df

            st.dataframe(filtered_df, use_container_width=True)
            
            # Download CSV
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data CSV",
                data=csv_data,
                file_name="hasil_komparasi_face_detection.csv",
                mime="text/csv"
            )
        else:
            st.error("File `results/comparison.csv` belum ditemukan. Jalankan `src/compare_detectors.py` terlebih dahulu.")

    # =========================================================================
    # MENU 3: PENJELASAN TEKNIS METODE
    # =========================================================================
    elif menu == "📖 Penjelasan Teknis Metode":
        st.subheader("Dokumentasi Teknis: MTCNN vs RetinaFace")
        doc_path = os.path.join(CURRENT_DIR, "docs", "metode_mtcnn_vs_retinaface.md")
        if os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf-8") as f:
                doc_content = f.read()
            st.markdown(doc_content)
        else:
            st.error("Dokumen teknis `docs/metode_mtcnn_vs_retinaface.md` tidak ditemukan.")

if __name__ == "__main__":
    main()
