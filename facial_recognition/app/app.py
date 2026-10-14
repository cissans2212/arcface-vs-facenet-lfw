"""
Streamlit GUI Application: Face Verification — ArcFace vs FaceNet
Membandingkan dua citra wajah untuk menentukan apakah orang yang sama atau berbeda.
Framework: DeepFace (ArcFace & FaceNet)
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st

# Setup encoding output
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

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Face Verification — ArcFace vs FaceNet",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolusi Path Direktori Project
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATASET_DIR = Path(os.getenv("DATASET_DIR", PROJECT_DIR.parent / "dataset")).resolve()
RESULTS_DIR = PROJECT_DIR / "results"
ARCFACE_EXP_DIR = PROJECT_DIR / "experiments" / "arcface"
FACENET_EXP_DIR = PROJECT_DIR / "experiments" / "facenet"

# Custom Styling CSS untuk Tampilan Bersih & Akademik
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .card-same {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .card-diff {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .metric-badge {
        font-size: 0.9rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        display: inline-block;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.6rem;
        border-radius: 8px;
        border: none;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_experiment_metadata():
    """Memuat threshold dan metrik hasil eksperimen sebelumnya secara dinamis."""
    arcface_metrics = None
    facenet_metrics = None
    comparison_df = None

    arc_path = ARCFACE_EXP_DIR / "metrics.json"
    if arc_path.exists():
        try:
            with open(arc_path, "r", encoding="utf-8") as f:
                arcface_metrics = json.load(f)
        except Exception:
            pass

    face_path = FACENET_EXP_DIR / "metrics.json"
    if face_path.exists():
        try:
            with open(face_path, "r", encoding="utf-8") as f:
                facenet_metrics = json.load(f)
        except Exception:
            pass

    csv_path = RESULTS_DIR / "model_comparison.csv"
    if csv_path.exists():
        try:
            comparison_df = pd.read_csv(csv_path)
        except Exception:
            pass

    # Ambil threshold hasil eksperimen Tahap 2 & 3
    arc_threshold = arcface_metrics.get("threshold", 0.0269) if arcface_metrics else 0.0269
    face_threshold = facenet_metrics.get("threshold", 0.4189) if facenet_metrics else 0.4189

    return {
        "arcface_metrics": arcface_metrics,
        "facenet_metrics": facenet_metrics,
        "comparison_df": comparison_df,
        "arc_threshold": float(arc_threshold),
        "face_threshold": float(face_threshold),
    }


def compute_cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Menghitung Cosine Distance: 1 - cosine_similarity."""
    dot = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 1.0
    cosine_sim = dot / (norm_u * norm_v)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    return float(1.0 - cosine_sim)


def get_embedding(img_np: np.ndarray, model_name: str) -> np.ndarray:
    """Mengekstrak embedding wajah menggunakan DeepFace dengan detector backend skip."""
    from deepface import DeepFace

    # DeepFace represent mengembalikan list of dicts
    reps = DeepFace.represent(
        img_path=img_np,
        model_name=model_name,
        detector_backend="skip",
        enforce_detection=False,
    )
    return np.array(reps[0]["embedding"], dtype=np.float32)


# ==========================================
# SIDEBAR: MODEL SETTINGS & EVALUATION
# ==========================================
metadata = load_experiment_metadata()

with st.sidebar:
    st.image("https://img.icons8.com/color/96/facial-recognition.png", width=64)
    st.markdown("### Konfigurasi Model")
    model_choice = st.selectbox(
        "Pilih Model Verifikasi:",
        options=["Compare Both", "ArcFace", "FaceNet"],
        index=0,
        help="Pilih model tunggal atau bandingkan performa kedua model sekaligus.",
    )

    st.markdown("---")
    st.markdown("### 📊 Model Evaluation")
    st.caption("Hasil evaluasi performa dari eksperimen LFW Test Set (Tahap 4):")

    if metadata["comparison_df"] is not None:
        st.dataframe(
            metadata["comparison_df"],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("File model_comparison.csv belum ditemukan.")

    with st.expander("ℹ️ Model Information"):
        st.markdown(
            f"""
            **ArcFace**:
            - Framework: DeepFace
            - Pretrained Model (ResNet Backbone)
            - Dimensi Embedding: **512**
            - Cosine Threshold: **{metadata['arc_threshold']:.4f}**
            - Metrik: Cosine Distance

            **FaceNet**:
            - Framework: DeepFace
            - Pretrained Model (Inception-ResNet)
            - Dimensi Embedding: **128**
            - Cosine Threshold: **{metadata['face_threshold']:.4f}**
            - Metrik: Cosine Distance
            """
        )

    with st.expander("📈 Grafik Evaluasi"):
        roc_path = RESULTS_DIR / "roc_comparison.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC Comparison (ArcFace vs FaceNet)", use_container_width=True)
        cm_arc_path = RESULTS_DIR / "confusion_matrix_arcface.png"
        cm_face_path = RESULTS_DIR / "confusion_matrix_facenet.png"
        if cm_arc_path.exists() and cm_face_path.exists():
            st.image(str(cm_arc_path), caption="Confusion Matrix: ArcFace", use_container_width=True)
            st.image(str(cm_face_path), caption="Confusion Matrix: FaceNet", use_container_width=True)


# ==========================================
# MAIN PAGE: HEADER & INPUT
# ==========================================
st.markdown('<div class="main-header">Face Verification — ArcFace vs FaceNet</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Sistem verifikasi wajah 1:1 menggunakan model deep representation <b>ArcFace</b> dan <b>FaceNet</b> pada framework DeepFace.</div>',
    unsafe_allow_html=True,
)

tab_upload, tab_sample = st.tabs(["📁 Upload Dua Gambar", "🧪 Gunakan Sampel Dataset LFW"])

img1_obj = None
img2_obj = None
label_info = None

with tab_upload:
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        file1 = st.file_uploader("Upload Image 1 (Wajah 1):", type=["jpg", "jpeg", "png"], key="upload_1")
    with col_u2:
        file2 = st.file_uploader("Upload Image 2 (Wajah 2):", type=["jpg", "jpeg", "png"], key="upload_2")

    if file1 and file2:
        try:
            img1_obj = Image.open(file1).convert("RGB")
            img2_obj = Image.open(file2).convert("RGB")
        except Exception as e:
            st.error(f"Gagal memuat file gambar: {e}")

with tab_sample:
    st.caption("Pilih pasangan gambar uji yang telah terverifikasi dari dataset LFW untuk demonstrasi instan:")
    sample_options = {
        "Contoh 1: Same Person (George W. Bush)": (
            "train/George_W_Bush/George_W_Bush_0001.jpg",
            "train/George_W_Bush/George_W_Bush_0002.jpg",
            "SAME PERSON (George W. Bush)",
        ),
        "Contoh 2: Same Person (Colin Powell)": (
            "train/Colin_Powell/Colin_Powell_0001.jpg",
            "train/Colin_Powell/Colin_Powell_0002.jpg",
            "SAME PERSON (Colin Powell)",
        ),
        "Contoh 3: Different Person (Gerhard Schroeder vs John Ashcroft)": (
            "test/Gerhard_Schroeder/Gerhard_Schroeder_0001.jpg",
            "test/John_Ashcroft/John_Ashcroft_0001.jpg",
            "DIFFERENT PERSON (Gerhard Schroeder vs John Ashcroft)",
        ),
        "Contoh 4: Different Person (Megawati vs Serena Williams)": (
            "train/Megawati_Sukarnoputri/Megawati_Sukarnoputri_0001.jpg",
            "train/Serena_Williams/Serena_Williams_0001.jpg",
            "DIFFERENT PERSON (Megawati vs Serena Williams)",
        ),
    }

    selected_sample = st.selectbox("Pilih Skenario Sampel:", options=list(sample_options.keys()))
    if st.button("Muat Gambar Sampel", key="btn_load_sample"):
        p1_rel, p2_rel, s_info = sample_options[selected_sample]
        p1_abs = DATASET_DIR / p1_rel
        p2_abs = DATASET_DIR / p2_rel

        if p1_abs.exists() and p2_abs.exists():
            img1_obj = Image.open(p1_abs).convert("RGB")
            img2_obj = Image.open(p2_abs).convert("RGB")
            st.session_state["sample_img1"] = img1_obj
            st.session_state["sample_img2"] = img2_obj
            st.session_state["sample_info"] = s_info
            st.success(f"Berhasil memuat: {s_info}")
        else:
            st.error(f"File sampel tidak ditemukan di {DATASET_DIR}")

    if "sample_img1" in st.session_state and "sample_img2" in st.session_state:
        img1_obj = st.session_state["sample_img1"]
        img2_obj = st.session_state["sample_img2"]
        label_info = st.session_state.get("sample_info", "")

# Tampilkan preview kedua gambar jika tersedia
st.markdown("### Preview Pasangan Gambar")
if img1_obj and img2_obj:
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(img1_obj, caption="Image 1 (Ref)", use_container_width=True)
    with col_img2:
        st.image(img2_obj, caption="Image 2 (Query)", use_container_width=True)
else:
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.info("Silakan upload Image 1 atau muat dari sampel.")
    with col_e2:
        st.info("Silakan upload Image 2 atau muat dari sampel.")

st.markdown("<br>", unsafe_allow_html=True)
btn_verify = st.button("🔍 VERIFY FACE", key="btn_verify_main")


# ==========================================
# PROSES VERIFIKASI & OUTPUT
# ==========================================
if btn_verify:
    if img1_obj is None or img2_obj is None:
        st.error("⚠️ Error: Harap upload kedua gambar terlebih dahulu sebelum melakukan verifikasi!")
    else:
        with st.spinner("Sedang memproses representasi embedding dan menghitung jarak kemiripan..."):
            np1 = np.array(img1_obj)
            np2 = np.array(img2_obj)

            try:
                results_data = {}

                # Ekstraksi ArcFace jika dipilih atau Compare Both
                if model_choice in ["ArcFace", "Compare Both"]:
                    emb1_arc = get_embedding(np1, "ArcFace")
                    emb2_arc = get_embedding(np2, "ArcFace")
                    dist_arc = compute_cosine_distance(emb1_arc, emb2_arc)
                    thresh_arc = metadata["arc_threshold"]
                    pred_arc = "SAME PERSON" if dist_arc <= thresh_arc else "DIFFERENT PERSON"
                    results_data["ArcFace"] = {
                        "distance": dist_arc,
                        "threshold": thresh_arc,
                        "prediction": pred_arc,
                        "embedding_dim": len(emb1_arc),
                    }

                # Ekstraksi FaceNet jika dipilih atau Compare Both
                if model_choice in ["FaceNet", "Compare Both"]:
                    emb1_face = get_embedding(np1, "Facenet")
                    emb2_face = get_embedding(np2, "Facenet")
                    dist_face = compute_cosine_distance(emb1_face, emb2_face)
                    thresh_face = metadata["face_threshold"]
                    pred_face = "SAME PERSON" if dist_face <= thresh_face else "DIFFERENT PERSON"
                    results_data["FaceNet"] = {
                        "distance": dist_face,
                        "threshold": thresh_face,
                        "prediction": pred_face,
                        "embedding_dim": len(emb1_face),
                    }

                st.markdown("---")
                st.markdown("### 📋 Hasil Verifikasi")

                # Penjelasan Cosine Distance
                st.info(
                    "💡 **Prinsip Cosine Distance**: Nilai yang **lebih kecil** menandakan vektor wajah **lebih mirip**.<br>"
                    "• Jika <b>Cosine Distance ≤ Threshold</b> ➔ <b>SAME PERSON</b> (Orang yang Sama)<br>"
                    "• Jika <b>Cosine Distance > Threshold</b> ➔ <b>DIFFERENT PERSON</b> (Orang Berbeda)",
                    icon="ℹ️",
                )

                if model_choice == "Compare Both":
                    # Tampilkan Tabel Perbandingan
                    table_rows = [
                        {
                            "Parameter / Metric": "Cosine Distance",
                            "ArcFace": f"{results_data['ArcFace']['distance']:.4f}",
                            "FaceNet": f"{results_data['FaceNet']['distance']:.4f}",
                        },
                        {
                            "Parameter / Metric": "Threshold Optimal",
                            "ArcFace": f"{results_data['ArcFace']['threshold']:.4f}",
                            "FaceNet": f"{results_data['FaceNet']['threshold']:.4f}",
                        },
                        {
                            "Parameter / Metric": "Dimensi Embedding",
                            "ArcFace": f"{results_data['ArcFace']['embedding_dim']}-d",
                            "FaceNet": f"{results_data['FaceNet']['embedding_dim']}-d",
                        },
                        {
                            "Parameter / Metric": "Hasil Prediksi",
                            "ArcFace": results_data["ArcFace"]["prediction"],
                            "FaceNet": results_data["FaceNet"]["prediction"],
                        },
                    ]
                    df_res = pd.DataFrame(table_rows)
                    st.table(df_res)

                    # Kartu Kesimpulan untuk Masing-masing Model
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        p_arc = results_data["ArcFace"]["prediction"]
                        card_class = "card-same" if p_arc == "SAME PERSON" else "card-diff"
                        icon = "✅" if p_arc == "SAME PERSON" else "❌"
                        st.markdown(
                            f"""
                            <div class="{card_class}">
                                <h4 style="margin:0 0 0.5rem 0;">{icon} ArcFace</h4>
                                <b>Prediksi:</b> {p_arc}<br>
                                <b>Distance:</b> {results_data['ArcFace']['distance']:.4f} &nbsp;|&nbsp; <b>Threshold:</b> {results_data['ArcFace']['threshold']:.4f}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col_r2:
                        p_face = results_data["FaceNet"]["prediction"]
                        card_class = "card-same" if p_face == "SAME PERSON" else "card-diff"
                        icon = "✅" if p_face == "SAME PERSON" else "❌"
                        st.markdown(
                            f"""
                            <div class="{card_class}">
                                <h4 style="margin:0 0 0.5rem 0;">{icon} FaceNet</h4>
                                <b>Prediksi:</b> {p_face}<br>
                                <b>Distance:</b> {results_data['FaceNet']['distance']:.4f} &nbsp;|&nbsp; <b>Threshold:</b> {results_data['FaceNet']['threshold']:.4f}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Kesimpulan Akhir
                    st.markdown("#### Kesimpulan:")
                    st.write(f"- **ArcFace**: `{p_arc}`")
                    st.write(f"- **FaceNet**: `{p_face}`")

                else:
                    # Tampilan untuk Model Tunggal (ArcFace atau FaceNet)
                    res = results_data[model_choice]
                    p = res["prediction"]
                    card_class = "card-same" if p == "SAME PERSON" else "card-diff"
                    icon = "✅" if p == "SAME PERSON" else "❌"

                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <h3 style="margin:0 0 0.5rem 0;">{icon} Model: {model_choice}</h3>
                            <h2 style="margin:0 0 1rem 0; color: {'#065F46' if p == 'SAME PERSON' else '#991B1B'};">
                                {p}
                            </h2>
                            <p style="margin:0; font-size:1.05rem;">
                                <b>Cosine Distance:</b> <code>{res['distance']:.4f}</code> &nbsp;|&nbsp;
                                <b>Threshold:</b> <code>{res['threshold']:.4f}</code> &nbsp;|&nbsp;
                                <b>Embedding:</b> <code>{res['embedding_dim']}-d</code>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Terjadi kesalahan saat inferensi verifikasi wajah: {str(e)}")
