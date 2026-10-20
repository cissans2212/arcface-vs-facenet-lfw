"""
Streamlit GUI Application: Face Verification — ArcFace vs FaceNet
Membandingkan dua citra wajah untuk menentukan apakah orang yang sama atau berbeda.
Framework: DeepFace (ArcFace & FaceNet)
Fitur Ekstensi: Eksperimen Preprocessing Ketahanan (Robustness) Brightness & Contrast
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageEnhance
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
    page_title="Face Verification — ArcFace vs FaceNet (Robustness Evaluator)",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolusi Path Direktori Project
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATASET_DIR = Path(os.getenv("DATASET_DIR", PROJECT_DIR.parent / "dataset")).resolve()
RESULTS_DIR = PROJECT_DIR / "results"
ARCFACE_EXP_DIR = PROJECT_DIR / "experiments" / "arcface"
FACENET_EXP_DIR = PROJECT_DIR / "experiments" / "facenet"
ROBUSTNESS_EXP_DIR = PROJECT_DIR / "experiments" / "robustness_brightness_contrast"
ROBUSTNESS_RES_DIR = RESULTS_DIR / "robustness_brightness_contrast"

try:
    from preprocessing import SCENARIOS, preprocess_image
except ImportError:
    SCENARIOS = {
        "baseline": {"name": "Skenario 1 — Baseline", "factor": 1.0, "type": "none"},
        "brightness_low": {"name": "Skenario 2 — Brightness Rendah", "factor": 0.7, "type": "brightness"},
        "brightness_high": {"name": "Skenario 3 — Brightness Tinggi", "factor": 1.3, "type": "brightness"},
        "contrast_low": {"name": "Skenario 4 — Contrast Rendah", "factor": 0.7, "type": "contrast"},
        "contrast_high": {"name": "Skenario 5 — Contrast Tinggi", "factor": 1.3, "type": "contrast"},
    }
    def preprocess_image(im, scen):
        return im.copy()

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
    .scenario-badge {
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        background-color: #EEF2F6;
        color: #1E293B;
        display: inline-block;
        border: 1px solid #CBD5E1;
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
    """Memuat threshold, metrik baseline, dan metrik robustness secara dinamis."""
    arcface_metrics = None
    facenet_metrics = None
    comparison_df = None
    robustness_df = None

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

    # Ambil tabel perbandingan 5 skenario robustness jika ada
    robust_csv = RESULTS_DIR / "brightness_contrast_comparison.csv"
    if not robust_csv.exists():
        robust_csv = ROBUSTNESS_RES_DIR / "brightness_contrast_comparison.csv"
    if robust_csv.exists():
        try:
            robustness_df = pd.read_csv(robust_csv)
        except Exception:
            pass

    arc_threshold = arcface_metrics.get("threshold", 0.0269) if arcface_metrics else 0.0269
    face_threshold = facenet_metrics.get("threshold", 0.4189) if facenet_metrics else 0.4189

    return {
        "arcface_metrics": arcface_metrics,
        "facenet_metrics": facenet_metrics,
        "comparison_df": comparison_df,
        "robustness_df": robustness_df,
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
    st.markdown("### 📊 Model Evaluation (Baseline)")
    st.caption("Hasil evaluasi performa baseline pada LFW Test Set (Tahap 4):")

    if metadata["comparison_df"] is not None:
        st.dataframe(
            metadata["comparison_df"],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("File model_comparison.csv belum ditemukan.")

    with st.expander("🔆 Robustness Analysis (Brightness & Contrast)"):
        st.caption("Hasil evaluasi ketahanan 5 skenario:")
        if metadata.get("robustness_df") is not None:
            sub_cols = ["Scenario_Name", "Factor", "ArcFace_Accuracy", "FaceNet_Accuracy", "ArcFace_F1", "FaceNet_F1"]
            avail_cols = [c for c in sub_cols if c in metadata["robustness_df"].columns]
            st.dataframe(
                metadata["robustness_df"][avail_cols],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Data robustness_comparison.csv sedang diproses atau belum dijalankan.")

        robust_acc_chart = RESULTS_DIR / "brightness_contrast_accuracy.png"
        if not robust_acc_chart.exists():
            robust_acc_chart = ROBUSTNESS_RES_DIR / "accuracy_comparison.png"
        if robust_acc_chart.exists():
            st.image(str(robust_acc_chart), caption="Akurasi ArcFace vs FaceNet (5 Skenario)", use_container_width=True)

        robust_err_chart = RESULTS_DIR / "brightness_contrast_error_rates.png"
        if not robust_err_chart.exists():
            robust_err_chart = ROBUSTNESS_RES_DIR / "error_rates_comparison.png"
        if robust_err_chart.exists():
            st.image(str(robust_err_chart), caption="Tingkat Kesalahan FAR vs FRR (5 Skenario)", use_container_width=True)

    with st.expander("🎓 Evaluasi Dataset Kelas Mahasiswa (Baru)"):
        st.caption("Hasil pengujian 200 pasangan dari 33 mahasiswa (5 Skenario):")
        kelas_csv = RESULTS_DIR / "kelas" / "kelas_comparison.csv"
        if kelas_csv.exists():
            try:
                kelas_df = pd.read_csv(kelas_csv)
                st.dataframe(kelas_df, use_container_width=True, hide_index=True)
            except Exception:
                pass

        kelas_acc_chart = RESULTS_DIR / "kelas" / "metrics_comparison.png"
        if kelas_acc_chart.exists():
            st.image(str(kelas_acc_chart), caption="Akurasi ArcFace vs FaceNet (Dataset Kelas)", use_container_width=True)

        kelas_err_chart = RESULTS_DIR / "kelas" / "far_frr_comparison.png"
        if kelas_err_chart.exists():
            st.image(str(kelas_err_chart), caption="Tingkat Kesalahan FAR vs FRR (Dataset Kelas)", use_container_width=True)

        kelas_roc_chart = RESULTS_DIR / "kelas" / "roc_comparison.png"
        if kelas_roc_chart.exists():
            st.image(str(kelas_roc_chart), caption="Kurva ROC (Dataset Kelas)", use_container_width=True)

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

    with st.expander("📈 Baseline Graphics"):
        roc_path = RESULTS_DIR / "roc_comparison.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC Comparison Baseline", use_container_width=True)
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
    '<div class="sub-header">Sistem verifikasi wajah 1:1 menggunakan <b>ArcFace</b> dan <b>FaceNet</b> dengan pengujian ketahanan (robustness) <b>Brightness</b> dan <b>Contrast</b>.</div>',
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


# ==========================================
# PENGATURAN PREPROCESSING (BRIGHTNESS & CONTRAST)
# ==========================================
st.markdown("### 🎛️ Pengaturan Skenario Preprocessing (Robustness Test)")

col_sc1, col_sc2 = st.columns([2, 1])
with col_sc1:
    scenario_display_options = {
        "Skenario 1 — Baseline (Citra Asli, Faktor 1.0)": "baseline",
        "Skenario 2 — Brightness Rendah (Faktor 0.7)": "brightness_low",
        "Skenario 3 — Brightness Tinggi (Faktor 1.3)": "brightness_high",
        "Skenario 4 — Contrast Rendah (Faktor 0.7)": "contrast_low",
        "Skenario 5 — Contrast Tinggi (Faktor 1.3)": "contrast_high",
        "Kustom — Slider Manual": "custom",
    }
    chosen_label = st.selectbox(
        "Pilih Skenario Pengujian:",
        options=list(scenario_display_options.keys()),
        index=0,
        help="Pilih skenario simulasi pencahayaan atau kontras untuk menguji ketahanan verifikasi wajah.",
    )
    selected_scen_key = scenario_display_options[chosen_label]

with col_sc2:
    apply_target = st.selectbox(
        "Terapkan Efek ke:",
        options=["Keduanya (Image 1 & Image 2)", "Image 2 saja (Query/Uji)", "Image 1 saja (Referensi)"],
        index=0,
        help="Tentukan apakah efek diterapkan pada kedua citra atau salah satunya.",
    )

custom_b = 1.0
custom_c = 1.0
if selected_scen_key == "custom":
    col_cu1, col_cu2 = st.columns(2)
    with col_cu1:
        custom_b = st.slider("Faktor Kecerahan (Brightness):", min_value=0.2, max_value=2.0, value=1.0, step=0.1)
    with col_cu2:
        custom_c = st.slider("Faktor Kontras (Contrast):", min_value=0.2, max_value=2.0, value=1.0, step=0.1)


def transform_image_ui(img, scen_key, b_factor, c_factor):
    """Menerapkan transformasi pada citra PIL untuk antarmuka GUI."""
    if img is None:
        return None
    if scen_key == "baseline":
        return img.copy()
    elif scen_key == "custom":
        res = img.copy()
        if b_factor != 1.0:
            res = ImageEnhance.Brightness(res).enhance(b_factor)
        if c_factor != 1.0:
            res = ImageEnhance.Contrast(res).enhance(c_factor)
        return res
    else:
        return preprocess_image(img, scen_key)


proc_img1 = img1_obj.copy() if img1_obj else None
proc_img2 = img2_obj.copy() if img2_obj else None

if selected_scen_key != "baseline":
    if apply_target in ["Keduanya (Image 1 & Image 2)", "Image 1 saja (Referensi)"] and img1_obj:
        proc_img1 = transform_image_ui(img1_obj, selected_scen_key, custom_b, custom_c)
    if apply_target in ["Keduanya (Image 1 & Image 2)", "Image 2 saja (Query/Uji)"] and img2_obj:
        proc_img2 = transform_image_ui(img2_obj, selected_scen_key, custom_b, custom_c)


# Tampilkan preview kedua gambar jika tersedia
st.markdown("### Preview Pasangan Gambar")
if proc_img1 and proc_img2:
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        cap1 = "Image 1 (Ref)"
        if selected_scen_key != "baseline" and apply_target in ["Keduanya (Image 1 & Image 2)", "Image 1 saja (Referensi)"]:
            cap1 += f" — [{chosen_label.split('(')[0].strip()}]"
        st.image(proc_img1, caption=cap1, use_container_width=True)
    with col_img2:
        cap2 = "Image 2 (Query)"
        if selected_scen_key != "baseline" and apply_target in ["Keduanya (Image 1 & Image 2)", "Image 2 saja (Query/Uji)"]:
            cap2 += f" — [{chosen_label.split('(')[0].strip()}]"
        st.image(proc_img2, caption=cap2, use_container_width=True)
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
    if proc_img1 is None or proc_img2 is None:
        st.error("⚠️ Error: Harap upload kedua gambar terlebih dahulu sebelum melakukan verifikasi!")
    else:
        with st.spinner("Sedang memproses representasi embedding dan menghitung jarak kemiripan..."):
            np1 = np.array(proc_img1)
            np2 = np.array(proc_img2)

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
                st.markdown(
                    f'<span class="scenario-badge">Skenario Aktif: <b>{chosen_label}</b> | Target: <b>{apply_target}</b></span>',
                    unsafe_allow_html=True,
                )

                st.info(
                    "💡 **Prinsip Cosine Distance**: Nilai yang **lebih kecil** menandakan vektor wajah **lebih mirip**.<br>"
                    "• Jika <b>Cosine Distance ≤ Threshold</b> ➔ <b>SAME PERSON</b> (Orang yang Sama)<br>"
                    "• Jika <b>Cosine Distance > Threshold</b> ➔ <b>DIFFERENT PERSON</b> (Orang Berbeda)",
                    icon="ℹ️",
                )

                if model_choice == "Compare Both":
                    table_rows = [
                        {
                            "Parameter / Metric": "Cosine Distance",
                            "ArcFace": f"{results_data['ArcFace']['distance']:.4f}",
                            "FaceNet": f"{results_data['FaceNet']['distance']:.4f}",
                        },
                        {
                            "Parameter / Metric": "Threshold Beku",
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

                    st.markdown("#### Kesimpulan:")
                    st.write(f"- **ArcFace**: `{p_arc}`")
                    st.write(f"- **FaceNet**: `{p_face}`")

                else:
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
