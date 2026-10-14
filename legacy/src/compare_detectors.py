import os
import sys
import glob
import time
import pandas as pd
from mtcnn_detector import MTCNNDetector
from retinaface_detector import RetinaFaceDetector

def main():
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(workspace, 'dataset', 'test')
    results_dir = os.path.join(workspace, 'results')
    mtcnn_vis_dir = os.path.join(results_dir, 'mtcnn')
    rf_vis_dir = os.path.join(results_dir, 'retinaface')
    csv_output_path = os.path.join(results_dir, 'comparison.csv')

    os.makedirs(mtcnn_vis_dir, exist_ok=True)
    os.makedirs(rf_vis_dir, exist_ok=True)

    # 1. Collect all test image paths
    image_paths = sorted(glob.glob(os.path.join(test_dir, '*', '*.jpg')))
    total_images = len(image_paths)
    print(f"Total test images found: {total_images}")

    if total_images == 0:
        print("Error: No test images found in", test_dir)
        return

    # 2. Select representative identities for visualization (10 diverse figures)
    vis_identities = [
        'George_W_Bush', 'Colin_Powell', 'Tony_Blair', 'Ariel_Sharon',
        'Serena_Williams', 'Arnold_Schwarzenegger', 'Megawati_Sukarnoputri',
        'Vladimir_Putin', 'Jennifer_Capriati', 'Hugo_Chavez'
    ]
    vis_saved = {k: False for k in vis_identities}

    # 3. Initialize detectors
    print("Initializing MTCNN detector...")
    mtcnn = MTCNNDetector(keep_all=True)

    print("Initializing RetinaFace detector...")
    retinaface = RetinaFaceDetector(threshold=0.9, allow_upscaling=False)

    print("Starting face detection evaluation across all test images...")
    records = []
    t_start = time.time()

    for idx, img_path in enumerate(image_paths, 1):
        rel_path = os.path.relpath(img_path, workspace)
        parts = os.path.normpath(rel_path).split(os.sep)
        identity = parts[-2]
        filename = parts[-1]

        # MTCNN detection
        mtcnn_detected, mtcnn_faces, mtcnn_time = mtcnn.detect(img_path)
        mtcnn_count = len(mtcnn_faces)

        # RetinaFace detection
        rf_detected, rf_faces, rf_time = retinaface.detect(img_path)
        rf_count = len(rf_faces)

        records.append({
            'image_path': rel_path.replace('\\', '/'),
            'identity': identity,
            'image_name': filename,
            'mtcnn_detected': 1 if mtcnn_detected else 0,
            'mtcnn_face_count': mtcnn_count,
            'mtcnn_time_sec': round(mtcnn_time, 4),
            'retinaface_detected': 1 if rf_detected else 0,
            'retinaface_face_count': rf_count,
            'retinaface_time_sec': round(rf_time, 4)
        })

        # Check if we should save visual detection for this image
        if identity in vis_saved and not vis_saved[identity]:
            # Save visual outputs
            out_mtcnn = os.path.join(mtcnn_vis_dir, f"{identity}_{filename}")
            out_rf = os.path.join(rf_vis_dir, f"{identity}_{filename}")
            mtcnn.draw_detections(img_path, mtcnn_faces, out_mtcnn)
            retinaface.draw_detections(img_path, rf_faces, out_rf)
            vis_saved[identity] = True

        if idx % 50 == 0 or idx == total_images:
            elapsed = time.time() - t_start
            print(f"[{idx}/{total_images}] Processed in {elapsed:.1f}s...")

    # 4. Save comparison CSV
    df = pd.DataFrame(records)
    df.to_csv(csv_output_path, index=False)
    print(f"\nSaved comparison results to: {csv_output_path}")

    # 5. Calculate Summary Statistics
    mtcnn_detected_total = df['mtcnn_detected'].sum()
    mtcnn_failed_total = total_images - mtcnn_detected_total
    mtcnn_avg_time = df['mtcnn_time_sec'].mean()

    rf_detected_total = df['retinaface_detected'].sum()
    rf_failed_total = total_images - rf_detected_total
    rf_avg_time = df['retinaface_time_sec'].mean()

    print("\n" + "="*60)
    print("RINGKASAN HASIL EKSPERIMEN FACE DETECTION")
    print("="*60)
    summary_data = [
        {
            'Metode': 'MTCNN',
            'Total Gambar': total_images,
            'Wajah Terdeteksi': mtcnn_detected_total,
            'Tidak Terdeteksi': mtcnn_failed_total,
            'Rata-rata Waktu (s)': round(mtcnn_avg_time, 4)
        },
        {
            'Metode': 'RetinaFace',
            'Total Gambar': total_images,
            'Wajah Terdeteksi': rf_detected_total,
            'Tidak Terdeteksi': rf_failed_total,
            'Rata-rata Waktu (s)': round(rf_avg_time, 4)
        }
    ]
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("="*60)

if __name__ == '__main__':
    main()
