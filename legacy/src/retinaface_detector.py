import time
import os
import cv2
from retinaface import RetinaFace
from retinaface.RetinaFace import build_model

class RetinaFaceDetector:
    def __init__(self, threshold=0.9, allow_upscaling=False):
        self.threshold = threshold
        self.allow_upscaling = allow_upscaling
        self.model = build_model()

    def detect(self, img_path):
        """
        Detect faces in image.
        Returns:
            detected (bool): True if at least one face detected
            faces (list of dict): List of detections with bbox [x1, y1, x2, y2], score, landmarks
            elapsed_time (float): Detection time in seconds
        """
        t0 = time.time()
        try:
            resp = RetinaFace.detect_faces(
                img_path,
                threshold=self.threshold,
                model=self.model,
                allow_upscaling=self.allow_upscaling
            )
        except Exception:
            resp = {}
        elapsed_time = time.time() - t0

        faces = []
        if isinstance(resp, dict):
            for key, val in resp.items():
                area = val.get('facial_area', [])
                score = float(val.get('score', 0.0))
                landmarks_dict = val.get('landmarks', {})
                lm_list = [
                    landmarks_dict.get('right_eye'),
                    landmarks_dict.get('left_eye'),
                    landmarks_dict.get('nose'),
                    landmarks_dict.get('mouth_right'),
                    landmarks_dict.get('mouth_left')
                ]
                lm_list = [p for p in lm_list if p is not None]

                if len(area) == 4:
                    faces.append({
                        'bbox': [int(area[0]), int(area[1]), int(area[2]), int(area[3])],
                        'score': score,
                        'landmarks': lm_list
                    })

        return len(faces) > 0, faces, elapsed_time

    @staticmethod
    def draw_detections(img_path, faces, output_path):
        """
        Draw bounding boxes and landmarks on image and save.
        """
        img = cv2.imread(img_path)
        if img is None:
            return

        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            score = face['score']
            # Draw bounding box (Green for RetinaFace)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"RetinaFace: {score:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Draw landmarks
            for pt in face.get('landmarks', []):
                cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
