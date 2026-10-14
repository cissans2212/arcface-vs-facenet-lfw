import time
import os
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

class MTCNNDetector:
    def __init__(self, keep_all=True, device='cpu'):
        self.device = device
        self.model = MTCNN(keep_all=keep_all, device=device)

    def detect(self, img_path):
        """
        Detect faces in image.
        Returns:
            detected (bool): True if at least one face detected
            faces (list of dict): List of detections with bbox [x1, y1, x2, y2], score, landmarks
            elapsed_time (float): Detection time in seconds
        """
        pil_img = Image.open(img_path).convert('RGB')
        t0 = time.time()
        boxes, probs, landmarks = self.model.detect(pil_img, landmarks=True)
        elapsed_time = time.time() - t0

        faces = []
        if boxes is not None:
            for i, box in enumerate(boxes):
                prob = float(probs[i]) if probs is not None else 0.0
                lm = landmarks[i].tolist() if landmarks is not None else []
                faces.append({
                    'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    'score': prob,
                    'landmarks': lm
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
            # Draw bounding box (Blue for MTCNN)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"MTCNN: {score:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
            
            # Draw landmarks
            for pt in face.get('landmarks', []):
                cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
