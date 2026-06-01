import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD, CLASSES


class YOLODetector:
    def __init__(self):
        model_to_load = MODEL_PATH
        try:
            with open(MODEL_PATH, "rb"):
                pass
        except FileNotFoundError:
            print(f"Warning: model file not found at '{MODEL_PATH}'. Falling back to 'yolov8s.pt'.")
            model_to_load = "yolov8s.pt"

        self.model = YOLO(model_to_load)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"YOLODetector using device: {self.device}")

        self.target_classes = {name.lower() for name in CLASSES}

    def detect(self, frame, frame_id):
        frame_h, frame_w = frame.shape[:2]
        frame_area = float(frame_w * frame_h) if frame_w > 0 and frame_h > 0 else 1.0

        result_data = {
            "fire": {
                "confidence": 0.0,
                "bbox": None,
                "area_ratio": 0.0,
            },
            "smoke": {
                "confidence": 0.0,
                "bbox": None,
                "area_ratio": 0.0,
            },
            "frame_id": int(frame_id),
        }

        results = self.model(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            device=self.device,
        )

        for result in results:
            boxes = result.boxes
            names = result.names
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                class_name = str(names.get(cls_id, "")).lower()
                if class_name not in self.target_classes:
                    continue
                if class_name not in ("fire", "smoke"):
                    continue

                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x1 = max(0, min(frame_w - 1, int(round(x1)))) if frame_w > 0 else 0
                y1 = max(0, min(frame_h - 1, int(round(y1)))) if frame_h > 0 else 0
                x2 = max(0, min(frame_w - 1, int(round(x2)))) if frame_w > 0 else 0
                y2 = max(0, min(frame_h - 1, int(round(y2)))) if frame_h > 0 else 0

                box_w = max(0, x2 - x1)
                box_h = max(0, y2 - y1)
                area_ratio = (box_w * box_h) / frame_area

                if conf > result_data[class_name]["confidence"]:
                    result_data[class_name] = {
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "area_ratio": float(area_ratio),
                    }

        return result_data

    def draw_boxes(self, frame, detection_result):
        annotated = frame.copy()

        fire_data = detection_result.get("fire", {})
        fire_bbox = fire_data.get("bbox")
        if fire_bbox is not None:
            x1, y1, x2, y2 = fire_bbox
            fire_conf = float(fire_data.get("confidence", 0.0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                f"FIRE {fire_conf:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        smoke_data = detection_result.get("smoke", {})
        smoke_bbox = smoke_data.get("bbox")
        if smoke_bbox is not None:
            x1, y1, x2, y2 = smoke_bbox
            smoke_conf = float(smoke_data.get("confidence", 0.0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (128, 128, 128), 2)
            cv2.putText(
                annotated,
                f"SMOKE {smoke_conf:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (128, 128, 128),
                2,
            )

        return annotated
