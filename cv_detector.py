import cv2
import numpy as np
from collections import deque

from config import (
    CONF_THRESHOLD,
    HSV_FLAME_LOWER,
    HSV_FLAME_UPPER,
    HSV_EMBER_LOWER,
    HSV_EMBER_UPPER,
    HSV_SMOKE_LOWER,
    HSV_SMOKE_UPPER,
    MIN_CONTOUR_AREA,
    MAX_SOLIDITY,
    MAX_CIRCULARITY,
    FLOW_MAG_SCALE,
    FLOW_VAR_SCALE,
    WEIGHT_HSV,
    WEIGHT_CONTOUR,
    WEIGHT_FLOW,
    WEIGHT_GROWTH,
)


class CVDetector:
    def __init__(self):
        self.prev_gray = None
        self.area_history = deque(maxlen=5)
        self.contour_memory = deque(maxlen=5)

        self.smoke_ratio = 0.0
        self.fire_bbox = None
        self.fire_area_ratio = 0.0
        self.last_smoke_mask = None
        self.last_scores = {"hsv": 0.0, "cnt": 0.0, "flow": 0.0, "grw": 0.0}

    def detect(self, frame, frame_id):
        hsv_score = self._analyze_hsv(frame)
        contour_score = self._analyze_contours(frame)
        flow_score = self._analyze_optical_flow(frame)
        growth_score = self._analyze_growth(self.fire_area_ratio)

        final_confidence = (
            (hsv_score * WEIGHT_HSV)
            + (contour_score * WEIGHT_CONTOUR)
            + (flow_score * WEIGHT_FLOW)
            + (growth_score * WEIGHT_GROWTH)
        )
        final_confidence = min(float(final_confidence), 1.0)

        smoke_bbox, smoke_area_ratio = self._extract_smoke_bbox_and_area(frame)
        # Smoke confidence should reflect selected plume region, not full-frame cloud coverage.
        smoke_confidence = min(max(float(smoke_area_ratio * 22.0), 0.0), 1.0)
        if smoke_bbox is None:
            smoke_confidence = 0.0

        self.last_scores = {
            "hsv": float(hsv_score),
            "cnt": float(contour_score),
            "flow": float(flow_score),
            "grw": float(growth_score),
        }

        return {
            "fire": {
                "confidence": final_confidence,
                "bbox": self.fire_bbox,
                "area_ratio": float(self.fire_area_ratio),
            },
            "smoke": {
                "confidence": float(smoke_confidence),
                "bbox": smoke_bbox,
                "area_ratio": float(smoke_area_ratio),
            },
            "frame_id": frame_id,
        }

    def _analyze_hsv(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        flame_lower = np.array(HSV_FLAME_LOWER, dtype=np.uint8)
        flame_upper = np.array(HSV_FLAME_UPPER, dtype=np.uint8)
        ember_lower = np.array(HSV_EMBER_LOWER, dtype=np.uint8)
        ember_upper = np.array(HSV_EMBER_UPPER, dtype=np.uint8)
        bright_lower = np.array([0, 200, 200], dtype=np.uint8)
        bright_upper = np.array([15, 255, 255], dtype=np.uint8)

        flame_mask = cv2.inRange(hsv, flame_lower, flame_upper)
        ember_mask = cv2.inRange(hsv, ember_lower, ember_upper)
        bright_mask = cv2.inRange(hsv, bright_lower, bright_upper)

        fire_mask = cv2.bitwise_or(flame_mask, ember_mask)
        fire_mask = cv2.bitwise_or(fire_mask, bright_mask)

        kernel_open_fire = np.ones((3, 3), dtype=np.uint8)
        kernel_close_fire = np.ones((7, 7), dtype=np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel_open_fire)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel_close_fire)

        total_pixels = frame.shape[0] * frame.shape[1]
        fire_pixels = cv2.countNonZero(fire_mask)
        fire_ratio = (fire_pixels / float(total_pixels)) if total_pixels > 0 else 0.0

        smoke_lower = np.array(HSV_SMOKE_LOWER, dtype=np.uint8)
        smoke_upper = np.array(HSV_SMOKE_UPPER, dtype=np.uint8)
        smoke_mask = cv2.inRange(hsv, smoke_lower, smoke_upper)
        bright_smoke_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 180], dtype=np.uint8),
            np.array([180, 75, 255], dtype=np.uint8),
        )
        smoke_mask = cv2.bitwise_or(smoke_mask, bright_smoke_mask)
        kernel_open_smoke = np.ones((3, 3), dtype=np.uint8)
        kernel_close_smoke = np.ones((5, 5), dtype=np.uint8)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel_open_smoke)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel_close_smoke)

        # Anti-cloud filter: ignore the upper sky band where static clouds dominate.
        frame_h = frame.shape[0]
        sky_cut = int(frame_h * 0.30)
        smoke_mask[:sky_cut, :] = 0

        smoke_pixels = cv2.countNonZero(smoke_mask)
        self.smoke_ratio = (smoke_pixels / float(total_pixels)) if total_pixels > 0 else 0.0
        self.last_smoke_mask = smoke_mask

        if fire_ratio < 0.0015:
            return 0.0
        if fire_ratio < 0.008:
            return float(fire_ratio * 45.0)
        if fire_ratio < 0.04:
            return float(0.36 + (fire_ratio - 0.008) * 10.0)
        return float(min(0.68 + (fire_ratio - 0.04) * 5.0, 1.0))

    def _analyze_contours(self, frame):
        fire_mask = self._build_fire_mask(frame)
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = float(frame.shape[0] * frame.shape[1]) if frame.size > 0 else 1.0
        fire_like_contours = []

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area <= MIN_CONTOUR_AREA:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = contour_area / hull_area

            x, y, w, h = cv2.boundingRect(contour)
            if h <= 0:
                continue
            aspect_ratio = w / float(h)

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * contour_area / (perimeter * perimeter)

            if (
                0.3 <= solidity <= MAX_SOLIDITY
                and 0.2 <= aspect_ratio <= 5.0
                and circularity < MAX_CIRCULARITY
            ):
                fire_like_contours.append((contour, contour_area))

        if not fire_like_contours:
            self.fire_bbox = None
            self.fire_area_ratio = 0.0
            self.area_history.append(self.fire_area_ratio)
            self.contour_memory.append(0)
            return 0.0

        largest_contour, _ = max(fire_like_contours, key=lambda c: c[1])
        x, y, w, h = cv2.boundingRect(largest_contour)
        self.fire_bbox = [int(x), int(y), int(x + w), int(y + h)]

        total_fire_like_area = sum(area for _, area in fire_like_contours)
        self.fire_area_ratio = float(total_fire_like_area / frame_area)

        self.area_history.append(self.fire_area_ratio)
        self.contour_memory.append(len(fire_like_contours))

        base = min(len(fire_like_contours) / 5.0, 1.0)
        size_bonus = min(self.fire_area_ratio * 10.0, 0.4)
        return float(min(base + size_bonus, 1.0))

    def _analyze_optical_flow(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_magnitude = float(np.mean(magnitude))
        angle_variance = float(np.var(angle))

        mag_score = min(mean_magnitude / FLOW_MAG_SCALE, 1.0)
        var_score = min(angle_variance / FLOW_VAR_SCALE, 1.0)
        flow_score = (mag_score * 0.5) + (var_score * 0.5)

        self.prev_gray = gray
        return float(flow_score)

    def _analyze_growth(self, current_area):
        self.area_history.append(float(current_area))
        if len(self.area_history) < 3:
            return 0.0

        first = float(self.area_history[0])
        last = float(self.area_history[-1])

        if first == 0.0:
            return 0.0

        growth_rate = (last - first) / first

        if growth_rate < 0.05:
            return 0.0
        if growth_rate < 0.20:
            return 0.3
        if growth_rate < 0.50:
            return 0.6
        return 1.0

    def draw_boxes(self, frame, detection_result):
        annotated = frame.copy()

        fire_data = detection_result.get("fire", {})
        fire_conf = float(fire_data.get("confidence", 0.0))
        fire_bbox = fire_data.get("bbox")

        if fire_bbox is not None and fire_conf > CONF_THRESHOLD:
            x1, y1, x2, y2 = fire_bbox
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
        smoke_conf = float(smoke_data.get("confidence", 0.0))
        if smoke_conf > CONF_THRESHOLD and self.last_smoke_mask is not None:
            contours, _ = cv2.findContours(
                self.last_smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            selected = self._select_smoke_contour(contours, frame.shape)
            if selected is not None:
                x, y, w, h = cv2.boundingRect(selected)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (128, 128, 128), 2)
                cv2.putText(
                    annotated,
                    f"SMOKE {smoke_conf:.2f}",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (128, 128, 128),
                    2,
                )

        overlay_text = (
            f"HSV: {self.last_scores.get('hsv', 0.0):.2f}  "
            f"CNT: {self.last_scores.get('cnt', 0.0):.2f}  "
            f"FLOW: {self.last_scores.get('flow', 0.0):.2f}  "
            f"GRW: {self.last_scores.get('grw', 0.0):.2f}"
        )

        text_x, text_y = 10, 25
        (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        box_x1, box_y1 = text_x - 5, text_y - th - 8
        box_x2, box_y2 = text_x + tw + 5, text_y + 6

        overlay = annotated.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0, annotated)
        cv2.putText(
            annotated,
            overlay_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return annotated

    def reset(self):
        self.prev_gray = None
        self.area_history.clear()
        self.contour_memory.clear()

        self.smoke_ratio = 0.0
        self.fire_bbox = None
        self.fire_area_ratio = 0.0
        self.last_smoke_mask = None
        self.last_scores = {"hsv": 0.0, "cnt": 0.0, "flow": 0.0, "grw": 0.0}

    def _build_fire_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        flame_mask = cv2.inRange(
            hsv,
            np.array(HSV_FLAME_LOWER, dtype=np.uint8),
            np.array(HSV_FLAME_UPPER, dtype=np.uint8),
        )
        ember_mask = cv2.inRange(
            hsv,
            np.array(HSV_EMBER_LOWER, dtype=np.uint8),
            np.array(HSV_EMBER_UPPER, dtype=np.uint8),
        )
        bright_mask = cv2.inRange(
            hsv,
            np.array([0, 200, 200], dtype=np.uint8),
            np.array([15, 255, 255], dtype=np.uint8),
        )

        fire_mask = cv2.bitwise_or(flame_mask, ember_mask)
        fire_mask = cv2.bitwise_or(fire_mask, bright_mask)
        fire_mask = cv2.morphologyEx(
            fire_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
        )
        fire_mask = cv2.morphologyEx(
            fire_mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)
        )
        return fire_mask

    def _extract_smoke_bbox_and_area(self, frame):
        if self.last_smoke_mask is None:
            return None, 0.0

        contours, _ = cv2.findContours(
            self.last_smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        selected = self._select_smoke_contour(contours, frame.shape)
        if selected is None:
            return None, 0.0

        x, y, w, h = cv2.boundingRect(selected)
        bbox = [int(x), int(y), int(x + w), int(y + h)]
        frame_area = float(frame.shape[0] * frame.shape[1]) if frame.size > 0 else 1.0
        smoke_area_ratio = float(cv2.contourArea(selected) / frame_area)
        return bbox, smoke_area_ratio

    def _select_smoke_contour(self, contours, frame_shape):
        if not contours:
            return None

        frame_h, frame_w = frame_shape[:2]
        best = None
        best_score = -1.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 180:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h < 16:
                continue
            if y + h > int(frame_h * 0.95):
                continue

            # Cloud rejection: very high-only blobs are likely clouds.
            if y < int(frame_h * 0.18) and (y + h) < int(frame_h * 0.55):
                continue

            aspect = w / float(max(h, 1))
            if aspect > 14.0 and h < 60:
                continue

            # Reject long horizon cloud bands.
            if w > int(frame_w * 0.55) and h < int(frame_h * 0.18):
                continue

            # Prefer plumes connected to lower scene (horizon/terrain side).
            if (y + h) < int(frame_h * 0.58) and h < int(frame_h * 0.20):
                continue

            # Prefer compact/taller smoke plumes over long thin horizon bands.
            compactness = h / float(max(w, 1))
            score = area * (1.0 + min(compactness, 1.5)) + (h * 5.0)
            if score > best_score:
                best = contour
                best_score = score

        return best
