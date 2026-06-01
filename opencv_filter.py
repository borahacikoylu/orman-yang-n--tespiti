import cv2
import numpy as np

from config import (
    HSV_MIN_PIXEL_RATIO,
    MOG2_HISTORY,
    MOG2_THRESHOLD,
    MOG2_DETECT_SHADOWS,
)


class OpenCVFilter:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS,
        )
        self.warmed_up = False
        self.frame_count = 0

    def is_suspicious(self, frame):
        self.frame_count += 1

        if self.frame_count < 30:
            self.bg_subtractor.apply(frame)
            return False

        self.warmed_up = True

        hsv_ok = self._check_hsv(frame)
        mog2_ok = self._check_mog2(frame)

        if hsv_ok and mog2_ok:
            return True
        if hsv_ok and not mog2_ok:
            return True
        if not hsv_ok and mog2_ok:
            return False
        return False

    def _check_hsv(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        flame_mask = cv2.inRange(
            hsv,
            np.array([0, 150, 150], dtype=np.uint8),
            np.array([25, 255, 255], dtype=np.uint8),
        )
        ember_mask = cv2.inRange(
            hsv,
            np.array([20, 150, 200], dtype=np.uint8),
            np.array([40, 255, 255], dtype=np.uint8),
        )
        smoke_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 100], dtype=np.uint8),
            np.array([180, 40, 200], dtype=np.uint8),
        )

        combined_mask = cv2.bitwise_or(flame_mask, ember_mask)
        combined_mask = cv2.bitwise_or(combined_mask, smoke_mask)

        total_pixels = frame.shape[0] * frame.shape[1]
        matching_pixels = cv2.countNonZero(combined_mask)
        ratio = matching_pixels / float(total_pixels) if total_pixels > 0 else 0.0

        return ratio > HSV_MIN_PIXEL_RATIO

    def _check_mog2(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)

        total_pixels = frame.shape[0] * frame.shape[1]
        moving_pixels = cv2.countNonZero(fg_mask)
        ratio = moving_pixels / float(total_pixels) if total_pixels > 0 else 0.0

        return ratio > 0.01

    def reset(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS,
        )
        self.frame_count = 0
        self.warmed_up = False
