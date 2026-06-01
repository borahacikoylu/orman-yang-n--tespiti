import cv2
import time

from config import INPUT_SIZE, FRAME_SKIP


class VideoReader:
    def __init__(self, source):
        self.source = source
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.fps >= 50:
            self.skip_rate = 4
        elif self.fps >= 25:
            self.skip_rate = 2
        elif self.fps > 0:
            self.skip_rate = 1
        else:
            # Fallback for missing/invalid FPS metadata.
            self.skip_rate = max(1, FRAME_SKIP)

        self.frame_counter = 0

    def read_frame(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                return None
            self.frame_counter += 1
            if self.frame_counter % self.skip_rate == 0:
                frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
                return frame, self.frame_counter

    def get_fps(self):
        return self.fps

    def release(self):
        if self.capture is not None:
            self.capture.release()
