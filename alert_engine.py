import collections
import time

from config import (
    WATCH_CONF_MIN,
    WARNING_CONF_MIN,
    CRITICAL_CONF_MIN,
    WATCH_MIN_FRAMES,
    WARNING_MIN_FRAMES,
    CRITICAL_MIN_FRAMES,
    RESET_FRAMES,
    SMOOTHING_WINDOW,
    AREA_WATCH_MIN,
    AREA_CRITICAL_MIN,
    AREA_GROWTH_THRESHOLD,
)


class AlertEngine:
    def __init__(self):
        self.current_level = "CLEAR"
        self.consecutive_detections = 0
        self.consecutive_clear = 0
        self.confidence_history = collections.deque(maxlen=SMOOTHING_WINDOW)
        self.area_history = collections.deque(maxlen=5)
        self.confirmed = False
        self.confirmation_start_time = None
        self.CONFIRMATION_WINDOW_SEC = 10
        self.pending_level = None
        self.pending_level_frames = 0
        self.LEVEL_CHANGE_STABLE_FRAMES = 4

    def update(self, detection_result):
        fire_data = detection_result.get("fire", {})
        smoke_data = detection_result.get("smoke", {})

        fire_conf = float(fire_data.get("confidence", 0.0))
        smoke_conf = float(smoke_data.get("confidence", 0.0))
        fire_area = float(fire_data.get("area_ratio", 0.0))
        smoke_area = float(smoke_data.get("area_ratio", 0.0))

        combined_confidence = max(fire_conf, smoke_conf)
        combined_area = fire_area + smoke_area

        if combined_confidence < WATCH_CONF_MIN:
            self.consecutive_clear += 1
            self.consecutive_detections = 0
            proposed_level = self.current_level
            if self.consecutive_clear >= RESET_FRAMES:
                proposed_level = "CLEAR"

            self.current_level = self._apply_level_hysteresis(proposed_level)
            confirmation_triggered = self._check_confirmation(self.current_level)
            return (self.current_level, confirmation_triggered)

        self.consecutive_clear = 0
        self.consecutive_detections += 1
        self.confidence_history.append(combined_confidence)
        self.area_history.append(combined_area)

        smoothed_conf = sum(self.confidence_history) / len(self.confidence_history)
        area_growing = self._is_area_growing()

        fire_detected = fire_conf > 0.0
        smoke_detected = smoke_conf > 0.0
        both_detected = fire_detected and smoke_detected

        proposed_level = self.current_level
        if (
            smoothed_conf >= CRITICAL_CONF_MIN
            and self.consecutive_detections >= CRITICAL_MIN_FRAMES
            and combined_area >= AREA_CRITICAL_MIN
        ):
            proposed_level = "CRITICAL"
        elif (
            smoothed_conf >= WARNING_CONF_MIN
            and self.consecutive_detections >= WARNING_MIN_FRAMES
            and (both_detected or area_growing)
        ):
            proposed_level = "WARNING"
        elif (
            smoothed_conf >= WATCH_CONF_MIN
            and self.consecutive_detections >= WATCH_MIN_FRAMES
            and combined_area >= AREA_WATCH_MIN
        ):
            proposed_level = "WATCH"

        if proposed_level == "CRITICAL" and len(self.area_history) >= 2:
            if self.area_history[-1] < self.area_history[0]:
                proposed_level = "WARNING"

        self.current_level = self._apply_level_hysteresis(proposed_level)

        confirmation_triggered = self._check_confirmation(self.current_level)
        return (self.current_level, confirmation_triggered)

    def get_level(self):
        return self.current_level

    def is_confirmed(self):
        return self.confirmed

    def reset(self):
        self.current_level = "CLEAR"
        self.consecutive_detections = 0
        self.consecutive_clear = 0
        self.confidence_history.clear()
        self.area_history.clear()
        self.confirmed = False
        self.confirmation_start_time = None
        self.pending_level = None
        self.pending_level_frames = 0

    def _is_area_growing(self):
        if len(self.area_history) < 2:
            return False
        first_area = self.area_history[0]
        last_area = self.area_history[-1]
        return last_area > first_area * (1 + AREA_GROWTH_THRESHOLD)

    def _check_confirmation(self, current_level):
        if current_level in ("WARNING", "CRITICAL"):
            if self.confirmation_start_time is None:
                self.confirmation_start_time = time.time()

            elapsed = time.time() - self.confirmation_start_time
            if elapsed >= self.CONFIRMATION_WINDOW_SEC and not self.confirmed:
                self.confirmed = True
                return True
        else:
            self.confirmation_start_time = None
            self.confirmed = False

        return False

    def _apply_level_hysteresis(self, proposed_level):
        if proposed_level == self.current_level:
            self.pending_level = None
            self.pending_level_frames = 0
            return self.current_level

        if self.pending_level != proposed_level:
            self.pending_level = proposed_level
            self.pending_level_frames = 1
            return self.current_level

        self.pending_level_frames += 1
        if self.pending_level_frames >= self.LEVEL_CHANGE_STABLE_FRAMES:
            self.current_level = proposed_level
            self.pending_level = None
            self.pending_level_frames = 0

        return self.current_level
