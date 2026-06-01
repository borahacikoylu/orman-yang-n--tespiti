import os
import json
import datetime
import threading

import config


class FireLogger:
    def __init__(self):
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self.cooldown_tracker = {
            "WATCH": None,
            "WARNING": None,
            "CRITICAL": None,
        }
        self._lock = threading.Lock()

    def _today_log_path(self):
        today = datetime.date.today().isoformat()
        return os.path.join(config.LOG_DIR, f"{today}.json")

    def log_alert(
        self,
        alert_level,
        frame_id,
        confidence_fire,
        confidence_smoke,
        area_ratio,
        video_source,
    ):
        now = datetime.datetime.now()

        with self._lock:
            last_triggered_time = self.cooldown_tracker.get(alert_level)
            if last_triggered_time is not None:
                elapsed = (now - last_triggered_time).total_seconds()
                if elapsed < config.ALARM_COOLDOWN_SEC:
                    return

            log_data = {
                "timestamp": now.isoformat(),
                "frame_id": frame_id,
                "alert_level": alert_level,
                "confidence_fire": confidence_fire,
                "confidence_smoke": confidence_smoke,
                "area_ratio": area_ratio,
                "video_source": video_source,
            }

            log_path = self._today_log_path()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

            self.cooldown_tracker[alert_level] = now

        alarm_thread = threading.Thread(
            target=self.play_alarm,
            args=(alert_level,),
            daemon=True,
        )
        alarm_thread.start()

    def play_alarm(self, alert_level):
        beep_count = {
            "WATCH": 1,
            "WARNING": 2,
            "CRITICAL": 3,
        }.get(alert_level, 1)

        played = False
        try:
            from playsound import playsound  # type: ignore

            sound_paths = {
                "WATCH": os.path.join("sounds", "watch.mp3"),
                "WARNING": os.path.join("sounds", "warning.mp3"),
                "CRITICAL": os.path.join("sounds", "critical.mp3"),
            }
            sound_path = sound_paths.get(alert_level)
            if sound_path and os.path.exists(sound_path):
                playsound(sound_path)
                played = True
        except Exception:
            played = False

        if not played:
            for _ in range(beep_count):
                print("\a", end="", flush=True)
            print()

    def get_today_logs(self):
        log_path = self._today_log_path()
        if not os.path.exists(log_path):
            return []

        logs = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return logs
