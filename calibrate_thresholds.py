import argparse
import math
from dataclasses import dataclass

import numpy as np

from alert_engine import AlertEngine
from cv_detector import CVDetector
from opencv_filter import OpenCVFilter
from video_reader import VideoReader


@dataclass
class VideoStats:
    path: str
    frames: int
    suspicious_frames: int
    smoke_only_frames: int
    warning_frames: int
    critical_frames: int
    confirmed_events: int
    smoke_conf_values: list
    smoke_area_values: list


def percentile(values, p):
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float32), p))


def analyze_video(path, max_fire_conf_for_smoke_only, smoke_over_fire_ratio):
    reader = VideoReader(path)
    filt = OpenCVFilter()
    detector = CVDetector()
    alert_engine = AlertEngine()

    frames = 0
    suspicious_frames = 0
    smoke_only_frames = 0
    warning_frames = 0
    critical_frames = 0
    confirmed_events = 0
    smoke_conf_values = []
    smoke_area_values = []

    try:
        while True:
            packet = reader.read_frame()
            if packet is None:
                break

            frame, frame_id = packet
            frames += 1

            if filt.is_suspicious(frame):
                suspicious_frames += 1
                det = detector.detect(frame, frame_id)
            else:
                det = {
                    "fire": {"confidence": 0.0, "bbox": None, "area_ratio": 0.0},
                    "smoke": {"confidence": 0.0, "bbox": None, "area_ratio": 0.0},
                    "frame_id": int(frame_id),
                }

            fire_conf = float(det["fire"]["confidence"])
            smoke_conf = float(det["smoke"]["confidence"])
            smoke_area = float(det["smoke"]["area_ratio"])

            smoke_dominant = (
                smoke_conf > 0.0
                and fire_conf <= max_fire_conf_for_smoke_only
                and smoke_conf >= (fire_conf * smoke_over_fire_ratio)
            )
            if smoke_dominant:
                smoke_only_frames += 1
                smoke_conf_values.append(smoke_conf)
                smoke_area_values.append(smoke_area)

            level, confirmed = alert_engine.update(det)
            if level == "WARNING":
                warning_frames += 1
            elif level == "CRITICAL":
                critical_frames += 1
            if confirmed:
                confirmed_events += 1
    finally:
        reader.release()

    return VideoStats(
        path=path,
        frames=frames,
        suspicious_frames=suspicious_frames,
        smoke_only_frames=smoke_only_frames,
        warning_frames=warning_frames,
        critical_frames=critical_frames,
        confirmed_events=confirmed_events,
        smoke_conf_values=smoke_conf_values,
        smoke_area_values=smoke_area_values,
    )


def find_best_warning_threshold(pos_conf, pos_area, neg_conf, neg_area):
    if not pos_conf:
        return 0.5, 0.015, 0.0, 0.0, 0.0

    grid_conf = np.arange(0.30, 0.91, 0.02)
    grid_area = np.arange(0.005, 0.061, 0.002)
    best = None

    p_conf = np.array(pos_conf, dtype=np.float32)
    p_area = np.array(pos_area, dtype=np.float32)
    n_conf = np.array(neg_conf, dtype=np.float32) if neg_conf else np.array([], dtype=np.float32)
    n_area = np.array(neg_area, dtype=np.float32) if neg_area else np.array([], dtype=np.float32)

    for conf_thr in grid_conf:
        for area_thr in grid_area:
            tp = int(np.sum((p_conf >= conf_thr) & (p_area >= area_thr)))
            fn = int(len(p_conf) - tp)

            if len(n_conf) > 0:
                fp = int(np.sum((n_conf >= conf_thr) & (n_area >= area_thr)))
            else:
                fp = 0

            precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2.0 * precision * recall / (precision + recall)

            score = (f1, precision, recall)
            if best is None or score > best[0]:
                best = (score, float(conf_thr), float(area_thr))

    if best is None:
        return 0.5, 0.015, 0.0, 0.0, 0.0

    f1, precision, recall = best[0]
    return best[1], best[2], f1, precision, recall


def build_recommendation(pos_conf, pos_area, neg_conf, neg_area):
    warning_conf, warning_area, f1, precision, recall = find_best_warning_threshold(
        pos_conf, pos_area, neg_conf, neg_area
    )

    neg_conf_p99 = percentile(neg_conf, 99)
    neg_area_p99 = percentile(neg_area, 99)
    pos_conf_p80 = percentile(pos_conf, 80)
    pos_area_p80 = percentile(pos_area, 80)

    critical_conf = max(warning_conf + 0.16, pos_conf_p80, neg_conf_p99 + 0.03)
    critical_area = max(warning_area * 1.8, pos_area_p80, neg_area_p99 + 0.003)

    critical_conf = min(0.98, round(critical_conf, 2))
    critical_area = min(0.20, round(critical_area, 3))
    warning_conf = round(warning_conf, 2)
    warning_area = round(warning_area, 3)

    if critical_conf <= warning_conf:
        critical_conf = min(0.98, round(warning_conf + 0.12, 2))
    if critical_area <= warning_area:
        critical_area = min(0.20, round(warning_area + 0.01, 3))

    return {
        "warning_conf": warning_conf,
        "warning_area": warning_area,
        "critical_conf": critical_conf,
        "critical_area": critical_area,
        "fit_f1": round(f1, 3),
        "fit_precision": round(precision, 3),
        "fit_recall": round(recall, 3),
    }


def print_video_summary(stats):
    suspicious_ratio = (stats.suspicious_frames / stats.frames) if stats.frames else 0.0
    smoke_only_ratio = (stats.smoke_only_frames / stats.frames) if stats.frames else 0.0

    print(f"\nVideo: {stats.path}")
    print(f"  frames={stats.frames} suspicious={stats.suspicious_frames} ({suspicious_ratio:.1%})")
    print(f"  smoke_only_frames={stats.smoke_only_frames} ({smoke_only_ratio:.1%})")
    print(
        "  current_alert_frames:"
        f" warning={stats.warning_frames} critical={stats.critical_frames}"
        f" confirmed={stats.confirmed_events}"
    )
    if stats.smoke_conf_values:
        print(
            "  smoke_only_conf p50/p90/p99="
            f"{percentile(stats.smoke_conf_values,50):.3f}/"
            f"{percentile(stats.smoke_conf_values,90):.3f}/"
            f"{percentile(stats.smoke_conf_values,99):.3f}"
        )
        print(
            "  smoke_only_area p50/p90/p99="
            f"{percentile(stats.smoke_area_values,50):.4f}/"
            f"{percentile(stats.smoke_area_values,90):.4f}/"
            f"{percentile(stats.smoke_area_values,99):.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate smoke-only alert thresholds using labeled videos."
    )
    parser.add_argument(
        "--positive",
        nargs="+",
        required=True,
        help="Videos that contain real fire/smoke incidents.",
    )
    parser.add_argument(
        "--negative",
        nargs="+",
        default=[],
        help="Videos that should not trigger fire alerts (fog/cloud/no-fire).",
    )
    parser.add_argument(
        "--max-fire-conf-for-smoke-only",
        type=float,
        default=0.25,
        help="Treat frame as smoke-dominant if fire confidence is below this value.",
    )
    parser.add_argument(
        "--smoke-over-fire-ratio",
        type=float,
        default=1.10,
        help="Smoke confidence should be at least this multiplier of fire confidence.",
    )
    args = parser.parse_args()

    pos_stats = [
        analyze_video(path, args.max_fire_conf_for_smoke_only, args.smoke_over_fire_ratio)
        for path in args.positive
    ]
    neg_stats = [
        analyze_video(path, args.max_fire_conf_for_smoke_only, args.smoke_over_fire_ratio)
        for path in args.negative
    ]

    print("\n=== VIDEO SUMMARIES ===")
    for s in pos_stats + neg_stats:
        print_video_summary(s)

    pos_conf = []
    pos_area = []
    for s in pos_stats:
        pos_conf.extend(s.smoke_conf_values)
        pos_area.extend(s.smoke_area_values)

    neg_conf = []
    neg_area = []
    for s in neg_stats:
        neg_conf.extend(s.smoke_conf_values)
        neg_area.extend(s.smoke_area_values)

    recommendation = build_recommendation(pos_conf, pos_area, neg_conf, neg_area)

    print("\n=== RECOMMENDED CONFIG UPDATE ===")
    print(f"SMOKE_ONLY_WARNING_CONF_MIN = {recommendation['warning_conf']}")
    print(f"SMOKE_ONLY_WARNING_AREA_MIN = {recommendation['warning_area']}")
    print(f"SMOKE_ONLY_CRITICAL_CONF_MIN = {recommendation['critical_conf']}")
    print(f"SMOKE_ONLY_CRITICAL_AREA_MIN = {recommendation['critical_area']}")
    print(
        "fit_metrics:"
        f" f1={recommendation['fit_f1']}"
        f" precision={recommendation['fit_precision']}"
        f" recall={recommendation['fit_recall']}"
    )


if __name__ == "__main__":
    main()
