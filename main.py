# python main.py orman_video.mp4
# python main.py orman_video.mp4 --debug
# python main.py orman_video.mp4 --yolo          # YOLO + CV fusion modu

import os
import sys
import argparse

from ui import FireDetectionUI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orman yangını tespit sistemi")
    parser.add_argument("video", help="Video dosya yolu")
    parser.add_argument("--debug", action="store_true", help="Debug çıktısını aç")
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="YOLO + CV fusion modunu etkinleştir (fire-trained model gerekir)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Hata: Video dosyası bulunamadı: {args.video}")
        sys.exit(1)

    print(f"Sistem başlatılıyor: {args.video}")
    if args.debug:
        print(f"Python sürümü: {sys.version}")
        print("Debug modu aktif")

    if args.yolo:
        print("Mod: YOLO + CV fusion")
    else:
        print("Mod: CV (OpenCV)")

    ui = FireDetectionUI(video_source=args.video, use_yolo=args.yolo)
    try:
        ui.run()
    except KeyboardInterrupt:
        ui.stop()
        print("Sistem kapatıldı.")
