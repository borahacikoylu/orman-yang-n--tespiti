# python main.py orman_video.mp4
# python main.py orman_video.mp4 --debug
# python main.py orman_video.mp4 --cv-only   (sadece OpenCV, modelsiz)

import os
import sys
import argparse

from ui import FireDetectionUI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orman yangını tespit sistemi")
    parser.add_argument("video", help="Video dosya yolu")
    parser.add_argument("--debug", action="store_true", help="Debug çıktısını aç")
    parser.add_argument(
        "--cv-only",
        action="store_true",
        help="Sadece OpenCV detektörünü kullan, modeli devre dışı bırak",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Hata: Video dosyası bulunamadı: {args.video}")
        sys.exit(1)

    print(f"Sistem başlatılıyor: {args.video}")
    if args.debug:
        print(f"Python sürümü: {sys.version}")
        print("Debug modu aktif")

    ui = FireDetectionUI(video_source=args.video, use_yolo=not args.cv_only)
    try:
        ui.run()
    except KeyboardInterrupt:
        ui.stop()
        print("Sistem kapatıldı.")
