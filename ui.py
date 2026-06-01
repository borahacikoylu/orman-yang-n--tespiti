import tkinter as tk
import tkinter.ttk as ttk
import cv2
import PIL.Image
import PIL.ImageTk
import queue
import threading

from config import INPUT_SIZE


class FireDetectionUI:
    def __init__(self, video_source):
        self.video_source = video_source
        self.root = tk.Tk()
        self.root.title("Orman Yangını Tespit Sistemi")

        self.frame_queue = queue.Queue(maxsize=2)
        self.alert_queue = queue.Queue(maxsize=10)

        self._stop_event = threading.Event()
        self.paused = False
        self._last_canvas_image = None
        self._log_entries = []
        self._last_logged_level = None
        self._last_confirmed_frame = None
        self._confirmation_popup_open = False
        self._display_level = "CLEAR"
        self._pending_display_level = None
        self._pending_display_frames = 0
        self._display_level_stable_frames = 5
        self._fire_bar_smooth = 0.0
        self._smoke_bar_smooth = 0.0

        self._build_layout()
        self._set_banner(self._display_level)

        self.worker_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.worker_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self.root.after(33, self._ui_update_loop)

    def _build_layout(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)

        self.banner_label = tk.Label(
            self.root,
            text="Sistem İzliyor...",
            font=("Helvetica", 20, "bold"),
            padx=10,
            pady=10,
        )
        self.banner_label.grid(row=0, column=0, sticky="ew")

        content = tk.Frame(self.root)
        content.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(
            content,
            width=INPUT_SIZE,
            height=INPUT_SIZE,
            bg="black",
            highlightthickness=1,
            highlightbackground="#999999",
        )
        self.video_canvas.grid(row=0, column=0, sticky="nw")

        right_panel = tk.Frame(content)
        right_panel.grid(row=0, column=1, sticky="n", padx=(12, 0))

        tk.Label(right_panel, text="Alev Güveni:").pack(anchor="w")
        self.fire_progress = ttk.Progressbar(right_panel, orient="horizontal", length=260, maximum=100)
        self.fire_progress.pack(anchor="w", pady=(0, 8))

        tk.Label(right_panel, text="Duman Güveni:").pack(anchor="w")
        self.smoke_progress = ttk.Progressbar(right_panel, orient="horizontal", length=260, maximum=100)
        self.smoke_progress.pack(anchor="w", pady=(0, 8))

        area_row = tk.Frame(right_panel)
        area_row.pack(anchor="w", fill="x", pady=2)
        tk.Label(area_row, text="Alan Oranı:").pack(side="left")
        self.area_value_label = tk.Label(area_row, text="0.0000")
        self.area_value_label.pack(side="left", padx=(8, 0))

        fps_row = tk.Frame(right_panel)
        fps_row.pack(anchor="w", fill="x", pady=2)
        tk.Label(fps_row, text="FPS:").pack(side="left")
        self.fps_value_label = tk.Label(fps_row, text="0.0")
        self.fps_value_label.pack(side="left", padx=(8, 0))

        seq_row = tk.Frame(right_panel)
        seq_row.pack(anchor="w", fill="x", pady=2)
        tk.Label(seq_row, text="Ardışık Frame:").pack(side="left")
        self.seq_value_label = tk.Label(seq_row, text="0")
        self.seq_value_label.pack(side="left", padx=(8, 0))

        ttk.Separator(right_panel, orient="horizontal").pack(fill="x", pady=10)

        self.toggle_button = tk.Button(right_panel, text="Durdur", width=14, command=self._toggle_pause)
        self.toggle_button.pack(anchor="w", pady=(0, 6))

        self.close_button = tk.Button(right_panel, text="Kapat", width=14, command=self.stop)
        self.close_button.pack(anchor="w")

        log_frame = tk.Frame(self.root)
        log_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=8, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.tag_configure("normal", foreground="#666666", font=("Helvetica", 10))
        self.log_text.tag_configure("confirmed", foreground="#cc2020", font=("Helvetica", 11, "bold"))

        log_scroll = tk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

    def _analysis_loop(self):
        import time
        from video_reader import VideoReader
        from opencv_filter import OpenCVFilter
        from cv_detector import CVDetector
        from alert_engine import AlertEngine
        from logger import FireLogger

        reader = None
        try:
            reader = VideoReader(self.video_source)
            filter_engine = OpenCVFilter()
            detector = CVDetector()
            alert_engine = AlertEngine()
            logger = FireLogger()

            prev_time = time.time()

            while not self._stop_event.is_set():
                if self.paused:
                    time.sleep(0.05)
                    continue

                frame_packet = reader.read_frame()
                if frame_packet is None:
                    break

                frame, frame_id = frame_packet

                suspicious = filter_engine.is_suspicious(frame)
                if suspicious:
                    detection_result = detector.detect(frame, frame_id)
                    frame_to_show = detector.draw_boxes(frame, detection_result)
                else:
                    detection_result = {
                        "fire": {"confidence": 0.0, "bbox": None, "area_ratio": 0.0},
                        "smoke": {"confidence": 0.0, "bbox": None, "area_ratio": 0.0},
                        "frame_id": int(frame_id),
                    }
                    frame_to_show = frame.copy()

                level, confirmed = alert_engine.update(detection_result)

                fire_conf = float(detection_result["fire"]["confidence"])
                smoke_conf = float(detection_result["smoke"]["confidence"])
                combined_area = float(detection_result["fire"]["area_ratio"]) + float(
                    detection_result["smoke"]["area_ratio"]
                )

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                if confirmed:
                    logger.log_alert(
                        level,
                        frame_id,
                        fire_conf,
                        smoke_conf,
                        combined_area,
                        self.video_source,
                    )

                self._queue_latest(self.frame_queue, (frame_to_show, frame_id))
                self.alert_queue.put(
                    {
                        "level": level,
                        "confirmed": confirmed,
                        "confidence_fire": detection_result["fire"]["confidence"],
                        "confidence_smoke": detection_result["smoke"]["confidence"],
                        "area_ratio": detection_result["fire"]["area_ratio"],
                        "frame_id": detection_result["frame_id"],
                    }
                )
        finally:
            if reader is not None:
                reader.release()

    def _ui_update_loop(self):
        if self._stop_event.is_set():
            return

        latest_frame = None
        while True:
            try:
                latest_frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break

        if latest_frame is not None:
            frame, _ = latest_frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = PIL.Image.fromarray(rgb)
            photo = PIL.ImageTk.PhotoImage(image=image)
            self._last_canvas_image = photo
            self.video_canvas.delete("all")
            self.video_canvas.create_image(0, 0, anchor="nw", image=photo)

        latest_info = None
        while True:
            try:
                latest_info = self.alert_queue.get_nowait()
            except queue.Empty:
                break

        if latest_info is not None:
            fire_pct = max(0.0, min(100.0, float(latest_info["confidence_fire"]) * 100.0))
            smoke_pct = max(0.0, min(100.0, float(latest_info["confidence_smoke"]) * 100.0))
            self._fire_bar_smooth = (self._fire_bar_smooth * 0.7) + (fire_pct * 0.3)
            self._smoke_bar_smooth = (self._smoke_bar_smooth * 0.7) + (smoke_pct * 0.3)
            self.fire_progress["value"] = self._fire_bar_smooth
            self.smoke_progress["value"] = self._smoke_bar_smooth
            self.area_value_label.config(text=f"{float(latest_info['area_ratio']):.4f}")
            self.seq_value_label.config(text=str(latest_info["frame_id"]))
            self._update_display_level(latest_info["level"])
            self._set_banner(self._display_level)

            if latest_info["level"] != self._last_logged_level:
                self._append_log(
                    f"[{latest_info['level']}] Frame {latest_info['frame_id']}",
                    kind="normal",
                )
                self._last_logged_level = latest_info["level"]

            if latest_info.get("confirmed") and latest_info["frame_id"] != self._last_confirmed_frame:
                self._last_confirmed_frame = latest_info["frame_id"]
                self._append_log(
                    f"⚠ YANGIN DOĞRULANDI — Frame {latest_info['frame_id']}",
                    kind="confirmed",
                )
                self.root.after(0, self._show_confirmation_popup)

        self.root.after(33, self._ui_update_loop)

    def run(self):
        self.root.mainloop()

    def stop(self):
        if self._stop_event.is_set():
            return

        self._stop_event.set()
        if self.worker_thread.is_alive() and threading.current_thread() is not self.worker_thread:
            self.worker_thread.join(timeout=2.0)

        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _toggle_pause(self):
        self.paused = not self.paused
        self.toggle_button.config(text="Devam" if self.paused else "Durdur")

    def _queue_latest(self, q, value):
        try:
            q.put_nowait(value)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(value)
            except queue.Full:
                pass

    def _append_log(self, line, kind="normal"):
        self._log_entries.append((line, kind))
        self._log_entries = self._log_entries[-20:]

        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        for text, text_kind in self._log_entries:
            tag = "confirmed" if text_kind == "confirmed" else "normal"
            self.log_text.insert(tk.END, f"{text}\n", tag)
        self.log_text.config(state="disabled")
        self.log_text.see(tk.END)

    def _show_confirmation_popup(self):
        if self._confirmation_popup_open:
            return
        self._confirmation_popup_open = True

        popup = tk.Toplevel(self.root)
        popup.title("Yangın Doğrulama")
        popup.configure(bg="#cc2020")
        popup.attributes("-topmost", True)
        popup.transient(self.root)
        popup.grab_set()

        width, height = 400, 200
        screen_w = popup.winfo_screenwidth()
        screen_h = popup.winfo_screenheight()
        pos_x = (screen_w - width) // 2
        pos_y = (screen_h - height) // 2
        popup.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
        popup.resizable(False, False)

        title = tk.Label(
            popup,
            text="🔴 YANGIN DOĞRULANDI",
            bg="#cc2020",
            fg="white",
            font=("Helvetica", 18, "bold"),
        )
        title.pack(pady=(24, 10))

        message = tk.Label(
            popup,
            text="Sistem yangını teyit etti. Lütfen bölgeyi kontrol edin.",
            bg="#cc2020",
            fg="white",
            font=("Helvetica", 12, "bold"),
            wraplength=340,
            justify="center",
        )
        message.pack(pady=(0, 16))

        def close_popup():
            self._confirmation_popup_open = False
            popup.destroy()

        button = tk.Button(
            popup,
            text="Anladım, Devam Et",
            command=close_popup,
            font=("Helvetica", 11, "bold"),
            padx=12,
            pady=6,
        )
        button.pack()
        popup.protocol("WM_DELETE_WINDOW", close_popup)

    def _set_banner(self, level):
        styles = {
            "CLEAR": {
                "bg": "#cccccc",
                "fg": "black",
                "text": "Sistem İzliyor...",
            },
            "WATCH": {
                "bg": "#f0c040",
                "fg": "black",
                "text": "⚠ İZLEME — Şüpheli Aktivite",
            },
            "WARNING": {
                "bg": "#e07020",
                "fg": "black",
                "text": "⚠⚠ UYARI — Olası Yangın",
            },
            "CRITICAL": {
                "bg": "#cc2020",
                "fg": "white",
                "text": "🔴 KRİTİK — YANGIN TESPİT EDİLDİ",
            },
        }
        style = styles.get(level, styles["CLEAR"])
        self.banner_label.config(bg=style["bg"], fg=style["fg"], text=style["text"])

    def _update_display_level(self, incoming_level):
        if incoming_level == self._display_level:
            self._pending_display_level = None
            self._pending_display_frames = 0
            return

        if self._pending_display_level != incoming_level:
            self._pending_display_level = incoming_level
            self._pending_display_frames = 1
            return

        self._pending_display_frames += 1
        if self._pending_display_frames >= self._display_level_stable_frames:
            self._display_level = incoming_level
            self._pending_display_level = None
            self._pending_display_frames = 0
