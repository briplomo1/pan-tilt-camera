#!/usr/bin/env python3
"""
Pan-tilt face tracker for Raspberry Pi 4.

- Camera: Raspberry Pi Camera Module (Picamera2) or a USB webcam
- Detector: OpenCV Haar cascade (fast enough on a Pi 4 at 640x480)
- Servos: GPIO pins (gpiozero), a PCA9685 servo board, or none (camera-only test)
- Live view: MJPEG stream at http://<pi-hostname>.local:8000 (works over SSH),
  or an OpenCV window with --display if you're on the Pi desktop / VNC

Examples:
  python face_tracker.py --servos none                 # test camera + detection only
  python face_tracker.py --servos gpio                 # servos on GPIO17 (pan) / GPIO27 (tilt)
  python face_tracker.py --servos pca9685              # servos on PCA9685 channels 0 / 1
  python face_tracker.py --camera 0 --servos none      # use a USB webcam instead
"""

import argparse
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

FRAME_W, FRAME_H = 640, 480


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
class Camera:
    def __init__(self, source: str, flip: bool):
        self.flip = flip
        self.picam = None
        self.cap = None

        if source == "picam":
            from picamera2 import Picamera2

            self.picam = Picamera2()
            config = self.picam.create_preview_configuration(
                main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}  # BGR order in memory -> OpenCV-ready
            )
            self.picam.configure(config)
            self.picam.start()
            time.sleep(1.0)  # let auto-exposure settle
        else:
            self.cap = cv2.VideoCapture(int(source))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            if not self.cap.isOpened():
                sys.exit(f"Could not open USB camera {source}. Try --camera 1, or check `ls /dev/video*`.")

    def read(self):
        if self.picam is not None:
            frame = self.picam.capture_array()
        else:
            ok, frame = self.cap.read()
            if not ok:
                return None
        if self.flip:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def close(self):
        if self.picam is not None:
            self.picam.stop()
        if self.cap is not None:
            self.cap.release()


# --------------------------------------------------------------------------
# Face detector
# --------------------------------------------------------------------------
def load_face_detector():
    name = "haarcascade_frontalface_default.xml"
    candidates = []
    if hasattr(cv2, "data"):
        candidates.append(os.path.join(cv2.data.haarcascades, name))
    candidates += [
        f"/usr/share/opencv4/haarcascades/{name}",
        f"/usr/share/opencv/haarcascades/{name}",
    ]
    for path in candidates:
        if os.path.exists(path):
            detector = cv2.CascadeClassifier(path)
            if not detector.empty():
                return detector
    sys.exit("Face model not found. Run: sudo apt install -y opencv-data")


def find_largest_face(detector, frame, scale=0.5):
    """Detect on a downscaled grayscale copy for speed; return (x, y, w, h) in full-res coords."""
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x / scale), int(y / scale), int(w / scale), int(h / scale)


# --------------------------------------------------------------------------
# Pan-tilt servos
# --------------------------------------------------------------------------
class PanTilt:
    MIN_ANGLE, MAX_ANGLE = 0.0, 180.0

    def __init__(self, mode, pan_pin, tilt_pin, pan_ch, tilt_ch):
        self.mode = mode
        self.pan = 90.0
        self.tilt = 90.0

        if mode == "gpio":
            from gpiozero import AngularServo

            # 0.5–2.5 ms pulse range suits most SG90 / MG90S hobby servos
            kwargs = dict(min_angle=0, max_angle=180, min_pulse_width=0.0005, max_pulse_width=0.0025)
            self.s_pan = AngularServo(pan_pin, **kwargs)
            self.s_tilt = AngularServo(tilt_pin, **kwargs)
        elif mode == "pca9685":
            from adafruit_servokit import ServoKit

            kit = ServoKit(channels=16)
            self.s_pan = kit.servo[pan_ch]
            self.s_tilt = kit.servo[tilt_ch]
            for s in (self.s_pan, self.s_tilt):
                s.set_pulse_width_range(500, 2500)

        self._apply()

    def _apply(self):
        if self.mode == "none":
            return
        self.s_pan.angle = self.pan
        self.s_tilt.angle = self.tilt

    def nudge(self, d_pan, d_tilt):
        self.pan = min(self.MAX_ANGLE, max(self.MIN_ANGLE, self.pan + d_pan))
        self.tilt = min(self.MAX_ANGLE, max(self.MIN_ANGLE, self.tilt + d_tilt))
        self._apply()

    def center(self):
        self.pan, self.tilt = 90.0, 90.0
        self._apply()

    def close(self):
        if self.mode == "gpio":
            self.s_pan.detach()
            self.s_tilt.detach()
        elif self.mode == "pca9685":
            self.s_pan.angle = None  # stop sending pulses
            self.s_tilt.angle = None


# --------------------------------------------------------------------------
# MJPEG stream (view in a browser on your laptop)
# --------------------------------------------------------------------------
class Streamer:
    def __init__(self, port):
        self.lock = threading.Lock()
        self.jpeg = None
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # keep the terminal quiet

            def do_GET(self):
                if self.path == "/":
                    body = (
                        b"<html><head><title>Face Tracker</title></head>"
                        b"<body style='margin:0;background:#111;display:flex;justify-content:center'>"
                        b"<img src='/stream' style='max-width:100%'></body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            with streamer.lock:
                                jpg = streamer.jpeg
                            if jpg is not None:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                                self.wfile.write(jpg + b"\r\n")
                            time.sleep(0.05)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_error(404)

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def update(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.lock:
                self.jpeg = buf.tobytes()

    def close(self):
        self.server.shutdown()


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Pan-tilt face tracker")
    p.add_argument("--camera", default="picam", help="'picam' for the Pi camera, or a USB index like 0")
    p.add_argument("--servos", choices=["none", "gpio", "pca9685"], default="none")
    p.add_argument("--pan-pin", type=int, default=17, help="GPIO (BCM) pin for pan servo")
    p.add_argument("--tilt-pin", type=int, default=27, help="GPIO (BCM) pin for tilt servo")
    p.add_argument("--pan-ch", type=int, default=0, help="PCA9685 channel for pan servo")
    p.add_argument("--tilt-ch", type=int, default=1, help="PCA9685 channel for tilt servo")
    p.add_argument("--invert-pan", action="store_true", help="flip pan direction if it moves away from you")
    p.add_argument("--invert-tilt", action="store_true", help="flip tilt direction if it moves away from you")
    p.add_argument("--flip", action="store_true", help="rotate image 180° (camera mounted upside down)")
    p.add_argument("--gain", type=float, default=5.0, help="max degrees moved per frame (higher = faster, jumpier)")
    p.add_argument("--deadzone", type=float, default=0.08, help="ignore errors smaller than this fraction of the frame")
    p.add_argument("--port", type=int, default=8000, help="port for the browser stream")
    p.add_argument("--display", action="store_true", help="also show an OpenCV window (Pi desktop/VNC only)")
    return p.parse_args()


def main():
    args = parse_args()

    camera = Camera(args.camera, args.flip)
    detector = load_face_detector()
    pantilt = PanTilt(args.servos, args.pan_pin, args.tilt_pin, args.pan_ch, args.tilt_ch)
    streamer = Streamer(args.port)

    print(f"Tracking. View the stream at http://<your-pi-hostname>.local:{args.port}  (Ctrl+C to stop)")

    pan_sign = 1 if args.invert_pan else -1
    tilt_sign = -1 if args.invert_tilt else 1
    cx, cy = FRAME_W / 2, FRAME_H / 2
    fps, last = 0.0, time.time()

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            face = find_largest_face(detector, frame)
            if face is not None:
                x, y, w, h = face
                fx, fy = x + w / 2, y + h / 2

                # Normalized error: -1 (left/top edge) .. +1 (right/bottom edge)
                err_x = (fx - cx) / cx
                err_y = (fy - cy) / cy

                d_pan = pan_sign * args.gain * err_x if abs(err_x) > args.deadzone else 0.0
                d_tilt = tilt_sign * args.gain * err_y if abs(err_y) > args.deadzone else 0.0
                pantilt.nudge(d_pan, d_tilt)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (int(fx), int(fy)), 4, (0, 255, 0), -1)

            # Crosshair at frame center
            cv2.line(frame, (int(cx) - 15, int(cy)), (int(cx) + 15, int(cy)), (255, 255, 255), 1)
            cv2.line(frame, (int(cx), int(cy) - 15), (int(cx), int(cy) + 15), (255, 255, 255), 1)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            status = f"FPS {fps:4.1f}  pan {pantilt.pan:5.1f}  tilt {pantilt.tilt:5.1f}"
            cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            streamer.update(frame)

            if args.display:
                cv2.imshow("Face Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        pantilt.center()
        time.sleep(0.3)
        pantilt.close()
        camera.close()
        streamer.close()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
