"""Face detectors that run on the PC (or the Pi): YuNet for auto-labelling, Haar to match the Pi baseline.

    detector = YuNet(score_threshold=0.5)
    faces = detector.detect(rgb)          # [(x0, y0, x1, y1, score), ...] in the image's pixels

YuNet is OpenCV's FaceDetectorYN with the opencv_zoo model, downloaded once to build/models/.
Neither is a candidate model for the ESP32; they label data and serve as PC-side backends for
tools/rig/virtual_gimbal.py.
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_PATH = ROOT / "build" / "models" / "face_detection_yunet_2023mar.onnx"

Detection = tuple[float, float, float, float, float]


def yunet_model(path: Path = YUNET_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading YuNet model to {path}")
        urllib.request.urlretrieve(YUNET_URL, path)
    return path


class YuNet:
    name = "yunet-pc"

    def __init__(self, score_threshold: float = 0.5, nms_threshold: float = 0.3, model: Path | None = None):
        import cv2

        self.cv2 = cv2
        self._detector = cv2.FaceDetectorYN.create(str(model or yunet_model()), "", (320, 320),
                                                   score_threshold, nms_threshold, 5000)
        self._size = (0, 0)

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        h, w = rgb.shape[:2]
        if (w, h) != self._size:
            self._detector.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._detector.detect(self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR))
        if faces is None:
            return []
        out = []
        for f in faces:
            x, y, fw, fh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])
            out.append((max(0.0, x), max(0.0, y), min(float(w), x + fw), min(float(h), y + fh), score))
        return sorted(out, key=lambda d: d[4], reverse=True)


class Haar:
    """The Pi baseline's detector (rsbpi/face_tracker.py detect_faces), on an already-small frame.
    Needs an OpenCV build with CascadeClassifier (OpenCV 4.x; OpenCV 5 dropped it from the main module)."""
    name = "haar-frontalface"

    def __init__(self, min_neighbors: int = 5):
        import os

        import cv2

        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError("This OpenCV has no CascadeClassifier (OpenCV 5). Use OpenCV 4.x, e.g. on the Pi.")
        name = "haarcascade_frontalface_default.xml"
        candidates = [os.path.join(cv2.data.haarcascades, name)] if hasattr(cv2, "data") else []
        candidates += [f"/usr/share/opencv4/haarcascades/{name}", f"/usr/share/opencv/haarcascades/{name}"]
        path = next((c for c in candidates if os.path.exists(c)), None)
        if not path:
            raise RuntimeError("Haar cascade file not found (sudo apt install opencv-data)")
        self.cv2 = cv2
        self._detector = cv2.CascadeClassifier(path)
        self.min_neighbors = min_neighbors

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        cv2 = self.cv2
        gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        boxes, _, weights = self._detector.detectMultiScale3(gray, scaleFactor=1.2, minNeighbors=self.min_neighbors,
                                                             minSize=(30, 30), outputRejectLevels=True)
        out = [(float(x), float(y), float(x + w), float(y + h), float(s)) for (x, y, w, h), s in zip(boxes, weights)]
        return sorted(out, key=lambda d: d[4], reverse=True)


def timed(detector, rgb: np.ndarray) -> tuple[list[Detection], int]:
    """Detections and the wall time they took, in microseconds."""
    t0 = time.perf_counter()
    faces = detector.detect(rgb)
    return faces, int((time.perf_counter() - t0) * 1e6)
