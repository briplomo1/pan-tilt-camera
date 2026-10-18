#!/usr/bin/env python3

import argparse
import json
import math
import os
import platform
import random
import socket
import statistics
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

FRAME_W, FRAME_H = 640, 480
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

class Camera:
    def __init__(self, source: str, flip: bool):
        self.flip = flip
        self.picam = None
        self.cap = None
        self.is_file = False

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
            try:
                index = int(source)
            except ValueError:
                index = None

            if index is None:
                self.is_file = True
                self.cap = cv2.VideoCapture(source)
            else:
                self.cap = cv2.VideoCapture(index)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            if not self.cap.isOpened():
                sys.exit(f"Could not open video source {source!r}. Try --camera 1, a video file path, or check `ls /dev/video*`.")

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
                return detector, path
    sys.exit("Face model not found. Run: sudo apt install -y opencv-data")


def detect_faces(detector, frame, scale=0.5, min_neighbors=5):
    """All candidate faces (x, y, w, h, score) in full-res coords, sorted by score desc,
    plus a preprocess/inference/postprocess timing breakdown in microseconds."""
    t0 = time.perf_counter()
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    t1 = time.perf_counter()

    boxes, _, weights = detector.detectMultiScale3(
        gray, scaleFactor=1.2, minNeighbors=min_neighbors, minSize=(30, 30), outputRejectLevels=True
    )
    t2 = time.perf_counter()

    candidates = [
        (int(x / scale), int(y / scale), int(w / scale), int(h / scale), float(score))
        for (x, y, w, h), score in zip(boxes, weights)
    ]
    candidates.sort(key=lambda c: c[4], reverse=True)
    t3 = time.perf_counter()

    timing = {
        "preprocess_us": int((t1 - t0) * 1e6),
        "inference_us": int((t2 - t1) * 1e6),
        "postprocess_us": int((t3 - t2) * 1e6),
        "total_us": int((t3 - t0) * 1e6),
    }
    return candidates, timing

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
# Metrics (speed + memory, same categories/shapes as the ESP32 track branch's
# BenchService/MemoryProbe + tools/bench/run_bench.py JSONL/summary.json)
# --------------------------------------------------------------------------
def _peak_rss_kb():
    try:
        import resource  # POSIX only; the Pi target has it
    except ImportError:
        return 0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def percentile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q / 100
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return s[f] + (s[c] - s[f]) * (k - f)


def latency_stats(values_us):
    ms = [v / 1000 for v in values_us]
    if not ms:
        return {}
    return {
        "n": len(ms),
        "mean_ms": statistics.fmean(ms),
        "median_ms": statistics.median(ms),
        "p95_ms": percentile(ms, 95),
        "p99_ms": percentile(ms, 99),
        "min_ms": min(ms),
        "max_ms": max(ms),
    }


class Metrics:
    """Per-frame JSONL + summary.json for the live tracker, disabled unless --metrics-out is set."""

    def __init__(self, out_dir, tag, every_n, model_path, load_us):
        self.enabled = out_dir is not None
        self.every_n = max(1, every_n)
        self.frame_index = 0
        self.totals_us = []
        if not self.enabled:
            return

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run = f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}"
        self.rows_path = self.out_dir / f"{self.run}.jsonl"
        self.rows = self.rows_path.open("w")
        self.rss_start_kb = _peak_rss_kb()
        self.model_bytes = os.path.getsize(model_path) if model_path else 0
        self.identity = {
            "model": "haar-frontalface",
            "backend": "opencv-haar",
            "precision": "float32",
            "model_bytes": self.model_bytes,
            "load_us": load_us,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "opencv_version": cv2.__version__,
        }

    def record_frame(self, frame_id, capture_time_us, state, timing, candidates, target_index,
                      pan, tilt, pan_target, tilt_target, command_us):
        if not self.enabled:
            return
        self.frame_index += 1
        self.totals_us.append(timing["total_us"])
        if self.frame_index % self.every_n != 0:
            return
        row = {
            "frame_id": frame_id,
            "time_us": capture_time_us,
            "state": state,
            "timing_us": timing,
            "faces": [{"box": [c[0], c[1], c[2], c[3]], "score": round(c[4], 4)} for c in candidates],
            "target_index": target_index,
            "pan": pan,
            "tilt": tilt,
            "pan_target": pan_target,
            "tilt_target": tilt_target,
            "capture_to_command_us": command_us,
        }
        self.rows.write(json.dumps(row) + "\n")

    def finalize(self, frame_count, wall_time_s):
        if not self.enabled:
            return
        self.rows.close()
        rss_peak_kb = _peak_rss_kb()
        stats = latency_stats(self.totals_us)
        summary = {
            "run": self.run,
            **self.identity,
            "frame_count": frame_count,
            "wall_time_s": round(wall_time_s, 1),
            "loop_fps_mean": round(frame_count / wall_time_s, 2) if wall_time_s > 0 else 0.0,
            "detect_total": stats,
            "memory": {
                "model_bytes": self.model_bytes,
                "rss_start_kb": self.rss_start_kb,
                "rss_peak_kb": rss_peak_kb,
                "rss_delta_kb": rss_peak_kb - self.rss_start_kb,
            },
        }
        (self.out_dir / f"{self.run}.summary.json").write_text(json.dumps(summary, indent=2))

        if stats:
            print(f"\nDetect time over {stats['n']} frames: median {stats['median_ms']:.2f} ms, "
                  f"mean {stats['mean_ms']:.2f}, p95 {stats['p95_ms']:.2f}, p99 {stats['p99_ms']:.2f}")
        print(f"Loop FPS mean: {summary['loop_fps_mean']:.1f}  Peak RSS: {rss_peak_kb} KB "
              f"(+{summary['memory']['rss_delta_kb']} KB)")
        print(f"Wrote {self.rows_path} and {self.run}.summary.json")


# --------------------------------------------------------------------------
# Accuracy mode: score a labeled image folder (Pi analog of the ESP32's
# run_bench.py + its planned score.py; same --gt JSON schema)
# --------------------------------------------------------------------------
def discover_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def iou(box_a, box_b):
    """Both boxes as (x0, y0, x1, y1)."""
    ix0, iy0 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix1, iy1 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def compute_ap(entries, iou_thresh):
    """entries: [(dets, gt_boxes), ...] with dets = [(box_x0y0x1y1, score), ...]."""
    all_dets = []
    total_gt = sum(len(gt) for _, gt in entries)
    if total_gt == 0:
        return 0.0
    for idx, (dets, _) in enumerate(entries):
        for box, score in dets:
            all_dets.append((score, idx, box))
    all_dets.sort(key=lambda d: d[0], reverse=True)

    matched = [[False] * len(gt) for _, gt in entries]
    tp_cum = fp_cum = 0
    precisions, recalls = [], []
    for score, idx, box in all_dets:
        gt_boxes = entries[idx][1]
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gt_boxes):
            if matched[idx][j]:
                continue
            v = iou(box, gt_box)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thresh and best_j >= 0:
            matched[idx][best_j] = True
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / total_gt)

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap, prev_recall = 0.0, 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_recall)
        prev_recall = r
    return ap


def bootstrap_ap_ci(entries, iou_thresh, n_resamples=1000, seed=0):
    n = len(entries)
    if n == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    samples = [compute_ap([entries[rng.randrange(n)] for _ in range(n)], iou_thresh) for _ in range(n_resamples)]
    return percentile(samples, 2.5), percentile(samples, 97.5)


def evaluate_at_threshold(entries, score_thresh, iou_thresh):
    total_gt = matched_total = fp_count = neg_images = 0
    centre_errors_px = []
    for dets, gt_boxes in entries:
        total_gt += len(gt_boxes)
        kept = [(box, score) for box, score in dets if score >= score_thresh]
        matched = [False] * len(gt_boxes)
        image_fp = 0
        for box, _ in kept:
            best_iou, best_j = 0.0, -1
            for j, gt_box in enumerate(gt_boxes):
                if matched[j]:
                    continue
                v = iou(box, gt_box)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= iou_thresh and best_j >= 0:
                matched[best_j] = True
                gt_box = gt_boxes[best_j]
                pred_c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                gt_c = ((gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2)
                centre_errors_px.append(math.hypot(pred_c[0] - gt_c[0], pred_c[1] - gt_c[1]))
            else:
                image_fp += 1
        matched_total += sum(matched)
        if not gt_boxes:
            neg_images += 1
            fp_count += image_fp
    recall = matched_total / total_gt if total_gt else 0.0
    fp_per_frame = fp_count / neg_images if neg_images else 0.0
    return recall, fp_per_frame, centre_errors_px


def run_accuracy_mode(args):
    root = Path(args.images)
    paths = discover_images(root)
    if not paths:
        sys.exit(f"No images found under {root}")
    gt_raw = json.loads(Path(args.gt).read_text())

    detector, model_path = load_face_detector()

    out_dir = Path(args.metrics_out or "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    run = f"{time.strftime('%Y%m%d-%H%M%S')}_{args.metrics_tag}"
    rows_path = out_dir / f"{run}.jsonl"

    first_frame = cv2.imread(str(paths[0]))
    for _ in range(args.warmup):
        detect_faces(detector, first_frame)

    rss_start_kb = _peak_rss_kb()
    entries = []
    totals_us = []
    started = time.monotonic()

    with rows_path.open("w") as rows:
        for path in paths:
            rel = path.relative_to(root).as_posix()
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            candidates, timing = detect_faces(detector, frame)
            totals_us.append(timing["total_us"])

            gt_boxes = [tuple(b) for b in gt_raw.get(rel, [])]
            dets = [((x, y, x + w, y + h), score) for x, y, w, h, score in candidates]
            entries.append((dets, gt_boxes))

            rows.write(json.dumps({
                "image": rel,
                "timing_us": timing,
                "faces": [{"box": list(box), "score": round(score, 4)} for box, score in dets],
                "gt_boxes": [list(b) for b in gt_boxes],
            }) + "\n")

    ap50 = compute_ap(entries, args.iou_thresh)
    ap_lo, ap_hi = bootstrap_ap_ci(entries, args.iou_thresh)
    recall, fp_per_frame, centre_errors = evaluate_at_threshold(entries, args.score_thresh, args.iou_thresh)
    rss_peak_kb = _peak_rss_kb()
    stats = latency_stats(totals_us)

    summary = {
        "run": run,
        "model": "haar-frontalface", "backend": "opencv-haar", "precision": "float32",
        "model_bytes": os.path.getsize(model_path) if model_path else 0,
        "host": socket.gethostname(), "platform": platform.platform(), "opencv_version": cv2.__version__,
        "images": str(root), "n_images": len(entries), "n_gt_boxes": sum(len(g) for _, g in entries),
        "iou_thresh": args.iou_thresh, "score_thresh": args.score_thresh,
        "detect_total": stats,
        "memory": {
            "rss_start_kb": rss_start_kb,
            "rss_peak_kb": rss_peak_kb,
            "rss_delta_kb": rss_peak_kb - rss_start_kb,
        },
        "accuracy": {
            "ap50": round(ap50, 4),
            "ap50_ci_lo": round(ap_lo, 4),
            "ap50_ci_hi": round(ap_hi, 4),
            "recall_at_threshold": round(recall, 4),
            "fp_per_frame": round(fp_per_frame, 4),
            "centre_error_px": round(statistics.median(centre_errors), 1) if centre_errors else None,
            "centre_error_pct": round(100 * statistics.median(centre_errors) / FRAME_W, 2) if centre_errors else None,
        },
        "wall_time_s": round(time.monotonic() - started, 1),
    }
    (out_dir / f"{run}.summary.json").write_text(json.dumps(summary, indent=2))

    if stats:
        print(f"Detect time over {stats['n']} images: median {stats['median_ms']:.2f} ms, "
              f"mean {stats['mean_ms']:.2f}, p95 {stats['p95_ms']:.2f}, p99 {stats['p99_ms']:.2f}")
    acc = summary["accuracy"]
    print(f"AP50 {acc['ap50']:.4f} (95% CI {acc['ap50_ci_lo']:.4f}-{acc['ap50_ci_hi']:.4f})  "
          f"recall@{args.score_thresh} {recall:.3f}  FP/frame {fp_per_frame:.3f}")
    print(f"Peak RSS: {rss_peak_kb} KB (+{summary['memory']['rss_delta_kb']} KB)")
    print(f"Wrote {rows_path} and {run}.summary.json")


def parse_args():
    p = argparse.ArgumentParser(description="Pan-tilt face tracker")
    p.add_argument("--camera", default="picam", help="'picam' for the Pi camera, a USB index like 0, or a video file path")
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
    p.add_argument("--metrics-out", default=None, help="write per-frame JSONL + summary.json metrics to this directory")
    p.add_argument("--metrics-every", type=int, default=1, help="log every Nth frame to metrics (reduces overhead)")
    p.add_argument("--metrics-tag", default="haar-default", help="label folded into the metrics run id")
    p.add_argument("--images", default=None, help="accuracy mode: score a labeled folder of images instead of live tracking")
    p.add_argument("--gt", default=None, help='ground-truth JSON for --images: {"relative/path.jpg": [[x0,y0,x1,y1], ...]}')
    p.add_argument("--iou-thresh", type=float, default=0.5, help="IoU for a detection to count as a match (AP50 default: 0.5)")
    p.add_argument("--score-thresh", type=float, default=0.05, help="score threshold for the recall/FP-per-frame operating point")
    p.add_argument("--warmup", type=int, default=5, help="unrecorded detections on the first image before timing starts (--images mode)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.images:
        if not args.gt:
            sys.exit("--images requires --gt")
        run_accuracy_mode(args)
        return

    camera = Camera(args.camera, args.flip)
    t_load0 = time.perf_counter()
    detector, model_path = load_face_detector()
    load_us = int((time.perf_counter() - t_load0) * 1e6)
    pantilt = PanTilt(args.servos, args.pan_pin, args.tilt_pin, args.pan_ch, args.tilt_ch)
    streamer = Streamer(args.port)
    metrics = Metrics(args.metrics_out, args.metrics_tag, args.metrics_every, model_path, load_us)

    print(f"Tracking. View the stream at http://<your-pi-hostname>.local:{args.port}  (Ctrl+C to stop)")

    pan_sign = 1 if args.invert_pan else -1
    tilt_sign = -1 if args.invert_tilt else 1
    cx, cy = FRAME_W / 2, FRAME_H / 2
    fps, last = 0.0, time.time()
    frame_id = 0
    started = time.monotonic()

    try:
        while True:
            capture_t0 = time.perf_counter()
            frame = camera.read()
            if frame is None:
                if camera.is_file:
                    break
                continue

            candidates, timing = detect_faces(detector, frame)
            face = max(candidates, key=lambda c: c[2] * c[3]) if candidates else None
            target_index = candidates.index(face) if face is not None else -1
            pan_target, tilt_target = pantilt.pan, pantilt.tilt

            if face is not None:
                x, y, w, h, score = face
                fx, fy = x + w / 2, y + h / 2

                err_x = (fx - cx) / cx
                err_y = (fy - cy) / cy

                d_pan = pan_sign * args.gain * err_x if abs(err_x) > args.deadzone else 0.0
                d_tilt = tilt_sign * args.gain * err_y if abs(err_y) > args.deadzone else 0.0
                pantilt.nudge(d_pan, d_tilt)
                pan_target, tilt_target = pantilt.pan, pantilt.tilt

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (int(fx), int(fy)), 4, (0, 255, 0), -1)

            command_us = int((time.perf_counter() - capture_t0) * 1e6)

            cv2.line(frame, (int(cx) - 15, int(cy)), (int(cx) + 15, int(cy)), (255, 255, 255), 1)
            cv2.line(frame, (int(cx), int(cy) - 15), (int(cx), int(cy) + 15), (255, 255, 255), 1)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            status = f"FPS {fps:4.1f}  pan {pantilt.pan:5.1f}  tilt {pantilt.tilt:5.1f}"
            cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            state = "track" if face is not None else "scan"
            metrics.record_frame(frame_id, int(capture_t0 * 1e6), state, timing, candidates, target_index,
                                  pantilt.pan, pantilt.tilt, pan_target, tilt_target, command_us)
            frame_id += 1

            streamer.update(frame)

            if args.display:
                cv2.imshow("Face Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        metrics.finalize(frame_id, time.monotonic() - started)
        pantilt.center()
        time.sleep(0.3)
        pantilt.close()
        camera.close()
        streamer.close()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
