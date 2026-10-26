"""Virtual gimbal: every model steers a camera window through the same wide video.

    python tools/rig/virtual_gimbal.py data/wide/walk.mp4 --hfov 90 --gt data/wide/walk.gt.json
    python tools/rig/virtual_gimbal.py data/wide/walk.mp4 --hfov 90 --gt data/wide/walk.gt.json \
        --model espdl-msrmnp-s8 --filter off --show --record results/vg_b0_nofilter.mp4
    python tools/rig/virtual_gimbal.py walk.mp4 --hfov 90 --backend haar      # on the Pi

How it works (docs/EXPERIMENT_PLAN.md §8.1):
  * The wide video is the world. A window with the OV2640's 54 x 41 deg view is cut from it where
    the virtual mount points, scaled to 320x240, and given to the model. The mount can turn as far
    as the window stays inside the video.
  * The controller is the firmware's own (tools/rig/controller.py mirrors src/Tracker.cpp,
    constants read from it); the virtual mount moves like src/PanTilt.cpp (rate-limited, 20 ms).
  * Model speed counts: the clock advances by each detection's real time (the device's own
    detect time over USB, or the measured time for a local backend) plus --overhead-ms for capture
    and the rest of the loop. The next frame is the video frame at that time, so slow models skip
    frames exactly as they would on the robot.

Backends: device (ESP32 over USB, any --model), yunet (PC, OpenCV), haar (Pi baseline; OpenCV 4).
Ground truth (optional): face boxes per video frame, keys "<video stem>/<frame:05d>"
(tools/data/autolabel.py + label.py make it). With it, RMS angular error, time on target, lock
ratio and reacquire time are measured. Output: results/vg-<time>_<model>.jsonl (+ .summary.json);
analyse with tools/eval/closed_loop.py.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.common.imaging import FRAME_HEIGHT, FRAME_WIDTH, rgb888_to_rgb565be  # noqa: E402
from tools.common.sources import load_ground_truth, video_key  # noqa: E402
from tools.rig import controller as ctl  # noqa: E402

PAN0 = 90.0                                # mount angles that look at the video centre
TILT0 = 90.0


class FrameProvider(Protocol):
    fps: float
    width: int
    height: int
    frame_count: int

    def frame(self, index: int) -> np.ndarray | None: ...
    def faces(self, index: int) -> list[list[float]] | None: ...


class VideoProvider:
    """Sequential access to a video: indices only ever increase, skipped frames are grabbed, not decoded."""

    def __init__(self, path: Path, gt: dict):
        import cv2

        self.cv2 = cv2
        self.stem = path.stem
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"Can't open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.gt = gt
        self.next_index = 0
        self.current: np.ndarray | None = None
        self.current_index = -1

    def frame(self, index: int) -> np.ndarray | None:
        if index == self.current_index:
            return self.current
        if index < self.next_index:
            raise ValueError("VideoProvider only moves forward")
        while self.next_index < index:
            if not self.cap.grab():
                return None
            self.next_index += 1
        ok, bgr = self.cap.read()
        if not ok:
            return None
        self.next_index += 1
        self.current = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        self.current_index = index
        return self.current

    def faces(self, index: int) -> list[list[float]] | None:
        return self.gt.get(video_key(self.stem, index))


@dataclass
class Geometry:
    """Pinhole model of the wide video and the camera window inside it."""
    width: int
    height: int
    hfov_deg: float

    def __post_init__(self):
        self.f = (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)
        self.vfov_deg = 2 * math.degrees(math.atan((self.height / 2) / self.f))
        self.half_w = self.f * math.tan(math.radians(ctl.C["kHFovDeg"]) / 2)
        self.half_h = self.f * math.tan(math.radians(ctl.C["kVFovDeg"]) / 2)
        if self.half_w > self.width / 2 or self.half_h > self.height / 2:
            raise SystemExit(f"The video ({self.hfov_deg:.0f} x {self.vfov_deg:.0f} deg) is narrower than the "
                             f"camera ({ctl.C['kHFovDeg']} x {ctl.C['kVFovDeg']} deg); record wider or check --hfov")
        self.pan_range = ctl.Range(PAN0 - (self.hfov_deg / 2 - ctl.C["kHFovDeg"] / 2),
                                   PAN0 + (self.hfov_deg / 2 - ctl.C["kHFovDeg"] / 2))
        self.tilt_range = ctl.Range(TILT0 - (self.vfov_deg / 2 - ctl.C["kVFovDeg"] / 2),
                                    TILT0 + (self.vfov_deg / 2 - ctl.C["kVFovDeg"] / 2))

    def centre_px(self, pan: float, tilt: float) -> tuple[float, float]:
        """Where a mount at (pan, tilt) points, in video pixels."""
        ax = math.radians(ctl.C["kPanDir"] * (pan - PAN0))
        ay = math.radians(ctl.C["kTiltDir"] * (tilt - TILT0))
        return self.width / 2 + self.f * math.tan(ax), self.height / 2 + self.f * math.tan(ay)

    def angles_of(self, x: float, y: float) -> tuple[float, float]:
        """Mount angles that would point at video pixel (x, y)."""
        ax = math.degrees(math.atan((x - self.width / 2) / self.f))
        ay = math.degrees(math.atan((y - self.height / 2) / self.f))
        return PAN0 + ctl.C["kPanDir"] * ax, TILT0 + ctl.C["kTiltDir"] * ay

    def window(self, pan: float, tilt: float) -> tuple[float, float, float, float]:
        cx, cy = self.centre_px(pan, tilt)
        return cx - self.half_w, cy - self.half_h, cx + self.half_w, cy + self.half_h

    def crop(self, rgb: np.ndarray, pan: float, tilt: float) -> np.ndarray:
        """The camera's 320x240 view; outside the video is black."""
        import cv2

        x0, y0, x1, y1 = self.window(pan, tilt)
        sx, sy = FRAME_WIDTH / (x1 - x0), FRAME_HEIGHT / (y1 - y0)
        m = np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]], dtype=np.float32)
        return cv2.warpAffine(rgb, m, (FRAME_WIDTH, FRAME_HEIGHT), flags=cv2.INTER_AREA,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def to_video(self, box: list[float], pan: float, tilt: float) -> list[float]:
        x0, y0, x1, y1 = self.window(pan, tilt)
        sx, sy = (x1 - x0) / FRAME_WIDTH, (y1 - y0) / FRAME_HEIGHT
        return [x0 + box[0] * sx, y0 + box[1] * sy, x0 + box[2] * sx, y0 + box[3] * sy]


# detector(rgb 320x240) -> ([(x0, y0, x1, y1, score)], detect time in us)
Detector = Callable[[np.ndarray], tuple[list[tuple[float, float, float, float, float]], int]]


@dataclass
class RunConfig:
    hfov: float
    overhead_us: int = 10_000
    options: ctl.Options | None = None
    start_s: float = 0.0
    duration_s: float | None = None


def gt_angles(geo: Geometry, faces: list[list[float]] | None) -> tuple[float | None, float | None]:
    """Direction of the (largest) labelled face; None when it's hidden or unlabelled."""
    if not faces:
        return None, None
    x0, y0, x1, y1 = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return geo.angles_of((x0 + x1) / 2, (y0 + y1) / 2)


def run(provider: FrameProvider, detect: Detector, cfg: RunConfig, emit: Callable[[dict], None],
        on_frame: Callable[[np.ndarray, dict, Geometry], bool] | None = None) -> None:
    geo = Geometry(provider.width, provider.height, cfg.hfov)
    mount = ctl.Mount(geo.pan_range, geo.tilt_range, TILT0, PAN0, TILT0)
    tracker = ctl.Tracker(mount, cfg.options)
    first = round(cfg.start_s * provider.fps)
    last = provider.frame_count if cfg.duration_s is None else min(provider.frame_count,
                                                                    first + round(cfg.duration_s * provider.fps))
    has_target = False

    def index_at(t_us: int) -> int:
        return first + int(t_us / 1e6 * provider.fps)

    def tick(t_us: int) -> None:
        index = index_at(t_us)
        if index >= last:
            return
        gp, gt_ = gt_angles(geo, provider.faces(index))
        emit({"kind": "tick", "t": round(t_us / 1e6, 4), "pan": round(mount.pan, 3), "tilt": round(mount.tilt, 3),
              "gt_pan": None if gp is None else round(gp, 3), "gt_tilt": None if gt_ is None else round(gt_, 3),
              "state": tracker.state, "locked": tracker.state == "TRACK" and has_target})

    t_us = 0
    while True:
        index = index_at(t_us)
        if index >= last:
            break
        rgb = provider.frame(index)
        if rgb is None:
            break
        pan_cap, tilt_cap = mount.pan, mount.tilt
        view = geo.crop(rgb, pan_cap, tilt_cap)
        dets, detect_us = detect(view)
        now = t_us + detect_us
        mount.advance(now, tick)
        faces = [ctl.Face(*d) for d in dets]
        result = tracker.step(faces, FRAME_WIDTH, FRAME_HEIGHT, t_us, pan_cap, tilt_cap, now)
        has_target = result.target_index >= 0
        gp, gtl = gt_angles(geo, provider.faces(index))
        row = {
            "kind": "frame", "t": round(t_us / 1e6, 4), "frame_index": index, "state": result.state,
            "pan": round(pan_cap, 3), "tilt": round(tilt_cap, 3),
            "pan_target": round(tracker.pan_target, 3), "tilt_target": round(tracker.tilt_target, 3),
            "target": [round(v, 1) for v in dets[result.target_index][:4]] if has_target else None,
            "faces": [{"box": [round(v, 1) for v in d[:4]], "score": round(d[4], 4)} for d in dets],
            "detect_ms": detect_us / 1000, "source": [FRAME_WIDTH, FRAME_HEIGHT], "coasting": result.coasting,
            "face_pan": result.face_pan, "face_tilt": result.face_tilt, "gt_pan": gp, "gt_tilt": gtl,
        }
        emit(row)
        if on_frame and not on_frame(rgb, row, geo):
            break
        t_us = now + cfg.overhead_us
        mount.advance(t_us, tick)


class Viewer:
    """Wide frame with the camera window, ground truth (green) and detections (red)."""

    def __init__(self, show: bool, record: Path | None, fps: float, gt_of: Callable[[int], list | None],
                 width: int = 960):
        import cv2

        self.cv2, self.show, self.fps, self.gt_of, self.width = cv2, show, fps, gt_of, width
        self.record = record
        self.writer = None
        self.frames_written = 0

    def __call__(self, rgb: np.ndarray, row: dict, geo: Geometry) -> bool:
        cv2 = self.cv2
        s = self.width / rgb.shape[1]
        canvas = cv2.cvtColor(cv2.resize(rgb, (self.width, round(rgb.shape[0] * s))), cv2.COLOR_RGB2BGR)

        def rect(box, colour, thick):
            cv2.rectangle(canvas, (round(box[0] * s), round(box[1] * s)), (round(box[2] * s), round(box[3] * s)),
                          colour, thick)

        for box in self.gt_of(row["frame_index"]) or []:
            rect(box, (80, 220, 80), 2)
        rect(geo.window(row["pan"], row["tilt"]), (255, 255, 255), 2)
        for face in row["faces"]:
            rect(geo.to_video(face["box"], row["pan"], row["tilt"]), (60, 60, 230), 2)
        label = (f"{row['state']}{' coast' if row['coasting'] else ''}  t {row['t']:.2f}s  frame {row['frame_index']}"
                 f"  detect {row['detect_ms']:.0f} ms  pan {row['pan']:.1f} tilt {row['tilt']:.1f}")
        cv2.putText(canvas, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if self.record:
            if self.writer is None:
                self.record.parent.mkdir(parents=True, exist_ok=True)
                h, w = canvas.shape[:2]
                self.writer = cv2.VideoWriter(str(self.record), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            # Repeat frames so the recording plays in real time, however slow the model is
            while self.frames_written <= row["t"] * self.fps:
                self.writer.write(canvas)
                self.frames_written += 1
        if self.show:
            cv2.imshow("virtual gimbal (white = camera, green = truth, red = model)", canvas)
            return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)
        return True

    def close(self) -> None:
        if self.writer:
            self.writer.release()
        if self.show:
            self.cv2.destroyAllWindows()


def device_detector(device, threshold: float | None) -> Detector:
    from tools.link import protocol as p

    frame_id = [0]

    def detect(view: np.ndarray):
        result = device.infer(rgb888_to_rgb565be(view), FRAME_WIDTH, FRAME_HEIGHT, frame_id[0])
        frame_id[0] += 1
        faces = [(f.x0, f.y0, f.x1, f.y1, f.score) for f in result.faces]
        return faces, result.timing.total_us

    return detect


def local_detector(backend, fixed_us: int | None) -> Detector:
    from tools.common.pc_detectors import timed

    def detect(view: np.ndarray):
        faces, us = timed(backend, view)
        return faces, fixed_us if fixed_us is not None else us

    return detect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--hfov", type=float, required=True, help="horizontal field of view of the video, degrees")
    parser.add_argument("--gt", help="face boxes per video frame (keys '<stem>/<frame:05d>')")
    parser.add_argument("--backend", choices=["device", "yunet", "haar"], default="device")
    parser.add_argument("--model", default="espdl-msrmnp-s8", help="device backend: model id")
    parser.add_argument("--port")
    parser.add_argument("--threshold", type=float, help="score threshold (device: the model's own by default)")
    parser.add_argument("--detect-ms", type=float, help="local backends: pretend each detection takes this long")
    parser.add_argument("--overhead-ms", type=float, default=10.0,
                        help="capture and loop time on top of detection (ESP32 B0: ~10 ms)")
    parser.add_argument("--filter", choices=["on", "off"], help="Kalman filter (default: Tracker.cpp's)")
    parser.add_argument("--lead-ms", type=float, help="filter lead (default: Tracker.cpp's)")
    parser.add_argument("--coast-ms", type=float, help="coast after a miss (default: Tracker.cpp's)")
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the video")
    parser.add_argument("--duration", type=float, help="seconds of video")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--record", help="MP4 of the wide view with window and boxes")
    parser.add_argument("--tag", default="", help="label folded into the run name")
    parser.add_argument("--out-dir", default=str(ctl.ROOT / "results"))
    args = parser.parse_args()

    options = ctl.Options()
    if args.filter:
        options.filter = args.filter == "on"
    if args.lead_ms is not None:
        options.lead_s = args.lead_ms / 1000
    if args.coast_ms is not None:
        options.coast_s = args.coast_ms / 1000
    cfg = RunConfig(args.hfov, round(args.overhead_ms * 1000), options, args.start, args.duration)
    provider = VideoProvider(Path(args.video), load_ground_truth(args.gt))

    model_name = args.model if args.backend == "device" else args.backend
    run_name = (f"vg-{time.strftime('%Y%m%d-%H%M%S')}_{args.tag + '_' if args.tag else ''}"
                f"{model_name.replace('/', '_')}_filter-{'on' if options.filter else 'off'}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{run_name}.jsonl"
    viewer = (Viewer(args.show, Path(args.record) if args.record else None, provider.fps, provider.faces)
              if args.show or args.record else None)

    frames = 0
    detect_ms: list[float] = []
    started = time.monotonic()
    with open(rows_path, "w") as out:
        def emit(row: dict) -> None:
            nonlocal frames
            if row["kind"] == "frame":
                frames += 1
                detect_ms.append(row["detect_ms"])
            out.write(json.dumps(row) + "\n")

        if args.backend == "device":
            from tools.link import protocol as p
            from tools.link.device import DeviceLink

            with DeviceLink(args.port, on_log=None) as device:
                device.hello()
                device.set_mode(p.Mode.BENCH)
                try:
                    info = device.load_model(args.model)
                    if not info.ok:
                        raise SystemExit(f"Model {args.model} failed to load")
                    if args.threshold is not None:
                        device.set_threshold(args.threshold)
                    run(provider, device_detector(device, args.threshold), cfg, emit, viewer)
                finally:
                    device.set_mode(p.Mode.TRACK)
        else:
            from tools.common import pc_detectors

            backend = (pc_detectors.YuNet(args.threshold if args.threshold is not None else 0.6)
                       if args.backend == "yunet" else pc_detectors.Haar())
            fixed = round(args.detect_ms * 1000) if args.detect_ms is not None else None
            run(provider, local_detector(backend, fixed), cfg, emit, viewer)
    if viewer:
        viewer.close()

    geo = Geometry(provider.width, provider.height, args.hfov)
    summary = {
        "run": run_name, "video": args.video, "hfov": args.hfov, "vfov": round(geo.vfov_deg, 2), "gt": args.gt,
        "backend": args.backend, "model": model_name, "threshold": args.threshold,
        "options": {"filter": options.filter, "lead_s": options.lead_s, "coast_s": options.coast_s},
        "overhead_ms": args.overhead_ms, "frames": frames,
        "detect_median_ms": float(np.median(detect_ms)) if detect_ms else None,
        "pan_range": [geo.pan_range.min, geo.pan_range.max], "tilt_range": [geo.tilt_range.min, geo.tilt_range.max],
        "controller": ctl.describe(), "wall_time_s": round(time.monotonic() - started, 1),
    }
    (out_dir / f"{run_name}.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{frames} frames processed; wrote {rows_path}")
    print(f"Analyse with:  python tools/eval/closed_loop.py {rows_path}")


if __name__ == "__main__":
    main()
