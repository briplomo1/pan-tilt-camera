"""Live view: the camera feed with the detector's boxes drawn on it, while the gimbal tracks.

    python tools/view/live_view.py
    python tools/view/live_view.py --format rgb565 --scale 1      # exact pixels, lower fps
    python tools/view/live_view.py --model espdl-msrmnp-s8 --threshold 0.6

Keys
    q / Esc   quit                      space   freeze the display
    s         snapshot (PNG + JSON)     r       start/stop recording (see below)
    m         next model                + / -   score threshold up / down
    p         next preview format       t       pause/resume tracking (bench/track mode;
                                                the feed stops while paused)
    f         Kalman filter on/off (A/B runs without a reflash; the HUD shows which)

The magenta cross is where the Kalman filter puts the face (from telemetry), next to the
detected box; "coast" in the HUD means the tracker is following the prediction through a miss.

A recording writes three files to --out:
    live-*.mp4            the annotated view (only the frames the preview managed to send)
    live-*.jsonl          one line per tracker frame from telemetry: every frame, for metrics
    live-*.preview.jsonl  one line per video frame, with every detected face

Streaming costs the device some CPU and USB bandwidth. Turn it off for measured runs
(docs/EXPERIMENT_PLAN.md, "observer effect").
"""
from __future__ import annotations

import argparse
import collections
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.common.imaging import decode  # noqa: E402
from tools.link import protocol as p  # noqa: E402
from tools.link.config import PROJECT_ROOT  # noqa: E402
from tools.link.device import DeviceError, DeviceLink  # noqa: E402
from tools.rig.controller import C as TRACKER, angle_to_offset  # noqa: E402

DEADBAND = TRACKER["kDeadband"]   # read from src/Tracker.cpp
FORMATS = {"jpeg": p.ImageFormat.JPEG, "rgb565": p.ImageFormat.RGB565BE, "gray": p.ImageFormat.GRAY8}
SCALES = {1: 0, 2: 1, 4: 2}   # downscale factor -> shift

GREEN = (80, 220, 80)
YELLOW = (40, 200, 240)
RED = (60, 60, 230)
WHITE = (255, 255, 255)
GREY = (160, 160, 160)
MAGENTA = (230, 60, 230)


class Rate:
    """Events per second over a sliding window."""

    def __init__(self, window: int = 30):
        self._samples: collections.deque[tuple[float, float]] = collections.deque(maxlen=window)

    def add(self, t: float, count: float = 1.0) -> None:
        self._samples.append((t, count))

    def value(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        (t0, c0), (t1, c1) = self._samples[0], self._samples[-1]
        return (c1 - c0) / (t1 - t0) if t1 > t0 else 0.0


def score_colour(score: float) -> tuple[int, int, int]:
    return YELLOW if score >= 0.7 else (40, 140, 240)


def draw(preview: p.Preview, image: np.ndarray, zoom: float, hud: list[str],
         telemetry: p.Telemetry | None = None) -> np.ndarray:
    width = round(preview.source_width * zoom)
    height = round(preview.source_height * zoom)
    canvas = cv2.cvtColor(cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR), cv2.COLOR_RGB2BGR)

    # Frame centre and the tracker's deadband (errors inside it are ignored)
    cx, cy = width // 2, height // 2
    dx, dy = round(DEADBAND * width / 2), round(DEADBAND * height / 2)
    cv2.rectangle(canvas, (cx - dx, cy - dy), (cx + dx, cy + dy), GREY, 1)
    cv2.line(canvas, (cx - 12, cy), (cx + 12, cy), GREY, 1)
    cv2.line(canvas, (cx, cy - 12), (cx, cy + 12), GREY, 1)

    for i, face in enumerate(preview.faces):
        target = i == preview.target_index
        colour = GREEN if target else score_colour(face.score)
        x0, y0, x1, y1 = (round(v * zoom) for v in face.box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 3 if target else 2)
        label = f"{'target ' if target else ''}face {face.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y0 - th - 6, 0)
        cv2.rectangle(canvas, (x0, top), (x0 + tw + 6, top + th + 6), colour, -1)
        cv2.putText(canvas, label, (x0 + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if target:
            fx, fy = (x0 + x1) // 2, (y0 + y1) // 2
            cv2.line(canvas, (cx, cy), (fx, fy), GREEN, 1, cv2.LINE_AA)

    # Filtered face direction, seen from where the mount pointed for this frame
    if telemetry and telemetry.filter_on and telemetry.filter_ready:
        fx = cx + angle_to_offset(telemetry.face_pan, preview.pan, "pan") * width / 2
        fy = cy + angle_to_offset(telemetry.face_tilt, preview.tilt, "tilt") * height / 2
        if 0 <= fx < width and 0 <= fy < height:
            cv2.drawMarker(canvas, (round(fx), round(fy)), MAGENTA, cv2.MARKER_CROSS, 18, 2)

    # Heads-up display on a dark strip
    strip = 20 * len(hud) + 8
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, strip), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)
    for i, line in enumerate(hud):
        colour = RED if line.startswith("REC") else WHITE
        cv2.putText(canvas, line, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return canvas


def metadata(preview: p.Preview) -> dict:
    return {
        "frame_id": preview.frame_id, "time_us": preview.time_us, "model": preview.model_id,
        "mode": preview.mode.name, "state": p.TRACKER_STATES.get(preview.tracker_state, preview.tracker_state),
        "pan": round(preview.pan, 2), "tilt": round(preview.tilt, 2), "timing_us": preview.timing.as_dict(),
        "free_internal": preview.free_internal, "source": [preview.source_width, preview.source_height],
        "target_index": preview.target_index,
        "faces": [{"box": f.box, "score": round(f.score, 4)} for f in preview.faces],
    }


def telemetry_row(t: p.Telemetry, source: list[int] | None) -> dict:
    target = None
    if t.target:
        target = {"box": [t.target.x0, t.target.y0, t.target.x1, t.target.y1], "score": round(t.target.score, 4)}
    return {
        "frame_id": t.frame_id, "time_us": t.time_us, "state": p.TRACKER_STATES.get(t.state, t.state),
        "pan": round(t.pan, 2), "tilt": round(t.tilt, 2),
        "pan_target": round(t.pan_target, 2), "tilt_target": round(t.tilt_target, 2),
        "face_count": t.face_count, "target": target, "timing_us": t.timing.as_dict(), "source": source,
        "filter": t.filter_on, "coasting": t.coasting,
        "face_pan": round(t.face_pan, 2) if t.filter_ready else None,
        "face_tilt": round(t.face_tilt, 2) if t.filter_ready else None,
        "face_pan_velocity": round(t.face_pan_velocity, 2) if t.filter_ready else None,
    }


class Recorder:
    """MP4 of the annotated view, every tracker frame's telemetry, and the video frames' detections."""

    def __init__(self, folder: Path, size: tuple[int, int], fps: float):
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.video_path = folder / f"live-{stamp}.mp4"
        self.writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.preview_log = open(folder / f"live-{stamp}.preview.jsonl", "w")
        # Telemetry arrives on the link's reader thread
        self.lock = threading.Lock()
        self.telemetry_log = open(folder / f"live-{stamp}.jsonl", "w")
        self.first_id: int | None = None
        self.last_id: int | None = None
        self.rows = 0

    def write(self, frame: np.ndarray, meta: dict) -> None:
        self.writer.write(frame)
        self.preview_log.write(json.dumps(meta) + "\n")

    def write_telemetry(self, row: dict) -> None:
        with self.lock:
            if self.telemetry_log.closed:
                return
            self.telemetry_log.write(json.dumps(row) + "\n")
            if self.first_id is None:
                self.first_id = row["frame_id"]
            self.last_id = row["frame_id"]
            self.rows += 1

    def close(self) -> None:
        self.writer.release()
        self.preview_log.close()
        with self.lock:
            self.telemetry_log.close()
        if self.first_id is not None:
            expected = self.last_id - self.first_id + 1
            print(f"Telemetry: {self.rows} of {expected} tracker frames logged ({expected - self.rows} missing)")
        else:
            print("Telemetry: no frames logged")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port")
    parser.add_argument("--format", choices=FORMATS, default="jpeg")
    parser.add_argument("--scale", type=int, choices=SCALES, default=1, help="device-side downscale")
    parser.add_argument("--every", type=int, default=1, help="send every Nth frame")
    parser.add_argument("--quality", type=int, default=70, help="JPEG quality 1-100")
    parser.add_argument("--zoom", type=float, default=2.0, help="window size vs. the 320x240 frame")
    parser.add_argument("--model", help="model to load first (default: keep the current one)")
    parser.add_argument("--threshold", type=float, help="score threshold to set first")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "recordings"), help="snapshots and recordings")
    parser.add_argument("--record-fps", type=float, default=15.0)
    args = parser.parse_args()

    latest: queue.Queue[p.Packet] = queue.Queue(maxsize=1)

    def on_preview(packet: p.Packet) -> None:
        # Keep only the newest frame; the display never falls behind the device
        try:
            latest.get_nowait()
        except queue.Empty:
            pass
        latest.put_nowait(packet)

    recorder: Recorder | None = None
    source: list[int] | None = None
    recent: collections.OrderedDict[int, p.Telemetry] = collections.OrderedDict()   # by frame id

    def on_telemetry(packet: p.Packet) -> None:
        # Every tracker frame, unlike the preview, which skips frames while it's still sending
        t = p.Telemetry.parse(packet.payload)
        recent[t.frame_id] = t
        while len(recent) > 64:
            recent.popitem(last=False)
        active = recorder
        if active:
            active.write_telemetry(telemetry_row(t, source))

    out_dir = Path(args.out)
    with DeviceLink(args.port) as device:
        hello = device.hello()
        print(f"Connected to {device.port}: firmware {hello.firmware}, model {hello.active_model}")
        models = [m for m, _ in device.list_models()]
        if args.model:
            info = device.load_model(args.model)
            print(f"Loaded {info.id} in {info.load_us / 1000:.0f} ms")
        threshold = args.threshold if args.threshold is not None else 0.5
        if args.threshold is not None:
            device.set_threshold(threshold)

        filter_on = bool(TRACKER["kFilterFace"])
        device.set_tracker(filter_on, TRACKER["kLeadS"], TRACKER["kCoastS"])
        formats = list(FORMATS.values())
        fmt = FORMATS[args.format]
        mode = p.Mode.TRACK
        device.set_mode(mode)
        device.on(p.Msg.PREVIEW, on_preview)
        device.on(p.Msg.TELEMETRY, on_telemetry)
        device.set_preview(True, fmt, SCALES[args.scale], args.every, args.quality)
        device.subscribe(True)

        view_rate, device_rate = Rate(), Rate()
        shown = 0
        frozen = False
        last_canvas: np.ndarray | None = None
        last_meta: dict | None = None
        window = "pan-tilt live view"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                try:
                    preview = p.Preview.parse(latest.get(timeout=0.1).payload)
                except queue.Empty:
                    preview = None

                if preview and not frozen:
                    image = decode(preview.image, preview.format, preview.width, preview.height)
                    shown += 1
                    view_rate.add(time.monotonic(), shown)
                    device_rate.add(preview.time_us / 1e6, preview.frame_id)
                    t = preview.timing
                    tel = recent.get(preview.frame_id)
                    hud = [
                        f"{preview.model_id}   {preview.mode.name.lower()} / "
                        f"{p.TRACKER_STATES.get(preview.tracker_state, '?').lower()}"
                        f"{' coast' if tel and tel.coasting else ''}   threshold {threshold:.2f}   "
                        f"filter {'on' if filter_on else 'off'}",
                        f"device {device_rate.value():4.1f} fps   view {view_rate.value():4.1f} fps   "
                        f"detect {t.total_us / 1000:5.1f} ms",
                        f"pan {preview.pan:6.1f}  tilt {preview.tilt:6.1f}   faces {len(preview.faces)}   "
                        f"free internal {preview.free_internal // 1024} KB   {preview.format.name.lower()} "
                        f"{preview.width}x{preview.height}",
                    ]
                    if recorder:
                        hud.append(f"REC {recorder.video_path.name}")
                    last_canvas = draw(preview, image, args.zoom, hud, tel)
                    last_meta = metadata(preview)
                    source = last_meta["source"]
                    if recorder:
                        recorder.write(last_canvas, last_meta)
                    cv2.imshow(window, last_canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                try:
                    if key == ord(" "):
                        frozen = not frozen
                    elif key == ord("s") and last_canvas is not None:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        stem = out_dir / f"snap-{time.strftime('%Y%m%d-%H%M%S')}-{last_meta['frame_id']}"
                        cv2.imwrite(f"{stem}.png", last_canvas)
                        Path(f"{stem}.json").write_text(json.dumps(last_meta, indent=2))
                        print(f"Saved {stem}.png")
                    elif key == ord("r") and last_canvas is not None:
                        if recorder:
                            recorder.close()
                            print(f"Saved {recorder.video_path}")
                            recorder = None
                        else:
                            h, w = last_canvas.shape[:2]
                            recorder = Recorder(out_dir, (w, h), args.record_fps)
                            print(f"Recording to {recorder.video_path}")
                    elif key == ord("m") and models:
                        current = last_meta["model"] if last_meta else hello.active_model
                        nxt = models[(models.index(current) + 1) % len(models)] if current in models else models[0]
                        info = device.load_model(nxt)
                        print(f"Loaded {info.id}: {info.load_us / 1000:.0f} ms, internal {info.internal_bytes:+} B, "
                              f"psram {info.psram_bytes:+} B")
                    elif key in (ord("+"), ord("=")):
                        threshold = min(1.0, round(threshold + 0.05, 2))
                        device.set_threshold(threshold)
                    elif key in (ord("-"), ord("_")):
                        threshold = max(0.0, round(threshold - 0.05, 2))
                        device.set_threshold(threshold)
                    elif key == ord("p"):
                        fmt = formats[(formats.index(fmt) + 1) % len(formats)]
                        device.set_preview(True, fmt, SCALES[args.scale], args.every, args.quality)
                        print(f"Preview format: {fmt.name}")
                    elif key == ord("f"):
                        filter_on = not filter_on
                        device.set_tracker(filter_on, TRACKER["kLeadS"], TRACKER["kCoastS"])
                        print(f"Kalman filter {'on' if filter_on else 'off'}")
                    elif key == ord("t"):
                        mode = p.Mode.BENCH if mode == p.Mode.TRACK else p.Mode.TRACK
                        device.set_mode(mode)
                        print("Gimbal paused" if mode == p.Mode.BENCH else "Gimbal tracking")
                except (DeviceError, TimeoutError) as e:
                    print(f"Command failed: {e}")
        except KeyboardInterrupt:
            pass
        finally:
            if recorder:
                recorder.close()
                print(f"Saved {recorder.video_path}")
            cv2.destroyAllWindows()
            try:
                device.subscribe(False)
                device.set_preview(False)
                device.set_mode(p.Mode.TRACK)
            except (DeviceError, TimeoutError, OSError):
                pass


if __name__ == "__main__":
    main()
