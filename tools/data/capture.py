"""Save frames from the device's own camera (dataset B in docs/EXPERIMENT_PLAN.md).

    python tools/data/capture.py --count 50 --interval 1.0 --session desk_1m --tag distance=1m --tag light=day
    python tools/data/capture.py --count 20 --format jpeg --keep-tracking
    python tools/data/capture.py --video 20 --session walk_backlit --tag light=backlit --show

Frames come through exactly the pipeline the models see (OV2640, 320x240). By default the gimbal
is paused while capturing so it holds still; --keep-tracking lets it follow you (pan/tilt are
logged per frame either way).

Stills (default): --count frames, --interval seconds apart. rgb565 is lossless and saved as PNG.
Video (--video SECONDS): frames as fast as the link allows (JPEG by default, ~10-15 fps), plus an
MP4 of them, for replay clips that run_bench.py can stream back to every model.

Each run writes one session folder, data/captures/<session or timestamp>/:
    frame_00000.png ...   the frames
    metadata.jsonl        per frame: file, device time, pan, tilt, size, tags
    session.json          tags and settings for the whole session
    <session>.mp4         (--video only)

Use one session per condition (distance, lighting, person...): tools/data/split.py keeps whole
sessions together, so near-identical frames never land in both calibration and test.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.common.imaging import decode  # noqa: E402
from tools.link import protocol as p  # noqa: E402
from tools.link.config import PROJECT_ROOT  # noqa: E402
from tools.link.device import DeviceLink  # noqa: E402

VIDEO_CHUNK = 10        # frames per Capture request in video mode


def parse_tags(items: list[str]) -> dict[str, str]:
    tags = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--tag expects key=value, got '{item}'")
        tags[key.strip()] = value.strip()
    return tags


class Saver:
    def __init__(self, out: Path, tags: dict, show: bool):
        self.out = out
        self.tags = tags
        self.show = show
        self.meta = open(out / "metadata.jsonl", "a")
        existing = sorted(out.glob("frame_*.*"))
        self.index = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0
        self.files: list[Path] = []

    def save(self, frame: p.CapturedFrame) -> Path:
        name = f"frame_{self.index:05d}.{'jpg' if frame.format == p.ImageFormat.JPEG else 'png'}"
        path = self.out / name
        if frame.format == p.ImageFormat.JPEG:
            path.write_bytes(frame.data)
        else:
            Image.fromarray(decode(frame.data, frame.format, frame.width, frame.height)).save(path)
        self.meta.write(json.dumps({"file": name, "time_us": frame.time_us, "pan": round(frame.pan, 2),
                                    "tilt": round(frame.tilt, 2), "width": frame.width, "height": frame.height,
                                    "tags": self.tags}) + "\n")
        self.meta.flush()
        self.index += 1
        self.files.append(path)
        if self.show:
            import cv2

            rgb = decode(frame.data, frame.format, frame.width, frame.height)
            cv2.imshow("capture", cv2.cvtColor(cv2.resize(rgb, (640, 480)), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        return path

    def close(self) -> None:
        self.meta.close()


def write_video(files: list[Path], path: Path, fps: float) -> None:
    import cv2

    first = cv2.imread(str(files[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in files:
        writer.write(cv2.imread(str(f)))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port")
    parser.add_argument("--count", type=int, default=10, help="stills to take")
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between stills")
    parser.add_argument("--video", type=float, metavar="SECONDS", help="capture continuously for this long")
    parser.add_argument("--format", choices=["rgb565", "jpeg"],
                        help="rgb565 is lossless (PNG); jpeg is smaller and faster (default: rgb565 stills, jpeg video)")
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--session", help="session name (default: timestamp); reusing a name appends to it")
    parser.add_argument("--tag", action="append", default=[], help="key=value describing the session (repeat)")
    parser.add_argument("--out", help="output folder (default: data/captures/<session>)")
    parser.add_argument("--keep-tracking", action="store_true", help="don't pause the gimbal")
    parser.add_argument("--show", action="store_true", help="display frames as they arrive")
    args = parser.parse_args()

    session = args.session or time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else PROJECT_ROOT / "data" / "captures" / session
    out.mkdir(parents=True, exist_ok=True)
    tags = parse_tags(args.tag)
    fmt_name = args.format or ("jpeg" if args.video else "rgb565")
    fmt = p.ImageFormat.JPEG if fmt_name == "jpeg" else p.ImageFormat.RGB565BE
    session_info = {"session": session, "tags": tags, "format": fmt_name, "quality": args.quality,
                    "keep_tracking": args.keep_tracking, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    saver = Saver(out, tags, args.show)
    with DeviceLink(args.port, on_log=None) as device:
        if not args.keep_tracking:
            device.set_mode(p.Mode.BENCH)
        try:
            if args.video:
                started = time.monotonic()
                first_us = last_us = None
                while time.monotonic() - started < args.video:
                    for frame in device.capture(VIDEO_CHUNK, fmt, args.quality):
                        saver.save(frame)
                        first_us = first_us if first_us is not None else frame.time_us
                        last_us = frame.time_us
                    print(f"\r{len(saver.files)} frames, {time.monotonic() - started:.1f} s", end="")
                print()
                fps = (len(saver.files) - 1) / ((last_us - first_us) / 1e6) if last_us and last_us > first_us else 10.0
                video_path = out / f"{session}.mp4"
                write_video(saver.files, video_path, fps)
                session_info.update(video=video_path.name, fps=round(fps, 2))
                print(f"Device delivered {fps:.1f} fps; wrote {video_path}")
            else:
                for i in range(args.count):
                    frame = device.capture(1, fmt, args.quality)[0]
                    path = saver.save(frame)
                    print(f"[{i + 1}/{args.count}] {path.name}  pan {frame.pan:.1f} tilt {frame.tilt:.1f}")
                    if i + 1 < args.count:
                        time.sleep(args.interval)
        finally:
            saver.close()
            if not args.keep_tracking:
                device.set_mode(p.Mode.TRACK)

    info_path = out / "session.json"
    history = json.loads(info_path.read_text()).get("runs", []) if info_path.exists() else []
    session_info["frames"] = len(saver.files)
    info_path.write_text(json.dumps({"session": session, "tags": tags, "runs": history + [session_info]}, indent=2))
    print(f"Saved {len(saver.files)} frames to {out}")
    print(f"Next: python tools/data/autolabel.py {out}  then  python tools/data/label.py {out}")


if __name__ == "__main__":
    main()
