"""Frames from an image folder, a video file or a device capture session, one interface for all.

    for frame in open_source("data/wider_test"):          # image folder
    for frame in open_source("clip.mp4", stride=2):       # video file
    for frame in open_source("data/captures/20261025-1"): # capture.py session (metadata.jsonl)

Every frame has a stable key that ground truth files use:
    image folder / capture session   relative path, e.g. "sub/img_001.jpg"
    video file                       "<video stem>/<frame index:05d>", e.g. "walk/00042"

Ground truth for every source is one JSON object {key: [[x0, y0, x1, y1], ...]} in the
original image's pixels; an empty list marks a frame known to have no face.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp"}
VIDEO_TYPES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


@dataclass
class SourceFrame:
    key: str
    rgb: np.ndarray                 # H x W x 3 uint8, original size
    time_s: float                   # position in the video, or capture time; 0 for plain images
    meta: dict = field(default_factory=dict)


def video_key(stem: str, index: int) -> str:
    return f"{stem}/{index:05d}"


class ImageFolder:
    def __init__(self, root: Path, limit: int | None = None):
        self.root = root
        self.paths = sorted(q for q in root.rglob("*") if q.suffix.lower() in IMAGE_TYPES)[:limit]
        # capture.py writes per-frame pan/tilt/time; keep it with the frame when present
        self.meta: dict[str, dict] = {}
        meta_path = root / "metadata.jsonl"
        if meta_path.exists():
            for line in meta_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.meta[row["file"]] = row

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[SourceFrame]:
        for path in self.paths:
            key = path.relative_to(self.root).as_posix()
            meta = self.meta.get(key, {})
            rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            yield SourceFrame(key, rgb, meta.get("time_us", 0) / 1e6, meta)


class VideoFile:
    """Decodes frames in order. `stride` keeps every Nth frame; `start`/`duration` in seconds."""

    def __init__(self, path: Path, stride: int = 1, start: float = 0.0, duration: float | None = None,
                 limit: int | None = None):
        import cv2

        self.path = path
        self.stride = max(1, stride)
        self.limit = limit
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise ValueError(f"Can't open video {path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.first = round(start * self.fps)
        self.last = self.frame_count if duration is None else min(self.frame_count, self.first + round(duration * self.fps))

    def __len__(self) -> int:
        n = max(0, (self.last - self.first + self.stride - 1) // self.stride)
        return min(n, self.limit) if self.limit is not None else n

    def __iter__(self) -> Iterator[SourceFrame]:
        import cv2

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.first)
        emitted = 0
        for index in range(self.first, self.last):
            ok, bgr = self._cap.read()
            if not ok:
                break
            if (index - self.first) % self.stride:
                continue
            yield SourceFrame(video_key(self.path.stem, index), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                              index / self.fps, {"frame_index": index})
            emitted += 1
            if self.limit is not None and emitted >= self.limit:
                break

    def frame_at(self, index: int) -> np.ndarray | None:
        """Random access (slower than iterating); RGB or None past the end."""
        import cv2

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = self._cap.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None

    def close(self) -> None:
        self._cap.release()


def open_source(path: str | Path, *, stride: int = 1, start: float = 0.0, duration: float | None = None,
                limit: int | None = None) -> ImageFolder | VideoFile:
    path = Path(path)
    if path.is_dir():
        return ImageFolder(path, limit)
    if path.suffix.lower() in VIDEO_TYPES:
        return VideoFile(path, stride, start, duration, limit)
    raise ValueError(f"{path} is neither a folder of images nor a video ({', '.join(sorted(VIDEO_TYPES))})")


def load_ground_truth(path: str | Path | None) -> dict[str, list[list[float]]]:
    return json.loads(Path(path).read_text()) if path else {}


def save_ground_truth(path: str | Path, gt: dict[str, list[list[float]]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rounded = {k: [[round(v, 1) for v in box] for box in boxes] for k, boxes in sorted(gt.items())}
    Path(path).write_text(json.dumps(rounded, indent=1))
