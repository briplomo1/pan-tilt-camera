"""Image conversions shared by the host tools.

Every dataset image is letterboxed to the camera's 320x240 and sent as big-endian RGB565,
exactly what the OV2640 delivers, so each model sees the same pixels it would see live.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

FRAME_WIDTH = 320
FRAME_HEIGHT = 240


@dataclass
class Letterbox:
    """How an image was fitted into the frame, to map boxes back to the original."""
    scale: float
    pad_x: int
    pad_y: int
    width: int      # original size
    height: int

    def to_frame(self, box: list[float]) -> list[float]:
        x0, y0, x1, y1 = box
        return [x0 * self.scale + self.pad_x, y0 * self.scale + self.pad_y,
                x1 * self.scale + self.pad_x, y1 * self.scale + self.pad_y]

    def to_original(self, box: list[float]) -> list[float]:
        x0, y0, x1, y1 = box
        return [(x0 - self.pad_x) / self.scale, (y0 - self.pad_y) / self.scale,
                (x1 - self.pad_x) / self.scale, (y1 - self.pad_y) / self.scale]

    def as_dict(self) -> dict:
        return {"scale": self.scale, "pad_x": self.pad_x, "pad_y": self.pad_y,
                "width": self.width, "height": self.height}


def letterbox(image: Image.Image, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> tuple[np.ndarray, Letterbox]:
    """Fit the image inside width x height keeping its aspect ratio, padding with black."""
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    new_w = max(1, round(image.width * scale))
    new_h = max(1, round(image.height * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (width, height))
    pad_x = (width - new_w) // 2
    pad_y = (height - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return np.asarray(canvas, dtype=np.uint8), Letterbox(scale, pad_x, pad_y, image.width, image.height)


def rgb888_to_rgb565be(rgb: np.ndarray) -> bytes:
    """H x W x 3 uint8 -> big-endian RGB565 bytes (the camera's byte order)."""
    r = rgb[..., 0].astype(np.uint16)
    g = rgb[..., 1].astype(np.uint16)
    b = rgb[..., 2].astype(np.uint16)
    value = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    return value.astype(">u2").tobytes()


def rgb565be_to_rgb888(data: bytes, width: int, height: int) -> np.ndarray:
    value = np.frombuffer(data, dtype=">u2", count=width * height).reshape(height, width).astype(np.uint16)
    r = ((value >> 11) & 0x1F) << 3
    g = ((value >> 5) & 0x3F) << 2
    b = (value & 0x1F) << 3
    # Repeat the top bits into the low bits so white stays 255
    rgb = np.stack([r | (r >> 5), g | (g >> 6), b | (b >> 5)], axis=-1)
    return rgb.astype(np.uint8)


def decode(data: bytes, fmt: int, width: int, height: int) -> np.ndarray:
    """Any device image format -> H x W x 3 RGB uint8."""
    from tools.link.protocol import ImageFormat  # local import keeps this module dependency-light

    fmt = ImageFormat(fmt)
    if fmt == ImageFormat.JPEG:
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
    if fmt == ImageFormat.RGB565BE:
        return rgb565be_to_rgb888(data, width, height)
    if fmt == ImageFormat.RGB888:
        return np.frombuffer(data, dtype=np.uint8, count=width * height * 3).reshape(height, width, 3).copy()
    if fmt == ImageFormat.GRAY8:
        gray = np.frombuffer(data, dtype=np.uint8, count=width * height).reshape(height, width)
        return np.repeat(gray[..., None], 3, axis=-1)
    raise ValueError(f"Unknown image format {fmt}")
