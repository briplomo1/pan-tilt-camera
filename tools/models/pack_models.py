"""Pack model files into one image for the `model` flash partition.

    python tools/models/pack_models.py model_a.tflite yunet=out/yunet_qat.tflite -o build/models.bin
    python tools/models/pack_models.py --dummy 3 --dummy-size 200000 -o build/dummy.bin   # for testing

Each file is stored under its name (file stem, or NAME=PATH) with its format taken from the
extension. The firmware's ModelStore reads this layout (src/models/ModelPack.h):

    Header(16) | Entry(56) x count | payloads, each 16-byte aligned
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.link.config import partition  # noqa: E402

MAGIC = b"MPK1"
VERSION = 1
ALIGNMENT = 16
MAX_ENTRIES = 64
HEADER = struct.Struct("<4sHHII")       # magic, version, count, tableCrc, reserved
ENTRY = struct.Struct("<32s8sIIII")     # name, format, offset, size, crc32, reserved


@dataclass
class PackedFile:
    name: str
    format: str
    data: bytes


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _fixed(text: str, size: int, what: str) -> bytes:
    raw = text.encode("utf-8")
    if not raw or len(raw) >= size:
        raise SystemExit(f"{what} '{text}' must be 1..{size - 1} bytes")
    return raw


def build_pack(files: list[PackedFile]) -> bytes:
    if len(files) > MAX_ENTRIES:
        raise SystemExit(f"At most {MAX_ENTRIES} files per pack")
    if len({f.name for f in files}) != len(files):
        raise SystemExit("Model names must be unique")

    offset = _align(HEADER.size + ENTRY.size * len(files))
    table = bytearray()
    body = bytearray()
    for f in files:
        crc = zlib.crc32(f.data) & 0xFFFFFFFF
        table += ENTRY.pack(_fixed(f.name, 32, "Name"), _fixed(f.format, 8, "Format"), offset, len(f.data), crc, 0)
        padding = _align(len(f.data)) - len(f.data)
        body += f.data + b"\xff" * padding       # 0xFF = erased flash
        offset += len(f.data) + padding

    header = HEADER.pack(MAGIC, VERSION, len(files), zlib.crc32(table) & 0xFFFFFFFF, 0)
    head = header + table
    return head + b"\xff" * (_align(len(head)) - len(head)) + body


def read_pack(image: bytes) -> list[dict]:
    """Parse a pack the same way the firmware does (used by the tests)."""
    magic, version, count, table_crc, _ = HEADER.unpack_from(image)
    if magic != MAGIC or version != VERSION:
        raise ValueError("not a model pack")
    table = image[HEADER.size:HEADER.size + count * ENTRY.size]
    if zlib.crc32(table) & 0xFFFFFFFF != table_crc:
        raise ValueError("table CRC mismatch")
    entries = []
    for i in range(count):
        name, fmt, offset, size, crc, _ = ENTRY.unpack_from(table, i * ENTRY.size)
        data = image[offset:offset + size]
        entries.append({"name": name.rstrip(b"\x00").decode(), "format": fmt.rstrip(b"\x00").decode(),
                        "offset": offset, "size": size, "crc32": crc,
                        "crc_ok": zlib.crc32(data) & 0xFFFFFFFF == crc})
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="PATH or NAME=PATH")
    parser.add_argument("-o", "--out", default="build/models.bin")
    parser.add_argument("--dummy", type=int, default=0, help="add N random test files")
    parser.add_argument("--dummy-size", type=int, default=100_000)
    args = parser.parse_args()

    files: list[PackedFile] = []
    for spec in args.files:
        name, _, path = spec.rpartition("=")
        path = Path(path)
        files.append(PackedFile(name or path.stem, path.suffix.lstrip(".").lower() or "bin", path.read_bytes()))
    for i in range(args.dummy):
        files.append(PackedFile(f"dummy{i}", "bin", os.urandom(args.dummy_size)))
    if not files:
        parser.error("nothing to pack")

    image = build_pack(files)
    _offset, capacity = partition("model")
    if len(image) > capacity:
        raise SystemExit(f"Pack is {len(image)} bytes but the model partition holds {capacity}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(image)
    for entry in read_pack(image):
        print(f"  {entry['name']:<32} {entry['format']:<8} {entry['size']:>9} bytes  crc {entry['crc32']:08x}")
    print(f"Wrote {out} ({len(image)} of {capacity} bytes, {100 * len(image) / capacity:.1f}% of the partition)")


if __name__ == "__main__":
    main()
