"""Project settings the host tools share: serial port and partition layout."""
from __future__ import annotations

import configparser
import csv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_port() -> str:
    """Serial port from $PAN_TILT_PORT, else monitor_port/upload_port in platformio.ini."""
    if os.environ.get("PAN_TILT_PORT"):
        return os.environ["PAN_TILT_PORT"]
    ini = configparser.ConfigParser()
    ini.read(PROJECT_ROOT / "platformio.ini")
    for section in ini.sections():
        for key in ("monitor_port", "upload_port"):
            if ini.has_option(section, key):
                return ini.get(section, key).strip()
    raise SystemExit("No serial port: pass --port, set PAN_TILT_PORT or set monitor_port in platformio.ini")


def _size(text: str) -> int:
    text = text.strip()
    units = {"K": 1024, "M": 1024 * 1024}
    if text[-1:].upper() in units:
        return int(text[:-1], 0) * units[text[-1].upper()]
    return int(text, 0)


def partition(name: str) -> tuple[int, int]:
    """(offset, size) of a partition in partitions.csv."""
    with open(PROJECT_ROOT / "partitions.csv", newline="") as f:
        for row in csv.reader(line for line in f if line.strip() and not line.startswith("#")):
            cells = [c.strip() for c in row]
            if cells and cells[0] == name:
                return _size(cells[3]), _size(cells[4])
    raise SystemExit(f"No partition named '{name}' in partitions.csv")
