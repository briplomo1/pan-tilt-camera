"""Write a model pack into the `model` partition (the app and settings are left alone).

    python tools/models/flash_models.py build/models.bin
    python tools/models/flash_models.py --erase          # wipe the partition

Close the live view / monitor first: esptool needs the serial port.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.link.config import default_port, partition  # noqa: E402


def esptool_command() -> list[str]:
    """esptool from this Python if installed, otherwise the copy PlatformIO ships."""
    if importlib.util.find_spec("esptool"):
        return [sys.executable, "-m", "esptool"]
    pio = Path.home() / ".platformio"
    script = pio / "packages" / "tool-esptoolpy" / "esptool.py"
    python = pio / "penv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if script.exists() and python.exists():
        return [str(python), str(script)]
    raise SystemExit("esptool not found: pip install esptool, or build once with PlatformIO")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pack", nargs="?", help="file from pack_models.py")
    parser.add_argument("--port")
    parser.add_argument("--erase", action="store_true", help="erase the model partition instead")
    args = parser.parse_args()

    offset, size = partition("model")
    command = esptool_command() + ["--chip", "esp32s3", "--port", args.port or default_port(), "--baud", "921600"]
    if args.erase:
        command += ["erase_region", hex(offset), hex(size)]
    else:
        if not args.pack:
            parser.error("give a pack file or --erase")
        pack = Path(args.pack)
        if pack.stat().st_size > size:
            raise SystemExit(f"{pack} is {pack.stat().st_size} bytes; the partition holds {size}")
        command += ["write_flash", hex(offset), str(pack)]

    print(" ".join(command))
    sys.exit(subprocess.call(command))


if __name__ == "__main__":
    main()
