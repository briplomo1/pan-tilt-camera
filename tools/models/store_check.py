"""Check the models flashed into the `model` partition from the device's side.

Lists the pack, then loads every file into every placement (flash-xip, psram, internal) N
times, checking the CRC each time and that free heap returns to where it started (no leaks).

    python tools/models/store_check.py
    python tools/models/store_check.py --cycles 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.link import protocol as p  # noqa: E402
from tools.link.device import DeviceLink  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port")
    parser.add_argument("--cycles", type=int, default=10)
    args = parser.parse_args()

    with DeviceLink(args.port, on_log=None) as device:
        device.set_mode(p.Mode.BENCH)   # pause tracking so heap use is steady
        try:
            files = device.list_store()
            if not files:
                print("No model pack flashed (see tools/models/pack_models.py + flash_models.py)")
                return
            for f in files:
                print(f"{f['name']:<32} {f['format']:<8} {f['size']:>9} bytes  crc {f['crc32']:08x}")

            before = device.hello()
            failures = 0
            print(f"\n{'file':<24} {'placement':<10} {'ok':>4} {'load ms':>9} {'internal B':>11} {'psram B':>9} {'address':>10}")
            for f in files:
                for placement in p.Placement:
                    infos = [device.load_blob(f["name"], placement) for _ in range(args.cycles)]
                    ok = sum(i.ok for i in infos)
                    failures += ok != len(infos)
                    last = infos[-1]
                    avg_ms = sum(i.load_us for i in infos) / len(infos) / 1000
                    print(f"{f['name']:<24} {placement.name.lower():<10} {ok:>4} {avg_ms:>9.2f} "
                          f"{last.internal_bytes:>11} {last.psram_bytes:>9} {last.address:>#10x}")
            after = device.hello()

            leak_internal = before.free_internal - after.free_internal
            leak_psram = before.free_psram - after.free_psram
            print(f"\nFree heap change over all cycles: internal {leak_internal:+} B, psram {leak_psram:+} B")
            print("PASS" if failures == 0 and leak_internal <= 0 and leak_psram <= 0 else "CHECK THE NUMBERS ABOVE")
        finally:
            device.set_mode(p.Mode.TRACK)


if __name__ == "__main__":
    main()
