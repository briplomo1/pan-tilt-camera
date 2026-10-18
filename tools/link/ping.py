"""Check the link: Hello, then echo payloads of several sizes and measure throughput.

    python tools/link/ping.py
    python tools/link/ping.py --rounds 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.link.device import DeviceLink  # noqa: E402

SIZES = [64, 1024, 16 * 1024, 64 * 1024, 320 * 240 * 2 + 12]   # last: one QVGA Infer request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    with DeviceLink(args.port, on_log=None) as device:
        start = time.perf_counter()
        hello = device.hello()
        print(f"Hello: {(time.perf_counter() - start) * 1000:.1f} ms round trip")
        print(f"  firmware {hello.firmware}, protocol v{hello.protocol}, mode {hello.mode.name}")
        print(f"  model {hello.active_model or '(none)'}")
        print(f"  internal RAM {hello.free_internal // 1024} / {hello.total_internal // 1024} KB free, "
              f"PSRAM {hello.free_psram // 1024} / {hello.total_psram // 1024} KB free")

        print(f"\nEcho ({args.rounds} rounds each; bytes go both ways)")
        print(f"{'size':>10} {'ms/round':>10} {'KB/s each way':>15} {'result':>8}")
        failures = 0
        for size in SIZES:
            payload = os.urandom(size)
            start = time.perf_counter()
            ok = True
            for _ in range(args.rounds):
                ok &= device.echo(payload) == payload
            elapsed = time.perf_counter() - start
            failures += not ok
            per_round = elapsed / args.rounds
            print(f"{size:>10} {per_round * 1000:>10.1f} {size / per_round / 1024:>15.0f} {'ok' if ok else 'MISMATCH':>8}")

        print(f"\nCorrupt frames seen by the host: {device.corrupt_frames}")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
