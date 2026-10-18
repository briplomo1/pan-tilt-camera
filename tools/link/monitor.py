"""Serial monitor that understands the link: prints device logs, optionally telemetry.

    python tools/link/monitor.py               # logs only
    python tools/link/monitor.py --telemetry   # plus one line per tracker frame

Use this instead of `pio device monitor` once the firmware's USB link is running.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.link import protocol as p  # noqa: E402
from tools.link.device import DeviceLink  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="serial port (default: from platformio.ini)")
    parser.add_argument("--telemetry", action="store_true", help="print tracker telemetry")
    args = parser.parse_args()

    with DeviceLink(args.port) as device:
        print(f"[monitor] {device.port} - Ctrl+C to quit")
        try:
            hello = device.hello()
            print(f"[monitor] firmware {hello.firmware}, protocol v{hello.protocol}, mode {hello.mode.name}, "
                  f"model {hello.active_model}, free internal {hello.free_internal // 1024} KB, "
                  f"psram {hello.free_psram // 1024} KB")
        except TimeoutError:
            print("[monitor] no reply to Hello yet (still booting, or old firmware) - showing raw output")

        if args.telemetry:
            def show(packet: p.Packet) -> None:
                t = p.Telemetry.parse(packet.payload)
                target = f"face {t.target.box} {t.target.score:.2f}" if t.target else "no face"
                print(f"[telemetry] #{t.frame_id} {p.TRACKER_STATES.get(t.state, t.state)} pan {t.pan:6.1f} "
                      f"tilt {t.tilt:6.1f} -> ({t.pan_target:6.1f}, {t.tilt_target:6.1f}) {target} "
                      f"detect {t.timing.total_us / 1000:.1f} ms")
            device.on(p.Msg.TELEMETRY, show)
            device.subscribe(True)
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            if args.telemetry:
                try:
                    device.subscribe(False)
                except (TimeoutError, OSError):
                    pass


if __name__ == "__main__":
    main()
