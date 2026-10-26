# Using the pan/tilt tracker (Phase 0)

How to build, flash and watch the baseline model, and how the host tools fit together.
The experiment design is in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).

## One-time setup

Firmware: PlatformIO (as before). Host tools: Python 3.10+ in a virtual env at `.venv`:

```
python -m venv .venv
.venv\Scripts\activate          # Git Bash: source .venv/Scripts/activate
pip install -r tools/requirements.txt
```

Activate it in each new terminal before running the tools (or call `.venv\Scripts\python` directly).
The training notebooks (Phase 3 onwards) get their own env, `.venv-train`, so their heavier
dependencies don't touch the live tools.

The serial port is read from `monitor_port` in `platformio.ini` (currently COM9). Override it
with `--port COMx` on any tool, or set `PAN_TILT_PORT`.

Run every command from the project root.

## Build and flash

```
pio run -t upload
```

Boot sequence (the first ~5 s print plain text):

1. 2 s pause so the USB port can reconnect.
2. **Calibration window (3 s):** hold CAL (D3) to calibrate the servos with UP/DOWN. Watch this
   with `pio device monitor` or `python tools/link/monitor.py`; both show plain text at this point.
3. Servos centre, camera starts, then **the USB link starts**. From here the port carries binary
   packets, so `pio device monitor` shows garbage. Use the tools below instead. Logs keep
   flowing as packets.
4. The baseline model `espdl-msrmnp-s8` loads and the gimbal starts scanning for faces. It works
   with no PC attached.

Only one program can hold the serial port at a time. Close the live view or monitor before
`pio run -t upload` or `flash_models.py`.

## Watch the baseline live

```
python tools/view/live_view.py
```

A window shows the camera with every detected face boxed and labelled with its score. The face
being tracked is green, with a line from the frame centre. The grey box is the tracker's deadband.
The text strip shows the model, mode/state, device loop FPS, view FPS, detect time, pan/tilt and
free internal RAM.

| Key | Action |
|---|---|
| `q` / Esc | quit (preview is switched off on the device) |
| space | freeze the display |
| `s` | snapshot: annotated PNG + JSON of the detections → `recordings/` |
| `r` | start/stop recording: MP4 + one JSON line per frame → `recordings/` |
| `m` | load the next model in the registry |
| `+` / `-` | score threshold ±0.05 |
| `p` | cycle preview format: JPEG → raw RGB565 → gray |
| `t` | pause/resume tracking (the feed stops while paused) |
| `f` | Kalman filter on/off; the magenta cross is the filter's face estimate |

Options: `--format rgb565 --scale 1` sends exact pixels (lower FPS). `--every 2` sends every
second frame. `--zoom 3` makes the window bigger.

Streaming the preview costs the device CPU and USB time. Measured runs keep it off.

## Other tools

| Command | What it does |
|---|---|
| `python tools/link/monitor.py [--telemetry]` | device logs; with `--telemetry`, one line per tracker frame |
| `python tools/link/ping.py` | Hello + echo of 64 B…153 KB payloads; link throughput and integrity |
| `python tools/data/capture.py --count 50 --interval 1` | save camera frames (PNG + pan/tilt metadata) to `data/captures/` |
| `python tools/bench/run_bench.py <image folder> [--repeat 5] [--show]` | run the model on every image over USB; per-image JSONL + latency/memory summary in `results/` |
| `python tools/models/pack_models.py <files…> -o build/models.bin` | build a model pack for the `model` partition |
| `python tools/models/flash_models.py build/models.bin` | write the pack (app and calibration untouched) |
| `python tools/models/store_check.py` | load every packed model into flash-XIP / PSRAM / internal RAM, check CRCs and leaks |
| `python -m unittest discover -s tools/tests -t .` | host tests: protocol, metrics (cross-checked with the Pi scorer), controller mirror, virtual gimbal |

Protocol v2 (2026-10-25): flash the current firmware before using these tools; v1 firmware is rejected.

## Comparing models

| Command | What it does |
|---|---|
| `python tools/bench/run_bench.py <folder or video.mp4> --model A --model B [--gt gt.json] [--show] [--record out.mp4]` | stream the same frames over USB to each model; one results JSONL + summary per model |
| `python tools/eval/score.py results/<run>.jsonl --gt gt.json [--cal-run <cal run>.jsonl]` | AP50 with 95 % CI, AP50:95, operating point (threshold from the calibration run), recall by face size, centre error |
| `python tools/eval/compare.py results/*.score.json --baseline espdl-msrmnp-s8 [--paired espdl-msrmnp-s8 --gt gt.json]` | one table (markdown + CSV) of accuracy, per-stage latency and memory, plus a Pareto plot |
| `python tools/rig/virtual_gimbal.py wide.mp4 --hfov 90 --gt wide.gt.json --model A [--filter off] [--show]` | each model steers a camera window through the same wide video with the firmware's controller; slow models skip frames |
| `python tools/eval/closed_loop.py <live-*.jsonl, clip-*.jsonl or vg-*.jsonl>` | tracking metrics: detected %, losses, centring, mount speed; with ground truth also RMS error, time on target, lock ratio, reacquire time |

Kalman filter A/B on the robot: in `live_view.py` press `f` to switch the filter, `r` to record each
condition, then compare the two `live-*.jsonl` files with `closed_loop.py`.

## Building datasets

| Command | What it does |
|---|---|
| `python tools/data/capture.py --session <name> --tag distance=2m --count 50` | stills from the robot's camera into `data/captures/<name>/` (`--video 20` for a clip) |
| `python tools/data/autolabel.py <session, folder or video>` | YuNet (PC) proposes boxes |
| `python tools/data/label.py <session, folder or video>` | review/correct boxes, writes `<source>.gt.json` (resumable) |
| `python tools/data/split.py data/captures --gt <session>.gt.json ... --out data/b` | B-cal / B-test, whole sessions on one side |
| `python tools/data/prepare_wider.py --wider <WIDER folder> --out data/a` | A-test (val), A-cal and A-train lists from WIDER FACE |
| `python tools/data/negatives.py <photos without people> --into data/a/test` | verified no-face frames for the false-positive rate |

## How the pieces fit

```
 PC tools ──USB (COBS packets, src/link/Protocol.h)── Link ── VisionLoop (core 1)
                                                         │        ├─ DetectorSlot ── IDetector ── EspDlDetector (B0)
                                                         │        │                          └─ (Phase 4: TfliteDetector)
                                                         │        ├─ Tracker ── PanTilt ── Servo ×2   (motion task, core 0)
                                                         │        └─ BenchService ── ModelStore ── `model` partition
                                                         └── PreviewStreamer (core 0, lowest priority)
```

* **Adding a model backend** (Phase 4): implement `IDetector`, then register a factory in
  `DetectorRegistry`. Built-in models use `add()`. Models flashed into the partition use
  `addStoreBackend("tflite", "tflite-micro", factory)`, which makes every `.tflite` in the pack
  appear as `tflite-micro/<name>` in `list_models`, the live view's `m` key and `run_bench.py --model`.
* **Deploying a model file:** `pack_models.py` → `flash_models.py`. No firmware rebuild.
* **Only one model is resident:** `DetectorSlot::load()` frees the old model before building the new
  one, and reports load time and heap use.
