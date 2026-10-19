# Quantized Face Detection on a Hardware-Constrained Edge Device
### Experiment plan and implementation plan — XIAO ESP32-S3 Sense pan/tilt face tracker

---

## 0. Status
| Phase | State |
|---|---|
| **Phase 0** — infrastructure + baseline | **Implemented**: baseline B0 behind `IDetector`, detector registry, `ModelStore` (model partition), memory probes, USB link (both directions), bench service, live view, host tools. Firmware builds. Waiting for on-device verification (§12) and first baseline numbers (§13). How to run it: `docs/USAGE.md`. |
| Phases 1–10 | Not started. Training work lives in `notebooks/` (placeholders, no code yet). |

**Notebooks** (`notebooks/`). Each starts with a header cell stating its purpose, the experiment IDs it covers, inputs/outputs, dependencies and the steps to be filled in later. There is no training code in them yet.

| Notebook | Covers |
|---|---|
| `00_toolchain_and_fidelity.ipynb` | Phase 3: PyTorch/PT2E → ai-edge-torch (fallback onnx2tf) → `.tflite`; fidelity test PyTorch vs TFLite-PC vs device |
| `01_datasets.ipynb` | Phase 1: WIDER subset (A-train/A-test), negatives, captured set B, letterbox → QVGA RGB565-BE, splits |
| `10_blazeface.ipynb` | P1 / Q1 |
| `11_ultralight_slim320.ipynb` | P2 / Q2 |
| `12_ultralight_rfb320.ipynb` | P3 / Q3 |
| `13_yunet.ipynb` | P4a/b / Q4 |
| `14_scrfd_500m.ipynb` | P5 / Q5 |
| `20_teacher_yolo11n.ipynb` | T0 (PC-only ceiling and distillation teacher) |
| `21_custom_detector.ipynb` | C1 (ReLU-only, QAT + KD) |
| `30_granularity_pot.ipynb` | G1–G4 (per-channel / per-tensor / PoT, PTQ vs QAT) |
| `31_precision.ipynb` | W1–W3 (16×8, W4A8, mixed) |
| `90_results_analysis.ipynb` | Report tables and figures from results JSONL |

---

## 1. Context

The project is a 2-axis servo gimbal with an OV2640 camera on a Seeed XIAO ESP32-S3 Sense that finds and follows a person's face in real time. It currently runs Espressif's stock ESP-DL face model (MSR+MNP, int8). The **main goal** is to study **quantization on edge hardware with hard resource limits**. We do that by building, deploying and measuring a family of face detectors that differ in architecture, input resolution, quantization method, numeric precision, runtime kernels and memory placement. Every result is compared against the stock model as the baseline.

The deliverable is a report of **accuracy vs speed vs memory**, plus **closed-loop tracking quality** (how well the gimbal keeps a moving face centred), for every combination. Each combination gets a stated hypothesis before it is run and a conclusion after.

**Implementation starts with Phase 0**: the baseline model plus all the base interfaces, meaning the pluggable detector API, model storage and deployment to flash, the USB streaming link and the benchmark service. After Phase 0, later phases only need to add models.

### Decisions already made in this session
| Decision | Choice | Reason |
|---|---|---|
| Training framework | PyTorch | User preference |
| Deployment stack for new models | **Stack B: TFLite Micro + ESP-NN** (int8) | Portability: `.tflite` int8 runs on most MCUs (Cortex-M/CMSIS-NN, STM32, Nordic, Coral). ESP-NN gives S3 vector kernels. |
| Baseline | ESP-DL MSR+MNP s8 (stock, rodata) | Vendor-optimised reference. Stays in its own ESP-DL backend. |
| PyTorch → TFLite | PT2E QAT/PTQ → `ai-edge-torch` (LiteRT); fallback ONNX → `onnx2tf` | Only path that carries PyTorch QAT into TFLite. Verified by a fidelity test (Phase 3). |
| Evaluation input | **USB streaming** of fixed image sets to the device | Same pixels for every model, real on-chip timing, no SD card |
| Persistence | `Storage` class for all settings and partitions (lazy NVS init) | Done this session |
| Servo range | Full 0–180° on both axes (calibrated pulse ends) | Done this session |
| Old `src/main.c` | Kept, commented out | Done this session |
| Live view | Device streams preview frames + detection metadata over the same USB link; the **host draws** boxes and labels in a window and can record MP4 | Keeps device cost to a frame copy (+ optional JPEG on core 0), keeps raw frames clean, reuses the link. WiFi/MJPEG was rejected: adds ~50–70 KB internal RAM and radio load to a memory-limited study. |

---

## 2. Background concepts (what is being explored and why)

**2.1 Quantization.** A real tensor is stored as integers plus a scale (and zero-point): `real ≈ scale × (q − zero_point)`. Int8 storage is 4× smaller than float32. More importantly here, the S3's SIMD unit executes int8 multiply-accumulates in parallel, while float math is scalar. Quantization therefore buys both **size** and **speed**, and costs **accuracy** through rounding and clipping error.

**2.2 PTQ vs QAT.**
- *Post-training quantization (PTQ)* takes a trained float model, runs a small calibration set to observe activation ranges, and picks scales. It's cheap, but layers with wide or outlier-heavy ranges (depthwise convs, the first and last layers) lose accuracy.
- *Quantization-aware training (QAT)* inserts "fake-quant" ops (quantize then dequantize, with a straight-through gradient) into training, so the weights learn to live on the integer grid. It costs extra training and usually recovers most of the PTQ loss.

**2.3 Granularity.** *Per-tensor* means one scale per tensor. *Per-channel* means one scale per output channel of a weight tensor. Per-channel fits channels with very different magnitudes far better. TFLite int8 uses per-channel weights and per-tensor asymmetric activations by default.

**2.4 Power-of-two (PoT) scales.** ESP-DL restricts scales to `2^e` (symmetric, per-tensor on the S3). Re-scaling between layers then becomes a bit shift instead of a fixed-point multiply. The cost is a coarse scale choice that can waste up to ~1 bit of range (e.g. max |w| = 0.51 forces scale 2⁻⁷, leaving half the int8 range unused). PoT per-tensor is a *subset* of what TFLite can express, so a PoT-constrained model can still be deployed in TFLite. That lets us measure the accuracy cost of the constraint in isolation (group G).

**2.5 Bit widths.** ESP-DL supports w8a8, w16a16 and w8a16. ESP-NN documents int8 (s8) kernels only. TFLite Micro supports a 16×8 mode (int16 activations, int8 weights), which we expect to fall back to slower reference kernels. **Int4 is supported by neither**, and the S3 has no 4-bit arithmetic. Int4 would be *weight-only* (W4A8): weights packed two per byte and unpacked to int8 inside a custom TFLite Micro kernel. It only helps speed if inference is memory-bandwidth-bound (decided by group M).

**2.6 Modern weight-rounding methods (GPTQ, AdaRound).** These choose how to round each weight using calibration data. GPTQ is not invalid on a constrained grid, but it was designed for large LLM matrices, quantizes weights only, and gains less on small CNNs with coarse grids. QAT is the primary method here. AdaRound-style rounding is an optional PTQ refinement.

**2.7 Runtimes and tools.**
- *ESP-PPQ* is Espressif's fork of OpenPPL's PPQ. It's a full PTQ pipeline (parse → assign rules per op → calibrate → simulate → export `.espdl`), not just a packager.
- *ESP-DL* is the runtime for `.espdl`.
- *ESP-NN* is a kernel library (S3 assembly) used underneath TFLite Micro.
- *ExecuTorch* is PyTorch's own edge runtime. No optimised S3 backend was found, so it's not used.

Stack A (ESP-PPQ/ESP-DL) is used only for the stock baseline and references.

**2.8 Where a model lives.** Flash is the chip's storage, divided into partitions.
- *Flash rodata* means the model is linked into the app image (the baseline today, 2.5 MB of the 3 MB app slot).
- *Model partition* means a separate 4 MB region at 0x310000 that is flashed independently. New models go there.

Both are read at runtime through the MMU/cache window (execute-in-place), with no copy. Alternatively, weights can be copied into PSRAM (8 MB octal) or internal SRAM (512 KB total, ~300 KB usable) to test whether memory bandwidth limits speed. NVS is a separate tiny key-value partition for settings (calibration), unrelated to models.

**2.9 Compute, not size, is the binding limit.** Espressif's published numbers put ESPDet-Pico-224 at 0.17 GFLOPs in 124 ms, about **1.4 GFLOP/s effective**. A 0.36 M-param model is ~0.4 MB at int8, far below the 4 MB partition. We expect models to hit the latency budget long before the size budget, so the study focuses on accuracy per millisecond.

---

## 3. Hardware budget (constraints every model must satisfy)

| Resource | Total | Already used / reserved | Available to a detector |
|---|---|---|---|
| CPU | 2 × Xtensa LX7 @ 240 MHz, PIE SIMD | Core 0: servo motion task (20 ms period), USB link RX | Core 1 (vision task) |
| Internal SRAM | 512 KB (~320 KB heap at boot) | ~35 KB static, drivers, task stacks, USB buffers | ≈ 150–200 KB (target ≤ 128 KB for arena + scratch) |
| PSRAM | 8 MB octal @ 80 MHz | 2 camera FBs (2 × 150 KB) + 2 link RX buffers (2 × 160 KB) + preview buffers when enabled (2 × 154 KB) ≈ 0.9 MB | ≈ 7 MB |
| Flash: app slot | 3 MB | 2.5 MB (incl. stock model) | 0.5 MB |
| Flash: `model` partition | 4 MB | — | 4 MB (≈ 4 M int8 params) |
| Latency budget | — | — | ≤ 66 ms for ≥ 15 FPS tracking loop (target); ≤ 40 ms matches baseline |

---

## 4. Research questions and global hypotheses

- **RQ1:** Can a small detector deployed in a portable int8 format (TFLite Micro + ESP-NN) match or beat the vendor-optimised baseline on accuracy *and* speed on the S3?
- **RQ2:** How much accuracy does each quantization choice cost on this hardware: PTQ vs QAT, per-tensor vs per-channel, PoT-constrained vs free scales, int8 vs int16 activations vs int4 weights?
- **RQ3:** Is S3 inference compute-bound or memory-bound, and does weight placement (flash-XIP / PSRAM / SRAM) change latency?
- **RQ4:** How does per-frame (open-loop) accuracy translate into closed-loop tracking quality, given the latency/FPS trade-off?

| ID | Global hypothesis | Rationale |
|---|---|---|
| GH1 | QAT does not change latency at all (same graph, same kernels), only accuracy | Quantization parameters are numbers in the same int8 ops |
| GH2 | PTQ → QAT gain is largest for the smallest, depthwise-heavy models | Depthwise layers have few weights per channel and wide per-channel ranges |
| GH3 | Per-channel > per-tensor > PoT-per-tensor in accuracy. PoT costs ≤ ~1 AP50 point after QAT. | PoT wastes ≤ 1 bit of range, and QAT compensates |
| GH4 | ESP-NN kernels are ≥ 3× faster than TFLM reference kernels. Float32 on-device is ≥ 5× slower than int8. | SIMD int8 MACs vs scalar |
| GH5 | Latency is mostly compute-bound. Moving weights from flash-XIP to SRAM changes latency < 15 %. | 64 KB data cache, small weights, conv reuse |
| GH6 | Closed-loop tracking is dominated by FPS/latency once recall is "good enough". A faster, slightly less accurate model tracks better than a slow accurate one. | Control loop with ~1–2 frame delay |
| GH7 | A Stack-B model at 320×240 beats the baseline on small/far faces (baseline runs MSR at 160×120) | Resolution sets the minimum detectable face size |

---

## 5. Experimental design

### 5.1 Variables
- **Independent:** model architecture, input resolution, quantization method (PTQ/QAT), granularity (per-tensor/per-channel/PoT), precision (int8, 16×8, W4A8, float32), runtime/kernels (ESP-DL, TFLM+ESP-NN, TFLM reference), weight placement (flash-XIP, PSRAM, SRAM).
- **Controlled:** CPU 240 MHz, same firmware build per group, same sdkconfig, same image sets and preprocessing (QVGA RGB565-BE letterboxed), detector score threshold 0.05 for curves, NMS IoU 0.5, camera settings locked for captured sets, servo speed limits, same tracker gains.
- **Dependent:** the metrics in §7.

### 5.2 Threats to validity (stated up front)
1. **Baseline confound:** B0 differs from Stack-B models in runtime, quantization scheme *and* architecture. Comparisons against B0 are system-level. **Single-variable conclusions come only from within-Stack-B comparisons** (same model, one variable changed).
2. **Vendor-set mismatch:** Espressif's published mAPs use their own validation set. We re-measure everything on our sets.
3. **Test-set leakage:** calibration (PTQ), threshold tuning and QAT training never use test images (§6 splits).
4. **Closed-loop ground truth** uses commanded servo angles, not measured shaft angles (servo lag is included in the error), and a 2-D face on a monitor. Live human trials are qualitative only.
5. **Timer resolution:** `esp_timer` µs. Each latency figure is from ≥ 500 frames after 20 warm-up frames.
6. **Observer effect of the live view:** streaming preview frames costs CPU (copy, optional JPEG) and USB bandwidth. **All measured runs use preview OFF** unless the table says otherwise. The preview overhead itself is measured once (loop FPS and detect p95 with preview off vs on).

### 5.3 Statistics
- Latency is reported as median, mean, p95 and p99 over ≥ 500 frames.
- AP is reported with a 95 % bootstrap CI (1000 resamples of images). A difference is "significant" only if the CIs do not overlap or the paired bootstrap p < 0.05.
- Closed-loop runs: 5 trials per (model, trajectory, speed). Report mean ± std.

---

## 6. Datasets

| Set | Source | Contents | Split use |
|---|---|---|---|
| **A-train** | WIDER FACE train | All images (for training / QAT) | Training, PTQ calibration (≤ 500 imgs) |
| **A-test** | WIDER FACE val, filtered to 1–3 faces, face height ≥ 12 px after letterbox to 320×240 (~400 imgs) + ~100 no-face images (rooms/outdoor/posters, from a no-person subset of a public set; licence checked) | Generic face detection | Test only |
| **B** | **Own captures** from the actual OV2640 through the device `Camera` pipeline (device CAPTURE mode), ~300 frames: distances 0.5–4 m, yaw/pitch, glasses, backlight, motion blur, partial occlusion, empty frames | Task-specific (this sensor, this room) | 30 % *B-cal* (calibration, threshold selection), 70 % *B-test* |
| **C** | Recorded tracking trajectories (closed-loop rig, §8) | Moving target | Closed-loop eval |

- **Preprocessing (all sets):** letterbox to 320×240 (keep aspect, pad with black), convert to RGB565 big-endian (what esp32-camera produces), and transform the GT boxes the same way. Each model then does its own resize from QVGA to its input size *on the device*, exactly as in live use.
- **Labelling set B:** a PC-side strong detector proposes boxes, then a human corrects them in CVAT or Label Studio.
- **Size buckets** (box height at 320×240): S < 24 px, M 24–64 px, L > 64 px.

---

## 7. Metrics — the final output table

Every experiment fills one row of this table. Columns are grouped by category.

| Category | Metric | Definition / how measured |
|---|---|---|
| **Identity** | Exp ID, model, input res, quant method, granularity, precision, runtime, placement | — |
| **Accuracy (open loop)** | AP50 (A-test), AP50 (B-test) | VOC-style AP at IoU ≥ 0.5, score thr 0.05, 95 % CI |
| | AP50:95 (B-test) | COCO-style, secondary |
| | Recall @ operating point | Threshold chosen on B-cal for FP/frame ≤ 0.05 |
| | FP per frame (negatives) | False detections per no-face frame at operating point |
| | Recall S / M / L | Recall per size bucket at operating point |
| | Centre error | Median ‖pred centre − GT centre‖, px and % of frame width (matched dets) |
| | Centre jitter | Std of centre on a static-face sequence (px) — becomes servo jitter |
| | Quantization drop ΔAP | AP50(int8 on-device) − AP50(float PyTorch) for the same model |
| | Fidelity | Mean abs output diff PyTorch-fake-quant vs TFLite-PC vs on-device |
| **Speed** | Pre / inference / post (ms) | Median, p95 per stage, on-device `esp_timer` |
| | Detect total (ms) | Median, p95, p99 |
| | Tracking loop FPS | Frames/s in TRACK mode, incl. capture |
| | Capture→command latency (ms) | Frame timestamp → `PanTilt::setTarget` |
| **Memory** | Model size (bytes) | File size in flash (`.tflite` / `.espdl`) |
| | Working memory internal / PSRAM (bytes) | Heap delta on load + peak during inference (heap monitor) |
| | Min free internal heap (bytes) | Low-water mark during inference — must stay > 32 KB |
| | App image delta (bytes) | Runtime code size added |
| **Closed loop** | RMS angular error (°) | Target angle vs commanded gimbal angle, per trajectory |
| | Time-on-target (%) | Fraction of time with angular error < 3° |
| | Lock ratio (%) | Fraction of time in TRACK state while target visible |
| | Max trackable speed (°/s) | Highest sine speed with lock ratio ≥ 90 % |
| | Reacquire time (s) | Target reappears after occlusion → TRACK |
| | Step settle time (s), overshoot (°) | 15° step target jump |
| **Derived** | AP50 per ms, Pareto-optimal (Y/N) | Pareto front on (AP50 B-test, detect p95, working memory) |
| | *(optional)* Energy per frame (mJ) | USB power meter, if available |
| **Instrumentation** | Preview overhead | Δ loop FPS and Δ detect p95, preview on vs off (measured once per backend) |

**Report figures:**
0. Short labelled videos (from the live view recorder) of each Pareto model tracking: qualitative evidence alongside the numbers.
1. Pareto plot: AP50 vs latency, bubble size = memory.
2. PR curves.
3. Recall by size bucket.
4. PTQ vs QAT vs granularity bars.
5. Latency by kernel/placement ablations.
6. Closed-loop error vs target speed.
7. Fidelity scatter.

---

## 8. Closed-loop tracking protocol (rig)

1. A monitor at a measured distance *d* displays a face image (from a fixed set of ~5 faces) moving along scripted trajectories:
   - horizontal sine at 10, 20, 40, 80, 160 °/s peak angular speed
   - vertical sine
   - 15° steps
   - random walk
   - 2 s occlusions (face hidden)
2. The target angle is `atan((x_px − x_centre)·pixel_pitch / d)`.
3. **Calibration step:** the gimbal is centred on the screen centre and the zero offset recorded.
4. **Sync:** the host sends a SYNC packet at trajectory start. The device timestamps it, and every TELEMETRY packet carries device time, so trajectories align.
5. The device streams TELEMETRY each frame: time, state, pan/tilt commanded, pan/tilt target, face box, score, detect µs.
6. The host computes the closed-loop metrics in §7.
7. Tracker gains (`kGain`, deadband, speeds) are held constant across models. A secondary run with gain re-tuned per model is reported separately, to show each model's best case.
   Fixed settings, set before the B0 baseline was recorded:

   | Setting | Value | Why |
   |---|---|---|
   | `kPanDir` / `kTiltDir` | −1 / +1 | Pan was reversed on this mount |
   | `kGain` | 0.6 | Fraction of the measured error corrected per frame |
   | `kDeadband` | 0.06 of half-frame | Stops jitter |
   | `kTrackSpeedDegS` | 90 °/s | 240 °/s blurred the face and caused misses |
   | Scan | slow sweep at 20 °/s with a 0.5 s pause every 20° of pan | A non-stop 30 °/s sweep blurred faces past detection; 90 °/s hops between pauses were jerky |
   | `kLostTimeoutUs` | 3 s | B0 misses runs of frames (up to ~1.3 s) with the face in view |
   | Scan tilt | level 150°, rows ±20° | Scan centred on the horizon, not the tilt-range middle |
   | Camera exposure | `ae_level` +1 | Faces against a bright window were too dark |

### 8.1 How tracking is compared across models: three tests

A replayed video can't move the camera, so tracking is measured in three steps, each holding a different thing constant:

| Test | Input | What moves | Measures | Removes |
|---|---|---|---|---|
| **Open loop** (§7 accuracy) | Labelled images and recorded videos, identical frames to every model (file on the Pi, USB on the ESP32) | Nothing; the face can be anywhere in frame, moving, small, turned or absent | AP50, recall, FP/frame, centre error, jitter, speed, memory | Camera, lighting, motors |
| **Virtual gimbal** (below) | One wide recorded video | A crop window, driven by the tracker | Closed-loop metrics with the model's real frame rate, but deterministic | Lighting, motors, run-to-run variation |
| **Physical rig** (this section) | Face on a monitor, scripted trajectories | The real gimbal | Closed-loop metrics including blur and servo response | Subject and trajectory variation |

**Virtual gimbal replay.**
1. Record a wide, high-resolution video (wider field of view than the OV2640) of a person moving, and label face positions in it (T0 auto-labels, then spot-check).
2. The model's camera is a window cropped from each video frame, sized to the OV2640's 54°×41° field of view and scaled to the model's input (320×240 on the ESP32).
3. The tracker controller (same gain, deadband and speed limits as §8 item 7) moves the window instead of the servos, at most `kTrackSpeedDegS`.
4. **Model speed counts.** A model only gets a new frame when it has finished the last one: a model whose detect time is 45 ms on a 30 fps video skips frames, a 15 ms model doesn't. On the ESP32 the crop is sent over USB and the detect time used is the device's own; the Pi runs the same loop locally.
5. The host computes the §7 closed-loop metrics against the labelled face positions: RMS error, time-on-target, lock ratio, reacquire time.

The same video is used for every model and both platforms, so differences come from detection accuracy and speed alone. It doesn't capture motion blur from a moving camera, servo dynamics or real lighting; the physical rig covers those.

---

## 9. Experiment matrix (combinations, hypotheses, results)

"Expected" values are hypotheses written *before* running. The **Result** and **Conclusion** columns are filled after each run, and each conclusion states whether the hypothesis held.

Notes:
- FLOP-based latency estimates assume ~1.4 GFLOP/s, measured for ESP-DL. TFLM+ESP-NN efficiency is unknown until K1.
- Candidate operator support and convertibility are unverified. Each must pass an export check first; a candidate that fails is recorded as "not deployable" with the reason, which is itself a result.

### Group B — Baseline and vendor references (ESP-DL, vendor PTQ, w8a8 PoT per-tensor, flash rodata)
| ID | Model | Input | Expected | Hypothesis / purpose | Result | Conclusion |
|---|---|---|---|---|---|---|
| **B0** | MSR+MNP s8 (stock) | 160×120 → 48×48 crops | ~36 ms model (vendor), ~40 ms total, ~20 FPS loop | **Baseline.** Weak on S-bucket faces (low res). Good closed loop due to speed. | TBD | TBD |
| R1 | ESPDet-Pico-224 face | 224×224 | ~131 ms, ~7 FPS, AP > B0 | Better per-frame accuracy but *worse* closed-loop at ≥ 40 °/s (tests GH6) | TBD | TBD |
| R2 | ESPDet-Pico-416 face | 416×416 | ~436 ms, ~2 FPS | Accuracy upper reference for on-device. Unusable for tracking. | TBD | TBD |

(R1/R2 need separate firmware builds, because flashing all three ESP-DL models would exceed the 3 MB app slot.)

### Group T — PC-only references (no device)
| ID | Model | Purpose | Expected |
|---|---|---|---|
| T0 | YOLO11n-face, float32, 640 | Accuracy ceiling. Distillation teacher. | AP50 far above all on-device models |
| T1 | Each Stack-B model in float32 (PyTorch) | Reference for ΔAP quantization drop | — |

### Group K — Runtime/kernel ablation (TFLM, first deployable small model, int8 PTQ per-channel)
| ID | Variant | Hypothesis | Result | Conclusion |
|---|---|---|---|---|
| K1 | TFLM + ESP-NN (S3 assembly) | Reference configuration | TBD | TBD |
| K2 | TFLM reference C kernels (ESP-NN off) | ≥ 3× slower than K1 (GH4) | TBD | TBD |
| K3 | TFLM float32 | ≥ 5× slower than K1, 4× model size, ΔAP ≈ 0 (GH4). Quantifies what int8 buys on this chip. | TBD | TBD |

### Group P — PTQ int8 per-channel, TFLM+ESP-NN, model partition (flash-XIP)
| ID | Model | Input | Approx. compute | Expected vs B0 | Hypothesis |
|---|---|---|---|---|---|
| P1 | BlazeFace (short-range) | 128×128 | very low | Faster than B0, lower AP on S faces | Fastest candidate. Small input limits range. |
| P2 | Ultra-Light-Fast slim-320 | 320×240 | low | ≈ B0 speed, better S recall | Plain conv+ReLU: smallest PTQ drop (GH2 control) |
| P3 | Ultra-Light-Fast RFB-320 | 320×240 | low–med | Slightly slower than P2, higher AP | Multi-branch RFB adds accuracy at modest cost |
| P4a/b | YuNet | 160×120 / 320×240 | very low | Best AP-per-ms candidate | Purpose-built tiny face detector. Depthwise-heavy, so a **large PTQ drop** (GH2). |
| P5 | SCRFD-0.5GF | 320×240 | ~0.1 GFLOPs | Highest AP of group, ~2× B0 latency | Strong small-face accuracy at a speed cost |

Result/Conclusion columns are the same as above for every row.

### Group Q — QAT (same models as P, int8 per-channel)
| ID | Model | Hypothesis |
|---|---|---|
| Q1–Q5 | QAT versions of P1–P5 | Latency identical to the P row (GH1). ΔAP recovered: P4/P1 gain most, P2 least (GH2). At least one Q model is Pareto-better than B0 (RQ1). |

### Group G — Granularity and scale constraints (best Q model; all QAT unless noted)
| ID | Scheme | Hypothesis |
|---|---|---|
| G1 | Per-channel weights (= the Q row) | Best accuracy |
| G2 | Per-tensor, free scales | −0.5 to −2 AP50 vs G1 |
| G3 | Per-tensor **PoT** (emulates ESP-DL constraint), deployed in TFLite | ≤ ~1 AP50 below G2 after QAT. Same latency (TFLM gains nothing from PoT), which isolates the accuracy cost of ESP-DL's constraint (GH3). |
| G4 | PoT per-tensor, **PTQ** | Largest drop of the group. Shows how much QAT matters under coarse grids. |

### Group R — Resolution sweep (best Q model)
| ID | Input | Hypothesis |
|---|---|---|
| R-160 | 160×120 | Latency ∝ pixels (~¼ of 320). S-bucket recall collapses. |
| R-224 | 224×168 | Knee of the accuracy/latency curve |
| R-320 | 320×240 | Best S recall. Max useful (camera native). |

### Group M — Memory placement (best Q model) — decides RQ3 and whether W2 is worth doing
| ID | Weights | Arena | Hypothesis |
|---|---|---|---|
| M1 | Flash-XIP (mmap) | PSRAM | Default |
| M2 | Copied to PSRAM | PSRAM | ≈ M1 |
| M3 | Copied to internal SRAM (if ≤ ~150 KB) | Internal | < 15 % faster than M1 if compute-bound (GH5). **If > 25 % faster: memory-bound, so pursue W2.** |
| M4 | Flash-XIP | Internal | Isolates activation-traffic cost |

### Group W — Precision
| ID | Precision | Hypothesis |
|---|---|---|
| W1 | 16×8 (int16 act, int8 weights) | +AP small, latency ≥ 2× (reference kernels). Not Pareto. |
| W2 | **W4A8** weight-only int4, custom TFLM op (only if M shows memory-bound) | Model size −40 to −50 %. Latency gain only if memory-bound. Needs QAT at 4-bit; 4-bit PTQ is expected to collapse. |
| W3 | Mixed: int8, sensitive layers 16×8 | Recovers most of W1's AP at a fraction of its latency cost |

### Group C — Custom model
| ID | Model | Hypothesis |
|---|---|---|
| C1 | Custom ReLU-only detector (MobileNet-style backbone, single-scale centre head, 320×240), QAT, KD from T0 | Narrow-task model with only ESP-NN-friendly ops matches P5 AP at ≤ B0 latency on B-test |

### Group L — Closed loop (Pareto set from all groups + B0, R1)
| ID | Hypothesis |
|---|---|
| L-* | Ranking by closed-loop RMS error follows FPS more than AP50 once recall ≥ ~0.9 (GH6). B0 outperforms R1 at ≥ 40 °/s. |

---

## 10. Phases and steps

### Phase 0 — Infrastructure + baseline (implement first, details in §11)
0.1 Copy this plan into the repo as `docs/EXPERIMENT_PLAN.md`. Results get appended there.
0.2 Vision interfaces: `ImageView`, `Face`, `IDetector`, `DetectTiming`, `ModelInfo`.
0.3 `EspDlDetector` backend (wraps `HumanFaceDetect`). Register B0 (R1/R2 behind build flags).
0.4 `DetectorRegistry` (named factories; one model resident at a time).
0.5 `MemoryProbe` (RAII heap measurement) and `HeapBuffer` (RAII `heap_caps` buffer).
0.6 Storage: RAII `MappedRegion` (fix the leaked mmap handle in `Partition::map`), ranged map.
0.7 `ModelStore`: model pack format in the `model` partition, `ModelBlob` with placement (XIP/PSRAM/internal), host `pack_models.py` and `flash_models.py`.
0.8 USB link: `ITransport`, `UsbSerialTransport`, COBS + CRC32 framing, `Protocol.h`, `Link` (RX task, packet pool, TX, log capture).
0.9 `VisionLoop` single owner task: Tracker refactor (`run()` → `step(IDetector&)`, pause/resume, telemetry observer). `BenchService` command handler.
0.10 Live view: device `PreviewStreamer` (§11.6) + host `view/live_view.py` (boxes, labels, HUD, recording).
0.11 Host tools: `link` (protocol, monitor, ping/echo), `bench/run_bench.py` (with `--show` to display GT vs predictions per image), `data/capture.py`.
0.12 Verification (§12) and first baseline numbers (B0 latency and memory on synthetic and captured frames).

### Phase 1 — Datasets
Prepare A-train/A-test (filter, letterbox, RGB565-BE, GT transform), capture and label B, split B-cal/B-test, write `score.py` (AP, PR, recall@op, size buckets, centre error, bootstrap CI). Unit-test the scorer on hand-made cases.

### Phase 2 — Baseline measurements
B0 on A-test and B-test (open loop). Build the closed-loop rig (`target_display.py`, telemetry recorder, metrics). Run L-B0. Optionally run R1 and R2.

### Phase 3 — PyTorch toolchain and fidelity
Python env (torch, ai-edge-torch/LiteRT, onnx2tf fallback). Tiny conv model → PT2E PTQ and QAT → `.tflite`. Compare PyTorch fake-quant vs TFLite-PC vs on-device outputs (needs Phase 4 backend). Gate: mean abs diff ≤ 1 quantization step.

### Phase 4 — TFLite Micro backend
Add `esp-tflite-micro` (+ ESP-NN) components. `TfliteDetector : IDetector` with model from `ModelStore`, tensor arena via `HeapBuffer`, op resolver limited to needed ops, per-model pre/post-processing strategy (anchor decode, NMS) behind a small `IDetectionHead` interface. Run K1–K3.

### Phase 5 — PTQ models (P1–P5)
Obtain or port each model to PyTorch, check export, calibrate on A-train + B-cal, deploy, measure.

### Phase 6 — QAT (Q1–Q5) and granularity (G1–G4)
Shared QAT recipe (fine-tune from float, fake-quant per §2, PoT option via a custom observer). Fixed epochs and seed per model.

### Phase 7 — Resolution (R-*) and memory placement (M1–M4)

### Phase 8 — Precision (W1, W3; W2 conditional on M)

### Phase 9 — Custom model (C1) with distillation from T0

### Phase 10 — Closed-loop on the Pareto set (L-*), final report
Fill all tables, plots and conclusions. Answer RQ1–RQ4.

---

## 11. Phase 0 implementation design

### 11.1 Code conventions (match the repo)
- Naming: `PascalCase` types, `camelCase` methods, members with trailing underscore, constants `kName` in an anonymous namespace, `constexpr const char *TAG` per file, `#pragma once`, `[[nodiscard]]` on queries, brief comments only where intent isn't obvious.
- **No exceptions and no RTTI** (sdkconfig has them off): no `throw` or `dynamic_cast`. Unrecoverable errors use `ESP_ERROR_CHECK`. Recoverable ones return `std::optional` or `bool`/`esp_err_t`.
- **Rule of five, explicit on every resource class:**
  - Owning hardware or OS handles that must stay unique (`UsbSerialTransport`, `Link`, `EspDlDetector`, `IDetector` base, `VisionLoop`): copy and move deleted.
  - Owning movable resources (`HeapBuffer`, `MappedRegion`, `ModelBlob`, `RxPacket`): move-only, copy deleted, move leaves the source empty (`std::exchange`), destructor releases.
  - Polymorphic bases: `virtual ~X() = default`, copy/move deleted to prevent slicing.
- **Memory discipline:**
  - No per-frame heap allocation in hot paths. Detectors return `std::span<const Face>` into a reserved internal vector, valid until the next `detect`.
  - Large buffers live in PSRAM via `heap_caps_aligned_alloc` wrapped in `HeapBuffer`.
  - Only one model is resident at a time (old one destroyed before the new one is created).
  - Wire structs are packed with `static_assert` sizes.
- C++20 (`std::span`, designated initialisers, `<chrono>` literals as in `Delay.h`).
- `src/CMakeLists.txt`: add `INCLUDE_DIRS "."` so the subfolders below can include `"vision/IDetector.h"`. The glob is already recursive.

### 11.2 File layout (as built)
```
src/
  main.cpp                   wiring: servos, calibration (Storage), mount, camera, Link, VisionLoop task
  Camera.h/.cpp              Frame::view() -> ImageView
  Tracker.h/.cpp             step(IDetector&), pause()/resume(), ITrackObserver (TrackSample per frame)
  Storage.h/.cpp             lazy NVS init, RAII MappedRegion, Partition::map(offset, size)
  servo_config.h             calibration struct + defaults (0-180 on both axes)
  vision/ImageView.h         PixelFormat, ImageView (non-owning view)
  vision/Face.h              Face struct
  vision/IDetector.h         IDetector, ModelInfo, DetectTiming
  vision/EspDlDetector.h/.cpp  ESP-DL backend: B0 baseline (+ Pico references behind CONFIG_FLASH_*)
  vision/DetectorRegistry.h/.cpp  named factories; addStoreBackend() for models in the partition
  vision/Jpeg.h/.cpp         allocation-free JPEG encode into a caller buffer
  memory/HeapBuffer.h        RAII heap_caps buffer (PSRAM / internal)
  memory/MemoryProbe.h/.cpp  RAII heap snapshot + local low-water-mark monitor
  models/ModelPack.h         pack layout shared with tools/models/pack_models.py
  models/ModelStore.h/.cpp   pack parser, ModelBlob (XIP / PSRAM / internal), CRC check
  link/ITransport.h          byte-pipe interface
  link/UsbSerialTransport.h/.cpp  USB-Serial-JTAG driver (RAII)
  link/Cobs.h/.cpp           streaming COBS encoder/decoder
  link/Protocol.h            message types, packed wire structs, kVersion
  link/Link.h/.cpp           RX task, 2-slot PSRAM pool, RxPacket (RAII), send(), log capture
  app/Wire.h                 Face/Timing/string <-> wire conversions
  app/DetectorSlot.h/.cpp    the one resident detector; load() measures time + memory
  app/VisionLoop.h/.cpp      vision task: commands, tracking, telemetry
  app/BenchService.h/.cpp    benchmark commands (models, infer, capture, store)
  app/PreviewStreamer.h/.cpp live view frames (non-blocking offer, core-0 JPEG + send)
tools/
  requirements.txt           pyserial, numpy, pillow, opencv-python, pytest
  link/protocol.py           protocol mirror (COBS, CRC, structs, parsers)
  link/device.py             DeviceLink: one class every tool uses
  link/config.py             port from platformio.ini, partitions.csv parsing
  link/monitor.py, ping.py   logs/telemetry monitor; Hello + echo throughput test
  common/imaging.py          letterbox to 320x240, RGB565-BE conversions, decoding
  view/live_view.py          live labelled video, HUD, snapshots, MP4 + JSONL recording
  bench/run_bench.py         stream an image folder -> results JSONL + summary
  data/capture.py            device camera frames -> PNG/JPEG + metadata (dataset B)
  models/pack_models.py      build a model pack; flash_models.py writes it; store_check.py tests it
  tests/test_protocol.py     COBS reference vectors, CRC vector, framing, struct sizes, pack format
notebooks/                   placeholder notebooks for Phases 1-10 (no code yet)
docs/EXPERIMENT_PLAN.md, docs/USAGE.md
```

### 11.3 Key interfaces (sketch)
```cpp
// vision/ImageView.h
enum class PixelFormat : uint8_t { Rgb565Be, Rgb888 };
struct ImageView { const uint8_t *data; uint16_t width; uint16_t height; PixelFormat format; };

// vision/IDetector.h
struct DetectTiming { uint32_t preprocessUs = 0, inferenceUs = 0, postprocessUs = 0, totalUs = 0; };
struct ModelInfo { std::string_view id, backend, quantization; uint16_t inputW, inputH; size_t modelBytes; };
class IDetector {
public:
    virtual ~IDetector() = default;
    IDetector(const IDetector &) = delete; IDetector &operator=(const IDetector &) = delete;
    IDetector(IDetector &&) = delete;      IDetector &operator=(IDetector &&) = delete;
    [[nodiscard]] virtual const ModelInfo &info() const = 0;
    virtual void setScoreThreshold(float thr) = 0;
    // Faces valid until the next detect() call; no allocation per frame
    [[nodiscard]] virtual std::span<const Face> detect(const ImageView &img, DetectTiming &timing) = 0;
protected:
    IDetector() = default;
};

// vision/DetectorRegistry.h
struct DetectorEntry { std::string_view id; std::unique_ptr<IDetector> (*create)(); };
[[nodiscard]] std::span<const DetectorEntry> detectors();
[[nodiscard]] const DetectorEntry *findDetector(std::string_view id);
```
- `EspDlDetector`: `HumanFaceDetect` with `lazy_load=false`. `setScoreThreshold` maps to `set_score_thr(thr, 0/1)`. ESP-DL gives one combined timing, so it fills `totalUs` and `inferenceUs`. `ModelInfo` reports internal/PSRAM usage from `dl::Model::get_memory_info()` via `get_raw_model(i)`.
- `MemoryProbe`: snapshot of `heap_caps_get_free_size(MALLOC_CAP_INTERNAL / MALLOC_CAP_SPIRAM)` + `heap_caps_monitor_local_minimum_free_size_start/stop` (available in IDF 6.1). Only one active at a time (asserted).
- `ModelStore`: pack = header {magic `"MPK1"`, version, count} + entries {name[32], format[8], offset, size, crc32}, payloads 16-byte aligned. `load(name, Placement) -> std::optional<ModelBlob>`. `ModelBlob` owns either a `MappedRegion` (XIP) or a `HeapBuffer` copy (PSRAM/internal) and exposes `std::span<const uint8_t>`. CRC is verified on open. Phase 0 tests it with dummy blobs (TFLite models arrive in Phase 4).

### 11.4 USB link protocol
- **Transport:** USB-Serial-JTAG (the existing console). The `UsbSerialTransport` constructor installs the driver (RX 8 KB, TX 4 KB ring buffers) and calls `usb_serial_jtag_vfs_use_driver()`. The destructor uninstalls.
- **Framing:** each packet = COBS(header + payload + CRC32) + `0x00` delimiter. CRC32 via ROM `esp_rom_crc32_le`, checked against Python `zlib.crc32` with shared test vectors.
- **Robustness:** text printed outside frames (early boot logs, panics, stray `printf`) is just bytes. The host prints anything that fails COBS/CRC as text, and the device drops invalid frames.
- **Header (8 B):** `uint8 version, uint8 type, uint16 seq, uint32 payloadLen`, little-endian.
- **Logs:** after the link starts, `esp_log_set_vprintf` routes ESP_LOG lines into `Log` packets (re-entrancy guard; dropped if TX is busy for more than 10 ms).
- **Flow:** stop-and-wait for host requests (host waits for the reply before sending the next frame). The pool holds **2 × 160 KB PSRAM buffers**, so the RX task can receive one packet while the vision task processes the other. `RxPacket` is an RAII handle that returns its buffer to the pool on destruction, the same pattern as the camera `Frame`.
- **TX of large payloads** (CAPTURE frames) is COBS-encoded in 512 B chunks straight from the camera buffer, with no full copy.

| Host → device | Payload | Device → host | Payload |
|---|---|---|---|
| `Hello` 0x01 | — | `HelloAck` 0x81 | protocol ver, fw build, heap totals |
| `ListModels` 0x02 | — | `ModelList` 0x82 | ids, backends |
| `LoadModel` 0x03 | id | `ModelInfo` 0x83 | info, load ms, heap Δ internal/PSRAM |
| `SetThreshold` 0x04 | float | `Ack` 0x84 | — |
| `Infer` 0x05 | frame id, w, h, fmt, pixels | `InferResult` 0x85 | frame id, `DetectTiming`, peak heap, faces[] |
| `Capture` 0x06 | count | `Frame` 0x86 | frame id, w, h, fmt, pixels, pan, tilt |
| `SetMode` 0x07 | Track/Bench | `Ack` 0x84 | — |
| `Subscribe` 0x08 | telemetry on/off | `Telemetry` 0x88 | t, state, pan, tilt, targets, face, detect µs |
| `Sync` 0x09 | marker | `SyncAck` 0x89 | marker, device time µs |
| `Echo` 0x0A | bytes | `EchoReply` 0x8A | same bytes |
| `SetPreview` 0x0B | enabled, format (JPEG / RGB565 / Gray8), scale (1 or ½), every-Nth, JPEG quality | `Preview` 0x87 | frame id, device time, w, h, format, pan, tilt, tracker state, `DetectTiming`, model id, faces[], image bytes |
| `ListStore` 0x0C | — | `StoreList` 0x8C | files in the model pack: name, format, size, crc |
| `LoadBlob` 0x0D | name, placement | `BlobInfo` 0x8D | ok, load µs, heap Δ, address (proves XIP vs RAM placement) |
| | | `Log` 0xF0 / `Error` 0xFF | text / code + text |

### 11.5 Tasks and ownership
| Task | Core | Prio | Owns |
|---|---|---|---|
| `pantilt` (existing) | 0 | 5 | servo motion |
| `link_rx` | 0 | 4 | USB RX, COBS decode → packet queue |
| `vision` (replaces `tracker`) | 1 | 3 | active `IDetector` (unique_ptr), `Tracker`, `BenchService`. Loop: drain commands; in Track mode `tracker.step(*detector)`; in Bench mode, block on command queue. |
| `preview` | 0 | 1 | Preview buffer, JPEG encode, send. Lowest priority, so it never delays servo motion or USB RX. |

All detector access happens on one task, so no locks are needed. A model swap is `detector.reset(); detector = entry->create();` measured by `MemoryProbe`. Bench mode pauses the tracker and holds the mount still. Track mode works with any loaded model, which closed-loop tests per model need.

### 11.6 Live view (labelled video while detection runs)
**Device — `PreviewStreamer`** (non-copyable, non-movable; owns its task and buffers):
- After each `detect()` (Track mode) or `Infer` (Bench mode), the vision loop calls `preview.offer(frameView, faces, timing, state, pan, tilt)`.
- `offer()` is **non-blocking**. If preview is disabled, this is not the Nth frame, or the previous preview is still being sent, the frame is dropped. Otherwise it copies the frame, decimating 2×2 if scale is ½ (QVGA → 160×120), into a preallocated PSRAM `HeapBuffer` together with the metadata, then notifies the preview task. The cost on the vision task is one copy (~0.5–1 ms at QVGA; measured).
- The `preview` task (core 0, prio 1) optionally JPEG-encodes with esp32-camera's streaming `fmt2jpg_cb` into a second preallocated PSRAM buffer (bounded, no per-frame malloc), then sends one `Preview` packet through `Link`.
- **Formats:** JPEG (default; QVGA q≈60 is a few KB, so ~10–20 FPS should fit the link), raw RGB565 (exact pixels, low FPS), Gray8 (debug).
- The camera frame buffer is returned to the driver immediately after the copy, so preview never holds the camera.
- **Memory:** 2 × 154 KB PSRAM (copy + JPEG out), allocated only while preview is enabled (RAII, freed on disable).

**Host — `tools/view/live_view.py`** (OpenCV window; owns the serial port, also prints `Log` packets):
- Draws on each frame:
  - each face box with label `face 0.87`, colour by score
  - the chosen track target highlighted
  - frame-centre crosshair and the tracker deadband rectangle
- HUD: model id, backend/quantization, tracker state (SCAN/TRACK), device loop FPS, preview FPS, detect ms (pre/infer/post), pan/tilt angles, free internal heap.
- Keys:
  - `space`: pause
  - `s`: snapshot PNG + JSON
  - `r`: toggle recording to MP4 + JSONL telemetry (for the report videos)
  - `m`: cycle the loaded model (`LoadModel`)
  - `+`/`-`: score threshold
  - `p`: preview format
  - `q`: quit
- Bench mode: `run_bench.py --show` draws **GT boxes (green) vs predictions (red)** per dataset image. That's useful for spotting labelling and preprocessing errors.
- **Note:** only one host program can open the serial port (COM9 at the time of writing; read from `platformio.ini`) at a time. The viewer, bench runner and capture tool share one Python `DeviceLink` class but run one at a time. The closed-loop recorder (Phase 2) embeds the viewer, so a rig run can be watched and recorded at once.

### 11.7 What existing code is reused
- `PanTilt` (`src/PanTilt.*`): unchanged. The motion task, `setTarget` and `pan()`/`tilt()` feed telemetry.
- `Servo`, calibration in `main.cpp`, `servo_config.h`: unchanged.
- `Camera` / `Frame` RAII (`src/Camera.*`): reused. Adds `view()`.
- `Tracker` scan/track logic (`src/Tracker.cpp`): kept. Only the loop structure changes (`run` → `step`) and it takes `IDetector&`.
- `FaceDetector.cpp` conversion code becomes `EspDlDetector::detect`.
- `Storage::getPartition` / `Partition` (`src/Storage.*`): base of `ModelStore`.
- `Delay.h` `delay()`.

---

## 12. Verification (Phase 0)
1. `pio run` builds cleanly. App size stays < 3 MB.
2. Host `pytest tools/tests`: COBS round-trip on edge cases (empty, 254/255-byte runs, all zeros) and CRC vectors identical to the device's.
3. Flash, then `ping.py`: `HelloAck` with the correct protocol version. Boot text and later `Log` packets both show in `monitor.py`.
4. Echo throughput: 1 KB–160 KB payloads, 100 % CRC pass, report KB/s (expect ≥ 300 KB/s, so ≤ 0.5 s per QVGA frame).
5. `LoadModel espdl-msrmnp-s8` 10× in a row: free heap returns to the same value each time (RAII / no-leak check).
6. Determinism: `Capture` one frame, then `Infer` it back. Boxes are identical across 10 repeats and match the live detection on that frame.
7. B0 timing sanity: detect total median within ~10 % of the vendor 36 ms + preprocessing.
8. `ModelStore`: pack 3 dummy blobs, flash them, list, open with each placement, CRC passes, 100 open/close cycles with no heap or mmap leak.
9. Tracking regression: with no host connected, the gimbal scans and tracks a face as before. With `Subscribe`, telemetry streams at the loop FPS.
10. Live view: `live_view.py` shows the camera with boxes and labels following a real face. Toggling the model with `m` works. Recording produces a playable MP4 + JSONL. Preview overhead is measured (loop FPS and detect p95, preview on vs off) and preview-off is confirmed to equal the no-preview baseline.
11. First numbers recorded in `docs/EXPERIMENT_PLAN.md`: B0 latency (median/p95), model memory, link throughput, preview overhead.

**Verification status**

| # | Check | How | Status |
|---|---|---|---|
| 1 | Firmware builds | `pio run` | **Passed** 2026-10-18: no warnings; static RAM 43 KB / 320 KB; app 2.54 MB / 3 MB |
| 2 | Host protocol tests | `python -m unittest discover -s tools/tests -t .` | **Passed** 2026-10-18: 11/11, incl. COBS reference vectors and CRC vector |
| 3–4 | Link up, echo throughput | `python tools/link/ping.py` | Pending (needs the board). Also checks the C++ COBS against Python byte for byte. |
| 5 | Model reload, no leak | `live_view.py` key `m`, or `run_bench.py` twice | Pending |
| 6 | Determinism | `capture.py` a frame, then `run_bench.py --repeat 10` on it | Pending |
| 7 | B0 timing sanity | `run_bench.py` on a folder of face images | Pending |
| 8 | ModelStore | `pack_models.py --dummy 3` → `flash_models.py` → `store_check.py --cycles 100` | Pending |
| 9 | Tracking regression, telemetry | board alone; then `monitor.py --telemetry` | Pending |
| 10 | Live view | `python tools/view/live_view.py` | Pending |

---

## 13. Phase 0 results (baseline B0)

Filled in from the first on-device runs. Every figure comes from a named results file.

| Quantity | Value | Source |
|---|---|---|
| Link: Hello round trip (ms) | TBD | `ping.py` |
| Link: echo throughput, 153 KB payload (KB/s each way) | TBD | `ping.py` |
| B0 load time (ms) | TBD | `run_bench.py` summary |
| B0 heap taken: internal / PSRAM (bytes) | TBD | `run_bench.py` summary |
| B0 weights in flash (bytes) | TBD | `run_bench.py` summary (`model_bytes`) |
| B0 detect total: median / p95 / p99 (ms) | TBD | `run_bench.py` summary |
| B0 lowest free internal heap during inference (bytes) | TBD | `run_bench.py` summary |
| Tracking loop FPS, preview off | TBD | tracker log line ("tracking, N fps") |
| Tracking loop FPS, preview on (JPEG, half scale) | TBD | live view HUD |
| Preview overhead: Δ loop FPS, Δ detect p95 | TBD | difference of the two rows above |
