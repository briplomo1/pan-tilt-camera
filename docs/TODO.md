# Project TODO: from here to the final report

Ordered, checkpointed steps to the end of the project. The design and the hypotheses behind each
step are in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) (§ numbers below); how to run the tools is in
[USAGE.md](USAGE.md).

**How to use this file.** Work top to bottom. Tick a box when it's done and write the result file
or commit next to it. A checkpoint is closed only when every "Done when" line holds; record the
numbers in EXPERIMENT_PLAN.md (§9 Result/Conclusion columns, §12, §13) as you go.

Tags: **[board]** needs the ESP32-S3 on USB · **[pi]** needs the Raspberry Pi · **[pc]** host only ·
**[gpu]** training (a CUDA GPU or a cloud notebook).

---

## CP0. Close out the tooling round and Phase 0 on the board

- [ ] [pc] Review and commit the tooling / dataset / Kalman work (2026-10-25 round)
- [ ] [board] Flash the protocol-v2 firmware (`pio run -t upload`); `python tools/link/ping.py` gets HelloAck v2
- [ ] [board] §12 checks 3–10, recorded in the §12 table with dates:
  - [ ] 3–4 link up, echo 64 B–153 KB, 100 % CRC, throughput in KB/s (`ping.py`)
  - [ ] 5 `LoadModel espdl-msrmnp-s8` ×10, free heap returns to the same value
  - [ ] 6 determinism: `capture.py` one frame, `run_bench.py --repeat 10` on it, identical boxes
  - [ ] 7 B0 timing on a folder of face images (`run_bench.py`), median within ~10 % of 36 ms + pre-processing
  - [ ] 8 `pack_models.py --dummy 3` → `flash_models.py` → `store_check.py --cycles 100`, no leak
  - [ ] 9 tracking with no host; telemetry with `monitor.py --telemetry`
  - [ ] 10 live view; preview overhead (loop fps and detect p95, preview on vs off)
- [ ] [board] Fill §13 (B0 load time, heap, weights, detect median/p95/p99, min free internal, loop fps, preview overhead)
- [ ] [board] Kalman filter A/B: same free-hand motion with `f` off and on in `live_view.py`, at least 3 recordings each
  (`r`); compare with `closed_loop.py`. Tune `kLeadS` / `kFaceAccelNoise` / `kFaceMeasNoise` only if clearly worse
- [ ] [pc] Freeze the controller: final values in the §8 settings table, "provisional" note removed
- [ ] [hardware] Separate servo power supply on both robots (brownouts, checkpoint report)

**Done when:** §12 is all "Passed", §13 is filled, and the controller is frozen with the A/B result written down.

---

## CP1. Datasets (Phase 1, §6)

- [ ] [pc] Download WIDER FACE (train, val, split files); `prepare_wider.py --wider <dir> --out data/a`
- [ ] [pc] Collect ≥ 150 no-face photos (rooms, offices, outdoors, posters without people; licence checked);
  `negatives.py <dir> --into data/a/test --count 100`
- [ ] [board] Capture set B, ~300 frames, one session per condition (`capture.py --session ... --tag ...`):
  - [ ] distances 0.5 / 1 / 2 / 3–4 m
  - [ ] head yaw and pitch (frontal, ¾, profile, looking down)
  - [ ] glasses / no glasses; at least 2–3 different people if possible
  - [ ] backlight (window), dim light, normal light
  - [ ] motion blur (`--keep-tracking`, moving), partial occlusion (hand, cup, hair)
  - [ ] empty frames (room without a person) for the false-positive rate
  - [ ] 2–3 short `--video` clips for replay tests (one static face for jitter, one moving)
- [ ] [pc] `autolabel.py` each session, then `label.py` to review every frame (set B is a test set: no `--accept`)
- [ ] [pc] `split.py data/captures --gt <each session>.gt.json --out data/b` (B-cal 30 % / B-test 70 %, by session)
- [ ] [pc] Record the wide video for the virtual gimbal: ≥ 90° horizontal view (phone 0.5× lens or action camera),
  1080p, a person walking, turning, sitting, leaving and re-entering, 1–2 min; note the lens HFOV
- [ ] [pc] Ground truth for it: `autolabel.py <video> --accept`, then `label.py <video> --stride 15 --base <video>.gt.json`
- [ ] [pc] Fill `notebooks/01_datasets.ipynb`: counts, faces per size bucket (S/M/L), sample grids with boxes

**Done when:** `data/a/test/gt.json` (~400 face + ~100 no-face images), `data/a/cal/gt.json`,
`data/b/gt_cal.json` / `gt_test.json`, and the wide video + ground truth exist, with counts in the notebook.

---

## CP2. Baseline measurements (Phase 2, group B, §8)

- [ ] [board] B0 open loop: `run_bench.py data/a/test --gt ... --threshold 0.05 --tag a_test` and the same for
  `data/b` (cal and test); `score.py` with `--cal-run` on the B-cal run
- [ ] [board] B0 on the replay clips (`run_bench.py <clip>.mp4`); centre jitter on the static clip
- [ ] [pi] Pi Haar on the same sets: `face_tracker.py --images ... --gt ... --metrics-out results/`, plus peak RSS;
  score with `score.py --images`
- [ ] [board] Virtual gimbal, B0, filter on and off, 5 runs each (`virtual_gimbal.py --model espdl-msrmnp-s8`)
- [ ] [pi] Virtual gimbal, Haar backend on the Pi (`--backend haar`), same video
- [ ] [pc] Physical rig (§8): `tools/rig/target_display.py` (face image moving on a monitor: sines 10–160 °/s,
  vertical sine, 15° steps, random walk, 2 s occlusions), host-side recorder that sends `Sync` and logs telemetry,
  target angle from monitor geometry; extend `closed_loop.py` to read rig logs
- [ ] [board] Rig calibration (centre on the screen, zero offset), then L-B0: 5 trials per trajectory and speed
- [ ] [board] Optional: R1 (ESPDet-Pico-224) build and run, for the GH6 comparison
- [ ] [pc] `compare.py` table B0 vs Pi Haar; group B rows filled in §9

**Done when:** B0 has AP50 (A-test, B-test) with CIs, latency per stage, memory, virtual-gimbal and rig
closed-loop numbers, all from named result files.

---

## CP3. Training toolchain and fidelity (Phase 3)

- [ ] [gpu] `.venv-train`: torch, torchvision, ai-edge-torch (LiteRT), onnx, onnx2tf (fallback), tensorflow (for the
  TFLite interpreter on the PC); pin versions in `notebooks/requirements-train.txt`
- [ ] [gpu] Tiny conv model → PT2E PTQ and QAT → int8 `.tflite`; check it holds only int8 ops ESP-NN supports
- [ ] [pc] Fidelity on the PC: PyTorch fake-quant vs TFLite interpreter, mean abs diff ≤ 1 quantization step
- [ ] [pc] Shared training code in `notebooks/common/` (or `tools/train/`): letterbox data loader (A-train, B-cal),
  loss/decode helpers, export function, KD loss hook, result logging
- [ ] [pc] `00_toolchain_and_fidelity.ipynb` filled in

**Done when:** a PyTorch model goes to an int8 `.tflite` by PTQ and by QAT, and matches on the PC within one quant step.

---

## CP4. TFLite Micro backend on the S3 (Phase 4, group K)

- [ ] [pc] Add `espressif/esp-tflite-micro` (with ESP-NN) to `src/idf_component.yml`; build size check (app < 3 MB)
- [ ] [pc] `src/vision/TfliteDetector.{h,cpp}`: `IDetector` over a `ModelBlob` from `ModelStore`, tensor arena in a
  `HeapBuffer` (PSRAM default, internal for group M), op resolver with only the needed ops, RGB565 → model input
  resize/normalise, timing per stage
- [ ] [pc] `IDetectionHead` interface with two implementations: SCRFD (anchor-free, strides 8/16/32) and YOLOv5
  (anchors, 3 scales); shared NMS; output into the reserved `Face` vector
- [ ] [pc] Register with `DetectorRegistry::addStoreBackend("tflite", "tflite-micro", ...)`; model metadata (head type,
  input size, score/NMS settings) stored next to the `.tflite` in the pack
- [ ] [board] Device fidelity: same input on PC TFLite and on the device, outputs match within one quant step
- [ ] [board] K1 (ESP-NN) / K2 (reference kernels) / K3 (float32) on the tiny model, then on S1 once it exists
- [ ] [board] Verify with the live view: `m` cycles to the TFLite model and it tracks

**Done when:** a `.tflite` flashed with `flash_models.py` appears as `tflite-micro/<name>`, runs in bench and track
mode, matches the PC, and K1–K3 are measured.

---

## CP5. Teachers and float models (Phase 5 part 1: T2, T3, S0, Y0)

- [ ] [gpu] T2 SCRFD-10GF and T3 YOLOv5s-face: get the official weights, evaluate float on A-test and B-test
  (PC detector backends, scored with `score.py`)
- [ ] [gpu] S0: SCRFD-0.5GF in PyTorch; drop the landmark branch; evaluate float at 320×240 / 224×168 / 160×120
- [ ] [gpu] Y0: YOLOv5n-0.5-face; drop landmarks; swap SiLU → ReLU and retrain/fine-tune to recover; evaluate float
- [ ] [pc] Export check for S0 and Y0: every op converts to int8 TFLite and is supported (record any blocker as a result)
- [ ] [pc] Confirm params/FLOPs from the repos and update the §9.0 candidates table
- [ ] [pc] Create `notebooks/15_yolov5n_face.ipynb`; fill `14_scrfd_500m.ipynb` and the teacher notebook

**Done when:** T2, T3, S0, Y0 have float AP50 on both test sets, and S0/Y0 pass the export check.

---

## CP6. PTQ on the device (S1, Y1)

- [ ] [gpu] S1, Y1: PTQ int8 per-channel, calibrated on A-cal + B-cal; export `.tflite`
- [ ] [board] `pack_models.py` → `flash_models.py`; `run_bench.py` on A-test, B-cal, B-test with
  `--model espdl-msrmnp-s8 --model tflite-micro/S1 --model tflite-micro/Y1`
- [ ] [pc] `score.py` (threshold from B-cal) and `compare.py --baseline espdl-msrmnp-s8 --paired espdl-msrmnp-s8`
- [ ] [pc] ΔAP vs float (S1 − S0, Y1 − Y0); latency and memory; S1/Y1 rows in §9
- [ ] [pc] Decision point: if S1 latency at 320×240 is far over budget, move the resolution sweep (CP8) earlier

**Done when:** S1 and Y1 rows are filled and compared with B0.

---

## CP7. QAT, distillation, fine-tuning, granularity (S2–S4, Y2–Y3, G1–G4)

- [ ] [gpu] QAT recipe (from the float weights, fixed epochs and seed): S2, Y2
- [ ] [gpu] KD loss from the family teacher (score and box outputs; temperature/weight fixed across models): S3, Y3
- [ ] [gpu] S4: S3 fine-tuned on B-cal (QAT kept on)
- [ ] [board] Deploy and measure each (same commands as CP6); check latency is unchanged from S1/Y1 (GH1)
- [ ] [gpu]+[board] Granularity on the best S recipe: G1 per-channel (= S2), G2 per-tensor, G3 PoT per-tensor (custom
  observer), G4 PoT PTQ; `30_granularity_pot.ipynb`
- [ ] [pc] Paired tests: S2 vs S1, S3 vs S2, S4 vs S3 (and Y), each with p-value; §9 rows and conclusions

**Done when:** every ladder rung up to S4/Y3 and G1–G4 has a measured row and a conclusion.

---

## CP8. Resolution, placement, runtime cross-check (S5, Y4, R, M, S-DL)

- [ ] [gpu]+[board] S5-160 and S5-224 (S4 recipe at lower input); R-160/224/320 rows
- [ ] [board] Y4 at the best resolution
- [ ] [board] M1–M4 on the best S model (weights flash/PSRAM/SRAM, arena PSRAM/internal); decide whether W2 runs
- [ ] [gpu]+[board] S-DL: best S model through ESP-PPQ → `.espdl`, register as an ESP-DL model, measure against B0
- [ ] [pc] Pareto front (`compare.py` plot) across B0, S*, Y*, S-DL

**Done when:** the resolution curve, placement result and S-DL row exist, and the Pareto set is known.

---

## CP9. Optional and conditional experiments

Do each only if its condition holds or there is time; otherwise record "not run" and why.

- [ ] W1 (16×8) and W3 (mixed int8 / 16×8) on the best S model
- [ ] W2 (W4A8, custom TFLM kernel) **only if** M3 is > 25 % faster than M1
- [ ] SCRFD-2.5GF and YOLOv5n-face (full width) through rungs 1–3 at 160×120
- [ ] C1 custom ReLU-only detector **only if** no S/Y model meets the latency target
- [ ] [pi] A learned detector on the Pi (YuNet) on the same sets, to separate model from hardware
- [ ] Deferred candidates (BlazeFace, Ultra-Light, YuNet on the S3)

---

## CP10. Closed loop on the Pareto set (group L)

- [ ] [board] Virtual gimbal: every Pareto model + B0 (+ R1 if built), filter on, 5 runs each, same video
- [ ] [board] Physical rig: the same models, all trajectories and speeds, 5 trials each
- [ ] [board] Secondary run with per-model retuned gain (reported separately, §8 item 7)
- [ ] [board] Live-view recordings of each Pareto model tracking (report videos)
- [ ] [pc] `closed_loop.py` tables; RMS error vs target speed, max trackable speed, reacquire time; test GH6

**Done when:** every Pareto model has virtual-gimbal and rig numbers next to B0.

---

## CP11. Final analysis and report

- [ ] [pc] `notebooks/90_results_analysis.ipynb`: the §7 table for every row, and figures 0–7
  (videos, Pareto, PR curves, recall by size, PTQ/QAT/KD bars, kernel/placement bars, closed-loop error vs speed,
  fidelity scatter)
- [ ] [pc] Every §9 row has Result and Conclusion; hypotheses GH1–GH7 marked held / not held
- [ ] [pc] Answer RQ1–RQ4; threats to validity revisited (§5.2)
- [ ] [pc] Final report (`ProjectReport.docx`) and presentation; reproducibility notes (commands, commits, seeds)
- [ ] [pc] Tag the final commit; archive `results/` and the key recordings

**Done when:** the report answers RQ1–RQ4 from named result files.
