# Checkpoint presentation: ESP32-S3 baseline results, comparison and conclusion (script draft)

**Section length:** about 7–8 minutes. **Presenter:** _[name]_
**Comes after:** Raspberry Pi baseline results (`rpi_results_script.md`). **This is the last section** of the recording.

Data source: one live tracking session recorded from the ESP32-S3 through `tools/view/live_view.py`
(`recordings/live-20261018-211259.mp4` and `.jsonl`, 11.7 s, 208 frames, every one of them logged).
The comparison uses the Pi's best clip, `recordings/clip-20261016-020936` (7.9 s, 281 frames), recomputed
with the same definitions, so the two columns compare directly.

> Text in quotes is spoken. Lines starting with **[Slide]** or **[Video]** are cues.

---

## 1. What the ESP32-S3 baseline is (≈45 s)

**[Slide]** Title: *ESP32-S3 baseline: a neural network on a microcontroller*. Photo of the ESP32 gimbal.
Bullets: Seeed XIAO ESP32-S3 Sense (240 MHz dual core, 512 KB on-chip RAM, 8 MB PSRAM) · OV2640 camera at 320×240 · 2 servos driven directly from the chip · Espressif ESP-DL face detector, 8-bit quantized.

"Our second robot is the one the project is really about. Instead of a Raspberry Pi it runs on an
ESP32-S3, a microcontroller that costs a few dollars, with no operating system and only half a
megabyte of fast on-chip memory.

The detector is Espressif's own face detection model, and unlike the Pi this one *is* a neural
network. It works in two stages. The first stage looks at a shrunk 160 by 120 copy of the image and
proposes places where a face might be. The second stage takes each of those small crops and decides
whether it really is a face, and where exactly. Both networks are quantized to 8-bit integers, so
together they take about 190 kilobytes of storage.

This model is our baseline for the rest of the project. Every model we build later has to beat it
on this exact board.

The Raspberry Pi plays a different role. It's our point of reference: a full computer with a
quad-core processor and gigabytes of memory, running a well-established detector. We don't expect
the microcontroller to beat it. What we want to know is how close a few-dollar chip can get, and the
gap between the two is what our future models try to close."

---

## 2. Same controller for every model (≈30 s)

**[Slide]** *Fixed tracking settings*. Small table: gain 0.6 · dead zone 6 % · tracking speed limit 90 °/s · hold 3 s before giving up · scan 20 °/s with a half-second pause every 20°.

"Before recording, we had to fix a few things on this robot, which I'll come back to in the problems
section. Once it tracked properly we froze the controller settings: the gain, the dead zone, the
speed limits, how long it holds after losing a face, and the scan pattern. Those stay the same for
every model from now on, so when we compare models later, any difference comes from the detector
and not from re-tuning the motors."

---

## 3. The video (≈40 s)

**[Video]** Play `live-20261018-211259.mp4` (12 s, full length).
Point out: the box is the detection with its score; the text shows state (SCAN/TRACK), pan/tilt and detect time; the dashed box in the middle is the dead zone.

"Here is the ESP32 tracking. It's locked on from the start, and as I move it follows me, keeping my
face near the middle box. The confidence scores in the corner are mostly around 0.9. These are real
probabilities from the network, unlike the Pi's Haar scores.

Between about three and four seconds, when I move fastest, you can see the box flicker: the
detector misses single frames. But it never loses me for more than a fifth of a second, so the
tracker never has to stop and wait."

---

## 4. Speed (≈45 s)

**[Slide]** *Speed: Pi vs ESP32-S3*. Table:

| Metric | Raspberry Pi (Haar) | ESP32-S3 (ESP-DL MSR+MNP) |
|---|---|---|
| Detection per frame, median | 15.6 ms | 45.3 ms |
| Detection, p95 / p99 | 22.1 / 24.4 ms | 49.3 / 49.3 ms |
| Tracking loop rate | ≈ 36 frames/s | ≈ 18 frames/s |
| Image the detector sees | 320×240 grayscale | 160×120 colour (first stage) |

"On speed the Pi wins, as expected. The ESP32 needs about 45 milliseconds per frame, three times
the Pi's 16, so the whole loop runs at about 18 frames per second instead of 36.

Two details are worth pointing out. First, the ESP32 is very consistent: its slowest 1 % of frames
take only about 9 % longer than the median, because nothing else runs on the chip. The Pi has an
operating system, so its slowest frames take about 60 % longer than its median. Second, the ESP32 is
faster when there's no face, at about 42 milliseconds, because the second stage only runs when the
first stage proposes something.

So 18 frames per second is enough to follow a person walking at a normal pace, but it means the
robot reacts about half as often as the Pi does."

---

## 5. Tracking quality (≈60 s)

**[Slide]** *Tracking: Pi vs ESP32-S3*. Table:

| Metric | Raspberry Pi (best clip, 7.9 s) | ESP32-S3 (11.7 s) |
|---|---|---|
| Face detected | 99 % | 93 % |
| Times the face was lost | 3 (all single frames) | 13 (all under 0.2 s) |
| Longest gap | 0.09 s | 0.17 s |
| Longest unbroken lock | 126 frames (≈ 3.5 s) | 33 frames (≈ 1.9 s) |
| Median distance from centre, horizontal / vertical | 10.0 % / 7.9 % | 12.5 % / 8.8 % |
| Face within the central quarter box | 95 % | 76 % |
| Median detection score | 6.1 (Haar weight, no unit) | 0.87 (probability) |
| Pan / tilt range covered | 61–107° / 123–131° | 61–102° / 126–142° |

_(Distance from centre is a percentage of half the frame width or height; 0 % is perfectly centred. The central quarter box is 80×60 px on the Pi's 640×480 frame and 40×30 px on the ESP32's 320×240.)_

"For tracking, we compare the Pi's best clip with the ESP32's latest recording.

In its best clip the Pi detects the face in 99 % of frames and drops it for a single frame only three
times. The ESP32 detects the face in 93 % of frames. It misses frames more often, 13 times, but every
miss is short: the longest is under a fifth of a second. So on these two recordings the Pi is ahead
on detection and centring as well as speed, which is what we'd expect from our reference.

The ESP32 isn't far behind, though, and the conditions weren't the same. The Pi clip is a face held
fairly steady and looking straight at the camera, which is exactly what a Haar cascade is good at.
In the ESP32 recording I was moving more, as the next slide shows. For context, in the Pi's other
clip with normal movement its detection fell to 60 %. The ESP32's 93 % under more movement is a
promising sign for the neural network, but we need the same input on both before we can claim it.

Centring is close: the face sits 10 to 12 % off centre horizontally on both, at the median. The Pi
keeps it inside the central box more often, 95 % against 76 %, because it updates twice as often."

---

## 5b. How much movement each robot needed (≈60 s)

**[Slide]** *Movement needed to keep the face centred*. Two tables:

| Mount movement | Raspberry Pi (best clip) | ESP32-S3 |
|---|---|---|
| Typical (median) mount speed | 18 °/s | 26 °/s |
| Fast moments (90th percentile) | 30 °/s | 49 °/s |
| Fastest | 46 °/s | 73 °/s |
| Pan movement per second | 14 °/s | 24 °/s |
| Tilt movement per second (within a range of) | 14 °/s (8°) | 13 °/s (16°) |
| Time spent faster than 35 °/s | 1 % | 33 % |

| Mount speed | Pi: detected / horizontal offset | ESP32-S3: detected / horizontal offset |
|---|---|---|
| 5–20 °/s | 99 % / 9.4 % | 98 % / 8.9 % |
| 20–35 °/s | 99 % / 11.2 % | 85 % / 9.1 % |
| 35–50 °/s | (4 frames only) | 96 % / 20.9 % |
| Over 50 °/s | (none) | 89 % / 25.0 % |

"To see how hard each robot had to work, we looked at how fast the mount itself was moving to keep
the face in the middle.

The ESP32 had to work harder. At a typical moment it was turning at 26 degrees per second against
the Pi's 18, and its pan moved 24 degrees every second against 14. The fast moments differ even
more: the Pi's best clip never needed more than 46 degrees per second and spent only 1 % of the
time above 35. The ESP32 reached 73 and spent a third of the recording above 35.

The Pi's tilt moved back and forth a lot inside a narrow 8-degree range, which is the controller
making small corrections every frame. The ESP32 moved its tilt about as much, but over twice the
range, so it was following real movement rather than correcting itself.

The second table compares the two at the same mount speed. Up to 20 degrees per second they're the
same: about 99 % detected, with the face under 10 % off centre. Between 20 and 35 the Pi detects
more, 99 against 85 %, and the two centre about equally.

Above 35 degrees per second only the ESP32 has data, so we can't compare there directly. What we can
see is that its detection holds up, 89 to 96 %, but the face drifts to between a fifth and a quarter
of the way to the edge. At 18 frames per second the face moves a long way between updates. So as the
movement increased, the neural network kept *finding* the face, and what limited the ESP32 was its
frame rate. That's exactly where the Pi's speed advantage should matter most, and the
virtual-gimbal and rig tests will measure it."

---

## 6. Problems on the ESP32 robot (≈30 s)

**[Slide]** *What we fixed*. Bullets: pan reversed · faces lost while in view (blur, backlight) · scan at the ceiling, then blurred and jerky · still to do: separate servo power, longer camera cable.

"A few fixes got the ESP32 to this point. The pan direction was reversed, so it first moved away from
faces. It lost faces that were still in view, because fast motion blurred them and a bright window
darkened them, so we capped the tracking speed, raised the exposure and added a 3-second hold. And
the scan first pointed at the ceiling, then blurred faces; it now sweeps slowly at eye level with
pauses. These settings are frozen for every future model. The remaining issues are hardware: the
servos share the USB power, and the board rides on the gimbal until we get a longer camera cable."

---

## 7. Conclusion (≈40 s)

**[Slide]** *Conclusions*. Two columns:
Pi + Haar: fast (16 ms, 36 FPS) · 99 % detected and 95 % centred on a steady frontal face · frontal faces only, no learned features.
ESP32-S3 + ESP-DL: 3× slower (45 ms, 18 FPS) · 93 % detected with faster movement, no miss longer than 0.2 s · runs in ~190 KB of 8-bit weights on a few-dollar chip · limited by frame rate and 160×120 input.

"To conclude: the Raspberry Pi is our reference point, and it has real advantages. It has many times
the processing power and memory, so it runs twice as many frames per second, reacts faster, and
has an operating system and network access that make it easy to develop and debug. On our best
recordings it is faster and, for a steady face looking at the camera, detects and centres it almost
perfectly. Its weakness is that the Haar cascade only knows frontal faces.

The ESP32-S3 is three times slower and misses more single frames, but it still found the face in
93 % of frames, never lost it for more than a fifth of a second, and did that with faster movement
than the Pi's clip. It runs a quantized neural network that fits in about 190 kilobytes, on a chip
with a tiny fraction of the Pi's memory and power. Where it falls behind is centring during fast
movement, and that comes down to frame rate."

"Both are working baselines now, and they're measured the same way, so everything we build next has
a fair reference point on each side."

---

## 8. Future work (≈90 s)

**[Slide]** *Next models and methods*. Bullets in three groups.

**Extending today's measurements: three tracking tests**
1. Replayed video and labelled images: the same frames into every model, camera still → accuracy, speed, memory.
2. Virtual gimbal: each model steers a window cropped from one wide video; slower models skip frames → tracking, perfectly repeatable.
3. Physical rig: the robot follows a face moving along scripted paths on a monitor → real tracking, repeated runs.

**Models to compare against these two baselines**
- ESP-DL's larger ESPDet-Pico detectors (224×224 and 416×416): more accurate, expected 3–10× slower.
- Small public face detectors converted to TensorFlow Lite Micro: BlazeFace, Ultra-Light-Fast, YuNet, SCRFD.
- Our own small detector, trained by distillation from a large PC model (YOLO11n-face).
- A learned detector on the Pi (e.g. YuNet in OpenCV) to separate the effect of the model from the hardware.

**Methods to test on them**
- Post-training quantization vs quantization-aware training, and per-channel vs per-tensor scales.
- Input resolution sweep: 160×120, 224×168, 320×240.
- Where the weights live (flash, PSRAM, on-chip RAM), to tell whether the chip is compute- or memory-bound.

**Robot improvements**
- Smoothing and predicting the face position (Kalman filter), then fixed for all models.
- Separate servo power supply and a longer camera cable.

"Looking ahead, we'll extend the measurements we built for this checkpoint so that every new model
is compared with both baselines on identical terms.

A replayed video can't move the camera, so we'll measure tracking in three steps. First, we stream
the same recorded frames and labelled images into every model with the camera still. The face can
be anywhere in the frame, and we check each frame against where the face really is. That gives
accuracy, speed and memory on identical pixels.

Second, a virtual gimbal. We record one wide, high-resolution video, and each model sees a small
window cut out of it. Our tracking controller moves that window instead of the servos, and a model
only gets a new frame once it has finished the last one, so a slow model misses frames just like on
the robot. Every model follows the same person through the same video, so tracking can be compared
with nothing else changing.

Third, the real robot. It follows a face on a monitor moving along scripted paths at set speeds,
with jumps and short occlusions. Because we know exactly where the face is at every moment, we can
measure the pointing error and the fastest movement each model can follow, and repeat each run
several times. That turns today's free-hand recordings into a controlled test.

On the ESP32 we'll try Espressif's larger detectors, several small public face detectors converted
to TensorFlow Lite Micro, and finally our own small model, trained by copying a large detector
running on a PC. We also want a learned detector on the Pi, so we can tell how much of today's
difference comes from the model and how much from the hardware.

For each model we'll test how it's quantized, what resolution it runs at, and where its weights are
stored in memory, because on a microcontroller those choices matter as much as the architecture.

On the robot side, a Kalman filter to predict where the face is going would help with the slower
frame rate, and a separate servo power supply should fix the brownouts.

That's our progress so far. Thank you."

---

## Notes for the presenter (not spoken)

- **Recordings used.** The comparison uses only the Pi's best clip (`clip-20261016-020936`) and the latest ESP32 recording (`live-20261018-211259`). The 60 % figure in section 5 is the Pi's other clip, quoted only as context for the movement difference; drop that sentence if you want the comparison to rest strictly on the two recordings.
- **Every frame is logged on both.** The Pi logs every frame itself. The ESP32 recording now logs every tracker frame from telemetry (208 of 208), so its 13 single-frame misses are counted just like the Pi's three. The video still only shows about every second ESP32 frame, so don't count frames from the video.
- **Mount speed** is how fast the pan/tilt angles changed over the previous 0.3 s. Frames where the mount was still (under 5 °/s) are left out of the speed table: the tracker stops when it loses a face, so still frames are mostly misses and would make both robots look worse for the wrong reason.
- **If asked about fast movement on the Pi:** its other clip did have some (85 frames above 35 °/s). It detected 84–100 % there with the face 19–22 % off centre, slightly better centred than the ESP32's 21–25 %. That fits the frame-rate explanation. It's not in the slides because the comparison uses only the best clip.
- **Different sessions.** The Pi and ESP32 numbers come from different sessions with different lengths (7.9 s vs 11.7 s), movement and lighting. They are a fair first comparison but not a controlled one; the three tests in future work (replayed video, virtual gimbal, scripted rig) fix that.
- **Other ESP32 recording from the same session** (`live-20261018-211234`, 23.7 s) is not used. It includes a 4.4 s loss and 74 % detection, and one frame of its log is missing.
- **Not measured yet on the ESP32:** time from capture to servo command (only detection time is logged), accuracy on labelled images, and the formal benchmark run (`run_bench.py`). The chip had at least 189 KB of on-chip RAM free throughout.
- **Fill in:** presenter name.
