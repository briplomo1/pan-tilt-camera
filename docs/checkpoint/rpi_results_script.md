# Checkpoint presentation: Raspberry Pi baseline results (script draft)

**Section length:** about 4 minutes. **Presenter:** _[name]_
**Comes after:** project overview / progress completed. **Comes before:** ESP32-S3 baseline results (to be added once recorded), then the Pi vs ESP32 comparison, difficulties and future work.

Data source: one live tracking session recorded on the Pi with `rsbpi/face_tracker.py`, two clips
(`recordings/clip-20261016-020908` and `clip-20261016-020936`), 37.9 s, 1,369 processed frames.
Every number below comes from the per-frame JSONL and summary files saved with those clips.

> Text in quotes is spoken. Lines starting with **[Slide]** or **[Video]** are cues.

---

## 1. What the Pi baseline is (≈40 s)

**[Slide]** Title: *Raspberry Pi baseline: classical detector, no neural network*. Photo of the Pi rig.
Bullets: Raspberry Pi _[model]_ · Camera Module 3 NoIR (IMX708) at 640×480 · PCA9685 servo driver, 2 servos · OpenCV Haar cascade (frontal face).

"Our first robot is the reference point for the whole project. It runs on a Raspberry Pi with the
Pi camera module, and it drives two hobby servos through a PCA9685 PWM board.

The detector is OpenCV's Haar cascade. It isn't a neural network: it slides a window over the image
and checks patterns of light and dark rectangles that faces usually have. To keep it fast, each
640 by 480 frame is shrunk to 320 by 240, converted to grayscale and contrast-equalised before
detection.

The tracker is a simple proportional controller. It measures how far the face is from the centre of
the frame and nudges the pan and tilt servos toward it, with a small dead zone so it doesn't
chase tiny errors."

---

## 2. How we measured it (≈30 s)

**[Slide]** *Measurement setup*. Diagram: camera → detect → servo command, with a timer on each stage.
Bullets: recording starts automatically when a face appears (2 s lead-in) · annotated video + per-frame log + summary · same metric categories as our experiment plan.

"For this checkpoint we added automatic recording to the Pi. As soon as it sees a face it saves a
video with the detections drawn on it, starting two seconds earlier so you see it find the face. It
also writes a log entry for every single frame: how long each detection stage took, every face it
found and its score, and where the servos were pointed.

We deliberately log the same categories our experiment plan uses for the ESP32: speed, tracking
quality, and later accuracy and memory. That way the two robots can be compared line by line."

---

## 3. The video (≈40 s)

**[Video]** Play `clip-20261016-020908.mp4` from the start (30 s; trim to ~15 s if short on time).
Point out: grey boxes = every detection with its score; green box and line = the face being followed; the yellow text = live FPS, pan/tilt, detect time.

"Here's the Pi tracking a face. For the first two seconds it hasn't found anyone yet, which is the
lead-in. Then the face appears, the green box locks on, and the line from the centre shows the
error the controller is correcting. You can see the pan and tilt angles updating in the corner,
and the detection time per frame, which stays around 15 milliseconds."

_(If showing clip 2 instead: "This second clip is a steadier stretch: the face is found in 99 % of
frames for the whole 8 seconds.")_

---

## 4. Speed results (≈45 s)

**[Slide]** *Speed*. Table:

| Metric | Median | p95 | p99 |
|---|---|---|---|
| Detection, total | 15.2 ms | 23.9 ms | 30.9 ms |
| · Haar detection step | 13.7 ms | 22.3 ms | — |
| · Resize + grayscale | 1.4 ms | 1.6 ms | — |
| Camera frame → servo command | 19.6 ms | 28.6 ms | — |
| Tracking loop rate | ≈ 37 frames/s | | |

"On speed, the Pi is comfortably real time. Detection takes about 15 milliseconds per frame at the
median, and even the slowest 1 % of frames finish in about 31 milliseconds. About 90 % of that time
is the Haar detection itself; the resizing and grayscale conversion are only about one and a half
milliseconds.

From the moment a frame is captured to the moment a servo command goes out is just under 20
milliseconds, and the whole loop runs at about 37 frames per second. So speed is not the
bottleneck on the Pi. That matters for the comparison, because the ESP32 has a far weaker processor
and we expect speed to be its main constraint."

---

## 5. Tracking results (≈60 s)

**[Slide]** *Tracking quality*. Table:

| Metric | Clip 1 (30 s) | Clip 2 (7.9 s) |
|---|---|---|
| Frames with the face detected | 57 % (60 % after first lock) | 99 % |
| Times the face was lost | 24 | 3 |
| Longest continuous lock | 132 frames (≈ 3.5 s) | 126 frames |
| Median distance from centre, horizontal | 9.7 % | 10.0 % |
| Median distance from centre, vertical | 10.4 % | 7.9 % |
| Pan / tilt range used | 62–110° / 90–131° | 61–107° / 123–131° |

_(Distance from centre is measured as a percentage of half the frame width or height; 0 % is perfectly centred.)_

"Tracking quality is where the classical detector shows its limits. When it has the face, it keeps
it close to the middle: the typical offset is about 10 % of the way from the centre to the edge,
and in 88 % of tracked frames the face centre stays within 80 pixels of the middle horizontally and
60 pixels vertically, on a 640 by 480 image.

But it loses the face often. In the first clip it dropped the face 24 times in 30 seconds. Most of
those are single-frame dropouts: 15 of the 24 gaps were shorter than a fifth of a second, which
is the Haar cascade simply missing a frame, often when the head turns slightly. A Haar cascade
only knows frontal faces, so a turned head, a tilt or a change in lighting is enough. The longest
gap was about two and a half seconds. The second clip, with the face held steadier toward the
camera, shows the best case: 99 % of frames detected."

---

## 6. What this means and limits of the data (≈30 s)

**[Slide]** *Takeaways*. Bullets: fast (≈15 ms, 37 FPS) · keeps a found face near centre · frequent dropouts from frontal-only detection · next: same measurements on the ESP32.

"To sum up: the classical baseline is fast and centres a face well once it has one, but it's
fragile. It misses faces whenever they aren't looking straight at the camera. That's exactly
the gap a learned detector should close, and it's what we'll measure on the ESP32 next.

Two caveats on these numbers. First, this is a single session of about 38 seconds, so treat it as
preliminary. Second, we haven't measured accuracy against labelled images or memory use on the Pi
yet. The script supports both, and they're part of the next round, so the final comparison will
have all four categories: accuracy, speed, memory and tracking."

**[Transition]** "Next, the same measurements on the ESP32-S3 with Espressif's own face detection model."

---

## Notes for the presenter (not spoken)

- **Detection scores** in the video (around 5–6, up to 12) are Haar confidence weights, not probabilities. Don't call them percentages.
- **The "57 %" for clip 1** includes the 2 s lead-in before any face was there and the time spent re-finding the face. After the first lock it is 60 %.
- **Clip dates** say 16 Oct because the Pi had no network time; the session was recorded on 18 Oct.
- **Still to add before submission:** Pi memory (`--metrics-out` gives peak RSS), Pi accuracy on a labelled image set (`--images ... --gt ...` gives AP50, recall, false positives), ESP32 numbers in the same tables, and a longer or repeated session so the tracking numbers have more than one run behind them.
- **Fill in:** the Pi model in section 1, and the presenter name.
