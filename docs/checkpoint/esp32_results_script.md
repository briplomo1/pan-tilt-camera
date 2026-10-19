# Checkpoint presentation: ESP32-S3 implementation, comparison, conclusion and future work (script)

**Length:** about 4½ minutes (≈ 620 spoken words). **Presenter:** _[name]_
**Comes after:** the Raspberry Pi results and its detection video demo (`rpi_results_script.md`). This is the last section.

Recordings compared: the Pi's best clip, `recordings/clip-20261016-020936` (7.9 s), and the latest ESP32-S3
recording, `recordings/live-20261018-211259` (11.7 s). Every frame is logged on both, and both were analysed with
the same script and definitions.

> Text in quotes is spoken. **[Slide]** and **[Video]** are cues.

---

## 1. The ESP32-S3 implementation (≈ 60 s)

**[Slide]** *ESP32-S3: a neural network on a microcontroller*. Photo of the gimbal.
Bullets: XIAO ESP32-S3 Sense, 240 MHz dual core, 512 KB on-chip RAM · OV2640 camera, 320×240 · Espressif ESP-DL face detector, 8-bit, ≈ 190 KB · fixed tracking controller for every model.

"Our second robot is the one the project is built around: an ESP32-S3, a microcontroller that costs
a few dollars, with no operating system and half a megabyte of fast memory.

Unlike the Pi's Haar cascade, its detector is a neural network, Espressif's own face model. A first
stage scans a 160 by 120 copy of the image for likely faces, and a second stage checks each one.
Both are quantized to 8-bit integers and fit in about 190 kilobytes. This is our microcontroller
baseline.

Getting it to track took a few fixes. The pan direction was reversed, and faces were lost when the
camera moved fast or was backlit, so we capped the tracking speed, raised the exposure, and made it
hold position for three seconds after a miss. We then froze these controller settings, so every
future model is tested under the same controller."

---

## 2. The ESP32-S3 video (≈ 30 s)

**[Video]** Play `live-20261018-211259.mp4` (12 s).

"Here it is tracking. Notice how much my face moves, much more than in the Pi video: side to side
and up and down. The box follows me the whole time. When I move fastest, around three
seconds in, it flickers for single frames, but it never loses me for more than a fifth of a second."

---

## 3. Comparing the two (≈ 90 s)

**[Slide]** *Pi vs ESP32-S3*.

| | Raspberry Pi (Haar) | ESP32-S3 (ESP-DL, int8) |
|---|---|---|
| Detection time (median) | 16 ms | 45 ms |
| Frame rate | 36 fps | 18 fps |
| Face detected | 99 % | 93 % |
| Longest miss | 0.09 s | 0.17 s |
| Face in central quarter of frame | 95 % | 76 % |
| Typical / fastest mount movement | 18 / 46 °/s | 26 / 73 °/s |
| Time moving faster than 35 °/s | 1 % | 33 % |

"The Pi is our reference point, and it has real advantages. Many times more processing power gives
it three times faster detection and twice the frame rate, so it updates more often and keeps the
face better centred: 95 % of the time in the central box, against 76 %. On these two clips it also
detects more frames, 99 % against 93 %.

But the two videos show the conditions were very different. In the Pi's clip the face stays mostly
still and looks straight at the camera, and as you saw, the moment the face turns away, the Pi's
detection drops out. A Haar cascade only knows frontal faces. In its other clip, with normal
movement, it detected only 60 % of frames.

The ESP32 had the much harder test. It moved a third of the time faster than 35 degrees per second,
where the Pi's clip barely did at all, and it still found the face in 93 % of frames. As the movement increased, its detection held up; what fell behind was centring, because at
18 frames per second the face moves further between updates. So its limit is speed, not detection."

---

## 4. Conclusion (≈ 30 s)

**[Slide]** Two columns. Pi: fast, well centred, frontal faces only. ESP32-S3: 3× slower, kept up with much more movement, 190 KB on a few-dollar chip.

"In short: the Pi is faster and centres better, but its classical detector fails as soon as the face
turns. The ESP32-S3 is three times slower, yet its small quantized network kept tracking a face that
moved far more, on a chip with a tiny fraction of the Pi's resources. Both baselines now run and are
measured the same way, which gives every future model a fair reference on each side."

---

## 5. Future work (≈ 60 s)

**[Slide]** *Next steps*. Three tests · next models · methods · robot improvements.

"Next, we extend these measurements so every model is compared on identical terms, in three tests.

First, we replay the same recorded frames and labelled images to every model, with the camera still,
to measure accuracy, speed and memory on identical pixels.

Second, a virtual gimbal: each model steers a window cut from one wide video, and a slow model skips
frames just as it would on the robot. That compares tracking with nothing else changing.

Third, the real robot follows a face moving along scripted paths on a monitor, repeated several
times, so we know exactly where the face was.

The models we'll test are Espressif's larger detectors, small public face detectors like BlazeFace,
YuNet and SCRFD, and finally our own small model. For each, we'll compare quantization methods,
input resolutions and where the weights are stored in memory. On the robot, a Kalman filter to
predict the face's movement should help the ESP32 with its lower frame rate.

That's our progress so far. Thank you."

---

## Notes for the presenter (not spoken)

- **Where the numbers come from.** Pi: best clip only. ESP32: the latest recording only. The 60 % in section 3 is the Pi's other clip, given as context; cut that sentence if you want to rely on the two clips alone.
- **"Drops as soon as the face turns away"** is what you can see in the Pi video. Point back to it if the demo is right before this section.
- **Mount speed** is how fast pan and tilt changed over the previous 0.3 s. The Pi's best clip has only 4 frames above 35 °/s, so the two can't be compared directly at high speed. Between 20 and 35 °/s the Pi detected 99 % and the ESP32 85 %, with similar centring.
- **The ESP32 video** shows only about every second frame (the stream can't keep up), but the metrics come from every frame (208 of 208).
- **Not measured yet:** accuracy on labelled images, memory on the Pi, and capture-to-servo delay on the ESP32. The ESP32 kept at least 189 KB of on-chip RAM free.
- **Fill in:** presenter name.
