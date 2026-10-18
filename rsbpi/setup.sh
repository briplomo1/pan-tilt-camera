#!/usr/bin/env bash
# One-time setup for the pan-tilt face tracker on Raspberry Pi OS (64-bit).
# Run on the Pi:  bash setup.sh
set -e

echo "==> Installing system packages (OpenCV, camera, GPIO)..."
sudo apt update
sudo apt install -y \
  python3-opencv opencv-data \
  python3-picamera2 \
  python3-gpiozero python3-lgpio \
  python3-venv python3-pip \
  i2c-tools

echo "==> Enabling I2C (needed for PCA9685 servo boards)..."
sudo raspi-config nonint do_i2c 0

echo "==> Creating virtual environment at ~/tracker-venv..."
# --system-site-packages lets the venv use the apt-installed OpenCV and Picamera2
python3 -m venv --system-site-packages ~/tracker-venv
~/tracker-venv/bin/pip install --upgrade pip
~/tracker-venv/bin/pip install adafruit-circuitpython-servokit

echo
echo "==> Checking for a camera..."
rpicam-hello --list-cameras 2>/dev/null || libcamera-hello --list-cameras 2>/dev/null || true
ls /dev/video* 2>/dev/null | head -n 3 || true

echo
echo "Done. Test with:"
echo "  ~/tracker-venv/bin/python face_tracker.py --servos none"
echo "Then open http://$(hostname).local:8000 in your laptop's browser."
