# Real-Time Driver Drowsiness Detector

A lightweight computer vision application built in Python that detects real-time driver fatigue and sustained eye closure using standard webcam feeds without requiring specialized biometric hardware.

## Features
* **Spatial Upper-Face ROI:** Restricts eye feature tracking to the upper region of detected face bounding boxes to optimize performance.
* **Temporal Frame Threshold:** Uses consecutive frame tracking to distinguish natural blinks from fatigue or microsleep onset.
* **Multi-Modal Alerts:** Triggers a visual HUD screen flash and a system audio alarm when eye closure exceeds safety thresholds.

## Tech Stack
* **Language:** Python 3
* **Libraries:** OpenCV (`opencv-python`), NumPy

## How to Run
1. Install dependencies:
   `pip install -r requirements.txt`
2. Run script:
   `python drowsiness_detector.py`
