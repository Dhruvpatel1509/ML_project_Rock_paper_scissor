
# Rock-Paper-Scissors Hand Gesture Recognition

> A real-time Rock-Paper-Scissors game controlled entirely via hand gestures captured through a webcam — where the robot always wins.

---
<img width="784" height="571" alt="image" src="https://github.com/user-attachments/assets/fd1c42bd-751d-4835-872d-a78f1c3aa6da" />
<img width="835" height="536" alt="image" src="https://github.com/user-attachments/assets/b017e9d7-1643-468b-a936-c68695173d01" />

## Overview

This project uses computer vision and machine learning to detect and classify hand gestures (Rock, Paper, Scissors) in real time. The game is intentionally rigged: the robot always counters the player's move.

The pipeline runs at ~20 FPS and processes live webcam frames through MediaPipe hand landmark detection, feeding normalized keypoint data into trained MLP classifiers.

---

## Architecture

```
Webcam (OpenCV)
    │
    ▼
MediaPipe HandLandmarker (hand_landmarker.task)
    │  21 landmarks per frame (x, y coords)
    ▼
Pre-processing (relative coords + normalization)
    ├──► KeyPointClassifier (MLP)        → Static gesture (Rock / Paper / Scissors)
    └──► PointHistoryClassifier (MLP)    → Dynamic gesture (finger movement trajectory)
    │
    ▼
Game Logic + OpenCV Overlay (app.py)
```
<img width="681" height="511" alt="image" src="https://github.com/user-attachments/assets/42348378-669b-43da-bf47-6fc25f3b0ebf" />

---

## Project Structure

```
├── app.py                                      # Main entry point
├── hand_landmarker.task                        # MediaPipe model (auto-downloaded)
├── model/
│   ├── keypoint_classifier/
│   │   ├── keypoint_classifier.py              # MLP for static gestures
│   │   ├── keypoint_classifier.tflite          # Trained TFLite model
│   │   ├── keypoint_classifier_label.csv       # Class labels
│   │   └── keypoint.csv                        # Training data
│   └── point_history_classifier/
│       ├── point_history_classifier.py         # MLP for dynamic gestures
│       ├── point_history_classifier.tflite     # Trained TFLite model
│       ├── point_history_classifier_label.csv  # Class labels
│       └── point_history.csv                   # Training data
└── utils/
    └── cvfpscalc.py                            # FPS calculation utility
```

---

## Models

### KeyPointClassifier — Static Gesture Recognition

- **Input:** Flattened 21 hand landmark coordinates (42 values), normalized relative to the wrist
- **Architecture:** Fully connected MLP
  ```
  Input(42) → Dropout(0.2) → Dense(20, ReLU) → Dropout(0.4) → Dense(10, ReLU) → Dense(N, Softmax)
  ```
- **Output:** Rock / Paper / Scissors (and other configurable gestures)
- **Training:** Supervised learning on labeled keypoint CSV data

### PointHistoryClassifier — Dynamic Gesture Recognition

- **Input:** Fingertip (index finger tip, landmark 8) movement trajectory over the last 16 frames
- **Architecture:** Second MLP trained on temporal motion patterns
- **Output:** Gesture motion class (e.g., Stop, Clockwise, etc.)
- **Purpose:** Stabilizes predictions when gestures involve movement across frames
<img width="779" height="255" alt="image" src="https://github.com/user-attachments/assets/e58243db-cbad-4109-881e-22b28a672e05" />

---

## Game Logic

The robot cheats deterministically:

| Player | Robot |
|--------|-------|
| Rock | Paper |
| Paper | Scissors |
| Scissors | Rock |

The result is displayed in real time in an overlay panel on the right side of the video feed.

---

## Requirements

```
Python >= 3.8
opencv-python
mediapipe >= 0.10.0
numpy
tensorflow (for training; TFLite runtime used for inference)
```

Install dependencies:

```bash
pip install opencv-python mediapipe numpy tensorflow
```

---

## Usage

```bash
python app.py
```

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | `0` | Camera device index |
| `--width` | `960` | Capture width |
| `--height` | `540` | Capture height |
| `--min_detection_confidence` | `0.7` | MediaPipe detection threshold |
| `--min_tracking_confidence` | `0.5` | MediaPipe tracking threshold |
| `--use_static_image_mode` | `False` | Disable tracking (use per-frame detection) |

The `hand_landmarker.task` model is downloaded automatically on first run.

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `ESC` | Exit |
| `n` | Normal mode |
| `k` | Keypoint logging mode |
| `h` | Point history logging mode |
| `0–9` | Select gesture label (in logging mode) |

---

## Training Custom Gestures

1. Launch `app.py` and press `k` to enter keypoint logging mode
2. Press a number key (`0–9`) to assign a label
3. Show your hand gesture — keypoints are logged to `model/keypoint_classifier/keypoint.csv`
4. Retrain the classifier using the updated CSV
5. Export the new model as a `.tflite` file and replace the existing one
6. Update `keypoint_classifier_label.csv` with the new label name

The same process applies to dynamic gestures using `h` mode and the point history classifier.

---

## How Hand Landmarks Work

MediaPipe extracts 21 landmarks per hand. Each landmark is converted to pixel coordinates, then normalized relative to landmark 0 (wrist) and scaled by the max absolute value in the vector.

<img width="549" height="460" alt="image" src="https://github.com/user-attachments/assets/1295a41e-dabb-43cd-a700-9ed49ef5d7bf" />
<img width="589" height="338" alt="image" src="https://github.com/user-attachments/assets/8dbb5278-b370-4fa1-bc66-bd72f7bb27b3" />


```

Landmark indices used in gesture classification:
  0  = WRIST          8  = INDEX_FINGER_TIP    16 = RING_FINGER_TIP
  4  = THUMB_TIP      12 = MIDDLE_FINGER_TIP   20 = PINKY_TIP
  ...
```

---

## Notes

- The project uses the MediaPipe Tasks API (`mediapipe >= 0.10`) with `VIDEO` running mode for webcam input
- Frame timestamps are approximated at 33ms intervals (~30 FPS)
- Point history uses a sliding window deque of length 16 frames
- The game panel is rendered as a semi-transparent overlay on the right 300px of the frame
