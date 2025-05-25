# Arabic-Sign-Language-Translation-using-Deep-Learning

This project focuses on building a real-time sign language gesture recognition system using the [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html) model for keypoint extraction and training deep learning models (LSTM, GRU, Transformer) for gesture classification.

---

## 📌 Project Overview

- **Input**: Live webcam feed capturing hand, face, and body landmarks.
- **Processing**: Extraction of keypoints (face, pose, left/right hand) via MediaPipe.
- **Output**: Trained deep learning model that classifies gestures into predefined actions.

---

## 🧠 Project Pipeline

### 1. Dependencies Installation

Install the required packages:

```bash
pip install opencv-python mediapipe matplotlib numpy tensorflow tensorflow-addons
