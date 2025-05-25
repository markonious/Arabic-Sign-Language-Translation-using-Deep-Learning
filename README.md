# Arabic-Sign-Language-Translation-using-Deep-Learning
# 🤟 Sign Language Gesture Recognition using MediaPipe and Deep Learning

A real-time sign language gesture recognition system using [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html) for keypoint extraction and deep learning models (LSTM, GRU, Transformer) for classification.

---

## 📌 Overview

This project detects and classifies sign language gestures in real time from a webcam feed using MediaPipe and TensorFlow. It extracts pose, hand, and face landmarks, then feeds them into a trained sequence model to classify the performed gesture.

---

## 🧠 Objectives

- Detect 3D keypoints from webcam using MediaPipe Holistic
- Collect and process time-series data for multiple sign gestures
- Train deep learning models (LSTM, GRU, Transformer) for classification
- Perform real-time gesture recognition

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **MediaPipe Holistic**
- **NumPy / Matplotlib**
- **TensorBoard**
- **TensorFlow Addons (for Transformer model)**

---

## 🗃️ Data Collection

### ➕ Actions Supported
The system recognizes the following sign gestures:

- `computers`
- `information`
- `student`
- `mansoura`
- `iloveyou`
- `in`
- `screen`

You can easily add more by modifying the `actions` list and collecting new data.

### 📦 Data Structure

Collected data is stored in a nested folder structure as follows:


pip install opencv-python mediapipe matplotlib numpy tensorflow tensorflow-addons

import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Holistic model
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=True) as holistic:

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        cv2.imshow('MediaPipe Holistic', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

