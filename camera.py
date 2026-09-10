import os
import sys
from pathlib import Path

# Hide TensorFlow/MediaPipe logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st

# Resolve paths relative to this file, not the process's working directory,
# so the app works no matter where `streamlit run` is invoked from.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_sign_model.joblib"
OUTPUT_FILE = BASE_DIR / "output" / "recognized_signs.txt"


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the trained model + label encoder once per session."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run scripts/train_model.py first to generate it."
        )
    model_data = joblib.load(MODEL_PATH)
    return model_data["model"], model_data["label_encoder"]


@st.cache_resource(show_spinner=False)
def load_hands():
    """Create the MediaPipe Hands detector once per session."""
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_hands, mp_draw, hands


@st.cache_resource(show_spinner=False)
def get_camera():
    """Open the webcam once per session with a lightweight resolution.

    On Windows, OpenCV's default backend (MSMF) is slow to open and adds
    noticeable per-frame latency for most webcams, so we explicitly request
    DirectShow (CAP_DSHOW) there. On Linux/macOS the default backend
    (V4L2 / AVFoundation) is fine, and CAP_DSHOW doesn't exist, so we leave
    the backend unspecified.
    """
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam (device 0). Check that a camera is "
            "connected and not in use by another application."
        )

    # MJPG capture is much faster than the raw formats many webcams default
    # to (e.g. YUY2), which is the other big source of lag at 640x480/30fps.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def prepare_frame(frame):
    """Resize large webcam frames before MediaPipe processing."""
    height, width = frame.shape[:2]
    max_width = 640
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(
            frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA
        )
    return frame


def process_frame(frame, last_saved_prediction):
    """
    Run hand detection + prediction on a single BGR frame.

    Returns: (annotated_frame_rgb, prediction, confidence, status, last_saved_prediction)
    """
    model, label_encoder = load_model()
    mp_hands, mp_draw, hands = load_hands()

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction = "--"
    confidence = 0.0
    status = "No Hand Detected"

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]
        min_x, min_y = min(xs), min(ys)

        features = []
        for lm in hand.landmark:
            features.append(lm.x - min_x)
            features.append(lm.y - min_y)

        sample = np.asarray(features).reshape(1, -1)
        prediction_index = model.predict(sample)[0]
        prediction = label_encoder.inverse_transform([prediction_index])[0]

        probabilities = model.predict_proba(sample)[0]
        confidence = float(np.max(probabilities) * 100)
        status = "Hand Detected"

        if prediction != last_saved_prediction:
            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_FILE, "a") as f:
                f.write(f"{prediction}, {confidence:.1f}%\n")
            last_saved_prediction = prediction

        cv2.putText(
            frame,
            f"{prediction} ({confidence:.1f}%)",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, prediction, confidence, status, last_saved_prediction
