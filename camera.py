import os
from pathlib import Path
import threading

# Hide TensorFlow/MediaPipe logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_sign_model.joblib"
OUTPUT_FILE = BASE_DIR / "output" / "recognized_signs.txt"


def load_model():
    """Load the trained XGBoost model and label encoder."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Make sure models/hand_sign_model.joblib is in the repository."
        )

    model_data = joblib.load(MODEL_PATH)
    return model_data["model"], model_data["label_encoder"]


def create_hands():
    """Create a MediaPipe Hands detector for this WebRTC processor."""
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_hands, mp_draw, hands


def prepare_frame(frame):
    """Keep processing fast by limiting very large camera frames."""
    height, width = frame.shape[:2]
    max_width = 640

    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(
            frame,
            (max_width, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    return frame


class LiveVideoProcessor(VideoProcessorBase):
    """Process each browser webcam frame continuously."""

    def __init__(self):
        self.model, self.label_encoder = load_model()
        self.mp_hands, self.mp_draw, self.hands = create_hands()

        self.last_saved_prediction = None

        # These values are read by the Streamlit UI thread.
        self.lock = threading.Lock()
        self.latest_prediction = "--"
        self.latest_confidence = 0.0
        self.latest_status = "No Hand Detected"

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = prepare_frame(image)

        # Mirror the webcam like a normal selfie camera.
        image = cv2.flip(image, 1)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        prediction = "--"
        confidence = 0.0
        status = "No Hand Detected"

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            self.mp_draw.draw_landmarks(
                image,
                hand,
                self.mp_hands.HAND_CONNECTIONS,
            )

            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]
            min_x, min_y = min(xs), min(ys)

            features = []
            for lm in hand.landmark:
                features.append(lm.x - min_x)
                features.append(lm.y - min_y)

            sample = np.asarray(features, dtype=np.float32).reshape(1, -1)

            prediction_index = self.model.predict(sample)[0]
            prediction = self.label_encoder.inverse_transform([prediction_index])[0]

            probabilities = self.model.predict_proba(sample)[0]
            confidence = float(np.max(probabilities) * 100)
            status = "Hand Detected"

            # Save a prediction only when the recognized sign changes.
            if prediction != self.last_saved_prediction:
                OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{prediction}, {confidence:.1f}%\n")
                self.last_saved_prediction = prediction

            cv2.putText(
                image,
                f"{prediction} ({confidence:.1f}%)",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        with self.lock:
            self.latest_prediction = str(prediction)
            self.latest_confidence = confidence
            self.latest_status = status

        # WebRTC expects a video frame, not a Streamlit image.
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def start_live_camera():
    """Start the browser webcam through WebRTC."""
    return webrtc_streamer(
        key="handsign-live-camera-v2",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveVideoProcessor,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
            ]
        },
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        video_html_attrs={
            "autoPlay": True,
            "controls": False,
            "muted": True,
            "playsInline": True,
            "style": {"width": "100%"},
        },
        async_processing=True,
    )
