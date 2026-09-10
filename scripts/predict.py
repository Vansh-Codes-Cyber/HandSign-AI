import joblib
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_sign_model.joblib"

# -----------------------------
# Load trained model
# -----------------------------
if not MODEL_PATH.exists():
    raise SystemExit(
        f"Model not found at {MODEL_PATH}. Run scripts/train_model.py first."
    )

model_dict = joblib.load(MODEL_PATH)
model = model_dict["model"]
label_encoder = model_dict["label_encoder"]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit(
        "Unable to access webcam. Check it's connected and not in use elsewhere."
    )

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

screenshot_count = 0

while True:
    data_aux = []
    x_ = []
    y_ = []
    message = "Show one hand gesture"
    color = (0, 0, 255)

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        for landmark in hand_landmarks.landmark:
            x_.append(landmark.x)
            y_.append(landmark.y)

        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))

        if len(data_aux) == 42:
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            sample = np.asarray(data_aux).reshape(1, -1)
            prediction = model.predict(sample)
            predicted_character = label_encoder.inverse_transform(prediction)[0]

            # Confidence score if available
            confidence_text = ""
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(sample)[0]
                confidence = float(np.max(probs)) * 100
                confidence_text = f" | Confidence: {confidence:.1f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
            cv2.putText(
                frame,
                f"Prediction: {predicted_character}{confidence_text}",
                (max(10, x1), max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            box_w = x2 - x1
            box_h = y2 - y1
            if box_w < 80 or box_h < 80:
                message = "Move hand closer to the camera"
                color = (0, 140, 255)
            else:
                message = "Hand detected"
                color = (0, 180, 0)

    cv2.putText(
        frame, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Press S to save screenshot | Press Q to quit",
        (20, H - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (50, 50, 50),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Sign Language Recognition System", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        screenshots_dir = PROJECT_ROOT / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        filename = screenshots_dir / f"prediction_{screenshot_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved as {filename}")
        screenshot_count += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
