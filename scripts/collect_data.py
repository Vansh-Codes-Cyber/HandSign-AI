import os
from pathlib import Path
import mediapipe as mp
import cv2

# CONFIGURATION

# Resolve relative to the project root (parent of this scripts/ folder) so
# this works regardless of the directory it's run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset" / "raw"
NUMBER_OF_CLASSES = 5
IMAGES_PER_CLASS = 500
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
BOUNDING_BOX_PADDING = 20
MIN_HAND_SIZE = 120

# INITIALIZE MEDIAPIPE

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

# CREATE DATASET DIRECTORY

os.makedirs(DATA_DIR, exist_ok=True)

# START CAMERA

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    hands.close()
    raise SystemExit(
        "Unable to access webcam. Check it's connected and not in use elsewhere."
    )

# READY SCREEN


def wait_for_user(class_name):

    while True:
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

        cv2.putText(
            frame,
            f"Press Q to start collecting {class_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Dataset Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# DATA COLLECTION


def collect_images(class_name):
    folder = os.path.join(DATA_DIR, class_name)
    os.makedirs(folder, exist_ok=True)
    image_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        save_frame = False
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                display,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            H, W, _ = frame.shape

            xs = [lm.x for lm in landmarks.landmark]
            ys = [lm.y for lm in landmarks.landmark]

            x1 = max(0, int(min(xs) * W) - BOUNDING_BOX_PADDING)
            y1 = max(0, int(min(ys) * H) - BOUNDING_BOX_PADDING)

            x2 = min(W, int(max(xs) * W) + BOUNDING_BOX_PADDING)
            y2 = min(H, int(max(ys) * H) + BOUNDING_BOX_PADDING)

            width = x2 - x1
            height = y2 - y1

            if width > MIN_HAND_SIZE and height > MIN_HAND_SIZE:

                save_frame = True

                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    display,
                    "Good Frame",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 180, 0),
                    2,
                )

            else:

                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 140, 255), 2)

                cv2.putText(
                    display,
                    "Move Hand Closer",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 140, 255),
                    2,
                )

        else:

            cv2.putText(
                display,
                "No Hand Detected",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            display,
            f"{class_name} : {image_count}/{IMAGES_PER_CLASS}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Dataset Collection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("Collection cancelled.")
            break

        if save_frame:

            filename = os.path.join(folder, f"{image_count}.jpg")

            print(f"Saving image {image_count}")

            cv2.imwrite(filename, frame)

            image_count += 1
            cv2.waitKey(20)

        if image_count >= IMAGES_PER_CLASS:
            print(f"{class_name} completed.")
            break


# MAIN


try:

    for i in range(NUMBER_OF_CLASSES):

        class_name = input(f"Enter name for class {i+1}: ").strip()

        if not class_name:
            class_name = str(i)

        print(f"\nCollecting images for {class_name}\n")

        wait_for_user(class_name)

        collect_images(class_name)

finally:

    cap.release()

    cv2.destroyAllWindows()

    hands.close()

    print("\nDataset collection completed successfully.")
