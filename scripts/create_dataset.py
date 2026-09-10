import os
import pickle
from pathlib import Path

import cv2
import mediapipe as mp

# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset" / "raw"
OUTPUT_FILE = PROJECT_ROOT / "dataset" / "data.pickle"

# ==========================================================
# INITIALIZE MEDIAPIPE
# ==========================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
)

# ==========================================================
# DATA STORAGE
# ==========================================================

data = []
labels = []

# ==========================================================
# READ DATASET
# ==========================================================

for class_name in sorted(os.listdir(DATA_DIR)):

    class_path = os.path.join(DATA_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"Processing class: {class_name}")

    for image_name in os.listdir(class_path):

        image_path = os.path.join(class_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            continue

        hand = results.multi_hand_landmarks[0]

        xs = []
        ys = []
        features = []

        for lm in hand.landmark:
            xs.append(lm.x)
            ys.append(lm.y)

        min_x = min(xs)
        min_y = min(ys)

        for lm in hand.landmark:
            features.append(lm.x - min_x)
            features.append(lm.y - min_y)

        if len(features) == 42:
            data.append(features)
            labels.append(class_name)

# ==========================================================
# SAVE DATASET
# ==========================================================

if not data:
    hands.close()
    raise SystemExit(
        f"No hand landmarks found in any image under {DATA_DIR}. "
        "Run scripts/collect_data.py first, or check the images are readable."
    )

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

hands.close()

print("\n====================================")
print("Dataset created successfully!")
print("Total Samples :", len(data))
print("Features      :", len(data[0]) if data else 0)
print("Classes       :", len(set(labels)))
print("Saved File    :", OUTPUT_FILE)
print("====================================")
