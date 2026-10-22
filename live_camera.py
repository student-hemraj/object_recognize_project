
# Live Object Classification using OpenCV

from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog


# ----------------------------------
# 1. Load the trained model
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Please run train_model.py first."
    )

saved = joblib.load(MODEL_PATH)

model = saved["model"]
class_names = saved["class_names"]
image_size = tuple(saved["image_size"])

print("Loaded model:", saved.get("model_name", "Unknown"))
print("Available classes:", class_names)


# ----------------------------------
# 2. Extract HOG features
# ----------------------------------

def extract_camera_features(frame):
    resized = cv2.resize(frame, image_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.reshape(1, -1)


# ----------------------------------
# 3. Open webcam
# ----------------------------------

camera = None

for camera_index in (0, 1):
    print(f"Trying webcam index {camera_index}...")

    candidate = cv2.VideoCapture(
        camera_index,
        cv2.CAP_DSHOW
    )

    if candidate.isOpened():
        success, test_frame = candidate.read()

        if success and test_frame is not None:
            camera = candidate
            print("Webcam opened successfully!")
            break

    candidate.release()

if camera is None:
    raise RuntimeError(
        "Cannot open webcam or read frames.\n"
        "Close other camera apps and check Windows camera permissions."
    )


# ----------------------------------
# 4. Live object classification
# ----------------------------------

try:
    while True:
        success, frame = camera.read()

        if not success or frame is None:
            print("Could not read camera frame.")
            break

        try:
            features = extract_camera_features(frame)

            prediction = model.predict(features)[0]
            label = f"Object: {prediction}"

            # Display model score if available
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                confidence = float(np.max(probabilities)) * 100
                label += f" | Score: {confidence:.1f}%"

            color = (0, 255, 0)

        except (ValueError, cv2.error) as error:
            label = "Prediction error"
            color = (0, 0, 255)
            print("Prediction error:", error)

        cv2.putText(
            frame,
            label,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "Live Object Classification - Press Q to quit",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")
