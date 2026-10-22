
from pathlib import Path
import csv
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"

MODEL_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (128, 128)
MIN_IMAGES_PER_CLASS = 50

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------- FEATURE EXTRACTION ----------------

def extract_features(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.resize(image, IMAGE_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.astype(np.float32)


# ---------------- LOAD DATASET ----------------

def load_dataset():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}"
        )

    class_dirs = sorted(
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    if len(class_dirs) != 10:
        raise ValueError(
            f"Expected exactly 10 class folders, found {len(class_dirs)}."
        )

    features = []
    labels = []

    for class_dir in class_dirs:
        image_paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        print(f"{class_dir.name}: {len(image_paths)} images")

        if len(image_paths) < MIN_IMAGES_PER_CLASS:
            raise ValueError(
                f"{class_dir.name} has only {len(image_paths)} images. "
                f"At least {MIN_IMAGES_PER_CLASS} are required."
            )

        for path in image_paths:
            try:
                feature_vector = extract_features(path)
                features.append(feature_vector)
                labels.append(class_dir.name)
            except (ValueError, cv2.error) as error:
                print(f"Skipping {path.name}: {error}")

    if not features:
        raise ValueError("No valid images found in the dataset.")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)

    # Ensure every class still has enough valid images
    for class_name in np.unique(y):
        count = int(np.sum(y == class_name))
        if count < MIN_IMAGES_PER_CLASS:
            raise ValueError(
                f"Only {count} valid images remain for {class_name}."
            )

    return X, y


# ---------------- SIX CLASSIFICATION MODELS ----------------

def create_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42)
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Naive Bayes": GaussianNB(),

        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5)
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "Support Vector Classifier": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf")
        )
    }


# ---------------- TRAIN AND EVALUATE ----------------

def main():
    print("\nLoading images and extracting HOG features...")

    X, y = load_dataset()

    class_names = sorted(np.unique(y).tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\nTotal images: {len(X)}")
    print(f"Training images: {len(X_train)}")
    print(f"Testing images: {len(X_test)}")
    print(f"Classes: {class_names}")

    models = create_models()
    results = []
    best_accuracy = -1
    best_model_name = None
    best_model = None

    report_path = EVAL_DIR / "classification_reports.txt"

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("OBJECT CLASSIFICATION EVALUATION\n")
        report_file.write("=" * 50 + "\n")

        for model_name, model in models.items():
            print(f"\nTraining: {model_name}")

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)

            results.append({
                "Model": model_name,
                "Accuracy": round(accuracy * 100, 2)
            })

            print(f"Test accuracy: {accuracy * 100:.2f}%")

            report_file.write(f"\n\nMODEL: {model_name}\n")
            report_file.write("-" * 50 + "\n")
            report_file.write(
                f"Accuracy: {accuracy * 100:.2f}%\n\n"
            )
            report_file.write(
                classification_report(
                    y_test,
                    predictions,
                    labels=class_names,
                    zero_division=0
                )
            )

            # Save a confusion matrix for each classifier
            matrix = confusion_matrix(
                y_test,
                predictions,
                labels=class_names
            )

            display = ConfusionMatrixDisplay(
                confusion_matrix=matrix,
                display_labels=class_names
            )

            fig, ax = plt.subplots(figsize=(10, 8))
            display.plot(
                ax=ax,
                xticks_rotation=45,
                cmap="Blues",
                colorbar=False
            )
            ax.set_title(f"Confusion Matrix - {model_name}")
            fig.tight_layout()

            safe_name = model_name.lower().replace(" ", "_")
            fig.savefig(
                EVAL_DIR / f"{safe_name}_confusion_matrix.png",
                dpi=150
            )
            plt.close(fig)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
                best_model = model

    # Save comparison table
    csv_path = EVAL_DIR / "model_comparison.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Model", "Accuracy"]
        )
        writer.writeheader()
        writer.writerows(results)

    # Save best model and metadata for live camera prediction
    joblib.dump(
        {
            "model": best_model,
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "feature_type": "HOG",
            "model_name": best_model_name
        },
        MODEL_DIR / "best_model.joblib"
    )

    # Print models sorted by accuracy
    results.sort(key=lambda item: item["Accuracy"], reverse=True)

    print("\n========== MODEL COMPARISON ==========")
    for result in results:
        print(f"{result['Model']}: {result['Accuracy']:.2f}%")

    print(f"\nBest model: {best_model_name}")
    print(f"Best test accuracy: {best_accuracy * 100:.2f}%")
    print(f"Reports saved in: {EVAL_DIR}")
    print(f"Trained model saved in: {MODEL_DIR / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
