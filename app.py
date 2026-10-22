
from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from io import BytesIO

app = Flask(__name__)

# Project and dataset paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

# Ten object classes
OBJECTS = [
    "pen",
    "pencil",
    "bottle",
    "scale",
    "eraser",
    "notebook",
    "mobile",
    "comb",
    "adapter",
    "school_bag"
]

IMAGES_PER_OBJECT = 50
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def create_dataset_folders():
    """Create dataset folders without deleting existing images."""

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for obj in OBJECTS:
        folder = DATASET_DIR / obj

        # Stop if a file exists where a folder is expected
        if folder.exists() and not folder.is_dir():
            raise RuntimeError(
                f"Cannot create dataset folder: {folder}\n"
                "A file with this name already exists. "
                "Rename that file and run the program again."
            )

        folder.mkdir(parents=True, exist_ok=True)


def get_image_count(obj):
    """Count supported image files in one object folder."""

    folder = DATASET_DIR / obj
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if not folder.is_dir():
        return 0

    return sum(
        1 for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def get_progress():
    """Return the number of collected images for each object."""

    return {
        obj: get_image_count(obj)
        for obj in OBJECTS
    }


@app.route("/")
def index():
    """Show the dataset collection page."""

    return render_template(
        "index.html",
        objects=OBJECTS,
        progress=get_progress()
    )


@app.route("/progress")
def progress():
    """Return current dataset collection progress."""

    return jsonify({
        "progress": get_progress(),
        "required_per_object": IMAGES_PER_OBJECT
    })


@app.route("/capture", methods=["POST"])
def capture():
    """Receive a photo and save it in the selected class folder."""

    obj = request.form.get("object", "").strip().lower()
    photo = request.files.get("photo")

    if obj not in OBJECTS:
        return jsonify({
            "error": "Please select a valid object."
        }), 400

    if photo is None or not photo.filename:
        return jsonify({
            "error": "Please capture or select a photo."
        }), 400

    # Read at most 10 MB plus one byte to enforce the size limit
    image_bytes = photo.stream.read(MAX_UPLOAD_BYTES + 1)

    if not image_bytes:
        return jsonify({
            "error": "The uploaded file is empty."
        }), 400

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        return jsonify({
            "error": "Image must be smaller than 10 MB."
        }), 400

    try:
        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            uploaded_image.verify()

        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            image = uploaded_image.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": "Invalid or unsupported image file."
        }), 400

    folder = DATASET_DIR / obj

    # Check the folder still exists and is a directory
    if not folder.is_dir():
        return jsonify({
            "error": f"Dataset folder for {obj} is missing."
        }), 500

    # Recount all supported image formats
    current_count = get_image_count(obj)

    if current_count >= IMAGES_PER_OBJECT:
        return jsonify({
            "error": f"{obj} already has {IMAGES_PER_OBJECT} images.",
            "progress": get_progress()
        }), 400

    # Choose an unused filename so existing images are not overwritten
    next_number = 1

    while (folder / f"{obj}_{next_number:03d}.jpg").exists():
        next_number += 1

    filename = f"{obj}_{next_number:03d}.jpg"
    save_path = folder / filename

    try:
        image.save(save_path, format="JPEG", quality=90)
    except OSError as error:
        return jsonify({
            "error": f"Could not save image: {error}"
        }), 500

    return jsonify({
        "message": f"Photo saved successfully: {filename}",
        "progress": get_progress(),
        "saved_path": str(save_path)
    })


if __name__ == "__main__":
    try:
        create_dataset_folders()
    except (OSError, RuntimeError) as error:
        print("\nDATASET FOLDER ERROR:")
        print(error)
        raise SystemExit(1)

    print("Dataset directory:", DATASET_DIR)
    print("Object classes:", ", ".join(OBJECTS))
    print("Required images per class:", IMAGES_PER_OBJECT)
    print("Open on this laptop: http://127.0.0.1:5000")
    print("For phone access, use your laptop's Wi-Fi IPv4 address.")
    
    app.run(host="0.0.0.0", port=5000, debug=False)
