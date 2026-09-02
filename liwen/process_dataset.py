"""Apply Liwen's pipeline to every image in the shared dataset."""

from pathlib import Path
import json
import time

import cv2

from pipeline import preprocess_image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FOLDER = PROJECT_ROOT / "dataset"
OUTPUT_FOLDER = Path(__file__).resolve().parent / "output"
CHECKPOINT_FILE = Path(__file__).resolve().parent / "checkpoint.json"
FAILED_IMAGES_FILE = Path(__file__).resolve().parent / "failed_images.txt"
FEATURES_FILE = Path(__file__).resolve().parent / "surface_features.jsonl"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_files():
    return sorted(
        path for path in DATASET_FOLDER.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_checkpoint():
    if not CHECKPOINT_FILE.exists():
        return {"last_completed_index": 0, "processed_count": 0, "failed_count": 0}
    try:
        return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"[WARNING] Could not load checkpoint: {error}")
        return {"last_completed_index": 0, "processed_count": 0, "failed_count": 0}


def save_checkpoint(last_completed_index, processed_count, failed_count):
    checkpoint = {
        "last_completed_index": last_completed_index,
        "processed_count": processed_count,
        "failed_count": failed_count,
    }
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=4), encoding="utf-8")


def process_one_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("OpenCV could not read image.")

    processed_image, features = preprocess_image(image)
    relative_path = image_path.relative_to(DATASET_FOLDER)
    output_path = OUTPUT_FOLDER / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), processed_image):
        raise IOError("OpenCV failed to save output image.")

    record = {"image": relative_path.as_posix(), **features}
    with FEATURES_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
    return output_path


def process_dataset():
    image_files = get_image_files()
    total = len(image_files)
    print("\n" + "=" * 70)
    print("LIWEN FULL DATASET PREPROCESSING")
    print("=" * 70)
    print(f"\nDataset:\n{DATASET_FOLDER}\n\nOutput:\n{OUTPUT_FOLDER}")
    print(f"\nTotal images found: {total:,}")
    if not image_files:
        print("\n[ERROR] No dataset images found.")
        return

    checkpoint = load_checkpoint()
    completed = int(checkpoint.get("last_completed_index", 0))
    processed = int(checkpoint.get("processed_count", 0))
    failed = int(checkpoint.get("failed_count", 0))
    start_time = time.time()

    try:
        for offset, image_path in enumerate(image_files[completed:], start=1):
            current = completed + offset
            try:
                process_one_image(image_path)
                processed += 1
                status = "OK"
            except Exception as error:
                failed += 1
                status = "FAILED"
                with FAILED_IMAGES_FILE.open("a", encoding="utf-8") as file:
                    file.write(f"{image_path} | {error}\n")

            save_checkpoint(current, processed, failed)
            relative_path = image_path.relative_to(DATASET_FOLDER)
            print(f"[{current:,}/{total:,}] {status:<6} {relative_path}")
    except KeyboardInterrupt:
        print("\nProcessing stopped. Checkpoint saved; run this file again to resume.")
        return

    elapsed_minutes = (time.time() - start_time) / 60
    print("\n" + "=" * 70)
    print("LIWEN PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nSuccessful: {processed:,}\nFailed: {failed:,}\nTotal: {total:,}")
    print(f"Elapsed time: {elapsed_minutes:.2f} minutes")
    print(f"\nProcessed dataset:\n{OUTPUT_FOLDER}")


if __name__ == "__main__":
    process_dataset()
