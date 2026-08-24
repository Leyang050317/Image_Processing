from pathlib import Path
import json
import time
import cv2

from pipeline import preprocess_image


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Original Kaggle dataset
DATASET_FOLDER = PROJECT_ROOT / "dataset"

# HY processed dataset
OUTPUT_FOLDER = Path(__file__).resolve().parent / "output"

# Runtime checkpoint
CHECKPOINT_FILE = Path(__file__).resolve().parent / "checkpoint.json"

# Failed image record
FAILED_FILE = Path(__file__).resolve().parent / "failed_images.txt"


# =========================================================
# Supported formats
# =========================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# =========================================================
# Checkpoint handling
# =========================================================

def load_checkpoint():
    """
    Load the previous processing checkpoint.

    Returns
    -------
    dict
        Example:
        {
            "last_completed_index": 1250,
            "processed": 1249,
            "failed": 1
        }
    """

    if not CHECKPOINT_FILE.exists():
        return {
            "last_completed_index": 0,
            "processed": 0,
            "failed": 0
        }

    try:
        with open(
            CHECKPOINT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (json.JSONDecodeError, OSError):
        print(
            "[WARNING] Checkpoint could not be read. "
            "Starting from the beginning."
        )

        return {
            "last_completed_index": 0,
            "processed": 0,
            "failed": 0
        }


def save_checkpoint(
    index,
    processed,
    failed
):
    """
    Immediately save progress after every processed image.
    """

    checkpoint = {
        "last_completed_index": index,
        "processed": processed,
        "failed": failed
    }

    # Write temporary checkpoint first
    temp_file = CHECKPOINT_FILE.with_suffix(".tmp")

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            checkpoint,
            file,
            indent=4
        )

    # Replace old checkpoint atomically
    temp_file.replace(
        CHECKPOINT_FILE
    )


# =========================================================
# Failure logging
# =========================================================

def log_failed_image(
    image_path,
    error
):
    """
    Store failed image paths without stopping the entire run.
    """

    with open(
        FAILED_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            f"{image_path} | {error}\n"
        )


# =========================================================
# Dataset discovery
# =========================================================

def get_image_files():
    """
    Recursively obtain all supported images.

    Folder structure is preserved later in processed_dataset/.
    """

    image_files = sorted(
        file
        for file in DATASET_FOLDER.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    return image_files


# =========================================================
# Save image
# =========================================================

def save_processed_image(
    image,
    source_path
):
    """
    Save output while maintaining the same directory structure.

    Example:

    dataset/
        train/
            ripe/
                banana001.jpg

    becomes:

    huaiyu/processed_dataset/
        train/
            ripe/
                banana001.jpg
    """

    relative_path = source_path.relative_to(
        DATASET_FOLDER
    )

    destination = OUTPUT_FOLDER / relative_path

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    success = cv2.imwrite(
        str(destination),
        image
    )

    if not success:
        raise IOError(
            f"Unable to save image: {destination}"
        )


# =========================================================
# Time formatting
# =========================================================

def format_time(seconds):
    """
    Convert seconds into HH:MM:SS.
    """

    seconds = int(seconds)

    hours = seconds // 3600

    minutes = (
        seconds % 3600
    ) // 60

    seconds = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# =========================================================
# Main dataset processor
# =========================================================

def process_dataset():

    print("=" * 70)
    print("HY BANANA PREPROCESSING PIPELINE")
    print("=" * 70)

    # -----------------------------------------------------
    # Validate dataset
    # -----------------------------------------------------

    if not DATASET_FOLDER.exists():

        print(
            "\n[ERROR] Dataset folder does not exist:"
        )

        print(
            DATASET_FOLDER
        )

        return


    # -----------------------------------------------------
    # Find images
    # -----------------------------------------------------

    image_files = get_image_files()

    # =========================================================
    # TEMPORARY BALANCED VALIDATION TEST
    # Select up to 25 images from each class folder
    # =========================================================

    TEST_IMAGES_PER_CLASS = 25

    class_groups = {}

    for image_path in image_files:

        class_name = image_path.parent.name

        if class_name not in class_groups:
            class_groups[class_name] = []

        if len(class_groups[class_name]) < TEST_IMAGES_PER_CLASS:
            class_groups[class_name].append(image_path)

    # Combine selected images
    image_files = []

    for class_name, images in sorted(class_groups.items()):
        image_files.extend(images)

        print(
            f"Validation class: {class_name} "
            f"→ {len(images)} image(s)"
        )

    total_images = len(image_files)

    if total_images == 0:

        print(
            "\n[ERROR] No images found inside:"
        )

        print(
            DATASET_FOLDER
        )

        return


    print(
        f"\nDataset:"
        f"\n{DATASET_FOLDER}"
    )

    print(
        f"\nOutput:"
        f"\n{OUTPUT_FOLDER}"
    )

    print(
        f"\nTotal images found: "
        f"{total_images:,}"
    )


    # -----------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------

    checkpoint = load_checkpoint()

    last_completed_index = checkpoint.get(
        "last_completed_index",
        0
    )

    processed_count = checkpoint.get(
        "processed",
        0
    )

    failed_count = checkpoint.get(
        "failed",
        0
    )


    # -----------------------------------------------------
    # Resume status
    # -----------------------------------------------------

    if last_completed_index > 0:

        print(
            "\nCheckpoint found."
        )

        print(
            f"Resuming directly from image "
            f"{last_completed_index + 1:,}"
        )

        print(
            f"Previously processed: "
            f"{processed_count:,}"
        )

        print(
            f"Previously failed: "
            f"{failed_count:,}"
        )

    else:

        print(
            "\nNo checkpoint found."
        )

        print(
            "Starting from image 1."
        )


    # -----------------------------------------------------
    # Already finished
    # -----------------------------------------------------

    if last_completed_index >= total_images:

        print(
            "\nDataset already completely processed."
        )

        return


    # -----------------------------------------------------
    # Only process remaining images
    #
    # IMPORTANT:
    # This is what avoids looping through images 1...1200
    # when resuming at image 1201.
    # -----------------------------------------------------

    remaining_images = image_files[
        last_completed_index:
    ]


    print(
        f"\nRemaining images: "
        f"{len(remaining_images):,}"
    )

    print(
        "\nPress Ctrl+C at any time to stop safely."
    )

    print("=" * 70)


    # -----------------------------------------------------
    # Runtime statistics
    # -----------------------------------------------------

    session_start = time.time()

    session_processed = 0

    session_failed = 0


    try:

        # -------------------------------------------------
        # Process only unfinished images
        # -------------------------------------------------

        for current_index, image_path in enumerate(
            remaining_images,
            start=last_completed_index + 1
        ):

            image_start = time.time()

            try:

                # -----------------------------------------
                # Load
                # -----------------------------------------

                image = cv2.imread(
                    str(image_path)
                )

                if image is None:
                    raise ValueError(
                        "OpenCV could not read image."
                    )


                # -----------------------------------------
                # HY preprocessing pipeline
                # -----------------------------------------

                processed_image, final_mask = (
                    preprocess_image(
                        image
                    )
                )


                # -----------------------------------------
                # Save final CNN-ready image
                # -----------------------------------------

                save_processed_image(
                    processed_image,
                    image_path
                )


                processed_count += 1

                session_processed += 1


                status = "OK"


            except Exception as error:

                failed_count += 1

                session_failed += 1

                status = "FAILED"

                log_failed_image(
                    image_path,
                    error
                )


            # ---------------------------------------------
            # Save checkpoint immediately
            # ---------------------------------------------

            save_checkpoint(
                current_index,
                processed_count,
                failed_count
            )


            # ---------------------------------------------
            # Runtime estimate
            # ---------------------------------------------

            elapsed = (
                time.time()
                - session_start
            )

            completed_this_session = (
                session_processed
                + session_failed
            )

            average_time = (
                elapsed
                / completed_this_session
            )

            remaining_count = (
                total_images
                - current_index
            )

            estimated_remaining = (
                average_time
                * remaining_count
            )


            image_time = (
                time.time()
                - image_start
            )


            # ---------------------------------------------
            # Progress
            # ---------------------------------------------

            print(
                f"[{current_index:,}"
                f"/{total_images:,}] "
                f"{status:<6} | "
                f"{image_path.name} | "
                f"{image_time:.2f}s | "
                f"ETA "
                f"{format_time(estimated_remaining)}"
            )


    # =====================================================
    # Ctrl+C handling
    # =====================================================

    except KeyboardInterrupt:

        elapsed = (
            time.time()
            - session_start
        )

        print(
            "\n\n"
            + "=" * 70
        )

        print(
            "PROCESSING STOPPED SAFELY"
        )

        print(
            "=" * 70
        )

        print(
            f"Last completed image: "
            f"{current_index:,}"
        )

        print(
            f"Processed this session: "
            f"{session_processed:,}"
        )

        print(
            f"Failed this session: "
            f"{session_failed:,}"
        )

        print(
            f"Session time: "
            f"{format_time(elapsed)}"
        )

        print(
            "\nCheckpoint has already been saved."
        )

        print(
            "Run process_dataset.py again "
            "to continue immediately."
        )

        return


    # =====================================================
    # Finished
    # =====================================================

    elapsed = (
        time.time()
        - session_start
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DATASET PROCESSING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Total dataset images: "
        f"{total_images:,}"
    )

    print(
        f"Successfully processed: "
        f"{processed_count:,}"
    )

    print(
        f"Failed images: "
        f"{failed_count:,}"
    )

    print(
        f"Processed this session: "
        f"{session_processed:,}"
    )

    print(
        f"Session duration: "
        f"{format_time(elapsed)}"
    )

    print(
        f"\nProcessed dataset saved to:"
        f"\n{OUTPUT_FOLDER}"
    )

    if failed_count > 0:

        print(
            f"\nFailed image list:"
            f"\n{FAILED_FILE}"
        )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    process_dataset()