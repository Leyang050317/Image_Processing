from pathlib import Path
import json
import time

import cv2

from pipeline import preprocess_image


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FOLDER = (
    PROJECT_ROOT
    / "dataset"
)

OUTPUT_FOLDER = (
    Path(__file__).resolve().parent
    / "output"
)

CHECKPOINT_FILE = (
    Path(__file__).resolve().parent
    / "checkpoint.json"
)

FAILED_IMAGES_FILE = (
    Path(__file__).resolve().parent
    / "failed_images.txt"
)


# =========================================================
# Configuration
# =========================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# =========================================================
# Dataset discovery
# =========================================================

def get_image_files():

    image_files = sorted(
        file
        for file in DATASET_FOLDER.rglob("*")
        if (
            file.is_file()
            and file.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    return image_files


# =========================================================
# Checkpoint handling
# =========================================================

def load_checkpoint():

    if not CHECKPOINT_FILE.exists():

        return {
            "last_completed_index": 0,
            "processed_count": 0,
            "failed_count": 0
        }

    try:

        with open(
            CHECKPOINT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            checkpoint = json.load(file)

        return checkpoint

    except Exception as error:

        print(
            f"[WARNING] Could not load checkpoint: {error}"
        )

        print(
            "Starting from the beginning."
        )

        return {
            "last_completed_index": 0,
            "processed_count": 0,
            "failed_count": 0
        }


def save_checkpoint(
    last_completed_index,
    processed_count,
    failed_count
):

    checkpoint = {
        "last_completed_index":
            last_completed_index,

        "processed_count":
            processed_count,

        "failed_count":
            failed_count
    }

    with open(
        CHECKPOINT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            checkpoint,
            file,
            indent=4
        )


# =========================================================
# Failed image logging
# =========================================================

def log_failed_image(
    image_path,
    error
):

    with open(
        FAILED_IMAGES_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            f"{image_path} | {error}\n"
        )


# =========================================================
# Process one image
# =========================================================

def process_one_image(
    image_path
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            "OpenCV could not read image."
        )

    processed_image = preprocess_image(
        image
    )

    # Example:
    #
    # dataset/ripe/banana001.jpg
    #
    # becomes:
    #
    # leyang/output/ripe/banana001.jpg

    relative_path = image_path.relative_to(
        DATASET_FOLDER
    )

    output_path = (
        OUTPUT_FOLDER
        / relative_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    success = cv2.imwrite(
        str(output_path),
        processed_image
    )

    if not success:

        raise IOError(
            "OpenCV failed to save output image."
        )

    return output_path


# =========================================================
# Main processor
# =========================================================

def process_dataset():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LEYANG FULL DATASET PREPROCESSING"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDataset:"
        f"\n{DATASET_FOLDER}"
    )

    print(
        f"\nOutput:"
        f"\n{OUTPUT_FOLDER}"
    )

    # -----------------------------------------------------
    # Find images
    # -----------------------------------------------------

    image_files = get_image_files()

    total_images = len(
        image_files
    )

    print(
        f"\nTotal images found: "
        f"{total_images:,}"
    )

    if total_images == 0:

        print(
            "\n[ERROR] No images found."
        )

        return

    # -----------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------

    checkpoint = load_checkpoint()

    last_completed_index = int(
        checkpoint.get(
            "last_completed_index",
            0
        )
    )

    processed_count = int(
        checkpoint.get(
            "processed_count",
            0
        )
    )

    failed_count = int(
        checkpoint.get(
            "failed_count",
            0
        )
    )

    if last_completed_index > 0:

        print(
            "\nCheckpoint found."
        )

        print(
            f"Already completed: "
            f"{last_completed_index:,} / "
            f"{total_images:,}"
        )

        print(
            f"Resuming directly from image "
            f"{last_completed_index + 1:,}."
        )

    else:

        print(
            "\nNo checkpoint found."
        )

        print(
            "Starting from image 1."
        )

    # IMPORTANT:
    # Slice the list immediately so previously completed
    # images are not processed again.

    remaining_images = image_files[
        last_completed_index:
    ]

    start_time = time.time()

    # -----------------------------------------------------
    # Processing loop
    # -----------------------------------------------------

    try:

        for offset, image_path in enumerate(
            remaining_images,
            start=1
        ):

            current_index = (
                last_completed_index
                + offset
            )

            try:

                process_one_image(
                    image_path
                )

                processed_count += 1

                status = "OK"

            except Exception as error:

                failed_count += 1

                status = "FAILED"

                log_failed_image(
                    image_path,
                    error
                )

            # Save after EVERY image.
            save_checkpoint(
                current_index,
                processed_count,
                failed_count
            )

            relative_path = (
                image_path.relative_to(
                    DATASET_FOLDER
                )
            )

            print(
                f"[{current_index:,}/"
                f"{total_images:,}] "
                f"{status:<6} "
                f"{relative_path}"
            )

    except KeyboardInterrupt:

        print(
            "\n\n"
            + "=" * 70
        )

        print(
            "PROCESSING STOPPED BY USER"
        )

        print(
            "=" * 70
        )

        print(
            "\nCheckpoint saved."
        )

        print(
            "Run process_dataset.py again "
            "to continue from the next image."
        )

        return

    # -----------------------------------------------------
    # Complete
    # -----------------------------------------------------

    elapsed_seconds = (
        time.time()
        - start_time
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LEYANG PREPROCESSING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSuccessful: "
        f"{processed_count:,}"
    )

    print(
        f"Failed: "
        f"{failed_count:,}"
    )

    print(
        f"Total: "
        f"{total_images:,}"
    )

    print(
        f"Elapsed time: "
        f"{elapsed_seconds / 60:.2f} minutes"
    )

    print(
        f"\nOutput dataset:"
        f"\n{OUTPUT_FOLDER}"
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    process_dataset()