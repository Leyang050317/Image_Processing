from pathlib import Path
import csv
import random
from collections import Counter, defaultdict


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_DATASET = PROJECT_ROOT / "dataset"

MANIFEST_FILE = (
    Path(__file__).resolve().parent
    / "split_manifest.csv"
)


# =========================================================
# Dataset configuration
# =========================================================

CLASS_NAMES = [
    "overripe",
    "ripe",
    "rotten",
    "unripe"
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# =========================================================
# Scan original dataset
# =========================================================

def scan_dataset():
    """
    Scan the original dataset and group image paths by class.

    Returns
    -------
    dict
        {
            "ripe": [Path(...), ...],
            ...
        }
    """

    if not ORIGINAL_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n"
            f"{ORIGINAL_DATASET}"
        )

    class_images = defaultdict(list)

    for class_name in CLASS_NAMES:

        class_folder = (
            ORIGINAL_DATASET / class_name
        )

        if not class_folder.exists():
            raise FileNotFoundError(
                f"Missing class folder:\n"
                f"{class_folder}"
            )

        images = sorted(
            file
            for file in class_folder.rglob("*")
            if file.is_file()
            and file.suffix.lower()
            in IMAGE_EXTENSIONS
        )

        class_images[class_name] = images

    return class_images


# =========================================================
# Split one class
# =========================================================

def split_class_images(
    images,
    random_generator
):
    """
    Split one class into train / validation / test.

    Splitting is done separately for every class so
    class proportions remain balanced.
    """

    images = list(images)

    random_generator.shuffle(
        images
    )

    total = len(images)

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    # Remaining images automatically become test
    test_count = (
        total
        - train_count
        - val_count
    )

    train_images = images[
        :train_count
    ]

    val_images = images[
        train_count:
        train_count + val_count
    ]

    test_images = images[
        train_count + val_count:
    ]

    assert (
        len(train_images)
        + len(val_images)
        + len(test_images)
        == total
    )

    return (
        train_images,
        val_images,
        test_images
    )


# =========================================================
# Create manifest
# =========================================================

def create_split_manifest():
    """
    Create split_manifest.csv.

    IMPORTANT:
    This should normally be created only ONCE.

    The same manifest must later be reused for:
    - Original baseline
    - HY preprocessing
    - JW preprocessing
    - LW preprocessing
    - LY preprocessing
    """

    if MANIFEST_FILE.exists():

        print(
            "\nSplit manifest already exists:"
        )

        print(
            MANIFEST_FILE
        )

        print(
            "\nExisting manifest will NOT be overwritten."
        )

        print(
            "This protects the fixed experimental split."
        )

        return


    class_images = scan_dataset()

    random_generator = random.Random(
        RANDOM_SEED
    )

    records = []


    # -----------------------------------------------------
    # Split each class independently
    # -----------------------------------------------------

    for class_name in CLASS_NAMES:

        images = class_images[
            class_name
        ]

        (
            train_images,
            val_images,
            test_images
        ) = split_class_images(
            images,
            random_generator
        )


        # ---------------------------------------------
        # Train
        # ---------------------------------------------

        for image_path in train_images:

            records.append(
                {
                    "relative_path":
                        image_path.relative_to(
                            ORIGINAL_DATASET
                        ).as_posix(),

                    "class":
                        class_name,

                    "split":
                        "train"
                }
            )


        # ---------------------------------------------
        # Validation
        # ---------------------------------------------

        for image_path in val_images:

            records.append(
                {
                    "relative_path":
                        image_path.relative_to(
                            ORIGINAL_DATASET
                        ).as_posix(),

                    "class":
                        class_name,

                    "split":
                        "validation"
                }
            )


        # ---------------------------------------------
        # Test
        # ---------------------------------------------

        for image_path in test_images:

            records.append(
                {
                    "relative_path":
                        image_path.relative_to(
                            ORIGINAL_DATASET
                        ).as_posix(),

                    "class":
                        class_name,

                    "split":
                        "test"
                }
            )


    # -----------------------------------------------------
    # Stable ordering
    # -----------------------------------------------------

    records.sort(
        key=lambda item: (
            item["split"],
            item["class"],
            item["relative_path"]
        )
    )


    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    with open(
        MANIFEST_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "relative_path",
                "class",
                "split"
            ]
        )

        writer.writeheader()

        writer.writerows(
            records
        )


    print(
        "\nSplit manifest created successfully:"
    )

    print(
        MANIFEST_FILE
    )

    print_split_summary(
        records
    )


# =========================================================
# Load manifest
# =========================================================

def load_manifest():
    """
    Load the existing split manifest.

    Returns
    -------
    list[dict]
    """

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            "split_manifest.csv does not exist.\n"
            "Run dataset_loader.py first."
        )

    records = []

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:
            records.append(
                row
            )

    return records


# =========================================================
# Obtain one split
# =========================================================

def get_split_records(
    split_name
):
    """
    Return records belonging to one split.

    Valid values:
        train
        validation
        test
    """

    valid_splits = {
        "train",
        "validation",
        "test"
    }

    if split_name not in valid_splits:

        raise ValueError(
            f"Invalid split: {split_name}"
        )

    records = load_manifest()

    return [
        record
        for record in records
        if record["split"]
        == split_name
    ]


# =========================================================
# Resolve experiment paths
# =========================================================

def resolve_image_path(
    source_root,
    relative_path
):
    """
    Convert a manifest relative path into an actual image path.

    Example
    -------

    Manifest:
        ripe/image001.jpg

    Baseline:
        dataset/ripe/image001.jpg

    HY:
        huaiyu/output/ripe/image001.jpg

    JW:
        jinwen/output/ripe/image001.jpg
    """

    source_root = Path(
        source_root
    )

    return (
        source_root
        / Path(relative_path)
    )


# =========================================================
# Validate experiment dataset
# =========================================================

def validate_source_dataset(
    source_root
):
    """
    Check whether every image listed in the manifest
    exists inside a particular experiment dataset.

    Useful before training HY / JW / LW / LY.
    """

    source_root = Path(
        source_root
    )

    records = load_manifest()

    missing_files = []

    for record in records:

        image_path = resolve_image_path(
            source_root,
            record["relative_path"]
        )

        if not image_path.exists():

            missing_files.append(
                image_path
            )


    total = len(records)

    available = (
        total
        - len(missing_files)
    )


    print(
        "\nDataset validation"
    )

    print(
        "=" * 60
    )

    print(
        f"Source: {source_root}"
    )

    print(
        f"Expected images: {total:,}"
    )

    print(
        f"Available images: {available:,}"
    )

    print(
        f"Missing images: {len(missing_files):,}"
    )


    if missing_files:

        print(
            "\nFirst missing files:"
        )

        for file in missing_files[:10]:

            print(
                f"  {file}"
            )


    return (
        len(missing_files) == 0
    )


# =========================================================
# Print split statistics
# =========================================================

def print_split_summary(
    records=None
):
    """
    Print overall and per-class split distribution.
    """

    if records is None:
        records = load_manifest()


    print(
        "\n"
        + "=" * 70
    )

    print(
        "DATASET SPLIT SUMMARY"
    )

    print(
        "=" * 70
    )


    # -----------------------------------------------------
    # Overall
    # -----------------------------------------------------

    split_counter = Counter(
        record["split"]
        for record in records
    )

    print(
        f"\nTotal images: "
        f"{len(records):,}"
    )

    print(
        f"Training: "
        f"{split_counter['train']:,}"
    )

    print(
        f"Validation: "
        f"{split_counter['validation']:,}"
    )

    print(
        f"Testing: "
        f"{split_counter['test']:,}"
    )


    # -----------------------------------------------------
    # Per class
    # -----------------------------------------------------

    print(
        "\nPer-class distribution:"
    )

    print(
        "-" * 70
    )

    print(
        f"{'Class':<15}"
        f"{'Train':>10}"
        f"{'Validation':>15}"
        f"{'Test':>10}"
        f"{'Total':>10}"
    )

    print(
        "-" * 70
    )


    for class_name in CLASS_NAMES:

        class_records = [
            record
            for record in records
            if record["class"]
            == class_name
        ]

        counter = Counter(
            record["split"]
            for record
            in class_records
        )

        total = len(
            class_records
        )

        print(
            f"{class_name:<15}"
            f"{counter['train']:>10}"
            f"{counter['validation']:>15}"
            f"{counter['test']:>10}"
            f"{total:>10}"
        )


    print(
        "-" * 70
    )


# =========================================================
# Main
# =========================================================

def main():

    create_split_manifest()

    # If already created, load it
    # and show the existing distribution.
    records = load_manifest()

    print_split_summary(
        records
    )


if __name__ == "__main__":
    main()