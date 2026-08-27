from pathlib import Path

import cv2

from pipeline import preprocess_image


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

JINWEN_FOLDER = Path(__file__).resolve().parent

INPUT_FOLDER = (
    PROJECT_ROOT
    / "input"
)

OUTPUT_FOLDER = (
    JINWEN_FOLDER
    / "each_step_output"
)


# =========================================================
# Demonstration image
# =========================================================
#
# Change only this filename if you want to demonstrate
# another banana image.
#
# =========================================================

TEST_IMAGE_NAME = "unripe_banana.jpg"


# =========================================================
# Save helper
# =========================================================

def save_image(
    filename,
    image
):
    """
    Save an image into each_step_output.
    """

    output_path = (
        OUTPUT_FOLDER
        / filename
    )

    success = cv2.imwrite(
        str(output_path),
        image
    )

    if not success:

        raise IOError(
            f"Failed to save: {output_path}"
        )

    print(
        f"Saved: {output_path.name}"
    )


# =========================================================
# Main
# =========================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "JINWEN PREPROCESSING DEMONSTRATION"
    )

    print(
        "=" * 70
    )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    image_path = (
        INPUT_FOLDER
        / TEST_IMAGE_NAME
    )

    print(
        f"\nInput image:"
        f"\n{image_path}"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not read image:\n"
            f"{image_path}"
        )

    # =====================================================
    # Run pipeline and request intermediate stages
    # =====================================================

    (
        final_image,
        colour_ratios,
        steps
    ) = preprocess_image(
        image,
        return_steps=True
    )

    # =====================================================
    # Save demonstration stages
    # =====================================================

    save_image(
        "01_original.jpg",
        steps[
            "original"
        ]
    )

    save_image(
        "02_resized.jpg",
        steps[
            "resized"
        ]
    )

    save_image(
        "03_background_removed.jpg",
        steps[
            "background_removed"
        ]
    )

    save_image(
        "04_white_balance.jpg",
        steps[
            "white_balance"
        ]
    )

    save_image(
        "05_clahe.jpg",
        steps[
            "clahe"
        ]
    )

    # -----------------------------------------------------
    # HSV must be converted back to BGR before saving
    # so Windows/PyCharm displays it correctly.
    # -----------------------------------------------------

    hsv_visualization = cv2.cvtColor(
        steps[
            "hsv"
        ],
        cv2.COLOR_HSV2BGR
    )

    save_image(
        "06_hsv.jpg",
        hsv_visualization
    )

    save_image(
        "07_normalized.jpg",
        final_image
    )

    # =====================================================
    # Save colour-ratio analysis
    # =====================================================

    features_file = (
        OUTPUT_FOLDER
        / "features.txt"
    )

    with open(
        features_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "JINWEN COLOUR RATIO ANALYSIS\n"
        )

        file.write(
            "=" * 40
            + "\n"
        )

        file.write(
            f"Input image: "
            f"{TEST_IMAGE_NAME}\n\n"
        )

        file.write(
            f"Green ratio: "
            f"{colour_ratios['green_ratio']:.6f}\n"
        )

        file.write(
            f"Yellow ratio: "
            f"{colour_ratios['yellow_ratio']:.6f}\n"
        )

        file.write(
            f"Brown ratio: "
            f"{colour_ratios['brown_ratio']:.6f}\n"
        )

    print(
        f"Saved: {features_file.name}"
    )

    # =====================================================
    # Summary
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DEMONSTRATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nGreen ratio: "
        f"{colour_ratios['green_ratio']:.4f}"
    )

    print(
        f"Yellow ratio: "
        f"{colour_ratios['yellow_ratio']:.4f}"
    )

    print(
        f"Brown ratio: "
        f"{colour_ratios['brown_ratio']:.4f}"
    )

    print(
        f"\nOutputs saved to:"
        f"\n{OUTPUT_FOLDER}"
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    main()