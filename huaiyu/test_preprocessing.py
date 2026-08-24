from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from pipeline import preprocess_image


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

INPUT_FOLDER = (
    PROJECT_ROOT / "input"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# =========================================================
# Process one image
# =========================================================

def process_image(image_path):

    print(
        f"\nProcessing: "
        f"{image_path.name}"
    )

    original = cv2.imread(
        str(image_path)
    )

    if original is None:

        print(
            f"[SKIPPED] "
            f"{image_path.name}"
        )

        return

    # -----------------------------------------------------
    # Run the ACTUAL HY pipeline
    # -----------------------------------------------------

    (
        processed_image,
        final_mask,
        debug
    ) = preprocess_image(
        original,
        return_debug=True
    )

    print(
        "[MASK DECISION] "
        + debug["mask_decision"]
    )

    # -----------------------------------------------------
    # Convert BGR images for display
    # -----------------------------------------------------

    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    resized_rgb = cv2.cvtColor(
        debug["resized"],
        cv2.COLOR_BGR2RGB
    )

    filtered_rgb = cv2.cvtColor(
        debug["filtered"],
        cv2.COLOR_BGR2RGB
    )

    processed_rgb = cv2.cvtColor(
        processed_image,
        cv2.COLOR_BGR2RGB
    )

    # =====================================================
    # Display
    # =====================================================

    plt.figure(
        figsize=(18, 15)
    )

    # -----------------------------------------------------
    # 1. Original
    # -----------------------------------------------------

    plt.subplot(3, 4, 1)

    plt.imshow(
        original_rgb
    )

    plt.title(
        "Original"
    )

    plt.axis("off")

    # -----------------------------------------------------
    # 2. Resize
    # -----------------------------------------------------

    plt.subplot(3, 4, 2)

    plt.imshow(
        resized_rgb
    )

    plt.title(
        "Resize + Padding"
    )

    plt.axis("off")

    # -----------------------------------------------------
    # 3. Bilateral
    # -----------------------------------------------------

    plt.subplot(3, 4, 3)

    plt.imshow(
        filtered_rgb
    )

    plt.title(
        "Bilateral Filter"
    )

    plt.axis("off")

    # -----------------------------------------------------
    # 4. HSV
    # -----------------------------------------------------

    plt.subplot(3, 4, 4)

    plt.imshow(
        debug["hsv_mask"],
        cmap="gray"
    )

    plt.title(
        "HSV Coarse Segmentation"
    )

    plt.axis("off")

    # -----------------------------------------------------
    # 5. GrabCut
    # -----------------------------------------------------

    plt.subplot(3, 4, 5)

    plt.imshow(
        debug["grabcut_mask"],
        cmap="gray"
    )

    plt.title(
        "GrabCut Refinement"
    )

    plt.axis("off")

    # -----------------------------------------------------
    # 6. Morphology
    # -----------------------------------------------------

    plt.subplot(3, 4, 6)

    plt.imshow(
        debug["morphology_mask"],
        cmap="gray"
    )

    plt.title(
        "Morphological Refinement"
    )

    plt.axis("off")

    # -----------------------------------------------------
    # 7. Hole Filling
    # -----------------------------------------------------

    plt.subplot(3, 4, 7)

    plt.imshow(
        debug["filled_mask"],
        cmap="gray"
    )

    plt.title(
        "Hole Filling"
    )

    plt.axis("off")

    # -----------------------------------------------------
    # 8. Largest Connected Component
    # -----------------------------------------------------

    plt.subplot(3, 4, 8)

    plt.imshow(
        debug[
            "largest_component_mask"
        ],
        cmap="gray"
    )

    plt.title(
        "Largest Connected Component"
    )

    plt.axis("off")

    # -----------------------------------------------------
    # 9. Contour Refinement
    # -----------------------------------------------------

    plt.subplot(3, 4, 9)

    plt.imshow(
        debug["contour_mask"],
        cmap="gray"
    )

    plt.title(
        "Contour Refinement"
    )

    plt.axis("off")

    # -----------------------------------------------------
    # 10. Final accepted mask
    # -----------------------------------------------------

    plt.subplot(3, 4, 10)

    plt.imshow(
        final_mask,
        cmap="gray"
    )

    plt.title(
        "Accepted Final Mask"
    )

    plt.axis("off")

    # -----------------------------------------------------
    # 11. Final background-removed image
    # -----------------------------------------------------

    plt.subplot(3, 4, 11)

    plt.imshow(
        processed_rgb
    )

    plt.title(
        "Background Removed"
    )

    plt.axis("off")

    # -----------------------------------------------------
    # 12. Mask decision
    # -----------------------------------------------------

    plt.subplot(3, 4, 12)

    plt.axis("off")

    plt.text(
        0.05,
        0.5,
        debug["mask_decision"],
        fontsize=11,
        wrap=True
    )

    # Figure title with enough spacing
    plt.suptitle(
        image_path.name,
        fontsize=18,
        y=0.98
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95
        ]
    )

    plt.show()

    plt.close()


# =========================================================
# Main
# =========================================================

def main():

    if not INPUT_FOLDER.exists():

        print(
            f"Input folder not found:\n"
            f"{INPUT_FOLDER}"
        )

        return

    image_files = sorted(
        file
        for file in INPUT_FOLDER.iterdir()
        if file.is_file()
        and file.suffix.lower()
        in IMAGE_EXTENSIONS
    )

    if not image_files:

        print(
            "No images found."
        )

        return

    print(
        f"\nFound "
        f"{len(image_files)} "
        f"image(s)."
    )

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        print(
            f"\n[{index}/"
            f"{len(image_files)}] "
            f"{image_path.name}"
        )

        process_image(
            image_path
        )

    print(
        "\nAll images processed."
    )


if __name__ == "__main__":
    main()