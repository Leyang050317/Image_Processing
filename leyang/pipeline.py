import cv2

from resize import resize_image
from background_removal import remove_background
from white_balance import apply_white_balance
from bilateral_filter import apply_bilateral_filter
from clahe import apply_clahe
from normalization import normalize_image


def preprocess_image(image):
    """
    Leyang preprocessing pipeline.

    Pipeline:
    Resize
        -> Background Removal
        -> White Balance
        -> Bilateral Filter
        -> Gentle CLAHE
        -> Normalization
    """

    if image is None:
        raise ValueError("Input image is None.")

    # -----------------------------------------------------
    # Step 1: Resize
    # -----------------------------------------------------
    resized = resize_image(
        image,
        width=224,
        height=224
    )

    # -----------------------------------------------------
    # Step 2: Background Removal
    # -----------------------------------------------------
    background_removed = remove_background(
        resized
    )

    # -----------------------------------------------------
    # Step 3: White Balance
    # -----------------------------------------------------
    balanced = apply_white_balance(
        background_removed
    )

    # -----------------------------------------------------
    # Step 4: Bilateral Filter
    # -----------------------------------------------------
    filtered = apply_bilateral_filter(
        balanced,
        diameter=3,
        sigma_color=20,
        sigma_space=20
    )

    # -----------------------------------------------------
    # Step 5: Gentle CLAHE
    # -----------------------------------------------------
    clahe_result = apply_clahe(
        filtered,
        clip_limit=0.5,
        tile_grid_size=(24, 24)
    )

    enhanced = cv2.addWeighted(
        filtered,
        0.85,
        clahe_result,
        0.15,
        0
    )

    # -----------------------------------------------------
    # Step 6: Normalization
    # -----------------------------------------------------
    normalized = normalize_image(
        enhanced
    )

    # Convert to normal uint8 before saving.
    # MobileNetV2 preprocessing is performed later by
    # the shared deep_learning training pipeline.
    final_image = (
        normalized * 255
    ).clip(
        0,
        255
    ).astype(
        "uint8"
    )

    return final_image