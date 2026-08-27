import cv2
import numpy as np

from resize import resize_image
from background_removal import remove_background
from white_balance import apply_white_balance
from clahe import apply_clahe
from hsv import convert_to_hsv
from colour_ratio import calculate_colour_ratios
from normalization import normalize_image


def preprocess_image(image, return_steps=False):
    """
    Jinwen preprocessing pipeline.

    Pipeline:
        Resize
        -> Background Removal
        -> Gray-World White Balance
        -> CLAHE
        -> HSV / Colour Ratio Analysis
        -> Normalization

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.

    return_steps : bool
        If True, return all intermediate stages
        for demonstration/debug purposes.

    Returns
    -------
    final_image : np.ndarray
        Final uint8 processed image.

    colour_ratios : dict
        Green, yellow and brown colour ratios.

    steps : dict, optional
        Intermediate preprocessing outputs.
    """

    if image is None:
        raise ValueError("Input image is None.")

    # =====================================================
    # Step 1: Resize
    # =====================================================

    resized = resize_image(
        image,
        width=224,
        height=224
    )

    # =====================================================
    # Step 2: Background Removal
    # =====================================================

    background_removed = remove_background(
        resized
    )

    # =====================================================
    # Step 3: Gray-World White Balance
    # =====================================================

    balanced = apply_white_balance(
        background_removed
    )

    # =====================================================
    # Step 4: CLAHE
    # =====================================================

    enhanced = apply_clahe(
        balanced
    )

    # =====================================================
    # Step 5: HSV + Colour Ratio Analysis
    # =====================================================

    hsv_image = convert_to_hsv(
        enhanced
    )

    colour_ratios = calculate_colour_ratios(
        hsv_image
    )

    # =====================================================
    # Step 6: Normalization
    # =====================================================

    normalized = normalize_image(
        enhanced
    )

    # Convert back to uint8 for storage.
    #
    # MobileNetV2-specific preprocessing will be applied
    # later inside deep_learning/train.py.
    final_image = (
        normalized * 255.0
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    # =====================================================
    # Optional intermediate results
    # =====================================================

    if return_steps:

        steps = {
            "original": image,
            "resized": resized,
            "background_removed": background_removed,
            "white_balance": balanced,
            "clahe": enhanced,
            "hsv": hsv_image,
            "normalized": final_image
        }

        return (
            final_image,
            colour_ratios,
            steps
        )

    return (
        final_image,
        colour_ratios
    )