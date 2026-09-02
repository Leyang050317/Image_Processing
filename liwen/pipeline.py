"""Liwen's surface-quality preprocessing pipeline."""

import numpy as np

from resize import resize_image
from median_filter import apply_median_filter
from clahe import apply_clahe
from hsv import convert_to_hsv
from blemish_segmentation import estimate_banana_mask, segment_blemishes
from morphology import refine_blemish_mask
from surface_analysis import calculate_surface_features
from normalization import normalize_image


def preprocess_image(image, return_steps=False):
    """Run all eight Liwen preprocessing stages on one OpenCV BGR image.

    Returns the storage-ready uint8 image and its surface measurements.
    Intermediate images and masks are also returned when ``return_steps`` is true.
    """
    if image is None:
        raise ValueError("Input image is None.")

    # 1. Resize
    resized = resize_image(image, width=250, height=250)

    # 2. Median filtering
    filtered = apply_median_filter(resized, kernel_size=5)

    # 3. CLAHE contrast enhancement
    enhanced = apply_clahe(filtered, clip_limit=2.0, tile_grid_size=(8, 8))

    # 4. HSV conversion
    hsv_image = convert_to_hsv(enhanced)

    # 5. Banana and blemish segmentation
    banana_mask = estimate_banana_mask(hsv_image, kernel_size=9)
    blemish_mask = segment_blemishes(
        hsv_image,
        banana_mask,
        brown_value_threshold=165,
        dark_value_threshold=85,
        inner_kernel_size=7,
    )

    # 6. Morphological refinement
    refined_blemish_mask = refine_blemish_mask(
        blemish_mask,
        kernel_size=3,
        open_iterations=1,
        close_iterations=1,
        min_region_area=12,
    )

    # 7. GLCM texture features and blemish ratio
    features = calculate_surface_features(
        enhanced,
        banana_mask,
        refined_blemish_mask,
        distances=(1,),
        angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
        levels=32,
    )

    # 8. Normalize, then convert back to uint8 for the processed dataset.
    normalized = normalize_image(enhanced)
    final_image = (normalized * 255.0).clip(0, 255).astype(np.uint8)

    if return_steps:
        steps = {
            "original": image,
            "resized": resized,
            "median_filter": filtered,
            "clahe": enhanced,
            "hsv": hsv_image,
            "banana_mask": banana_mask,
            "blemish_mask": blemish_mask,
            "refined_blemish_mask": refined_blemish_mask,
            "normalized": final_image,
        }
        return final_image, features, steps

    return final_image, features
