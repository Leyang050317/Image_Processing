"""LW: Noise Reduction and Surface Quality Analysis pipeline.

Run from the project root:
    python liwen/pipeline.py --all-inputs
"""

import argparse
import json
import os

import cv2

from resize import resize_image
from median_filter import apply_median_filter
from clahe import apply_clahe
from hsv import convert_to_hsv
from blemish_segmentation import estimate_banana_mask, segment_blemishes
from morphology import refine_blemish_mask
from surface_analysis import calculate_surface_features
from normalization import normalize_image


def process_image(image_path, output_root):
    """Run every LW stage for one image and save its final normalized image."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    # 1. Resize
    resized = resize_image(image, width=250, height=250)

    # 2. Median Filter
    filtered = apply_median_filter(resized, kernel_size=5)

    # 3. CLAHE
    enhanced = apply_clahe(filtered, clip_limit=2.0, tile_grid_size=(8, 8))

    # 4. HSV
    hsv_image = convert_to_hsv(enhanced)

    # 5. Blemish Segmentation
    banana_mask = estimate_banana_mask(hsv_image, kernel_size=9)
    blemish_mask = segment_blemishes(
        hsv_image, banana_mask, brown_value_threshold=165,
        dark_value_threshold=85, inner_kernel_size=7
    )

    # 6. Morphological Refinement
    refined_blemish_mask = refine_blemish_mask(
        blemish_mask, kernel_size=3, open_iterations=1, close_iterations=1,
        min_region_area=12,
    )

    # 7. GLCM + Blemish Ratio
    features = calculate_surface_features(
        enhanced, banana_mask, refined_blemish_mask,
        distances=(1,), angles=(0, 0.785398, 1.570796, 2.356194), levels=32,
    )

    # 8. Normalization
    normalized = normalize_image(enhanced)

    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(output_root, os.path.basename(image_path))
    final_image = (normalized * 255).astype("uint8")
    if not cv2.imwrite(output_path, final_image):
        raise IOError(f"Unable to save final output image: {output_path}")

    return features


def main():
    parser = argparse.ArgumentParser(description="Run the LW surface-quality pipeline.")
    parser.add_argument("--image", help="Path to one input image.")
    parser.add_argument("--all-inputs", action="store_true", help="Process all image files in input/.")
    arguments = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "..", "input")
    output_root = os.path.join(base_dir, "output")

    if arguments.all_inputs:
        image_paths = [
            os.path.join(input_dir, name)
            for name in sorted(os.listdir(input_dir))
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    elif arguments.image:
        image_paths = [os.path.abspath(arguments.image)]
    else:
        image_paths = [os.path.join(input_dir, "overripe_banana.jpg")]

    if not image_paths:
        raise FileNotFoundError("No supported images found in input/.")

    for image_path in image_paths:
        features = process_image(image_path, output_root)
        print(f"Processed: {image_path}")
        print(json.dumps(features, indent=2))


if __name__ == "__main__":
    main()
