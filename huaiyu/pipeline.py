# huaiyu/pipeline.py

import cv2 as cv
import os

from resize import resize_image
from background_removal import remove_background
from morphology import refine_mask, apply_mask
from clahe import apply_clahe


# Input image
image_path = "../input/banana1.png"

image = cv.imread(image_path)


# --------------------------------
# Step 1: Resize
# --------------------------------

resized = resize_image(image)


# --------------------------------
# Step 2: Background Removal
# --------------------------------

foreground, raw_mask = remove_background(resized)


# --------------------------------
# Step 3: Morphological Refinement
# --------------------------------

refined_mask = refine_mask(raw_mask)

segmented = apply_mask(
    foreground,
    refined_mask
)


# --------------------------------
# Step 4: CLAHE
# --------------------------------

enhanced = apply_clahe(segmented)


# --------------------------------
# Save outputs
# --------------------------------

output_folder = "output"

os.makedirs(
    output_folder,
    exist_ok=True
)

cv.imwrite(
    f"{output_folder}/01_resized.jpg",
    resized
)

cv.imwrite(
    f"{output_folder}/02_raw_mask.jpg",
    raw_mask
)

cv.imwrite(
    f"{output_folder}/03_refined_mask.jpg",
    refined_mask
)

cv.imwrite(
    f"{output_folder}/04_background_removed.jpg",
    segmented
)

cv.imwrite(
    f"{output_folder}/05_clahe.jpg",
    enhanced
)


print("HY preprocessing pipeline completed.")