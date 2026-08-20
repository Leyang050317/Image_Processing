import os
import cv2

from resize import resize_image
from white_balance import apply_white_balance
from bilateral_filter import apply_bilateral_filter
from clahe import apply_clahe
from hsv import convert_to_hsv
from segmentation import segment_banana, segment_blemishes
from morphology import refine_mask, refine_blemish_mask, apply_mask
from surface_analysis import calculate_banana_features
from normalization import normalize_image

# Read image
base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "..", "input", "overripe_banana.jpg")
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found.")

output_folder = os.path.join(base_dir, "output")
os.makedirs(output_folder, exist_ok=True)

filename = os.path.basename(image_path)

# Step 1: Resize
resized = resize_image(image)

# Step 2: White Balance
balanced = apply_white_balance(resized)

# Step 3: Bilateral Filter
filtered = apply_bilateral_filter(balanced)

# Step 4: HSV for initial banana segmentation
initial_hsv = convert_to_hsv(filtered)

# Step 5: Banana segmentation
banana_mask = segment_banana(initial_hsv)

# Step 6: Morphology and banana ROI extraction
refined_banana_mask = refine_mask(banana_mask)
banana_roi = apply_mask(filtered, refined_banana_mask)

# Step 7: CLAHE on banana ROI
enhanced_roi = apply_clahe(
    banana_roi,
    clip_limit=1.0,
    tile_grid_size=(16, 16)
)

# Step 8: HSV and blemish segmentation
final_hsv = convert_to_hsv(enhanced_roi)
blemish_mask = segment_blemishes(
    final_hsv,
    enhanced_roi,
    refined_banana_mask
)
refined_blemish_mask = refine_blemish_mask(
    blemish_mask,
    kernel_size=3
)

# Step 9: Colour / texture / blemish analysis
features = calculate_banana_features(
    final_hsv,
    enhanced_roi,
    refined_banana_mask,
    refined_blemish_mask
)

# Step 10: Normalization
normalized = normalize_image(enhanced_roi)
display_image = (normalized * 255).astype("uint8")

output_path = os.path.join(output_folder, filename)
features_path = os.path.join(output_folder, "features.txt")

cv2.imwrite(os.path.join(output_folder, "01_resized.jpg"), resized)
cv2.imwrite(os.path.join(output_folder, "02_white_balance.jpg"), balanced)
cv2.imwrite(os.path.join(output_folder, "03_bilateral_filter.jpg"), filtered)
cv2.imwrite(os.path.join(output_folder, "04_initial_hsv.jpg"), initial_hsv)
cv2.imwrite(os.path.join(output_folder, "05_banana_mask.jpg"), refined_banana_mask)
cv2.imwrite(os.path.join(output_folder, "06_banana_roi.jpg"), banana_roi)
cv2.imwrite(os.path.join(output_folder, "07_clahe_roi.jpg"), enhanced_roi)
cv2.imwrite(os.path.join(output_folder, "08_final_hsv.jpg"), final_hsv)
cv2.imwrite(os.path.join(output_folder, "09_blemish_mask.jpg"), refined_blemish_mask)
cv2.imwrite(os.path.join(output_folder, "10_normalized.jpg"), display_image)

with open(features_path, "w", encoding="utf-8") as file:
    for name, value in features.items():
        file.write(f"{name}: {value:.4f}\n")

cv2.imwrite(output_path, display_image)

print("Saved:", output_path)
print("Features saved:", features_path)
print("LY preprocessing pipeline completed.")
