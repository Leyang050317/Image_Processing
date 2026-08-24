import os
import cv2

from resize import resize_image
from background_removal import remove_background
from white_balance import apply_white_balance
from clahe import apply_clahe
from hsv import convert_to_hsv
from colour_ratio import calculate_colour_ratios
from normalization import normalize_image

# Read image
base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "..", "input", "unripe_banana.jpg")
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found.")

output_folder = os.path.join(base_dir, "output")
os.makedirs(output_folder, exist_ok=True)

# Step 1: Resize
resized = resize_image(
    image,
    width=224,
    height=224
)

# Step 2: Background Removal
background_removed = remove_background(resized)

# Step 3: Gray-World White Balance
balanced = apply_white_balance(background_removed)

# Step 4: CLAHE
enhanced = apply_clahe(balanced)

# Step 5: HSV and colour ratio analysis
hsv_image = convert_to_hsv(enhanced)
colour_ratios = calculate_colour_ratios(hsv_image)

# Step 6: Normalization for MobileNetV2 input
normalized = normalize_image(enhanced)
final_image = (normalized * 255).astype("uint8")

output_path = os.path.join(output_folder, "output.png")

cv2.imwrite(output_path, final_image)

print("Saved:", output_path)
print("Colour ratios:", colour_ratios)
print("JW MobileNetV2 input image completed.")
