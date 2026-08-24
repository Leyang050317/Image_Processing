import os
import cv2

from resize import resize_image
from background_removal import remove_background
from white_balance import apply_white_balance
from bilateral_filter import apply_bilateral_filter
from clahe import apply_clahe
from normalization import normalize_image

# Read image
base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "..", "input", "overripe_banana.jpg")
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

# Step 3: White Balance
balanced = apply_white_balance(background_removed)

# Step 4: Bilateral Filter
filtered = apply_bilateral_filter(
    balanced,
    diameter=3,
    sigma_color=20,
    sigma_space=20
)

# Step 5: Gentle CLAHE
clahe_roi = apply_clahe(
    filtered,
    clip_limit=0.5,
    tile_grid_size=(24, 24)
)
enhanced = cv2.addWeighted(
    filtered,
    0.85,
    clahe_roi,
    0.15,
    0
)

# Step 6: Normalization for MobileNetV2 input
normalized = normalize_image(enhanced)
final_image = (normalized * 255).astype("uint8")

output_path = os.path.join(output_folder, "output.png")

cv2.imwrite(output_path, final_image)

print("Saved:", output_path)
print("LY MobileNetV2 input image completed.")
