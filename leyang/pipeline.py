import os
import cv2

from resize import resize_image
from background_removal import remove_background
from clahe import apply_clahe
from hsv import convert_to_hsv
from normalization import normalize_image

# Read image
image_path = "../input/banana1.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found.")

# Step 1: Resize
image = resize_image(image)

# Step 2: Remove background
image = remove_background(image)

# Step 3: CLAHE.
image = apply_clahe(image)

# Step 4: HSV
image = convert_to_hsv(image)

# Step 5: Normalization
image = normalize_image(image)

# Convert back to 0-255 ONLY for saving
image = (image * 255).astype("uint8")

# HSV → BGR for normal display
image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

# Show result
cv2.imshow("Output", image)

# Save
output_folder = "output"

filename = os.path.basename(image_path)

output_path = os.path.join(output_folder, filename)

cv2.imwrite(output_path, image)

print("Saved:", output_path)

cv2.waitKey(0)
cv2.destroyAllWindows()