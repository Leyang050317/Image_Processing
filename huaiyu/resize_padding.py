import cv2
import numpy as np


def resize_with_padding(
    image,
    target_size=(224, 224),
    pad_value=(0, 0, 0)
):
    """
    Resize an image while preserving its aspect ratio,
    then add padding to reach the required target size.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    target_width, target_height = target_size

    height, width = image.shape[:2]

    # Calculate scale while preserving aspect ratio
    scale = min(
        target_width / width,
        target_height / height
    )

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    # Resize
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Calculate required padding
    delta_width = target_width - new_width
    delta_height = target_height - new_height

    left = delta_width // 2
    right = delta_width - left

    top = delta_height // 2
    bottom = delta_height - top

    # Apply padding
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=pad_value
    )

    return padded