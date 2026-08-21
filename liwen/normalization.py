import numpy as np


def normalize_image(image):
    """Normalize BGR pixel values to float32 in the [0.0, 1.0] range."""
    if image is None:
        raise ValueError("Input image is None.")

    return image.astype(np.float32) / 255.0
