import numpy as np


def normalize_image(image):
    """
    Normalize image pixel values to range 0.0 - 1.0.

    Parameters:
        image (numpy.ndarray): Input image

    Returns:
        numpy.ndarray: Normalized float32 image
    """

    normalized_image = image.astype(np.float32) / 255.0

    return normalized_image