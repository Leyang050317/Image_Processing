import cv2
import numpy as np


def remove_background(image, mask, background_color=(0, 0, 0)):
    """
    Apply a binary mask to an image and remove the background.

    Parameters
    ----------
    image : numpy.ndarray
        Input BGR image.

    mask : numpy.ndarray
        Binary mask where banana = 255 and background = 0.

    background_color : tuple
        BGR colour used for the removed background.
        Default is black.

    Returns
    -------
    numpy.ndarray
        Background-removed BGR image.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    if mask is None:
        raise ValueError("Mask cannot be None.")

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            "Image and mask dimensions must match."
        )

    # Create output background
    output = np.full_like(
        image,
        background_color
    )

    # Copy banana pixels only
    output[mask > 0] = image[mask > 0]

    return output