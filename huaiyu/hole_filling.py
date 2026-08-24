import cv2
import numpy as np


def fill_internal_holes(mask):
    """
    Fill enclosed holes inside the segmented banana region.

    This is useful when dark or low-saturation banana regions
    are mistakenly classified as background during segmentation.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask where foreground = 255 and background = 0.

    Returns
    -------
    numpy.ndarray
        Binary mask with internal holes filled.
    """

    if mask is None:
        raise ValueError("Input mask cannot be None.")

    binary_mask = np.where(
        mask > 0,
        255,
        0
    ).astype(np.uint8)

    # Add a small black border so flood fill always starts
    # from a guaranteed background location.
    padded = cv2.copyMakeBorder(
        binary_mask,
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=0
    )

    flood_filled = padded.copy()

    # Flood-fill external background
    flood_mask = np.zeros(
        (
            padded.shape[0] + 2,
            padded.shape[1] + 2
        ),
        dtype=np.uint8
    )

    cv2.floodFill(
        flood_filled,
        flood_mask,
        seedPoint=(0, 0),
        newVal=255
    )

    # Invert so enclosed holes become white
    inverted_background = cv2.bitwise_not(
        flood_filled
    )

    # Combine holes with original foreground
    filled = cv2.bitwise_or(
        padded,
        inverted_background
    )

    # Remove temporary border
    filled = filled[
        1:-1,
        1:-1
    ]

    return filled