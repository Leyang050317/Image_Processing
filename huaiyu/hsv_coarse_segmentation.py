import cv2
import numpy as np


def segment_banana_hsv(image):
    """
    Produce a coarse banana mask using HSV colour thresholds.

    The first range targets green/yellow banana regions.
    The second range retains brown/overripe regions.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    # Convert BGR image to HSV
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # -----------------------------------------------------
    # Green / yellow banana regions
    # -----------------------------------------------------

    lower_main = np.array(
        [15, 25, 40]
    )

    upper_main = np.array(
        [95, 255, 255]
    )

    mask_main = cv2.inRange(
        hsv,
        lower_main,
        upper_main
    )

    # -----------------------------------------------------
    # Brown / overripe banana regions
    # -----------------------------------------------------

    lower_brown = np.array(
        [0, 20, 20]
    )

    upper_brown = np.array(
        [25, 255, 220]
    )

    mask_brown = cv2.inRange(
        hsv,
        lower_brown,
        upper_brown
    )

    # -----------------------------------------------------
    # Combine masks
    # -----------------------------------------------------

    combined_mask = cv2.bitwise_or(
        mask_main,
        mask_brown
    )

    return combined_mask