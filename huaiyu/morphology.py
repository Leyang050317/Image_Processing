# huaiyu/morphology.py

import cv2 as cv
import numpy as np


def refine_mask(mask):
    kernel = np.ones((5, 5), np.uint8)

    opened_mask = cv.morphologyEx(
        mask,
        cv.MORPH_OPEN,
        kernel
    )

    refined_mask = cv.morphologyEx(
        opened_mask,
        cv.MORPH_CLOSE,
        kernel
    )

    return refined_mask


def apply_mask(image, mask):
    foreground = cv.bitwise_and(
        image,
        image,
        mask=mask
    )

    # Create white background
    white_background = np.ones_like(image) * 255

    inverse_mask = cv.bitwise_not(mask)

    background = cv.bitwise_and(
        white_background,
        white_background,
        mask=inverse_mask
    )

    result = cv.add(
        foreground,
        background
    )

    return result