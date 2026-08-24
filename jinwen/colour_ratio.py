import cv2 as cv
import numpy as np


def calculate_colour_ratios(hsv_image):
    """
    Calculate green, yellow and brown ratios from an HSV image.
    """

    if hsv_image is None:
        raise ValueError("Input image is None.")

    valid_mask = cv.inRange(
        hsv_image,
        np.array([0, 20, 20]),
        np.array([179, 255, 255])
    )

    valid_area = max(cv.countNonZero(valid_mask), 1)

    green_mask = cv.inRange(
        hsv_image,
        np.array([35, 30, 30]),
        np.array([95, 255, 255])
    )

    yellow_mask = cv.inRange(
        hsv_image,
        np.array([20, 30, 30]),
        np.array([35, 255, 255])
    )

    brown_mask = cv.inRange(
        hsv_image,
        np.array([0, 30, 20]),
        np.array([25, 255, 180])
    )

    return {
        "green_ratio": cv.countNonZero(cv.bitwise_and(green_mask, valid_mask)) / valid_area,
        "yellow_ratio": cv.countNonZero(cv.bitwise_and(yellow_mask, valid_mask)) / valid_area,
        "brown_ratio": cv.countNonZero(cv.bitwise_and(brown_mask, valid_mask)) / valid_area,
    }
