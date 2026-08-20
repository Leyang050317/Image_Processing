import cv2
import numpy as np


def calculate_colour_ratios(hsv_image, banana_mask):
    """
    Calculate green, yellow and brown colour ratios inside the banana mask.
    """

    if hsv_image is None:
        raise ValueError("Input image is None.")

    if banana_mask is None:
        raise ValueError("Input mask is None.")

    banana_area = max(cv2.countNonZero(banana_mask), 1)

    green_mask = cv2.inRange(
        hsv_image,
        np.array([35, 35, 35]),
        np.array([95, 255, 255])
    )

    yellow_mask = cv2.inRange(
        hsv_image,
        np.array([20, 35, 35]),
        np.array([35, 255, 255])
    )

    brown_mask = cv2.inRange(
        hsv_image,
        np.array([0, 25, 20]),
        np.array([25, 255, 180])
    )

    return {
        "green_ratio": cv2.countNonZero(cv2.bitwise_and(green_mask, banana_mask)) / banana_area,
        "yellow_ratio": cv2.countNonZero(cv2.bitwise_and(yellow_mask, banana_mask)) / banana_area,
        "brown_ratio": cv2.countNonZero(cv2.bitwise_and(brown_mask, banana_mask)) / banana_area,
    }


def calculate_texture_features(gray_image, banana_mask):
    """
    Calculate simple texture features for the banana region.
    """

    if gray_image is None:
        raise ValueError("Input image is None.")

    if banana_mask is None:
        raise ValueError("Input mask is None.")

    region = cv2.bitwise_and(
        gray_image,
        gray_image,
        mask=banana_mask
    )

    pixels = region[banana_mask > 0]

    if pixels.size == 0:
        return {
            "texture_mean": 0.0,
            "texture_std": 0.0,
        }

    return {
        "texture_mean": float(np.mean(pixels)),
        "texture_std": float(np.std(pixels)),
    }


def calculate_banana_features(hsv_image, bgr_image, banana_mask, blemish_mask):
    """
    Combine colour, texture and blemish measurements for LY hybrid analysis.
    """

    if bgr_image is None:
        raise ValueError("Input image is None.")

    if blemish_mask is None:
        raise ValueError("Input mask is None.")

    gray_image = cv2.cvtColor(
        bgr_image,
        cv2.COLOR_BGR2GRAY
    )

    banana_area = max(cv2.countNonZero(banana_mask), 1)
    blemish_inside_banana = cv2.bitwise_and(
        blemish_mask,
        banana_mask
    )
    blemish_area = cv2.countNonZero(blemish_inside_banana)

    features = {}
    features.update(calculate_colour_ratios(hsv_image, banana_mask))
    features.update(calculate_texture_features(gray_image, banana_mask))
    features["blemish_ratio"] = blemish_area / banana_area

    return features
