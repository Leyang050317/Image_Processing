import cv2
import numpy as np


def segment_banana(hsv_image):
    """
    Segment banana pixels from an HSV image.

    Returns:
        numpy.ndarray: Binary banana mask
    """

    if hsv_image is None:
        raise ValueError("Input image is None.")

    peel_mask = cv2.inRange(
        hsv_image,
        np.array([15, 12, 35]),
        np.array([95, 255, 255])
    )

    dark_peel_mask = cv2.inRange(
        hsv_image,
        np.array([0, 20, 20]),
        np.array([35, 255, 210])
    )

    mask = cv2.bitwise_or(peel_mask, dark_peel_mask)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return mask

    largest_contour = max(contours, key=cv2.contourArea)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(
        filled_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    return filled_mask


def segment_blemishes(hsv_image, bgr_image, banana_mask):
    """
    Segment darker brown or dull blemish areas inside the banana region.

    Returns:
        numpy.ndarray: Binary blemish mask
    """

    if hsv_image is None:
        raise ValueError("Input image is None.")

    if bgr_image is None:
        raise ValueError("Input image is None.")

    if banana_mask is None:
        raise ValueError("Input mask is None.")

    kernel = np.ones((9, 9), np.uint8)
    inner_banana_mask = cv2.erode(
        banana_mask,
        kernel,
        iterations=1
    )

    h_channel, s_channel, v_channel = cv2.split(hsv_image)

    brown_hue_mask = cv2.inRange(
        hsv_image,
        np.array([0, 45, 0]),
        np.array([28, 255, 255])
    )

    low_value_mask = cv2.inRange(
        v_channel,
        0,
        145
    )

    brown_hsv_mask = cv2.bitwise_and(
        brown_hue_mask,
        low_value_mask
    )

    very_dark_mask = cv2.inRange(
        hsv_image,
        np.array([0, 35, 0]),
        np.array([179, 255, 70])
    )

    lab_image = cv2.cvtColor(
        bgr_image,
        cv2.COLOR_BGR2LAB
    )
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    inner_pixels = l_channel[inner_banana_mask > 0]
    if inner_pixels.size == 0:
        return np.zeros_like(banana_mask)

    brightness_threshold = max(
        35,
        min(115, int(np.mean(inner_pixels) - 55))
    )

    low_brightness_mask = cv2.inRange(
        l_channel,
        0,
        brightness_threshold
    )

    brown_lab_mask = cv2.inRange(
        lab_image,
        np.array([0, 128, 138]),
        np.array([160, 175, 205])
    )

    dull_brown_mask = cv2.bitwise_and(
        low_brightness_mask,
        brown_lab_mask
    )

    brown_defect_mask = cv2.bitwise_or(
        brown_hsv_mask,
        dull_brown_mask
    )

    combined_mask = cv2.bitwise_or(
        brown_defect_mask,
        very_dark_mask
    )

    blemish_mask = cv2.bitwise_and(
        combined_mask,
        inner_banana_mask
    )

    return blemish_mask
