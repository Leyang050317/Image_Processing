import cv2
import numpy as np


def estimate_banana_mask(
    hsv_image,
    peel_lower=(10, 50, 35),
    peel_upper=(100, 255, 255),
    brown_lower=(0, 20, 20),
    brown_upper=(35, 255, 220),
    kernel_size=9,
):
    """Estimate the banana region from broad peel and brown HSV ranges.

    This classical method assumes the banana is the largest connected object
    with green/yellow/brown peel-like colours. It intentionally includes brown
    pixels so dark blemishes remain part of the denominator for blemish ratio.
    """
    if hsv_image is None:
        raise ValueError("Input HSV image is None.")

    peel_mask = cv2.inRange(
        hsv_image,
        np.array(peel_lower, dtype=np.uint8),
        np.array(peel_upper, dtype=np.uint8),
    )
    brown_mask = cv2.inRange(
        hsv_image,
        np.array(brown_lower, dtype=np.uint8),
        np.array(brown_upper, dtype=np.uint8),
    )
    candidate_mask = cv2.bitwise_or(peel_mask, brown_mask)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    candidate_mask = cv2.morphologyEx(
        candidate_mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    contours, _ = cv2.findContours(
        candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    banana_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(banana_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return banana_mask


def segment_blemishes(
    hsv_image,
    banana_mask,
    brown_lower=(0, 45, 20),
    brown_upper=(18, 255, 210),
    brown_value_threshold=165,
    dark_value_threshold=85,
    inner_kernel_size=7,
):
    """Find dark brown or very dark pixels inside the estimated banana surface.

    Potential blemishes are defined by dark brown hue/saturation characteristics
    or very low brightness. Brown pixels must also be dark; this avoids treating
    healthy yellow peel as a blemish. Eroding the banana mask excludes the peel
    boundary, which is commonly dark because of shadows rather than damage.
    """
    if hsv_image is None or banana_mask is None:
        raise ValueError("HSV image and banana mask are required.")

    inner_kernel = np.ones((inner_kernel_size, inner_kernel_size), dtype=np.uint8)
    inner_banana_mask = cv2.erode(banana_mask, inner_kernel, iterations=1)

    brown_mask = cv2.inRange(
        hsv_image,
        np.array(brown_lower, dtype=np.uint8),
        np.array(brown_upper, dtype=np.uint8),
    )
    value_channel = hsv_image[:, :, 2]
    brown_value_mask = cv2.inRange(value_channel, 0, brown_value_threshold)
    dark_mask = cv2.inRange(value_channel, 0, dark_value_threshold)
    dark_brown_mask = cv2.bitwise_and(brown_mask, brown_value_mask)
    candidate_mask = cv2.bitwise_or(dark_brown_mask, dark_mask)

    return cv2.bitwise_and(candidate_mask, inner_banana_mask)
