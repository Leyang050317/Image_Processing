import cv2
import numpy as np


def refine_blemish_mask(
    mask,
    kernel_size=3,
    open_iterations=1,
    close_iterations=1,
    min_region_area=12,
):
    """Remove isolated noise, close small blemish gaps, and discard tiny blobs."""
    if mask is None:
        raise ValueError("Input mask is None.")
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive.")

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    refined_mask = cv2.morphologyEx(
        binary_mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations
    )
    refined_mask = cv2.morphologyEx(
        refined_mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations
    )

    contours, _ = cv2.findContours(
        refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cleaned_mask = np.zeros_like(refined_mask)
    for contour in contours:
        if cv2.contourArea(contour) >= min_region_area:
            cv2.drawContours(cleaned_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return cleaned_mask
